#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""几何分析器 —— 把"逐帧检测框"变成"带证据的场上事件候选"。

这是几何层的算法入口: 场上的每一帧进来, 出去的是
【接触 / 出界 / 进球 / 越位位置】四类候选, 每条都带着证据和"还缺什么"。

★★ 最重要的行为约定: 【没有标定就什么都不报】★★

    历史版本的 PitchAnalyzer 用"整幅画面四角 ↔ 球场四角"造了一个假的单应性,
    于是把像素当米用, 输出了一堆看起来合理、实际毫无意义的"米"。
    本版本:
        · 标定必须来自显式来源（配置 / sidecar 文件 / 标定目录）;
        · 没有标定 → calibrated=False, 四类候选一律不产出, 并在 summary 里说明原因;
        · calibrated_pitch 这个能力标志因此不再撒谎。

★ 所有可调参数来自 GeometrySettings / TrackingSettings（对应 football.yaml 的
  pitch / calibration / rules / analyzer / tracking 各节）, 本文件里不出现
  任何阈值字面量。

配套:
    · tracking.MultiObjectTracker  —— 维护跨帧轨迹（接触去重、球归属历史需要它）
    · geometric.rules.RuleEngine   —— 规则判据（越位 / 出界 / 进球 / 接触）
    · geometric.calibration        —— 标定及其质量校验
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from od_platform.common.config_utils import read_section
from od_platform.geometric.calibration import (
    CalibrationError, PitchCalibration, default_calibration_dir, discover,
)
from od_platform.geometric.homography import FieldHomography
from od_platform.geometric.rules import RuleEngine
from od_platform.geometric.settings import GeometrySettings, default_config_path
from od_platform.tracking.settings import TrackingSettings
from od_platform.tracking.tracker import MultiObjectTracker

logger = logging.getLogger(__name__)

Point = Tuple[float, float]


class PitchAnalyzer:
    """场地几何 + 规则候选分析器。

    Args:
        width, height: 画面尺寸（像素）; 用于校验标定记录的机位分辨率
        platform_root: ODPlatform 仓库根（保留参数, 兼容旧调用）
        config: football.yaml 的内容; 为 None 时自动加载 configs/runtime/football.yaml
        media_path: 输入素材路径; 用于发现同名 sidecar 标定文件
        calibration: 直接给一份标定（优先级最高, 测试与离线分析用）
        settings: 直接给一套 GeometrySettings（给了就不再从 config 构造）

    注意: 本类【不抛异常】。构造失败只是 ok=False + skip_reason, 由调用方决定
    怎么降级 —— 几何分析失败不应该让整个推理任务失败。
    """

    def __init__(self, width: int, height: int, platform_root: str = "",
                 config: Optional[Mapping[str, Any]] = None, *,
                 media_path: Optional[Path] = None,
                 calibration: Optional[PitchCalibration] = None,
                 settings: Optional[GeometrySettings] = None,
                 tracking_settings: Optional[TrackingSettings] = None) -> None:
        self.width = int(width)
        self.height = int(height)
        self.platform_root = platform_root
        self.media_path = Path(media_path) if media_path else None

        self.ok = False
        self.calibrated = False
        self.skip_reason: Optional[str] = None
        self.calibration: Optional[PitchCalibration] = None
        self.homography: Optional[FieldHomography] = None
        self.rules: Optional[RuleEngine] = None

        self._cooldown: Dict[Any, int] = {}
        self._event_counts: Dict[str, int] = {}
        self._last_touch_team: Optional[str] = None
        self._last_touch_frame = 0
        self._offside_skip_reason: Optional[str] = None

        self._raw_config: Mapping[str, Any] = (
            config if config is not None else _load_config_file())
        try:
            self.settings = settings or GeometrySettings.from_config(self._raw_config)
        except (ValueError, TypeError) as exc:
            logger.warning("几何配置不合法, 全部改用默认值: %s", exc)
            self.settings = GeometrySettings()

        try:
            tracker_settings = tracking_settings or TrackingSettings.from_app_config(
                self._raw_config)
            self._tracker: Optional[MultiObjectTracker] = MultiObjectTracker(tracker_settings)
        except (ValueError, TypeError) as exc:
            logger.warning("跟踪配置不合法, 退化为无轨迹模式: %s", exc)
            self._tracker = None

        try:
            self._init_geometry(calibration)
        except Exception as exc:  # noqa: BLE001 —— 构造期绝不抛
            self.skip_reason = f"几何模块初始化失败: {type(exc).__name__}: {exc}"
            logger.warning("[PitchAnalyzer] %s", self.skip_reason)

    # ------------------------------------------------------------------ 初始化

    def _init_geometry(self, calibration: Optional[PitchCalibration]) -> None:
        calibration = (calibration
                       or self._calibration_from_config()
                       or self._discover_calibration())
        if calibration is None:
            self.skip_reason = (
                "未找到球场标定（配置 homography.points / 素材同名 "
                "<stem>.calibration.json / configs/runtime/calibrations/）—— "
                "为避免把像素当米用, 几何事件一律不产出")
            logger.warning("[PitchAnalyzer] %s", self.skip_reason)
            return

        if calibration.image_size and tuple(calibration.image_size) != (self.width, self.height):
            self.skip_reason = (f"标定记录的分辨率 {calibration.image_size} 与当前画面 "
                                f"{(self.width, self.height)} 不一致, 拒绝使用")
            logger.warning("[PitchAnalyzer] %s", self.skip_reason)
            return

        self.calibration = calibration
        self.homography = FieldHomography.from_calibration(calibration)
        self.rules = RuleEngine(homography=self.homography, settings=self.settings.rules)
        # ok 与 calibrated 同义: 只有真标定才算"几何可用"
        self.ok = bool(self.homography.calibrated and self.rules.usable)
        self.calibrated = self.ok
        if not self.ok:
            self.skip_reason = self.rules.skip_reason
        logger.info("几何分析器就绪: 标定来源=%s 重投影误差=%.3f m",
                    calibration.source, calibration.reprojection_error_m)

    def _calibration_from_config(self) -> Optional[PitchCalibration]:
        section = read_section(self._raw_config, "homography")
        if not section:
            return None
        file_name = section.get("calibration_file")
        if file_name:
            path = Path(file_name)
            if not path.is_absolute():
                path = default_config_path().parent / path
            return PitchCalibration.load(path, pitch=self.settings.pitch,
                                         settings=self.settings.calibration)
        if not (section.get("points") or section.get("image_points")):
            return None
        return PitchCalibration.from_config(section, pitch=self.settings.pitch,
                                            settings=self.settings.calibration)

    def _discover_calibration(self) -> Optional[PitchCalibration]:
        image_size = (self.width, self.height)
        if self.media_path is not None:
            found = discover(self.media_path, image_size=image_size,
                             pitch=self.settings.pitch,
                             settings=self.settings.calibration)
            if found is not None:
                return found
        # 没有素材路径时（例如直接拿像素尺寸构造）: 在标定目录里找
        # 唯一一个分辨率匹配的标定。多份都匹配就不能猜, 交给调用方指定。
        directory = default_calibration_dir()
        if not directory.is_dir():
            return None
        matches: List[PitchCalibration] = []
        for path in sorted(directory.glob("*.json")):
            try:
                calibration = PitchCalibration.load(path, pitch=self.settings.pitch,
                                                    settings=self.settings.calibration)
            except CalibrationError as exc:
                logger.warning("标定文件 %s 不可用: %s", path, exc)
                continue
            if calibration.image_size and tuple(calibration.image_size) == image_size:
                matches.append(calibration)
        if len(matches) == 1:
            logger.info("按画面分辨率匹配到唯一标定（来源 %s）", matches[0].source)
            return matches[0]
        if len(matches) > 1:
            logger.warning("标定目录里有 %d 份分辨率匹配的标定, 无法确定用哪一份; "
                           "请用 sidecar 或 homography.calibration_file 指定", len(matches))
        return None

    # ------------------------------------------------------------------ 主入口

    def analyze(self, frame: Dict[str, Any]) -> List[Dict[str, Any]]:
        """分析一帧, 返回事件候选列表（未标定时恒为空）。"""
        if not self.ok or self.rules is None or self.homography is None:
            return []
        boxes = [box for box in frame.get("boxes", []) if isinstance(box, dict)]
        if not boxes:
            return []

        frame_index = int(frame.get("frame_index", 0))
        timestamp = frame.get("timestamp")
        if not self._cooldown:            # 首次分析: 以本帧为"从未触球"的基准
            self._last_touch_frame = frame_index - self.settings.analyzer.possession_memory_frames

        track_ids = self._track(boxes, frame_index)
        players: List[Tuple[Point, Optional[str], Any]] = []
        ball: Optional[Point] = None
        for index, box in enumerate(boxes):
            try:
                meter = self.homography.pixel_to_meter(float(box["cx"]), float(box["y2"]))
            except (KeyError, TypeError, ValueError):
                continue
            role = str(box.get("cls", ""))
            track_id = track_ids.get(index, box.get("track_id"))
            if role == "player":
                players.append((meter, box.get("team"), track_id))
            elif role == "ball":
                ball = meter

        events: List[Dict[str, Any]] = []
        events += self._offside_events(players, ball, frame_index, timestamp)
        events += self._contact_events(players, frame_index, timestamp)
        events += self._ball_events(players, ball, frame_index, timestamp)
        for event in events:
            self._event_counts[event["type"]] = self._event_counts.get(event["type"], 0) + 1
        return events

    # ------------------------------------------------------------------ 跟踪

    def _track(self, boxes: Sequence[Dict[str, Any]], frame_index: int) -> Dict[int, int]:
        """用自研跟踪器给本帧框分配稳定 ID（调用方已有 ID 时只作补充）。"""
        if self._tracker is None:
            return {}
        detections = [{"cls_name": str(box.get("cls", "")),
                       "bbox": [float(box.get(k, 0.0)) for k in ("x1", "y1", "x2", "y2")],
                       "conf": float(box.get("conf", 0.0))} for box in boxes]
        try:
            self._tracker.update(detections, frame_index)
        except (ValueError, TypeError) as exc:   # 跟踪失败不应拖垮几何分析
            logger.warning("跟踪更新失败（frame %s）: %s", frame_index, exc)
            return {}
        return dict(self._tracker.last_assignment)

    # ------------------------------------------------------------------ 越位

    def _offside_events(self, players, ball, frame_index, timestamp) -> List[Dict[str, Any]]:
        if self.rules is None or ball is None:
            return []
        teams: Dict[str, List[Tuple[Point, Any]]] = {"A": [], "B": []}
        for position, team, track_id in players:
            if team in ("A", "B"):
                teams[team].append((position, track_id))
        if not teams["A"] or not teams["B"]:
            self._note_offside_skip(
                "越位需要球队归属（A/B）, 当前检测结果里没有可用的队伍标签")
            return []

        rule_settings = self.settings.rules
        attacking = str(rule_settings.attacking_team).upper()
        if attacking not in ("A", "B"):
            attacking = self._team_nearest_to(teams, ball)
            if attacking is None:
                return []
        defending = "B" if attacking == "A" else "A"

        direction = str(rule_settings.attacking_direction).lower()
        direction_inferred = direction not in ("left", "right")
        if direction_inferred:
            direction = self._infer_attacking_direction(
                teams[attacking], ball,
                self.settings.analyzer.min_direction_delta_m)
            if direction is None:
                self._note_offside_skip(
                    "进攻方向无法由站位推断（球与该队平均位置几乎重合）, "
                    "请用 rules.attacking_direction 显式指定")
                return []

        candidates = self.rules.check_offside(
            [position for position, _ in teams[attacking]],
            [position for position, _ in teams[defending]],
            ball, attacking_direction=direction)
        events: List[Dict[str, Any]] = []
        for candidate in candidates:
            index = candidate["player_idx"]
            track_id = teams[attacking][index][1]
            key = ("offside", track_id if track_id is not None else index)
            if not self._passes_cooldown(key, frame_index):
                continue
            events.append({
                "type": "offside",
                "frame_index": frame_index,
                "timestamp": timestamp,
                "track_id": track_id,
                "team": attacking,
                "attacking_direction": direction,
                "attacking_direction_inferred": direction_inferred,
                **{k: v for k, v in candidate.items() if k != "player_idx"},
            })
        return events

    @staticmethod
    def _team_nearest_to(teams, ball) -> Optional[str]:
        """离球最近的队伍视为控球方（进攻方）; 没有球员时返回 None。"""
        best_team, best_distance = None, float("inf")
        for team, members in teams.items():
            for position, _ in members:
                distance = float(np.hypot(position[0] - ball[0], position[1] - ball[1]))
                if distance < best_distance:
                    best_team, best_distance = team, distance
        return best_team

    @staticmethod
    def _infer_attacking_direction(members, ball, min_delta_m: float) -> Optional[str]:
        """球在该队平均位置的那一侧, 就是他们进攻的方向。

        站姿推断天然有歧义, 所以只有当球明显偏离该队平均位置（超过 min_delta_m）
        时才给结论; 否则返回 None, 由上层记为"证据不足"而不是猜一个方向。
        """
        if not members:
            return None
        mean_x = float(np.mean([position[0] for position, _ in members]))
        if abs(ball[0] - mean_x) < min_delta_m:
            return None
        return "right" if ball[0] > mean_x else "left"

    def _note_offside_skip(self, reason: str) -> None:
        if self._offside_skip_reason is None:
            self._offside_skip_reason = reason
            logger.info("[PitchAnalyzer] %s —— 越位候选不会产出", reason)

    # ------------------------------------------------------------------ 接触

    def _contact_events(self, players, frame_index, timestamp) -> List[Dict[str, Any]]:
        if self.rules is None or len(players) < 2:
            return []
        positions = [position for position, _, _ in players]
        events: List[Dict[str, Any]] = []
        for contact in self.rules.check_collision(positions):
            track_a = players[contact["p1"]][2]
            track_b = players[contact["p2"]][2]
            key = ("contact",) + tuple(sorted(
                (str(track_a), str(track_b)) if track_a is not None and track_b is not None
                else (f"idx{contact['p1']}", f"idx{contact['p2']}")))
            if not self._passes_cooldown(key, frame_index):
                continue
            events.append({
                "type": "collision",
                "frame_index": frame_index,
                "timestamp": timestamp,
                "track_a": track_a,
                "track_b": track_b,
                "distance_m": contact["dist_m"],
                "position_m": contact["position"],
                "decision_status": contact["decision_status"],
                "requires_evidence": contact["requires_evidence"],
                "note": contact["note"],
            })
        return events

    # ------------------------------------------------------------------ 出界 / 进球

    def _ball_events(self, players, ball, frame_index, timestamp) -> List[Dict[str, Any]]:
        if self.rules is None or ball is None:
            return []
        rule_settings = self.settings.rules
        tolerance = rule_settings.ball_assignment_tolerance_m
        nearest_team, nearest = None, float("inf")
        for position, team, _ in players:
            distance = self.homography.distance_m(position, ball)
            if distance < nearest:
                nearest_team, nearest = team, distance
        if nearest <= tolerance and nearest_team in ("A", "B"):
            self._last_touch_team = nearest_team
            self._last_touch_frame = frame_index

        last_touch = (self._last_touch_team
                      if frame_index - self._last_touch_frame
                      <= self.settings.analyzer.possession_memory_frames
                      else None)

        events: List[Dict[str, Any]] = []
        goal = self.rules.check_goal(ball, last_touch_team=last_touch)
        if goal is not None and self._passes_cooldown(("goal", goal["side"]), frame_index):
            events.append({"type": "goal", "frame_index": frame_index,
                           "timestamp": timestamp, "last_touch_team": last_touch,
                           "last_touch_inferred": True, **goal})

        out = self.rules.check_ball_out(
            ball, last_touch_team=last_touch,
            goal_defending_team=rule_settings.goal_defending_team)
        if out is not None and self._passes_cooldown(("ball_out", out["side"]), frame_index):
            events.append({"type": "ball_out", "frame_index": frame_index,
                           "timestamp": timestamp, "last_touch_inferred": True, **out})
        return events

    # ------------------------------------------------------------------ 工具

    def _passes_cooldown(self, key: Any, frame_index: int) -> bool:
        last = self._cooldown.get(key)
        if last is not None and frame_index - last < self.settings.analyzer.event_cooldown_frames:
            return False
        self._cooldown[key] = frame_index
        return True

    def summary(self) -> Dict[str, Any]:
        """给结果文件 / 审计用的分析器自述（含本次生效的全部参数）。"""
        return {
            "ok": self.ok,
            "calibrated": self.calibrated,
            "skip_reason": self.skip_reason,
            "calibration": self.calibration.quality() if self.calibration else None,
            "tracker": self._tracker.summary() if self._tracker else None,
            "event_counts": dict(self._event_counts),
            "offside_skipped_reason": self._offside_skip_reason,
            "settings": self.settings.to_audit(),
        }


def _load_config_file() -> Mapping[str, Any]:
    """默认读取 configs/runtime/football.yaml（此前这个文件从未被加载过）。"""
    path = default_config_path()
    if not path.is_file():
        logger.info("未找到 %s, 使用内置默认配置", path)
        return {}
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{path} 顶层不是 mapping: {type(data).__name__}")
    logger.info("已加载运行配置: %s", path)
    return data