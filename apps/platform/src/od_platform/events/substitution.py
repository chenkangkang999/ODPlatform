#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : substitution.py
# @Project   : ODPlatform / events
# @Function  : 换人事件的判定（第四官员举换人牌 + 边线邻近判定）

"""换人事件的判定。

★ 能力边界（同 ``od_platform.events`` 的说明, 这里再强调一次）

    换人靠的是**第四官员举起的换人牌**。要检出它, 就必须有一个训练了该
    （或"牌/牌子"一类）类别的模型 —— 当前足球权重只有 ``player`` / ``football``,
    **检不出换人牌, 所以本模块在现有权重上不会产出任何事件**。

★ 判据链路

    1. **主判据**: 检出换人牌, 且检出框的形状/置信度合格;
    2. **位置判据（可选、但推荐）**: 换人发生在**边线**（球场两条长边）附近,
       所以把牌的**接地点**（框下边缘中点 —— 不是框中心, 中心在透视下会系统性
       偏向场内）用场地单应映射成米制坐标, 再看到边线的距离是否在阈值内;
    3. **没有可信标定时必须降级**: 只报**候选**, 并在判据里明写"未标定",
       置信度也被压住 —— **不假装确定**, 这是本模块最重要的一条纪律。

    ★ 关于"可信标定": 只看 ``homography is not None`` 是不够的。
      :class:`~od_platform.geometric.homography.FieldHomography` 允许用两组点
      随手构造（``src_points`` / ``dst_points``）, 那种映射**没有场上物理意义**,
      其 ``to_audit()["calibrated"]`` 为 False。本模块认这个标记: 不可信的标定
      与"没有标定"同等对待, 一律降级。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from od_platform.events import (
    KIND_SEVERITY,
    KIND_SUBSTITUTION,
    ROLE_SUBSTITUTION_BOARD,
    Detection,
    PenaltyEvent,
    cap_confidence,
    image_size_of,
    in_area_ratio_range,
    parse_detections,
)
from od_platform.events.settings import EventSettings
from od_platform.geometric.pitch import DEFAULT_PITCH, PitchSpec

logger = logging.getLogger(__name__)

_KIND_NAME = "换人"


class SubstitutionDetector:
    """逐帧判定"是否发生了换人"。

    用法::

        detector = SubstitutionDetector(EventSettings.load(), homography=my_homography)
        for index, dets in enumerate(source):
            for event in detector.detect(dets, frame_index=index):
                logger.info(event.describe())

    切换视频/输入源时必须调 :meth:`reset`。

    Args:
        settings: 全部阈值; None 时用默认值
        homography: 场地单应（``FieldHomography``）; **可以为 None** —— 此时
            位置判据做不了, 事件降级为候选而不是猜
        pitch: 场地尺寸。**必须与标定所用的一致**, 否则边线位置会算错;
            默认用 :data:`~od_platform.geometric.pitch.DEFAULT_PITCH`
    """

    def __init__(self, settings: Optional[EventSettings] = None, *,
                 homography: Any = None, pitch: Optional[PitchSpec] = None):
        self.settings = settings or EventSettings()
        self.homography = homography
        self.pitch = pitch or DEFAULT_PITCH
        self._last_frame_of_kind: Dict[str, int] = {}
        self._tick = 0

    # ---------------------------------------------------------------- 状态
    def reset(self) -> None:
        """清空冷却记忆。"""
        self._last_frame_of_kind.clear()
        self._tick = 0

    def _next_tick(self, frame_index: Optional[int]) -> int:
        if frame_index is not None:
            return int(frame_index)
        self._tick += 1
        return self._tick

    # ---------------------------------------------------------------- 主流程
    def detect(self, raw_detections: Any, *, frame: Any = None,
               frame_index: Optional[int] = None) -> List[PenaltyEvent]:
        """判定当前帧的换人事件。

        Args:
            raw_detections: ``[{"cls","conf","xyxy"}]``（与 from_yolo_results 同格式）
            frame: BGR 图像; 只在需要按"面积占画面比"过滤时用, 不给也能跑
                （那一步校验会被跳过, 而不是把"不知道画面尺寸"当成不合格）
            frame_index: 帧号; 用于冷却。不给则按内部调用计数

        Returns:
            本帧确认或降级的事件（可能为空）
        """
        detections = parse_detections(raw_detections)
        if not detections:
            return []
        cfg = self.settings.substitution
        classes = self.settings.classes
        image_size = image_size_of(frame)
        tick = self._next_tick(frame_index)

        boards = [det for det in detections
                  if classes.resolve_role(det.cls) == ROLE_SUBSTITUTION_BOARD]
        if not boards:
            return []

        calibrated, calibration_evidence = self._calibration_state()
        events: List[PenaltyEvent] = []
        for board in boards:
            reject = self._reject_reason(board, image_size)
            if reject:
                logger.debug("跳过疑似换人牌 %s: %s", board.cls, reject)
                continue

            if not self._cooldown_allows(tick):
                continue

            reasons: List[str] = [f"检出换人牌(类别 {board.cls})"]
            evidence: Dict[str, Any] = {
                "board_class": board.cls,
                "detection_confidence": board.conf,
                "aspect_ratio": round(board.aspect_ratio(), 4),
                "substitution_min_area_ratio": cfg.min_area_ratio,
                "substitution_max_area_ratio": cfg.max_area_ratio,
                "touchline_margin_m": cfg.touchline_margin_m,
                "cooldown_frames": cfg.cooldown_frames,
                **calibration_evidence,
            }

            if calibrated:
                position = self._measure_touchline_distance(board)
                evidence.update(position["evidence"])
                if position["ok"] and cfg.require_touchline_proximity \
                        and not position["near_touchline"]:
                    # 标定可用且明确"不在边线附近" → 不构成换人
                    logger.debug("换人牌距边线 %.2f m 超过阈值 %.2f m, 不构成换人",
                                 position["distance_m"], cfg.touchline_margin_m)
                    continue
                confirmed = True
                reasons.append(
                    f"接地点距边线 {position['distance_m']:.2f} m"
                    f"(阈值 {cfg.touchline_margin_m:g} m), 位于场边, 判定为{_KIND_NAME}")
            elif not cfg.allow_uncalibrated_candidates:
                logger.debug("未标定且配置不允许无标定候选, 跳过换人牌 %s", board.cls)
                continue
            else:
                # 没有可信标定 → 只报候选, 绝不假装确定
                confirmed = not cfg.require_calibration
                reasons.append(
                    "未标定（或标定不可信）: 无法判断是否靠近边线, "
                    + ("只能作为候选, 待人工确认" if not confirmed
                       else "按配置不要求标定, 直接采信检出"))

            ceiling = cfg.unconfirmed_confidence_ceiling
            base_conf = board.conf if board.conf is not None else ceiling
            evidence["severity"] = KIND_SEVERITY.get(KIND_SUBSTITUTION, 0)
            events.append(PenaltyEvent(
                kind=KIND_SUBSTITUTION, confirmed=confirmed,
                confidence=cap_confidence(base_conf, confirmed, ceiling),
                bbox=board.bbox, frame_index=frame_index, reasons=tuple(reasons),
                evidence=evidence,
            ))
            self._last_frame_of_kind[KIND_SUBSTITUTION] = tick
        return events

    # ---------------------------------------------------------------- 标定状态
    def _calibration_state(self) -> Tuple[bool, Dict[str, Any]]:
        """标定是否**可信**。不可信与没有标定同等对待。"""
        if self.homography is None:
            return False, {"calibrated": False, "calibration_source": "none",
                           "calibration_note": "未提供场地单应(homography=None)"}
        try:
            audit = dict(self.homography.to_audit())
        except Exception as exc:              # noqa: BLE001 —— 不是本模块的接口
            return False, {"calibrated": False, "calibration_source": "unknown",
                           "calibration_note": f"单应缺少可读的 to_audit(): {exc}"}
        calibrated = bool(audit.get("calibrated"))
        return calibrated, {
            "calibrated": calibrated,
            "calibration_source": audit.get("source"),
            "calibration_points": audit.get("points"),
            "calibration_reprojection_error_m": audit.get("reprojection_error_m"),
            "calibration_note": audit.get("notes"),
        }

    # ---------------------------------------------------------------- 位置判据
    def _measure_touchline_distance(self, board: Detection) -> Dict[str, Any]:
        """把牌的接地点映射成场地坐标, 算它到最近一条边线的距离。

        Returns:
            ``{"ok": bool, "distance_m": float, "near_touchline": bool, "evidence": {...}}``
            ``ok=False`` 表示映射失败（点不可映射/落在无效区域）, 调用方应据此降级。
        """
        x_px, y_px = board.bottom_centre
        evidence: Dict[str, Any] = {
            "board_bottom_centre_px": [round(x_px, 2), round(y_px, 2)],
            "pitch_length_m": self.pitch.length,
            "pitch_width_m": self.pitch.width,
        }
        try:
            metre_x, metre_y = self.homography.pixel_to_meter(x_px, y_px)
        except Exception as exc:              # noqa: BLE001 —— 单应可能拒绝对某些点求值
            evidence["position_error"] = f"{type(exc).__name__}: {exc}"
            return {"ok": False, "distance_m": float("inf"),
                    "near_touchline": False, "evidence": evidence}
        if metre_x is None or metre_y is None:
            evidence["position_error"] = "映射结果为空"
            return {"ok": False, "distance_m": float("inf"),
                    "near_touchline": False, "evidence": evidence}

        # 边线是球场的两条【长边】(y = 0 与 y = 宽度); 点即使在场地外也按到线的距离算
        distance = min(abs(float(metre_y)), abs(self.pitch.width - float(metre_y)))
        near = distance <= self.settings.substitution.touchline_margin_m
        evidence.update({
            "board_position_m": [round(float(metre_x), 3), round(float(metre_y), 3)],
            "distance_to_touchline_m": round(distance, 4),
            "near_touchline": near,
            "inside_pitch": self.pitch.within(float(metre_x), float(metre_y)),
        })
        return {"ok": True, "distance_m": distance, "near_touchline": near,
                "evidence": evidence}

    # ---------------------------------------------------------------- 内部工具
    def _reject_reason(self, board: Detection,
                       image_size: Optional[Tuple[int, int]]) -> str:
        """"这压根不像换人牌"的判断; 返回空串表示通过。

        不知道画面尺寸时, 面积比的校验会被**跳过**而不是判成不合格 ——
        "不知道"不等于"不符合", 这条纪律和 Detection.conf 的处理一致。
        """
        cfg = self.settings.substitution
        if board.conf is not None and board.conf < cfg.min_confidence:
            return f"置信度 {board.conf:.3f} 低于阈值 {cfg.min_confidence:g}"
        ratio_ok = in_area_ratio_range(board, cfg.min_area_ratio, cfg.max_area_ratio,
                                       image_size)
        if ratio_ok is False:
            return (f"面积占比 {board.normalised_area(image_size):.2e} 不在 "
                    f"[{cfg.min_area_ratio:g}, {cfg.max_area_ratio:g}] 内")
        return ""

    def _cooldown_allows(self, tick: int) -> bool:
        last = self._last_frame_of_kind.get(KIND_SUBSTITUTION)
        return last is None or (tick - last) >= self.settings.substitution.cooldown_frames