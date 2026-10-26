#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""几何子系统的配置模型 —— 所有可调数字的唯一声明处。

设计原则（对应"全局不要硬编码"）:
    1. 默认值只写在 dataclass 字段上, 逻辑代码里【不出现任何字面量】;
    2. 每一项都能被 football.yaml 覆盖, 覆盖键名与字段同名;
    3. 配置缺项/类型不对时给出明确报错或回退默认, 不静默用半个配置。

football.yaml 里对应的节:
    pitch:        场地尺寸（默认 IFAB 国际比赛标准场地）
    calibration:  标定的质量门槛与求解参数
    rules:        越位/接触/出界/进球的判据阈值
    analyzer:     事件冷却、球归属记忆、方向推断灵敏度
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from od_platform.common.config_utils import build_settings, read_section, report_unknown_keys
from od_platform.geometric.pitch import PitchSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibrationSettings:
    """标定求解与质量校验参数。"""

    min_points: int = 4                  # 求解单应性所需的最少对应点
    max_reprojection_error_m: float = 1.0  # 重投影误差上限（米）, 超过即拒绝该标定
    ransac_reprojection_px: float = 5.0  # >=5 点时 RANSAC 的内点阈值（像素）
    full_frame_tolerance_px: float = 2.0  # 判定"画面四角凑数"的容差（像素）
    point_range_factor: float = 2.0      # 像素点允许超出画面的倍数（防点错坐标）
    finite_margin: float = 1e-6          # 场地坐标的浮点容差（米）
    coordinate_precision_px: int = 3     # 标定文件里像素坐标的小数位
    coordinate_precision_m: int = 4      # 标定文件里场地坐标的小数位


@dataclass(frozen=True)
class RuleSettings:
    """规则判据阈值。"""

    offside_margin_m: float = 0.0        # 越位容差（米）; 标定误差大时可放宽
    collision_distance_m: float = 2.0    # 接触候选的距离阈值（米）
    ball_assignment_tolerance_m: float = 0.5   # 用"谁离球最近"估计球归属的距离（米）
    ball_radius_m: float = 0.11          # 5 号球半径（米）, 整球过线判定的容差来源
    goal_line_tolerance_m: Optional[float] = None  # 门线容差; 不给则取 ball_radius_m
    attacking_direction: str = "auto"    # auto | left | right
    attacking_team: str = "auto"         # auto | A | B
    goal_defending_team: Optional[str] = None       # 该端球门的防守方; 不给则出界只能报 unknown

    def __post_init__(self) -> None:
        if self.collision_distance_m <= 0:
            raise ValueError("collision_distance_m 必须为正数")
        if self.ball_assignment_tolerance_m <= 0:
            raise ValueError("ball_assignment_tolerance_m 必须为正数")
        if self.offside_margin_m < 0:
            raise ValueError("offside_margin_m 不能为负")
        for name, value in (("attacking_direction", self.attacking_direction),
                            ("attacking_team", self.attacking_team)):
            allowed = ("auto", "left", "right") if name == "attacking_direction" else ("auto", "a", "b")
            if str(value).lower() not in allowed:
                raise ValueError(f"{name} 只能是 {allowed}, 收到 {value!r}")

    @property
    def effective_goal_line_tolerance_m(self) -> float:
        """门线容差: 没显式配就用球半径（整球过线的物理含义）。"""
        return (self.ball_radius_m if self.goal_line_tolerance_m is None
                else float(self.goal_line_tolerance_m))


@dataclass(frozen=True)
class AnalyzerSettings:
    """分析器的时序与去重参数。"""

    event_cooldown_frames: int = 30      # 同类事件在该帧数内只报一次
    possession_memory_frames: int = 15   # 球归属结论保留多少帧
    min_direction_delta_m: float = 1.0   # 推断进攻方向时球至少偏离多少米


@dataclass(frozen=True)
class GeometrySettings:
    """几何子系统的完整配置。"""

    pitch: PitchSpec = field(default_factory=PitchSpec)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    rules: RuleSettings = field(default_factory=RuleSettings)
    analyzer: AnalyzerSettings = field(default_factory=AnalyzerSettings)

    @classmethod
    def from_config(cls, config: Any, *, where: str = "football.yaml") -> "GeometrySettings":
        """从整份配置（football.yaml 的内容）构造。"""
        if not isinstance(config, dict):
            return cls()
        settings = cls(
            pitch=PitchSpec.from_config(read_section(config, "pitch")),
            calibration=_build(CalibrationSettings, config, "calibration", where),
            rules=_build(RuleSettings, config, "rules", where,
                         aliases={"offside_margin": "offside_margin_m",
                                  "collision_dist": "collision_distance_m",
                                  "ball_tol": "ball_assignment_tolerance_m",
                                  "goal_line_x_tolerance": "goal_line_tolerance_m"}),
            analyzer=_build(AnalyzerSettings, config, "analyzer", where),
        )
        if config.get("event_cooldown_frames") is not None:
            # 历史键曾经放在顶层, 继续认它, 避免旧配置静默失效
            settings = GeometrySettings(
                pitch=settings.pitch, calibration=settings.calibration,
                rules=settings.rules,
                analyzer=AnalyzerSettings(
                    event_cooldown_frames=int(config["event_cooldown_frames"]),
                    possession_memory_frames=settings.analyzer.possession_memory_frames,
                    min_direction_delta_m=settings.analyzer.min_direction_delta_m))
        return settings

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "GeometrySettings":
        """从 yaml 文件加载; 默认用 configs/runtime/football.yaml。"""
        path = Path(path) if path else default_config_path()
        if not path.is_file():
            logger.info("未找到几何配置 %s, 全部采用默认值", path)
            return cls()
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} 顶层不是 mapping: {type(data).__name__}")
        logger.info("已加载几何配置: %s", path)
        return cls.from_config(data, where=str(path))

    def to_audit(self) -> Dict[str, Any]:
        """给结果文件/审计用的配置快照（说明"这次是按什么参数跑的"）。"""
        return {
            "pitch": {"length": self.pitch.length, "width": self.pitch.width,
                      "goal_width": self.pitch.goal_width,
                      "penalty_area_depth": self.pitch.penalty_area_depth},
            "calibration": {"min_points": self.calibration.min_points,
                            "max_reprojection_error_m": self.calibration.max_reprojection_error_m,
                            "ransac_reprojection_px": self.calibration.ransac_reprojection_px},
            "rules": {"offside_margin_m": self.rules.offside_margin_m,
                      "collision_distance_m": self.rules.collision_distance_m,
                      "ball_assignment_tolerance_m": self.rules.ball_assignment_tolerance_m,
                      "goal_line_tolerance_m": self.rules.effective_goal_line_tolerance_m,
                      "attacking_direction": self.rules.attacking_direction,
                      "attacking_team": self.rules.attacking_team,
                      "goal_defending_team": self.rules.goal_defending_team},
            "analyzer": {"event_cooldown_frames": self.analyzer.event_cooldown_frames,
                         "possession_memory_frames": self.analyzer.possession_memory_frames,
                         "min_direction_delta_m": self.analyzer.min_direction_delta_m},
        }


def default_config_path() -> Path:
    """football.yaml 的默认位置。

    真正的定义在 :mod:`od_platform.common.football_config` —— 四个子系统
    (geometry / tracking / detection / action_recognition) 必须指向同一个文件,
    所以路径只在那里写一次。这里保留同名入口是为了兼容既有调用方。
    """
    from od_platform.common.football_config import default_config_path as canonical
    return canonical()


def _build(cls, config: Dict[str, Any], section: str, where: str,
           aliases: Optional[Dict[str, str]] = None):
    unknown: List[str] = []
    settings = build_settings(cls, read_section(config, section), aliases=aliases,
                              unknown_sink=unknown)
    report_unknown_keys(f"{where} 的 {section} 节", unknown)
    return settings