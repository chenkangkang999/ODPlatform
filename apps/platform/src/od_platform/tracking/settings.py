#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""跟踪子系统的配置模型 —— 所有可调数字的唯一声明处。

对应 football.yaml 的 tracking: 节::

    tracking:
      backend: botsort
      min_confidence: 0.25
      high_threshold: 0.5
      iou_threshold: 0.45
      max_age: 30
      min_hits: 3
      max_tracks: 256
      small_object_extent_px: 48
      center_distance_gate: 1.0
      history_length: 64
      distance_cost_base: 1.0
      invalid_cost: 10.0
      kalman:
        std_weight_position: 0.05
        std_weight_velocity: 0.00625
        initiate_position_factor: 2.0
        initiate_velocity_factor: 10.0
        aspect_std: 0.01
        aspect_velocity_std: 1.0e-5
        measurement_aspect_std: 0.1

设计说明:
    · 卡尔曼噪声权重沿用 SORT/DeepSORT 的经典取值（位置 1/20、速度 1/160）,
      但它们仍然是【可配置项】—— 不同分辨率的视频需要不同的过程噪声。
    · distance_cost_base / invalid_cost 不是"调参", 而是代价矩阵的【结构常量】:
      IoU 代价落在 [0,1), 中心距离兜底代价落在 [distance_cost_base, 2),
      invalid_cost 必须大于任何有效代价, 用来表示"禁止匹配"。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from od_platform.common.config_utils import build_settings, report_unknown_keys

# 出厂默认的主题色 —— 这里只声明一次, 供文档与测试引用
DEFAULT_BACKEND = "botsort"
SUPPORTED_BACKENDS = ("botsort", "bytetrack")


@dataclass(frozen=True)
class KalmanSettings:
    """卡尔曼滤波的过程/观测噪声参数。"""

    std_weight_position: float = 1.0 / 20.0
    std_weight_velocity: float = 1.0 / 160.0
    initiate_position_factor: float = 2.0      # 新建轨迹时位置方差放大倍数
    initiate_velocity_factor: float = 10.0     # 新建轨迹时速度方差放大倍数
    aspect_std: float = 1.0e-2                 # 宽高比的过程噪声
    aspect_velocity_std: float = 1.0e-5        # 宽高比速度的过程噪声
    measurement_aspect_std: float = 1.0e-1     # 宽高比的观测噪声


@dataclass(frozen=True)
class TrackingSettings:
    """多目标跟踪器的全部可调参数。"""

    backend: str = DEFAULT_BACKEND
    min_confidence: float = 0.25        # 进入跟踪的最低检测置信度
    high_threshold: float = 0.5         # 高分阶段阈值（ByteTrack 的 track_thresh）
    iou_threshold: float = 0.45         # 关联所需 IoU 下限
    max_age: int = 30                   # 丢失后保持多少帧
    min_hits: int = 3                   # 连续命中多少帧才确认
    max_tracks: int = 256               # 活跃轨迹上限（保护内存）
    small_object_extent_px: float = 48.0   # 小于该尺度的目标允许用中心距离兜底
    center_distance_gate: float = 1.0      # 中心距离兜底的归一化上限（相对目标尺度）
    history_length: int = 64            # 每条轨迹保留多少帧历史（有界内存）
    distance_cost_base: float = 1.0     # 中心距离兜底代价的起点（保证排在 IoU 之后）
    invalid_cost: float = 10.0          # 禁止匹配的哨兵代价, 必须大于任何有效代价
    kalman: KalmanSettings = field(default_factory=KalmanSettings)

    def __post_init__(self) -> None:
        if self.backend.lower() not in SUPPORTED_BACKENDS:
            raise ValueError(f"不支持的跟踪后端: {self.backend!r}（可选 {SUPPORTED_BACKENDS}）")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold 必须在 [0, 1], 收到 {self.iou_threshold}")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(f"min_confidence 必须在 [0, 1], 收到 {self.min_confidence}")
        if self.high_threshold < self.min_confidence:
            raise ValueError(
                f"high_threshold({self.high_threshold}) 不能低于 "
                f"min_confidence({self.min_confidence})")
        if self.max_age < 0:
            raise ValueError("max_age 不能为负")
        if self.min_hits < 1:
            raise ValueError("min_hits 至少为 1")
        if self.max_tracks < 1:
            raise ValueError("max_tracks 至少为 1")
        if self.small_object_extent_px <= 0 or self.center_distance_gate <= 0:
            raise ValueError("small_object_extent_px 与 center_distance_gate 必须为正数")
        if self.history_length < 1:
            raise ValueError("history_length 至少为 1")
        if self.invalid_cost <= self.distance_cost_base + 1.0:
            raise ValueError(
                f"invalid_cost({self.invalid_cost}) 必须明显大于任何有效代价"
                f"（有效代价上限约为 {self.distance_cost_base + 1.0}）")

    @classmethod
    def from_config(cls, section: Any, *, where: str = "football.yaml 的 tracking 节"
                    ) -> "TrackingSettings":
        """从 football.yaml 的 tracking 节构造（未给的字段走默认值）。"""
        unknown: list = []
        kalman_section = section.get("kalman") if isinstance(section, dict) else None
        settings = build_settings(cls, _without(section, "kalman"), unknown_sink=unknown)
        report_unknown_keys(where, unknown)
        if isinstance(kalman_section, dict):
            kalman_unknown: list = []
            kalman = build_settings(KalmanSettings, kalman_section, unknown_sink=kalman_unknown)
            report_unknown_keys(f"{where} 的 kalman 子节", kalman_unknown)
            settings = TrackingSettings(
                **{**{f: getattr(settings, f) for f in settings.__dataclass_fields__
                      if f != "kalman"}, "kalman": kalman})
        return settings

    @classmethod
    def from_app_config(cls, config: Any) -> "TrackingSettings":
        """从整份 football.yaml 内容里取 tracking 节。"""
        from od_platform.common.config_utils import read_section
        return cls.from_config(read_section(config, "tracking"))


def _without(section: Any, key: str) -> Optional[dict]:
    if not isinstance(section, dict):
        return None
    return {k: v for k, v in section.items() if k != key}