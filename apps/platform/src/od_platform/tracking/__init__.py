"""多目标跟踪子系统 —— 自实现的 ByteTrack 风格跟踪（两阶段关联 + 卡尔曼滤波）。

不依赖任何推理框架: 输入 list[dict] 检测、输出 list[Track]，可脱离 torch 单独测试。
所有参数来自 TrackingSettings（对应 football.yaml 的 tracking 节）。
"""
from .settings import DEFAULT_BACKEND, KalmanSettings, TrackingSettings
from .tracker import MultiObjectTracker, Track

__all__ = [
    "MultiObjectTracker",
    "Track",
    "TrackingSettings",
    "KalmanSettings",
    "DEFAULT_BACKEND",
]