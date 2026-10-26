#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : settings.py
# @Project   : ODPlatform
# @Function  : 检测层参数（默认值只声明一次, 逻辑代码不出现字面量）

"""检测层可调参数 —— 对应 ``football.yaml`` 的 ``detection:`` 节。

★ 为什么要有这个模块

    这些值此前【写死在 ``apps/football-web/football_portal/bridge_worker.py`` 里】:
    ``conf=0.25``、``iou=0.7``(硬编码)、``imgsz=960``、``max_det=300``、
    ``max_frames=1500``、还有一串校验边界。后果是 ``football.yaml`` 的
    ``detection`` 节写着 ``iou_threshold: 0.45`` / ``imgsz: 1280`` ——
    **和实际跑的完全不是一回事**, 改配置没有任何效果。
    现在默认值只在本文件出现一次, 运行期可被任务的 request.json 覆盖。

约定:
    · 用户可调的量 → dataclass 字段, 配置键名与字段同名;
    · "哪些值明显不合法"的护栏 → 本模块的具名常量（不是业务决策, 但不许当魔法数字用）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from od_platform.common import paths
from od_platform.common.config_utils import build_settings, read_section, report_unknown_keys
from od_platform.common.football_config import load_config

SECTION = "detection"

# ---------------------------------------------------------------- 合法性护栏
# 这些不是"调参", 是"挡掉明显错的值"。有名字, 不许在逻辑里当字面量。
CONF_MIN: float = 0.001
CONF_MAX: float = 0.99
IOU_MIN: float = 0.0
IOU_MAX: float = 1.0
IMGSZ_MIN: int = 320
IMGSZ_MAX: int = 1536
IMGSZ_STEP: int = 32          # ultralytics 要求输入边长是 32 的倍数
STRIDE_MIN: int = 1
STRIDE_MAX: int = 30
MAX_FRAMES_MIN: int = 1
MAX_DET_MIN: int = 1


@dataclass(frozen=True)
class DetectionSettings:
    """检测（+跟踪器选择）的可调参数。

    字段说明见 ``football.yaml`` 的 ``detection:`` 节。
    """

    conf: float = 0.25            # 进入结果的置信度下限
    iou: float = 0.7              # NMS 的 IoU 阈值（此前硬编码在 Worker 里）
    imgsz: int = 960              # 推理输入边长
    max_det: int = 300            # 每帧最多保留多少个框
    stride: int = 1               # 视频抽帧步长
    max_frames: int = 1500        # 单任务处理帧数上限
    device: str = "cpu"           # "cpu" / "0" / "0,1"
    tracking: bool = True         # 视频是否开启多目标跟踪
    tracker: str = "botsort.yaml"  # 跟踪器配置名或路径, 见 resolved_tracker()
    classes: Tuple[int, ...] = (0, 1)   # 0=player, 1=football
    model: str = ""               # 默认权重（说明性; 实际由 Web 侧 FOOTBALL_MODEL_PATH 决定）

    def __post_init__(self) -> None:
        if not CONF_MIN <= self.conf <= CONF_MAX:
            raise ValueError(f"conf 必须在 [{CONF_MIN}, {CONF_MAX}] 内, 收到 {self.conf}")
        if not IOU_MIN < self.iou <= IOU_MAX:
            raise ValueError(f"iou 必须在 ({IOU_MIN}, {IOU_MAX}] 内, 收到 {self.iou}")
        if not IMGSZ_MIN <= self.imgsz <= IMGSZ_MAX or self.imgsz % IMGSZ_STEP:
            raise ValueError(
                f"imgsz 必须在 [{IMGSZ_MIN}, {IMGSZ_MAX}] 内且是 {IMGSZ_STEP} 的倍数, "
                f"收到 {self.imgsz}")
        if not STRIDE_MIN <= self.stride <= STRIDE_MAX:
            raise ValueError(f"stride 必须在 [{STRIDE_MIN}, {STRIDE_MAX}] 内, 收到 {self.stride}")
        if self.max_frames < MAX_FRAMES_MIN:
            raise ValueError(f"max_frames 必须 >= {MAX_FRAMES_MIN}, 收到 {self.max_frames}")
        if self.max_det < MAX_DET_MIN:
            raise ValueError(f"max_det 必须 >= {MAX_DET_MIN}, 收到 {self.max_det}")
        if not self.tracker:
            raise ValueError("tracker 不能为空")

    def resolved_tracker(self) -> str:
        """返回喂给 ultralytics 的 tracker 参数。

        若本仓库 ``configs/runtime/`` 下存在同名文件（例如 ``botsort.yaml``）就用它,
        否则把名字原样交给 ultralytics（它会在自己的 cfg/trackers 目录里找）。

        为什么这样做: 这样 BoT-SORT 的 10 个参数就能放在本仓库里被版本管理,
        又不必把绝对路径写进任何被 git 跟踪的文件（CI 的硬编码守卫会拦）。
        """
        candidate = paths.RUNTIME_CONFIGS_DIR / self.tracker
        return str(candidate) if candidate.is_file() else self.tracker

    @classmethod
    def from_config(cls, config: Any, *, where: str = f"football.yaml 的 {SECTION} 节"
                    ) -> "DetectionSettings":
        """从 football.yaml 的 detection 节构造（未给的字段走默认值）。"""
        unknown: list[str] = []
        settings = build_settings(cls, read_section(config, SECTION), unknown_sink=unknown)
        report_unknown_keys(where, unknown)
        return settings

    @classmethod
    def from_app_config(cls, app_config: Any) -> "DetectionSettings":
        """兼容两种入参: 整份 football.yaml, 或直接就是 detection 节。"""
        if isinstance(app_config, dict) and SECTION in app_config:
            return cls.from_config(app_config)
        return build_settings(cls, app_config if isinstance(app_config, dict) else {})

    @classmethod
    def load(cls, path: Optional[Any] = None) -> "DetectionSettings":
        """从 yaml 文件加载; 默认 configs/runtime/football.yaml。"""
        return cls.from_config(load_config(path))

    def to_audit(self) -> Dict[str, Any]:
        """给结果文件用的参数快照（说明"这次是按什么参数跑的"）。"""
        return {
            "conf": self.conf, "iou": self.iou, "imgsz": self.imgsz,
            "max_det": self.max_det, "stride": self.stride, "max_frames": self.max_frames,
            "device": self.device, "tracking": self.tracking,
            "tracker": self.resolved_tracker(), "classes": list(self.classes),
        }