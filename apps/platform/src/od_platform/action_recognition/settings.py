#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : settings.py
# @Project   : ODPlatform
# @Function  : 动作识别参数（默认值只声明一次）

"""动作识别的可调参数 —— 对应 ``football.yaml`` 的 ``action_recognition:`` 节。

★ 能力边界（必须如实说明, 别在文档/答辩里写成已支持）

    实际加载的权重是 ``MCG-NJU/videomae-base-finetuned-kinetics``, 即
    **VideoMAE V1 base + Kinetics-400 预训练**（400 类通用动作）。

    · ``model_name`` 只是【分派键】, 不决定加载哪个模型 —— 加载什么完全由
      ``checkpoint`` 决定。此前配置里写的 ``videomae_v2_base`` 和
      ``num_classes: 10`` 都是错的: 前者暗示是 V2, 后者与真实的 400 类对不上。
    · ``slowfast`` / ``custom`` 是【预留位】: 传进来只会降级为空结果并打 warning,
      **没有实现**。
    · 真实类别数不写死在配置里, 由加载后的模型自己报告（见 ActionRecognizer.num_classes）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from od_platform.common.config_utils import build_settings, read_section, report_unknown_keys
from od_platform.common.football_config import load_config

SECTION = "action_recognition"

# 已实现的模型族（分派用前缀）
SUPPORTED_FAMILIES: tuple[str, ...] = ("videomae",)
# 文档里提过、代码里【没有】实现的族 —— 单独列出来, 免得被当成能力
PLANNED_FAMILIES: tuple[str, ...] = ("slowfast", "custom")

CLIP_LENGTH_MIN: int = 1
TOP_K_MIN: int = 1
CONF_MIN: float = 0.0
CONF_MAX: float = 1.0


@dataclass(frozen=True)
class ActionRecognitionSettings:
    """动作识别参数。

    字段说明见 ``football.yaml`` 的 ``action_recognition:`` 节。
    """

    model_name: str = "videomae"       # 分派键; 目前只实现 videomae
    checkpoint: str = ""               # 本地目录或 HuggingFace repo id; 空 = 降级
    clip_length: int = 16              # 送入模型的帧数（VideoMAE base 训练时即 16）
    confidence_threshold: float = 0.3  # 低于此值的类别不产出动作事件
    top_k: int = 5                     # 只看置信度最高的前 K 类

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("model_name 不能为空")
        if self.clip_length < CLIP_LENGTH_MIN:
            raise ValueError(f"clip_length 必须 >= {CLIP_LENGTH_MIN}, 收到 {self.clip_length}")
        if self.top_k < TOP_K_MIN:
            raise ValueError(f"top_k 必须 >= {TOP_K_MIN}, 收到 {self.top_k}")
        if not CONF_MIN <= self.confidence_threshold <= CONF_MAX:
            raise ValueError(f"confidence_threshold 必须在 [{CONF_MIN}, {CONF_MAX}] 内, "
                             f"收到 {self.confidence_threshold}")

    @property
    def family(self) -> str:
        """把 model_name 归到已实现的族; 未实现返回空串。"""
        lowered = self.model_name.strip().lower()
        for name in SUPPORTED_FAMILIES:
            if lowered.startswith(name):
                return name
        return ""

    @property
    def implemented(self) -> bool:
        return bool(self.family)

    def resolved_checkpoint(self) -> str:
        """返回本地存在的 checkpoint 路径; 不存在则原样返回（交给 transformers 当 repo id）。"""
        if not self.checkpoint:
            return ""
        from pathlib import Path

        path = Path(self.checkpoint)
        return str(path) if path.exists() else self.checkpoint

    @classmethod
    def from_config(cls, config: Any, *, where: str = f"football.yaml 的 {SECTION} 节"
                    ) -> "ActionRecognitionSettings":
        """从 football.yaml 的 action_recognition 节构造。"""
        unknown: list[str] = []
        settings = build_settings(cls, read_section(config, SECTION), unknown_sink=unknown)
        report_unknown_keys(where, unknown)
        return settings

    @classmethod
    def from_app_config(cls, app_config: Any) -> "ActionRecognitionSettings":
        """兼容两种入参: 整份 football.yaml, 或直接就是 action_recognition 节。"""
        if isinstance(app_config, Mapping) and SECTION in app_config:
            return cls.from_config(app_config)
        return build_settings(cls, app_config if isinstance(app_config, Mapping) else {})

    @classmethod
    def load(cls, path: Optional[Any] = None) -> "ActionRecognitionSettings":
        """从 yaml 文件加载; 默认 configs/runtime/football.yaml。"""
        return cls.from_config(load_config(path))

    def to_audit(self) -> Dict[str, Any]:
        """给结果文件用的参数快照。"""
        return {
            "model_name": self.model_name,
            "family": self.family or "(未实现)",
            "implemented": self.implemented,
            "checkpoint": self.resolved_checkpoint(),
            "clip_length": self.clip_length,
            "confidence_threshold": self.confidence_threshold,
            "top_k": self.top_k,
        }