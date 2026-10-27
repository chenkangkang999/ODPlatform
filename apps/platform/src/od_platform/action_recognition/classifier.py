#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : classifier.py
# @Project   : ODPlatform
# @Function  : 视频动作识别封装层（参数全部来自 ActionRecognitionSettings）

"""视频动作识别封装层。

★ 参数一律来自 :class:`ActionRecognitionSettings`（``football.yaml`` 的
  ``action_recognition:`` 节）, 本文件不出现可调字面量。

★ 实际加载的是 ``MCG-NJU/videomae-base-finetuned-kinetics``, 即
  **VideoMAE V1 base + Kinetics-400**。真实类别数由模型自己报告, 不写死在配置里
  （此前配置写的 ``num_classes: 10`` 与真实的 400 类对不上）。

★ ``custom`` 是预留位、``slowfast`` 已按需求清单删除: 两者都不会被加载,
  传进来只会降级为空结果并打 warning。详见 settings 模块。

用法::

    recognizer = ActionRecognizer(settings)      # 或 ActionRecognizer({...节...})
    if recognizer.ready:
        for item in recognizer.predict(clip):    # clip: (T, H, W, 3) BGR
            print(item["cls"], item["conf"], item["raw"])
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np

from od_platform.action_recognition.settings import ActionRecognitionSettings

logger = logging.getLogger(__name__)

# Kinetics 标签里与足球比赛相关的动作 -> 我们关心的比赛动作
_KINETICS_TO_FOOTBALL = {
    "running": "running",
    "jogging": "running",
    "jumping": "jumping",
    "kicking field goal": "kicking",
    "kicking soccer ball": "kicking",
    "passing soccer ball": "passing",
    "shooting goal (soccer)": "shooting",
    "diving": "diving",
    "drop kicking": "kicking",
    "punching person (boxing)": "collision",
    "wrestling": "collision",
    "tackling": "tackling",
    "stretching leg": "stretching",
}

# 少于这么多帧就没什么可识别的（有名字, 不做魔法数字）
MIN_CLIP_FRAMES: int = 2


class ActionRecognizer:
    """视频动作识别器。

    Args:
        config: 配置；可以是 ``ActionRecognitionSettings``、football.yaml 的
            ``action_recognition`` 节、或整份 football.yaml
        settings: 直接给一套 settings（给了就不再从 config 构造）
    """

    def __init__(self, config: Optional[Union[Mapping[str, Any], ActionRecognitionSettings]] = None,
                 *, settings: Optional[ActionRecognitionSettings] = None):
        if settings is None:
            if isinstance(config, ActionRecognitionSettings):
                settings = config
            else:
                settings = ActionRecognitionSettings.from_app_config(config)
        self.settings = settings
        # 便于调用方沿用旧属性名（只读）
        self.model_name = settings.model_name
        self.checkpoint = settings.checkpoint
        self.conf_threshold = settings.confidence_threshold
        self.clip_length = settings.clip_length
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._labels: Dict[int, str] = {}
        self._init_model()

    # ---------------------------------------------------------------- 加载
    def _init_model(self) -> None:
        """延迟加载模型；checkpoint 为空或模型族未实现时降级（predict 返回 []）。"""
        if not self.settings.checkpoint:
            logger.warning("未配置动作识别 checkpoint，动作识别降级为空结果")
            return
        if not self.settings.implemented:
            logger.warning(
                "动作识别模型族 %r 未实现（已实现: %s；预留: %s），降级为空结果",
                self.settings.model_name, list(_SUPPORTED_HINT()), list(_PLANNED_HINT()))
            return
        try:
            import torch
            from transformers import AutoImageProcessor, VideoMAEForVideoClassification

            checkpoint = self.settings.resolved_checkpoint()
            logger.info("加载 VideoMAE: %s", checkpoint)
            self._processor = AutoImageProcessor.from_pretrained(checkpoint)
            self._model = VideoMAEForVideoClassification.from_pretrained(checkpoint)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device).eval()
            self._labels = {v: k for k, v in self._model.config.label2id.items()}
            self._warn_on_frame_mismatch()
            logger.info("VideoMAE 就绪，类别数=%d，设备=%s", len(self._labels), self._device)
        except Exception as exc:  # noqa: BLE001 —— 模型拉不起来不该让整条流水线挂掉
            logger.warning("动作识别模型加载失败: %s，降级为空", exc)
            self._model = None

    def _warn_on_frame_mismatch(self) -> None:
        """配置的 clip_length 与模型期望的帧数不一致时报警（不静默）。"""
        expected = getattr(self._processor, "num_frames", None)
        if expected is None:
            size = getattr(self._processor, "size", None) or {}
            expected = size.get("num_frames")
        if expected is not None and int(expected) != self.settings.clip_length:
            logger.warning("clip_length=%d 与模型期望的帧数 %s 不一致，识别结果可能失真",
                           self.settings.clip_length, expected)

    # ---------------------------------------------------------------- 能力
    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def num_classes(self) -> int:
        """真实类别数（由加载后的模型报告; 未就绪时为 0）。"""
        return len(self._labels)

    # ---------------------------------------------------------------- 推理
    def predict(self, video_clip: np.ndarray) -> List[Dict[str, Any]]:
        """输入一段视频片段，返回 Top-K 动作类别。

        Args:
            video_clip: 形状 (T, H, W, 3) 的 BGR numpy 数组（OpenCV 顺序）

        Returns:
            ``[{"cls": "kicking", "conf": 0.85, "raw": "kicking soccer ball"}, ...]``
        """
        if self._model is None or video_clip is None or len(video_clip) < MIN_CLIP_FRAMES:
            return []

        import cv2
        import torch

        length = self.settings.clip_length
        # VideoMAE 需要 RGB；均匀采样到 clip_length 帧
        indices = np.linspace(0, len(video_clip) - 1, num=length).astype(int)
        frames_rgb = [cv2.cvtColor(np.ascontiguousarray(video_clip[i]), cv2.COLOR_BGR2RGB)
                      for i in indices]

        inputs = self._processor(frames_rgb, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

        results: List[Dict[str, Any]] = []
        top = min(self.settings.top_k, probs.shape[0])
        for idx in torch.argsort(probs, descending=True)[:top].tolist():
            conf = float(probs[idx])
            if conf < self.settings.confidence_threshold:
                continue
            raw = self._labels.get(int(idx), str(idx))
            results.append({"cls": _KINETICS_TO_FOOTBALL.get(raw.lower(), raw),
                            "conf": round(conf, 3), "raw": raw})
        return results

    def to_audit(self) -> Dict[str, Any]:
        """给结果文件用的参数 + 能力快照。"""
        audit = dict(self.settings.to_audit())
        audit.update({"ready": self.ready, "num_classes": self.num_classes, "device": self._device})
        return audit


def _SUPPORTED_HINT():
    from od_platform.action_recognition.settings import SUPPORTED_FAMILIES
    return SUPPORTED_FAMILIES


def _PLANNED_HINT():
    from od_platform.action_recognition.settings import PLANNED_FAMILIES
    return PLANNED_FAMILIES