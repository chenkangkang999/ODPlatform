#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""视频动作识别封装层。

底层模型从配置读，不硬编码：
- model_name: "videomae_v2_base" | "slowfast" | "custom"
- num_classes: 动作类别数
- checkpoint: 预训练权重路径或 HuggingFace 模型名

简单跑通阶段默认加载 Kinetics 预训练 VideoMAE，识别通用动作
（跑、跳、踢、扑救等）；真实足球专门动作需要后续用足球视频微调。
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

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


class ActionRecognizer:
    """视频动作识别器。

    Args:
        config: 配置字典（football.yaml 的 action_recognition 节）
    """

    def __init__(self, config: dict[str, Any]):
        self.model_name = config.get("model_name", "videomae_v2_base")
        self.num_classes = int(config.get("num_classes", 10))
        self.checkpoint = config.get("checkpoint", "")
        self.conf_threshold = float(config.get("confidence_threshold", 0.5))
        self.clip_length = int(config.get("clip_length", 16))
        self._model = None
        self._processor = None
        self._labels = {}
        self._init_model()

    def _init_model(self):
        """延迟加载模型，避免启动时占用显存。

        checkpoint 为空时降级（predict 返回 unknown），不强制下载。
        """
        if not self.checkpoint:
            logger.warning("未配置动作识别 checkpoint，动作识别降级为空结果")
            return
        try:
            if self.model_name.startswith("videomae"):
                import torch
                from transformers import AutoImageProcessor, VideoMAEForVideoClassification

                logger.info(f"加载 VideoMAE: {self.checkpoint}")
                self._processor = AutoImageProcessor.from_pretrained(self.checkpoint)
                self._model = VideoMAEForVideoClassification.from_pretrained(self.checkpoint)
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model.to(self._device).eval()
                self._labels = {v: k for k, v in self._model.config.label2id.items()}
                logger.info(f"VideoMAE 就绪，类别数={len(self._labels)}，设备={self._device}")
            else:
                logger.warning(f"暂不支持的模型: {self.model_name}，降级为空")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"动作识别模型加载失败: {e}，降级为空")
            self._model = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def predict(self, video_clip: np.ndarray) -> list[dict]:
        """输入一段视频片段，返回Top-K动作类别。

        Args:
            video_clip: 形状 (T, H, W, 3) 的 BGR numpy数组（OpenCV顺序）

        Returns:
            [{"cls": "kicking", "conf": 0.85, "raw": "kicking soccer ball"}, ...]
        """
        if self._model is None or video_clip is None or len(video_clip) < 2:
            return []

        import cv2
        import torch

        # VideoMAE 需要 RGB；均匀采样到 clip_length 帧
        indices = np.linspace(0, len(video_clip) - 1, num=self.clip_length).astype(int)
        frames_rgb = [cv2.cvtColor(np.ascontiguousarray(video_clip[i]), cv2.COLOR_BGR2RGB)
                      for i in indices]

        inputs = self._processor(frames_rgb, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

        results = []
        for idx in torch.argsort(probs, descending=True)[:5].tolist():
            raw = self._labels.get(int(idx), str(idx))
            conf = float(probs[idx])
            if conf < self.conf_threshold:
                continue
            football_cls = _KINETICS_TO_FOOTBALL.get(raw.lower(), raw)
            results.append({"cls": football_cls, "conf": round(conf, 3), "raw": raw})
        return results
