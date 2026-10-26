#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""视频动作识别封装层。

底层模型从配置读，不硬编码：
- model_name: "videomae_v2_base" | "slowfast" | "custom"
- num_classes: 动作类别数
- checkpoint: 预训练权重路径
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ActionRecognizer:
    """视频动作识别器。

    Args:
        config: 配置字典（football.yaml 的 action_recognition 节）
    """

    def __init__(self, config: dict[str, Any]):
        self.model_name = config.get("model_name", "videomae_v2_base")
        self.num_classes = int(config.get("num_classes", 10))
        self.checkpoint = config.get("checkpoint", "")
        self.conf_threshold = float(config.get("confidence_threshold", 0.6))
        self._model = None
        self._init_model()

    def _init_model(self):
        """延迟加载模型，避免启动时占用显存"""
        try:
            if self.model_name.startswith("videomae"):
                logger.info(f"加载 VideoMAE V2: {self.checkpoint}")
                # 实际用 transformers 加载
                # from transformers import VideoMAEForVideoClassification
                # self._model = VideoMAEForVideoClassification.from_pretrained(self.checkpoint)
            elif self.model_name == "slowfast":
                logger.info("加载 SlowFast（需mmaction2）")
            else:
                logger.warning(f"未知模型: {self.model_name}，降级为空识别")
        except Exception as e:
            logger.warning(f"动作识别模型加载失败: {e}，降级为空")

    def predict(self, video_clip) -> list[dict]:
        """输入一段视频片段，返回Top-K动作类别。

        Args:
            video_clip: 形状 (T, H, W, 3) 的 numpy数组

        Returns:
            [{"cls": "header_kick", "conf": 0.85}, ...]
        """
        if self._model is None:
            return [{"cls": "unknown", "conf": 0.0}]

        # 实际推理代码（占位）
        logger.debug("动作识别推理中...")
        return [{"cls": "header_kick", "conf": 0.0}]
