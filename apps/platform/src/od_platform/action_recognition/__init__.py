#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""视频动作识别。

**已实现**：VideoMAE（Kinetics-400 预训练）的动作分类 —— 输入一段帧序列，
输出 top-K 的通用动作类别，再按映射表折算成足球相关动作名。

**未实现**：``slowfast`` / ``custom``。文档里出现过这两个名字，但代码里没有实现，
传进来只会降级为空结果并打 warning。不要在对外材料里把它们当成已支持的能力。
（见 :data:`~od_platform.action_recognition.settings.PLANNED_FAMILIES`）
"""
from .classifier import ActionRecognizer
from .settings import PLANNED_FAMILIES, SUPPORTED_FAMILIES, ActionRecognitionSettings

__all__ = [
    "ActionRecognizer",
    "ActionRecognitionSettings",
    "SUPPORTED_FAMILIES",
    "PLANNED_FAMILIES",
]