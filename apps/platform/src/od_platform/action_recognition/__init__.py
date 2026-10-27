#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""视频动作识别。

**已实现**：VideoMAE（Kinetics-400 预训练）的动作分类 —— 输入一段帧序列，
输出 top-K 的通用动作类别，再按映射表折算成足球相关动作名。

**未实现**：``custom``（唯一剩下的预留位）。传进来只会降级为空结果并打 warning，
不要在对外材料里把它当成已支持的能力。
（见 :data:`~od_platform.action_recognition.settings.PLANNED_FAMILIES`）

**已删除**：``slowfast`` —— 它在本项目里从未实现过，且与 VideoMAE 职责完全重叠
（都是视频动作分类），已按需求清单移出。它现在不在任何"预留位"清单里；旧配置若仍
传这个名字，行为与任意未知名字一致（降级为空结果，**不报错**），所以别指望它替你
兜住笔误。
"""
from .classifier import ActionRecognizer
from .settings import PLANNED_FAMILIES, SUPPORTED_FAMILIES, ActionRecognitionSettings

__all__ = [
    "ActionRecognizer",
    "ActionRecognitionSettings",
    "SUPPORTED_FAMILIES",
    "PLANNED_FAMILIES",
]