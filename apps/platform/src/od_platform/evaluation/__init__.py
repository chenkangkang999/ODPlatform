#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : __init__.py
# @Project   : ODPlatform
# @Function  : 模型评估子系统（独立评估已训好的检测模型）

"""模型评估子系统 —— 拿一个**已经训好的**检测模型单独跑评估, 出结构化报告。

★ 为什么要独立成一个子系统

    训练时顺带打印的指标有三个问题: 只在训练那一刻存在、格式随框架变、
    想复现得把训练重跑一遍。而论文/答辩里的数字必须能**随时重跑**出来 ——
    所以评估要有自己的入口（``odp-eval``）、自己的配置（``eval.yaml``）、
    自己的报告（``runs/detect_eval/<run_id>/``）。

★ 三个子系统一起用

    :mod:`od_platform.evaluation.settings`  ``EvalSettings`` —— 全部可调项
    :mod:`od_platform.evaluation.dataset`   ``build_dataset_view`` —— 只读安全的数据集视图
    :mod:`od_platform.evaluation.metrics`   指标整理（纯函数, 可离线单测）
    :mod:`od_platform.evaluation.report`    ``EvaluationReport`` —— 三种出口
    :mod:`od_platform.evaluation.service`   ``evaluate_model`` —— 端到端串联

对外只导出下面这几个名字, 其余是实现细节。
"""
from od_platform.evaluation.dataset import DatasetView, build_dataset_view
from od_platform.evaluation.report import EvaluationReport
from od_platform.evaluation.service import evaluate_model, load_model
from od_platform.evaluation.settings import EvalSettings

__all__ = [
    "DatasetView",
    "EvalSettings",
    "EvaluationReport",
    "build_dataset_view",
    "evaluate_model",
    "load_model",
]
