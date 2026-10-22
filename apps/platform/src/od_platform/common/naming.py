"""运行产物的命名规则(单一真相)。
规则: <阶段>-<run_id>-<数据集>-<模型>
本模块故意不 import datetime —— 能力剥夺, 杜绝第二个真相。
"""
from __future__ import annotations

from pathlib import PurePath
from typing import Optional


def sanitize(text: str) -> str:
    """文件名安全化: 只留字母数字和 _ -, 其余换成 _。"""
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in text)


def model_stem(model: str) -> str:
    """模型引用取干净主干: 'models/pretrained/yolov8n.pt' -> 'yolov8n'。"""
    return sanitize(PurePath(model).stem)


def run_stem(*, stage: str, run_id: str,
             dataset: Optional[str] = None, model: Optional[str] = None) -> str:
    """一次运行的完整身份 stem。"""
    parts = [sanitize(stage.replace("_", "-")), run_id]
    if dataset:
        parts.append(sanitize(dataset))
    if model:
        parts.append(model_stem(model))
    return "-".join(parts)
