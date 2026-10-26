#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : football_config.py
# @Project   : ODPlatform
# @Function  : football.yaml 的唯一入口（路径与加载只在这里定义一次）

"""足球运行配置的统一加载口。

为什么独立成模块:
    geometry / tracking / detection / action_recognition 四个子系统都要读同一份
    ``configs/runtime/football.yaml``。如果每处各写一遍 ``RUNTIME_CONFIGS_DIR /
    "football.yaml"``, 一旦文件改名或搬目录, 就会有一处【静默回落到默认值】——
    那正是此前"配置里写着 0.45、实际跑的是 0.7"这类事故的成因。
    所以文件名与加载逻辑收在这里, 大家共用。

约定:
    · 配置缺失/损坏一律【返回空 dict + 打日志】, 绝不抛异常打断推理;
      调用方拿到空 dict 会自然走各自的默认值。
    · 拿到的可能是"配置内容"也可能是"配置路径"(Web 侧的 request.json 两种都可能传),
      统一用 :func:`coerce_config` 归一。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from od_platform.common import paths

logger = logging.getLogger(__name__)

CONFIG_NAME = "football.yaml"


def default_config_path() -> Path:
    """football.yaml 的默认位置。"""
    return paths.RUNTIME_CONFIGS_DIR / CONFIG_NAME


def load_config(path: Optional[Any] = None) -> Dict[str, Any]:
    """读整份 football.yaml。

    Args:
        path: yaml 路径; None 时用 :func:`default_config_path`

    Returns:
        配置字典; 文件不存在或解析失败时返回空 dict（调用方全部走默认值）
    """
    target = Path(path) if path else default_config_path()
    if not target.is_file():
        logger.info("未找到足球运行配置 %s, 全部采用默认值", target)
        return {}
    import yaml

    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001 —— 配置坏掉不该让推理挂死
        logger.warning("足球运行配置解析失败, 全部采用默认值: %s (%s)", target, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("足球运行配置顶层不是 mapping, 全部采用默认值: %s", target)
        return {}
    logger.info("已加载足球运行配置: %s", target)
    return data


def coerce_config(value: Optional[Any]) -> Dict[str, Any]:
    """把"配置内容 / 配置路径 / None"统一成配置字典。"""
    if isinstance(value, Mapping):
        return dict(value)
    if value:
        return load_config(value)
    return load_config(None)