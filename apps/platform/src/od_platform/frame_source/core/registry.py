#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : registry.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : 源注册表
"""源注册表: 配置类型是派发轴; @register_source 让源自注册。"""
from __future__ import annotations

import logging
from typing import Callable, NamedTuple

from .base   import FrameSource
from .config import SourceConfig


logger = logging.getLogger(__name__)


class StrRule(NamedTuple):
    name:      str
    matcher:   Callable[[str], bool]
    to_config: Callable[[str], SourceConfig]


_BY_CONFIG: dict[type[SourceConfig], type[FrameSource]] = {}
_STR_RULES: list[StrRule] = []


def register_source(
    config_cls: type[SourceConfig],
    *,
    str_matcher:   Callable[[str], bool] | None = None,
    str_to_config: Callable[[str], SourceConfig] | None = None,
):
    if not (isinstance(config_cls, type) and issubclass(config_cls, SourceConfig)):
        raise TypeError(f"config_cls 必须是 SourceConfig 的子类, 收到: {config_cls!r}")
    if str_matcher is not None and str_to_config is None:
        raise ValueError("提供 str_matcher 时必须同时提供 str_to_config")

    def decorator(source_cls: type[FrameSource]) -> type[FrameSource]:
        if not (isinstance(source_cls, type) and issubclass(source_cls, FrameSource)):
            raise TypeError(f"被注册的类必须是 FrameSource 的子类, 收到: {source_cls!r}")

        existing = _BY_CONFIG.get(config_cls)
        if existing is not None and existing is not source_cls:
            raise ValueError(
                f"配置 {config_cls.__name__} 已被 {existing.__name__} 注册, "
                f"拒绝用 {source_cls.__name__} 覆盖"
            )

        _BY_CONFIG[config_cls] = source_cls
        if str_matcher is not None:
            _STR_RULES.append(StrRule(source_cls.__name__, str_matcher, str_to_config))

        logger.debug(
            f"已注册源: {config_cls.__name__} → {source_cls.__name__}"
            + (" (+字符串规则)" if str_matcher is not None else "")
        )
        return source_cls

    return decorator


def build_from_config(config: SourceConfig) -> FrameSource:
    if not isinstance(config, SourceConfig):
        raise TypeError(f"build_from_config 需要 SourceConfig 实例, 收到 {type(config).__name__}")

    source_cls = _BY_CONFIG.get(type(config))
    if source_cls is None:
        raise ValueError(
            f"未注册的配置类型: {type(config).__name__}。"
            f"已注册: {sorted(c.__name__ for c in _BY_CONFIG)}"
        )
    return source_cls(config)


def build_from_string(source: str) -> FrameSource:
    if not isinstance(source, str):
        raise TypeError(f"build_from_string 需要 str, 收到 {type(source).__name__}")

    hits = [rule for rule in _STR_RULES if rule.matcher(source)]

    if not hits:
        raise ValueError(
            f"无法识别的源字符串: {source!r}。"
            f"已注册的字符串规则: {[r.name for r in _STR_RULES]}"
        )
    if len(hits) > 1:
        raise ValueError(
            f"源字符串 {source!r} 命中多个规则(歧义): {[h.name for h in hits]}。"
        )

    rule = hits[0]
    try:
        config = rule.to_config(source)
    except Exception as e:
        raise ValueError(f"由 {source!r} 构造 {rule.name} 的配置失败: {e}") from e

    return build_from_config(config)


def list_sources() -> dict[str, str]:
    return {cfg.__name__: src.__name__ for cfg, src in _BY_CONFIG.items()}
