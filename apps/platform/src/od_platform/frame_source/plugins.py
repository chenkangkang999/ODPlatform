#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : plugins.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : 第三方源发现
"""第三方源发现(可选)。"""
from __future__ import annotations

import logging
from importlib.metadata import entry_points

logger = logging.getLogger(__name__)

PLUGIN_GROUP = "frame_source.sources"


def load_plugins(group: str = PLUGIN_GROUP, *, raise_on_error: bool = False) -> list[str]:
    loaded: list[str] = []
    for ep in entry_points(group=group):
        try:
            ep.load()
            loaded.append(ep.name)
            logger.info(f"已加载源插件: {ep.name}")
        except Exception as e:
            logger.error(f"加载源插件 {ep.name} 失败: {e}")
            if raise_on_error:
                raise
    return loaded
