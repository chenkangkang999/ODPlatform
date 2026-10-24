#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : factory.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : 便捷工厂
"""便捷工厂: frame_source 对外的主入口。"""
from __future__ import annotations

from .core.config   import SourceConfig
from .core.registry import build_from_config, build_from_string
from .core.base     import FrameSource
from .wrappers      import ThreadedSource, AsyncSource, BufferStrategy


def _resolve(source: str | SourceConfig) -> FrameSource:
    if isinstance(source, SourceConfig):
        return build_from_config(source)
    if isinstance(source, str):
        return build_from_string(source)
    raise TypeError(
        f"source 必须是 str 或 SourceConfig, 收到 {type(source).__name__}。"
    )


def create_frame_source(source: str | SourceConfig, *, stride: int = 1) -> FrameSource:
    """创建一个帧源(最常用入口)。"""
    if stride < 1:
        raise ValueError(f"stride 必须 ≥ 1, 收到: {stride}")
    src = _resolve(source)
    if stride > 1:
        src.set_stride(stride)
    return src


def create_threaded_source(
    source: str | SourceConfig,
    *,
    stride: int = 1,
    strategy: BufferStrategy = BufferStrategy.LATEST,
    maxsize: int = 1,
    warmup: bool = True,
) -> ThreadedSource:
    """创建后台线程采集的源。"""
    if stride < 1:
        raise ValueError(f"stride 必须 ≥ 1, 收到: {stride}")
    inner = _resolve(source)
    if stride > 1:
        inner.set_stride(stride)
    return ThreadedSource(inner, strategy=strategy, maxsize=maxsize, warmup=warmup)


def create_async_source(
    source: str | SourceConfig,
    *,
    stride: int = 1,
    threaded: bool = False,
    strategy: BufferStrategy = BufferStrategy.LATEST,
    maxsize: int = 1,
) -> AsyncSource:
    """创建异步接口的源。"""
    if stride < 1:
        raise ValueError(f"stride 必须 ≥ 1, 收到: {stride}")
    inner = _resolve(source)
    if stride > 1:
        inner.set_stride(stride)
    if threaded:
        inner = ThreadedSource(inner, strategy=strategy, maxsize=maxsize)
    return AsyncSource(inner)
