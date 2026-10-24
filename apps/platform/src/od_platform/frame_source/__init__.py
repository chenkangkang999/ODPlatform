#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : __init__.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : frame_source 包入口
"""
frame_source: 统一帧输入源抽象层(可插拔 · 多模态)。
"""
from __future__ import annotations

__version__ = "0.2.0"

from .core import (
    SourceType, FrameInfo, Frame, CameraIntrinsics, EndReason, is_recoverable,
    IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, STREAM_SCHEMES, is_stream_url,
    SourceConfig, CameraConfig, VideoConfig, ImageConfig, ImageFolderConfig,
    CameraBackend, CameraCodec,
    FrameSource,
    register_source, build_from_config, build_from_string, list_sources,
)

from .sources import CameraSource, VideoSource, ImageSource, ImageFolderSource

from .wrappers import ThreadedSource, BufferStrategy, AsyncSource, ReconnectingSource

from .factory import create_frame_source, create_threaded_source, create_async_source

from .plugins import load_plugins

__all__ = [
    "__version__",
    "SourceType", "FrameInfo", "Frame", "CameraIntrinsics", "EndReason", "is_recoverable",
    "IMAGE_EXTENSIONS", "VIDEO_EXTENSIONS", "STREAM_SCHEMES", "is_stream_url",
    "SourceConfig", "CameraConfig", "VideoConfig", "ImageConfig", "ImageFolderConfig",
    "CameraBackend", "CameraCodec",
    "FrameSource", "register_source", "build_from_config", "build_from_string", "list_sources",
    "CameraSource", "VideoSource", "ImageSource", "ImageFolderSource",
    "ThreadedSource", "BufferStrategy", "AsyncSource", "ReconnectingSource",
    "create_frame_source", "create_threaded_source", "create_async_source",
    "load_plugins",
]
