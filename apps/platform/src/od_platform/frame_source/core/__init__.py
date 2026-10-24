#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : __init__.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : core 子包入口
"""core 子包: 数据类型、每源配置、抽象协议、注册表。"""
from __future__ import annotations

from .types  import (
    SourceType, FrameInfo, Frame, CameraIntrinsics, EndReason, is_recoverable,
    IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, STREAM_SCHEMES, is_stream_url,
)
from .config import (
    SourceConfig, CameraConfig, VideoConfig, ImageConfig, ImageFolderConfig,
    CameraBackend, CameraCodec,
)
from .base   import FrameSource
from .registry import (
    register_source, build_from_config, build_from_string, list_sources,
)

__all__ = [
    "SourceType", "FrameInfo", "Frame", "CameraIntrinsics",
    "IMAGE_EXTENSIONS", "VIDEO_EXTENSIONS", "STREAM_SCHEMES", "is_stream_url",
    "EndReason", "is_recoverable",
    "SourceConfig", "CameraConfig", "VideoConfig", "ImageConfig", "ImageFolderConfig",
    "CameraBackend", "CameraCodec",
    "FrameSource",
    "register_source", "build_from_config", "build_from_string", "list_sources",
]
