#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : types.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : 核心数据类型 (SourceType / FrameInfo / Frame / CameraIntrinsics)
"""
frame_source 核心数据类型定义(多模态版)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class SourceType(str, Enum):
    CAMERA       = "camera"
    VIDEO        = "video"
    IMAGE        = "image"
    IMAGE_FOLDER = "image_folder"


IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff",
})
VIDEO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm",
})

STREAM_SCHEMES: tuple[str, ...] = ("rtsp://", "rtmp://", "http://", "https://")


def is_stream_url(source: str) -> bool:
    return source.lower().startswith(STREAM_SCHEMES)


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float = 1.0
    baseline: Optional[float] = None


@dataclass(frozen=True)
class FrameInfo:
    width: int
    height: int
    source_type: str
    source_path: str
    frame_index: int = 0
    total_frames: Optional[int] = None
    timestamp: float = 0.0
    fps: Optional[float] = None
    filename: Optional[str] = None

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)


@dataclass
class Frame:
    image: np.ndarray
    info: FrameInfo
    depth: Optional[np.ndarray] = None
    aux: dict[str, np.ndarray] = field(default_factory=dict)
    intrinsics: Optional[CameraIntrinsics] = None

    @property
    def has_depth(self) -> bool:
        return self.depth is not None

    @property
    def resolution(self) -> tuple[int, int]:
        return self.info.resolution

    @property
    def width(self) -> int:
        return self.info.width

    @property
    def height(self) -> int:
        return self.info.height


class EndReason(str, Enum):
    NOT_STARTED  = "not_started"
    RUNNING      = "running"
    COMPLETED    = "completed"
    DISCONNECTED = "disconnected"
    ERROR        = "error"


_RECOVERABLE: frozenset = frozenset({EndReason.DISCONNECTED, EndReason.ERROR})


def is_recoverable(reason: "EndReason") -> bool:
    return reason in _RECOVERABLE
