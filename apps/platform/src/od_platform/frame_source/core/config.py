#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : config.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : 每源配置 (Pydantic v2)
"""源配置(基于 Pydantic v2 BaseModel)。"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .types import is_stream_url


CameraBackend = Literal["auto", "msmf", "dshow", "v4l2"]
CameraCodec   = Literal["MJPG", "YUYV", "H264", "MP4V"]


class SourceConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    @property
    def identity(self) -> str:
        return self.__class__.__name__


class CameraConfig(SourceConfig):
    camera_id: int = Field(default=0,    ge=0)
    width:     int = Field(default=1280, gt=0, le=7680)
    height:    int = Field(default=720,  gt=0, le=4320)
    fps:       int = Field(default=30,   gt=0, le=1000)
    backend: CameraBackend = Field(default="auto")
    codec:   CameraCodec   = Field(default="MJPG")

    @property
    def identity(self) -> str:
        return f"camera:{self.camera_id}"

    def get_resolution(self) -> tuple[int, int, int]:
        return (self.width, self.height, self.fps)


class VideoConfig(SourceConfig):
    path: str = Field(description="视频文件路径或网络流 URL")

    @field_validator("path")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        if is_stream_url(v):
            return v
        if not Path(v).is_file():
            raise ValueError(f"视频文件不存在: {v}")
        return v

    @property
    def identity(self) -> str:
        return self.path


class ImageConfig(SourceConfig):
    path: str = Field(description="本地图片文件路径")

    @field_validator("path")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        if is_stream_url(v):
            raise ValueError(f"图片源不支持 URL: {v}")
        if not Path(v).is_file():
            raise ValueError(f"图片文件不存在: {v}")
        return v

    @property
    def identity(self) -> str:
        return self.path


class ImageFolderConfig(SourceConfig):
    path: str = Field(description="图片文件夹路径")

    @field_validator("path")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        if not Path(v).is_dir():
            raise ValueError(f"不是有效文件夹: {v}")
        return v

    @property
    def identity(self) -> str:
        return self.path
