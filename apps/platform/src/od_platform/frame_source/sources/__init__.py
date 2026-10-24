#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : __init__.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : sources 子包入口
"""sources 子包。import 即触发各源自注册。"""
from __future__ import annotations

from .camera import CameraSource
from .video  import VideoSource
from .image  import ImageSource, ImageFolderSource

__all__ = ["CameraSource", "VideoSource", "ImageSource", "ImageFolderSource"]
