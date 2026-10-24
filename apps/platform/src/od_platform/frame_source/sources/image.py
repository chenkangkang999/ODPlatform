#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : image.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : 图片源 —— 单张图片 + 图片文件夹
"""图片输入源。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2

from ..core.base     import FrameSource
from ..core.config   import ImageConfig, ImageFolderConfig
from ..core.registry import register_source
from ..core.types    import Frame, FrameInfo, SourceType, EndReason, IMAGE_EXTENSIONS, is_stream_url


logger = logging.getLogger(__name__)


@register_source(
    ImageConfig,
    str_matcher=lambda s: (not is_stream_url(s)) and Path(s).suffix.lower() in IMAGE_EXTENSIONS,
    str_to_config=lambda s: ImageConfig(path=s),
)
class ImageSource(FrameSource):
    """单张图片源。read() 只产一帧。"""

    def __init__(self, config: ImageConfig):
        super().__init__(config)
        self._path: str = config.path
        self._image: Optional["cv2.typing.MatLike"] = None
        self._read_count: int = 0

    def open(self) -> bool:
        self._read_count = 0
        self._frame_index = 0
        self._end_reason = EndReason.RUNNING
        self._image = cv2.imread(self._path)
        if self._image is None:
            self._end_reason = EndReason.ERROR
            logger.error(f"无法读取图片: {self._path}")
            return False
        return True

    def read(self) -> Optional[Frame]:
        if self._image is None:
            self._end_reason = EndReason.NOT_STARTED
            return None
        if self._read_count > 0:
            self._end_reason = EndReason.COMPLETED
            return None
        self._read_count += 1

        h, w = self._image.shape[:2]
        info = FrameInfo(
            width=w, height=h,
            source_type=SourceType.IMAGE,
            source_path=self._path,
            frame_index=0, total_frames=1,
            timestamp=0.0,
            filename=Path(self._path).name,
        )
        return Frame(image=self._image, info=info)

    def close(self) -> None:
        self._image = None

    def get_source_type(self) -> SourceType:
        return SourceType.IMAGE


@register_source(
    ImageFolderConfig,
    str_matcher=lambda s: Path(s).is_dir(),
    str_to_config=lambda s: ImageFolderConfig(path=s),
)
class ImageFolderSource(FrameSource):
    """图片文件夹源。按排序后的文件列表逐张产出。"""

    def __init__(self, config: ImageFolderConfig):
        super().__init__(config)
        self._folder: str = config.path
        self._files: list[Path] = []
        self._current_index: int = 0

    def open(self) -> bool:
        self._current_index = 0
        self._frame_index = 0

        folder = Path(self._folder)
        self._files = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self._files:
            logger.warning(f"文件夹内没有支持的图片: {self._folder}")
            return False
        logger.info(f"文件夹共 {len(self._files)} 张图片: {self._folder}")
        self._end_reason = EndReason.RUNNING
        return True

    def read(self) -> Optional[Frame]:
        if self._stride > 1 and self._current_index > 0:
            self._current_index += self._stride - 1

        while self._current_index < len(self._files):
            path = self._files[self._current_index]
            img = cv2.imread(str(path))
            if img is None:
                logger.warning(f"跳过无法读取的图片: {path}")
                self._current_index += 1
                continue

            idx = self._current_index
            self._current_index += 1
            self._frame_index = idx

            h, w = img.shape[:2]
            info = FrameInfo(
                width=w, height=h,
                source_type=SourceType.IMAGE_FOLDER,
                source_path=self._folder,
                frame_index=idx, total_frames=len(self._files),
                timestamp=0.0,
                filename=path.name,
            )
            return Frame(image=img, info=info)

        self._end_reason = EndReason.COMPLETED
        return None

    def close(self) -> None:
        self._files = []

    def get_source_type(self) -> SourceType:
        return SourceType.IMAGE_FOLDER

    @property
    def seekable(self) -> bool:
        return True

    def seek(self, frame: Optional[int] = None, time_sec: Optional[float] = None) -> bool:
        if frame is None:
            logger.warning("文件夹源只支持按帧索引 seek(frame=N)")
            return False
        if not (0 <= frame < len(self._files)):
            logger.warning(f"seek 越界: {frame} 不在 [0, {len(self._files)})")
            return False
        self._current_index = frame
        return True
