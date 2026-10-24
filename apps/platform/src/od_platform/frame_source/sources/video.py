#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : video.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : 视频源 —— 本地视频文件 + rtsp/rtmp/http(s) 网络流
"""视频输入源(VideoSource)。"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import cv2

from ..core.base     import FrameSource
from ..core.config   import VideoConfig
from ..core.registry import register_source
from ..core.types    import Frame, FrameInfo, SourceType, EndReason, VIDEO_EXTENSIONS, is_stream_url


logger = logging.getLogger(__name__)


@register_source(
    VideoConfig,
    str_matcher=lambda s: is_stream_url(s) or Path(s).suffix.lower() in VIDEO_EXTENSIONS,
    str_to_config=lambda s: VideoConfig(path=s),
)
class VideoSource(FrameSource):
    """视频文件 / 网络流源。"""

    def __init__(self, config: VideoConfig):
        super().__init__(config)
        self._path: str = config.path
        self._is_stream: bool = is_stream_url(config.path)
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: Optional[float] = None
        self._total: Optional[int] = None
        self._just_sought: bool = False

    def open(self) -> bool:
        self._frame_index = 0
        self._just_sought = False
        self._end_reason = EndReason.RUNNING

        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            logger.error(f"无法打开视频源: {self._path}")
            self._cap = None
            return False

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0 or math.isnan(fps):
            logger.warning(f"视频 fps 缺失或异常({fps}): {self._path}")
            self._fps = None
        else:
            self._fps = float(fps)

        if self._is_stream:
            self._total = None
            logger.info(f"网络流已打开(不可 seek, 无总帧数): {self._path}")
        else:
            n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._total = n if n > 0 else None
        return True

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            self._end_reason = EndReason.NOT_STARTED
            return None

        if self._stride > 1 and self._frame_index > 0 and not self._just_sought:
            for _ in range(self._stride - 1):
                if not self._cap.grab():
                    self._end_reason = (EndReason.DISCONNECTED if self._is_stream
                                        else EndReason.COMPLETED)
                    return None
        self._just_sought = False

        ret, img = self._cap.read()
        if not ret:
            if self._is_stream:
                self._end_reason = EndReason.DISCONNECTED
            elif self._total is not None and self._frame_index < self._total:
                self._end_reason = EndReason.ERROR
            else:
                self._end_reason = EndReason.COMPLETED
            return None

        if self._is_stream:
            idx = self._frame_index
        else:
            idx = max(0, int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        self._frame_index = idx + 1

        h, w = img.shape[:2]
        timestamp = (idx / self._fps) if self._fps else 0.0
        info = FrameInfo(
            width=w, height=h,
            source_type=SourceType.VIDEO,
            source_path=self._path,
            frame_index=idx, total_frames=self._total,
            timestamp=timestamp, fps=self._fps,
            filename=(self._path if self._is_stream else Path(self._path).name),
        )
        return Frame(image=img, info=info)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_source_type(self) -> SourceType:
        return SourceType.VIDEO

    @property
    def seekable(self) -> bool:
        return self._cap is not None and not self._is_stream

    def seek(self, frame: Optional[int] = None, time_sec: Optional[float] = None) -> bool:
        if not self.seekable:
            reason = "网络流" if self._is_stream else "未打开的视频"
            logger.warning(f"{reason}不支持 seek")
            return False
        if frame is None and time_sec is None:
            logger.warning("seek 需要 frame 或 time_sec 之一")
            return False

        if time_sec is not None:
            if self._fps is None:
                logger.warning("视频无 fps, 无法按秒 seek")
                return False
            frame = int(time_sec * self._fps)

        frame = max(0, frame)
        ok = self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame))
        if ok:
            self._frame_index = frame
            self._just_sought = True
        return ok
