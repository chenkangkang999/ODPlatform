#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : base.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : FrameSource 抽象基类
"""帧源抽象基类。"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Iterator, Optional

from .config import SourceConfig
from .types  import Frame, SourceType, EndReason


logger = logging.getLogger(__name__)


class FrameSource(ABC):
    def __init__(self, config: SourceConfig):
        self.config       = config
        self.source_path  = config.identity
        self._frame_index = 0
        self._stride      = 1
        self._start_time  = time.time()
        self._end_reason: EndReason = EndReason.NOT_STARTED

    @abstractmethod
    def open(self) -> bool:
        ...

    @abstractmethod
    def read(self) -> Optional[Frame]:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    @abstractmethod
    def get_source_type(self) -> SourceType:
        ...

    @property
    def end_reason(self) -> EndReason:
        return self._end_reason

    @property
    def modalities(self) -> frozenset[str]:
        return frozenset({"image"})

    def seek(self, frame=None, time_sec=None) -> bool:
        logger.warning(f"{self.__class__.__name__} 不支持 seek 操作")
        return False

    @property
    def seekable(self) -> bool:
        return False

    def set_stride(self, stride: int) -> None:
        if stride < 1:
            raise ValueError(f"stride 必须 ≥ 1, 收到: {stride}")
        self._stride = stride
        if stride > 1:
            logger.debug(f"{self.__class__.__name__}: stride 设为 {stride}")

    @property
    def stride(self) -> int:
        return self._stride

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def __iter__(self) -> Iterator[Frame]:
        return self

    def __next__(self) -> Frame:
        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame
