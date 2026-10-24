#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : threaded.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : ThreadedSource —— 后台线程采集
"""线程化包装器。"""
from __future__ import annotations

import logging
import queue
import threading
from enum import Enum
from typing import Optional

from ..core.base  import FrameSource
from ..core.types import Frame, SourceType


logger = logging.getLogger(__name__)


class BufferStrategy(str, Enum):
    LATEST  = "latest"
    BOUNDED = "bounded"


class ThreadedSource(FrameSource):
    """后台线程采集包装器。"""

    def __init__(
        self,
        inner: FrameSource,
        *,
        strategy: BufferStrategy = BufferStrategy.LATEST,
        maxsize: int = 1,
        warmup: bool = True,
        read_timeout: float = 5.0,
    ):
        super().__init__(inner.config)
        self._inner   = inner
        self._strategy = strategy
        qsize = 1 if strategy == BufferStrategy.LATEST else max(1, maxsize)
        self._queue: "queue.Queue[Frame]" = queue.Queue(maxsize=qsize)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._eos  = threading.Event()
        self._warmup = warmup
        self._read_timeout = read_timeout
        self._timeout_warned = False

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            frame = self._inner.read()
            if frame is None:
                self._eos.set()
                break
            if self._strategy == BufferStrategy.LATEST:
                try: self._queue.get_nowait()
                except queue.Empty: pass
                try: self._queue.put_nowait(frame)
                except queue.Full: pass
            else:
                while not self._stop.is_set():
                    try:
                        self._queue.put(frame, timeout=0.1)
                        break
                    except queue.Full:
                        continue

    def _drain(self) -> None:
        while not self._queue.empty():
            try: self._queue.get_nowait()
            except queue.Empty: break

    def open(self) -> bool:
        self._frame_index = 0
        if not self._inner.open():
            return False
        self._stop.clear(); self._eos.clear(); self._timeout_warned = False
        self._drain()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        if self._warmup:
            for _ in range(int(self._read_timeout * 20)):
                if not self._queue.empty() or self._eos.is_set():
                    break
                threading.Event().wait(0.05)
        return True

    def read(self) -> Optional[Frame]:
        if self._thread is None:
            return None
        while not self._stop.is_set():
            try:
                return self._queue.get(timeout=self._read_timeout)
            except queue.Empty:
                if self._eos.is_set():
                    return None
                if not self._timeout_warned:
                    logger.warning(
                        f"ThreadedSource: {self._read_timeout}s 未取到帧"
                    )
                    self._timeout_warned = True
        return None

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._inner.close()
        self._drain()

    def get_source_type(self) -> SourceType:
        return self._inner.get_source_type()

    @property
    def modalities(self) -> frozenset[str]:
        return self._inner.modalities

    @property
    def seekable(self) -> bool:
        return self._inner.seekable

    def seek(self, frame=None, time_sec=None) -> bool:
        return self._inner.seek(frame=frame, time_sec=time_sec)

    def set_stride(self, stride: int) -> None:
        self._inner.set_stride(stride)

    @property
    def stride(self) -> int:
        return self._inner.stride
