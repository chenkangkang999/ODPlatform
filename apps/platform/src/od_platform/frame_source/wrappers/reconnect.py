#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : reconnect.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : ReconnectingSource —— 断线重连
"""断线重连包装器。"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from ..core.base  import FrameSource
from ..core.types import Frame, SourceType, EndReason, is_recoverable


logger = logging.getLogger(__name__)


class ReconnectingSource(FrameSource):
    """断线重连包装器(适用于网络流 / 摄像头)。"""

    def __init__(
        self,
        inner: FrameSource,
        *,
        max_retries: Optional[int] = None,
        backoff_initial: float = 1.0,
        backoff_max: float = 30.0,
        backoff_factor: float = 2.0,
        on_reconnect: Optional[Callable[[int, EndReason], None]] = None,
    ):
        super().__init__(inner.config)
        self._inner = inner
        self._max_retries = max_retries
        self._backoff_initial = backoff_initial
        self._backoff_max = backoff_max
        self._backoff_factor = backoff_factor
        self._on_reconnect = on_reconnect
        self._opened = False
        self._stop = threading.Event()

    def _sleep(self, seconds: float) -> bool:
        return self._stop.wait(seconds)

    def open(self) -> bool:
        self._frame_index = 0
        self._stop.clear()
        self._end_reason = EndReason.RUNNING
        self._opened = self._open_with_retry()
        return self._opened

    def _open_with_retry(self) -> bool:
        attempt = 0
        backoff = self._backoff_initial
        while not self._stop.is_set():
            if self._inner.open():
                return True
            attempt += 1
            if self._max_retries is not None and attempt > self._max_retries:
                logger.error(f"首次打开失败, 已重试 {attempt - 1} 次仍不成, 放弃")
                self._end_reason = EndReason.DISCONNECTED
                return False
            logger.warning(f"打开失败, {backoff:.1f}s 后第 {attempt} 次重试: {self.source_path}")
            if self._sleep(backoff):
                return False
            backoff = min(backoff * self._backoff_factor, self._backoff_max)
        return False

    def read(self) -> Optional[Frame]:
        if not self._opened:
            self._end_reason = EndReason.NOT_STARTED
            return None

        frame = self._inner.read()
        if frame is not None:
            return frame

        reason = self._inner.end_reason
        if not is_recoverable(reason):
            self._end_reason = reason
            return None

        return self._reconnect_and_read(reason)

    def _reconnect_and_read(self, reason: EndReason) -> Optional[Frame]:
        attempt = 0
        backoff = self._backoff_initial
        while not self._stop.is_set():
            attempt += 1
            if self._max_retries is not None and attempt > self._max_retries:
                logger.error(f"重连放弃: 已重试 {attempt - 1} 次")
                self._end_reason = EndReason.DISCONNECTED
                return None

            logger.warning(f"源中断({reason.value}), {backoff:.1f}s 后第 {attempt} 次重连: {self.source_path}")
            if self._sleep(backoff):
                self._end_reason = EndReason.DISCONNECTED
                return None

            try:
                self._inner.close()
            except Exception as e:
                logger.debug(f"重连前关闭旧句柄出错(忽略): {e}")

            if self._inner.open():
                if self._on_reconnect is not None:
                    try: self._on_reconnect(attempt, reason)
                    except Exception as e: logger.debug(f"on_reconnect 回调出错(忽略): {e}")
                logger.info(f"第 {attempt} 次重连成功: {self.source_path}")
                frame = self._inner.read()
                if frame is not None:
                    return frame
            backoff = min(backoff * self._backoff_factor, self._backoff_max)

        self._end_reason = EndReason.DISCONNECTED
        return None

    def close(self) -> None:
        self._opened = False
        self._stop.set()
        self._inner.close()

    def get_source_type(self) -> SourceType:
        return self._inner.get_source_type()

    @property
    def modalities(self) -> frozenset[str]:
        return self._inner.modalities

    @property
    def seekable(self) -> bool:
        return False

    def set_stride(self, stride: int) -> None:
        self._inner.set_stride(stride)

    @property
    def stride(self) -> int:
        return self._inner.stride
