"""推理生命周期回调."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameEvent:
    frame_idx: int
    image: np.ndarray
    annotated: np.ndarray
    n_detections: int
    detections: list = None


@dataclass
class ProgressEvent:
    frame_idx: int
    total_frames: Optional[int]
    elapsed_sec: float
    fps_loop: float
    fps_infer: float
    detections_total: int


@dataclass
class InferHooks:
    on_frame:    Optional[Callable[[FrameEvent],    None]] = None
    on_progress: Optional[Callable[[ProgressEvent], None]] = None
    on_complete: Optional[Callable[[Any],           None]] = None
    on_error:    Optional[Callable[[Exception],     None]] = None
    progress_interval_frames: int = 30

    def fire_frame(self, evt: FrameEvent) -> None:
        if self.on_frame is not None:
            try:
                self.on_frame(evt)
            except Exception as e:
                logger.warning(f"on_frame 回调异常 (已吞): {e}")

    def fire_progress(self, evt: ProgressEvent) -> None:
        if self.on_progress is not None:
            try:
                self.on_progress(evt)
            except Exception as e:
                logger.warning(f"on_progress 回调异常 (已吞): {e}")

    def fire_complete(self, result: Any) -> None:
        if self.on_complete is not None:
            try:
                self.on_complete(result)
            except Exception as e:
                logger.warning(f"on_complete 回调异常 (已吞): {e}")

    def fire_error(self, exc: Exception) -> None:
        if self.on_error is not None:
            try:
                self.on_error(exc)
            except Exception as e:
                logger.warning(f"on_error 回调异常 (已吞): {e}")
