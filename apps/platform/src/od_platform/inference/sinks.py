"""推理结果输出适配器."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from od_platform.frame_source import SourceType

from .defaults import (
    DEFAULT_VIDEO_FILENAME,
    DEFAULT_VIDEO_FOURCC,
    DEFAULT_VIDEO_FPS,
)

logger = logging.getLogger(__name__)


class OutputSink(ABC):
    @abstractmethod
    def open(self, output_dir: Path, source_type: SourceType) -> None: ...

    @abstractmethod
    def write(self, frame, annotated: np.ndarray) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class LocalFileSink(OutputSink):
    """把结果落到本地: 视频/摄像头源写成一个 mp4, 图像源写成 jpg。

    文件名、编码器、兜底帧率都可以从 pipeline 配置覆盖
    (``configs/runtime/infer_pipeline.yaml`` 的 output 段), 默认值集中在
    :mod:`od_platform.inference.defaults` —— 这里不写魔法数。
    """

    def __init__(self, *, filename: str = DEFAULT_VIDEO_FILENAME,
                 fourcc: str = DEFAULT_VIDEO_FOURCC,
                 default_fps: float = DEFAULT_VIDEO_FPS) -> None:
        self.filename = filename
        self.fourcc = fourcc
        self.default_fps = default_fps
        self.output_dir: Path | None = None
        self._is_stream: bool = False
        self._video = None
        self._count: int = 0

    def open(self, output_dir: Path, source_type: SourceType) -> None:
        self.output_dir = output_dir
        self._is_stream = source_type in (SourceType.VIDEO, SourceType.CAMERA)

    def write(self, frame, annotated) -> None:
        import cv2
        try:
            if self._is_stream:
                if self._video is None:
                    h, w = annotated.shape[:2]
                    fps = (float(frame.info.fps)
                           if getattr(frame.info, "fps", None) else self.default_fps)
                    out = self.output_dir / self.filename
                    fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
                    self._video = cv2.VideoWriter(str(out), fourcc, fps, (w, h))
                self._video.write(annotated)
            else:
                fname = frame.info.filename or f"frame_{frame.info.frame_index:06d}"
                cv2.imwrite(str(self.output_dir / f"{Path(fname).stem}.jpg"), annotated)
            self._count += 1
        except Exception as e:
            logger.warning(f"LocalFileSink.write 失败, 跳过: {e}")

    def close(self) -> None:
        if self._video is not None:
            try:
                self._video.release()
            except Exception as e:
                logger.warning(f"LocalFileSink.close release 失败 (已吞): {e}")
            finally:
                self._video = None


class NullSink(OutputSink):
    def open(self, output_dir: Path, source_type: SourceType) -> None: ...
    def write(self, frame, annotated: np.ndarray) -> None: ...
    def close(self) -> None: ...
