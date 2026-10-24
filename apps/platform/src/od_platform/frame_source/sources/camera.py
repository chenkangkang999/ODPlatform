#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : camera.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : 摄像头源 —— 跨平台后端协商
"""摄像头输入源(CameraSource)。"""
from __future__ import annotations

import logging
import time
from typing import Optional

import cv2

from ..core.base     import FrameSource
from ..core.config   import CameraConfig
from ..core.registry import register_source
from ..core.types    import Frame, FrameInfo, SourceType, EndReason


logger = logging.getLogger(__name__)

_BACKEND_FLAGS = {
    "auto":  cv2.CAP_ANY,
    "msmf":  cv2.CAP_MSMF,
    "dshow": cv2.CAP_DSHOW,
    "v4l2":  cv2.CAP_V4L2,
}


@register_source(
    CameraConfig,
    str_matcher=lambda s: s.isdigit(),
    str_to_config=lambda s: CameraConfig(camera_id=int(s)),
)
class CameraSource(FrameSource):
    """cv2 摄像头源, 带后端协商与参数验证。"""

    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self._cap: Optional[cv2.VideoCapture] = None
        self._actual_fps: Optional[float] = None

    def open(self) -> bool:
        self._frame_index = 0
        self._end_reason = EndReason.RUNNING
        cfg: CameraConfig = self.config

        backend_flag = _BACKEND_FLAGS.get(cfg.backend, cv2.CAP_ANY)
        self._cap = cv2.VideoCapture(cfg.camera_id, backend_flag)
        if not self._cap.isOpened():
            logger.error(f"无法打开摄像头 {cfg.camera_id}(backend={cfg.backend})")
            self._cap = None
            return False

        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.codec))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self._cap.set(cv2.CAP_PROP_FPS,          cfg.fps)

        ret, _ = self._cap.read()
        if not ret:
            logger.error(f"摄像头 {cfg.camera_id} 已打开但读不到帧")
            self._cap.release()
            self._cap = None
            return False

        aw = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        af = self._cap.get(cv2.CAP_PROP_FPS)
        if (aw, ah) != (cfg.width, cfg.height):
            logger.warning(f"分辨率: 请求 {cfg.width}x{cfg.height}, 实际 {aw}x{ah}")
        if af and abs(af - cfg.fps) > 1:
            logger.warning(f"帧率: 请求 {cfg.fps}, 实际 {af:.1f}")
        self._actual_fps = float(af) if af and af > 0 else None
        logger.info(f"摄像头 {cfg.camera_id} 就绪: {aw}x{ah} @ {af:.1f}fps(backend={cfg.backend})")
        return True

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            self._end_reason = EndReason.NOT_STARTED
            return None
        ret, img = self._cap.read()
        if not ret:
            self._end_reason = EndReason.DISCONNECTED
            return None

        idx = self._frame_index
        self._frame_index += 1
        h, w = img.shape[:2]
        info = FrameInfo(
            width=w, height=h,
            source_type=SourceType.CAMERA,
            source_path=self.source_path,
            frame_index=idx, total_frames=None,
            timestamp=time.time() - self._start_time,
            fps=self._actual_fps,
            filename=self.source_path,
        )
        return Frame(image=img, info=info)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_source_type(self) -> SourceType:
        return SourceType.CAMERA

    def set_stride(self, stride: int) -> None:
        if stride != 1:
            logger.warning(
                "摄像头是实时源, 不支持 stride 跳采。"
                "要降频请在消费端丢帧, 或用 ThreadedSource 始终取最新帧。"
            )
