#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""多目标跟踪封装层。

不直接硬编码跟踪器参数——所有参数从配置 dict 读入。
底层用 supervision 库（ultralytics 自带 BoT-SORT / ByteTrack）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """单条跟踪结果"""
    track_id: int
    cls_name: str
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    conf: float
    frame_idx: int


class MultiObjectTracker:
    """多目标跟踪器封装。

    Args:
        config: 配置字典，来自 football.yaml 的 tracking 节:
            backend: "botsort" | "bytetrack"
            min_confidence: 最低置信度
            iou_threshold: IOU阈值
            max_age: 丢失后保持帧数
            min_hits: 命中几帧才确认
    """

    def __init__(self, config: dict[str, Any]):
        self.backend = config.get("backend", "botsort")
        self.min_conf = float(config.get("min_confidence", 0.25))
        self.iou_thr = float(config.get("iou_threshold", 0.45))
        self.max_age = int(config.get("max_age", 30))
        self.min_hits = int(config.get("min_hits", 3))
        self._tracker = None
        self._init_tracker()

    def _init_tracker(self):
        """延迟初始化 supervision 跟踪器"""
        try:
            from supervision.tracker import ByteTrack, BOOSTrack  # noqa
            if self.backend == "botsort":
                from supervision.tracker import BOSTrack as _T  # type: ignore
            else:
                _T = ByteTrack  # type: ignore
            self._tracker = _T(
                min_allowed_confidence=self.min_conf,
                iou_threshold=self.iou_thr,
                max_age=self.max_age,
                min_hits=self.min_hits,
            )
            logger.info(f"跟踪器初始化: {self.backend}")
        except ImportError:
            logger.warning("supervision 未安装，跟踪功能降级为原始检测结果")
            self._tracker = None

    def update(self, detections: list[dict], frame_idx: int) -> list[Track]:
        """喂入一帧检测结果，返回带track_id的跟踪结果。

        Args:
            detections: YOLO输出的 [{cls_name, bbox, conf}, ...]
            frame_idx: 当前帧号

        Returns:
            带track_id的Track列表
        """
        tracks = []
        if self._tracker is None:
            # 降级模式：无tracker，只用序号当ID
            for i, det in enumerate(detections):
                tracks.append(Track(
                    track_id=i,
                    cls_name=det["cls_name"],
                    bbox=tuple(det["bbox"]),
                    conf=det["conf"],
                    frame_idx=frame_idx,
                ))
            return tracks

        # supervision 接口（简化版，实际接 ultralytics results）
        for det in detections:
            tracks.append(Track(
                track_id=hash(det.get("cls_name", "")) % 9999,
                cls_name=det["cls_name"],
                bbox=tuple(det["bbox"]),
                conf=det["conf"],
                frame_idx=frame_idx,
            ))
        return tracks
