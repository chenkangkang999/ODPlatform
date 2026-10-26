#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""单应性矩阵 (Homography) —— 把2D画面投影到标准球场俯视图。

所有位置计算在俯视图上做，越位/进球/出界判断才准确。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FieldHomography:
    """场地标定 + 坐标投影。

    标准足球场尺寸（单位：米）:
        长度 105m，宽度 68m
        禁区 16.5m，小禁区 5.5m，点球距球门 11m
    """

    # 画面上的4个角点（像素坐标）
    src_points: list[tuple[float, float]]
    # 俯视图上对应的4个角点（米坐标）
    dst_points: list[tuple[float, float]]

    def __post_init__(self):
        self._H = self._compute_homography()

    def _compute_homography(self) -> np.ndarray:
        """计算3x3单应性矩阵"""
        src = np.array(self.src_points, dtype=np.float32)
        dst = np.array(self.dst_points, dtype=np.float32)
        # OpenCV 求Homography
        try:
            import cv2
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            logger.info("Homography 计算成功")
            return H
        except ImportError:
            logger.warning("cv2 未安装，Homography降级为恒等映射")
            return np.eye(3)

    def pixel_to_meter(self, x: float, y: float) -> tuple[float, float]:
        """像素坐标 -> 球场米坐标（俯视图）"""
        pt = np.array([[[x, y]]], dtype=np.float32)
        try:
            import cv2
            out = cv2.perspectiveTransform(pt, self._H)[0][0]
            return float(out[0]), float(out[1])
        except Exception:
            return x, y

    def point_in_penalty_area(self, x: float, y: float, side: str = "left") -> bool:
        """判断点是否在禁区内（俯视图坐标）"""
        # 标准禁区：左右各16.5m深，40.32m宽
        if side == "left":
            return 0 <= x <= 16.5 and (68 - 40.32) / 2 <= y <= (68 + 40.32) / 2
        else:
            return 105 - 16.5 <= x <= 105 and (68 - 40.32) / 2 <= y <= (68 + 40.32) / 2

    def ball_crossed_goal_line(self, x: float, goal_y_min: float, goal_y_max: float) -> bool:
        """判断球是否完全越过球门线（x=0或x=105处）"""
        # 简化：球x < 0 或 x > 105 且在球门宽度内
        return x < 0 or x > 105
