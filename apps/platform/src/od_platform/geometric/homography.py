#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""单应性矩阵 (Homography) —— 把画面像素投影到标准球场俯视图。

★ 本模块最重要的一条纪律: 【区分"标定"与"随手映射"】

    from_calibration(...)  → 由显式标定构造, calibrated=True, 可以算"米"
    ad_hoc(src, dst)       → 由调用方随手给的两组点构造, calibrated=False,
                             只保证数学上可逆, 不保证有物理意义

历史版本用"整幅画面的四个角 ↔ 球场四个角"去构造 FieldHomography, 那等于
假设"这张照片恰好完整框住了球场且没有透视"。转播机位显然不是这样, 于是
所有以米为单位的结论（球员间距、出界、越位线）全部失真, 而 capabilities
却报告 calibrated_pitch=True —— 这是本项目最危险的一处"看起来对"。

现在: 没有真标定就是没有。上层（analyzer）必须据此拒绝输出几何事件,
而不是拿错误标定的数字继续往下算。

★ 场地尺寸一律来自 PitchSpec（pitch.py 的默认值或 football.yaml 的 pitch 节）,
  本文件里不出现任何场地尺寸字面量。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from od_platform.geometric.calibration import (
    CalibrationError, PitchCalibration, apply_homography, solve_homography,
)
from od_platform.geometric.pitch import DEFAULT_PITCH, PitchSpec

logger = logging.getLogger(__name__)

Point = Tuple[float, float]


class FieldHomography:
    """场地标定 + 坐标投影。

    推荐用法（有真标定时）::

        FieldHomography.from_calibration(PitchCalibration.load(path))

    兼容用法（没有标定时的临时映射，会明确标记为未标定）::

        FieldHomography(src_points=..., dst_points=...)
    """

    def __init__(
        self,
        src_points: Optional[Sequence[Point]] = None,
        dst_points: Optional[Sequence[Point]] = None,
        *,
        calibration: Optional[PitchCalibration] = None,
        image_size: Optional[Tuple[int, int]] = None,
        pitch: PitchSpec = DEFAULT_PITCH,
    ) -> None:
        if calibration is not None and (src_points or dst_points):
            raise CalibrationError("不能同时给 calibration 和 src_points/dst_points")
        if calibration is not None:
            self._init_from_calibration(calibration, image_size)
        elif src_points and dst_points:
            self._init_ad_hoc(src_points, dst_points, image_size, pitch)
        else:
            raise CalibrationError(
                "构造 FieldHomography 需要 calibration= 或 src_points/dst_points；"
                "本项目不允许默认一个'整幅画面'的假标定")

    # ---------------------------------------------------------------- 构造

    def _init_from_calibration(self, calibration: PitchCalibration,
                               image_size: Optional[Tuple[int, int]]) -> None:
        self.calibration = calibration
        self.pitch = calibration.pitch
        self.calibrated = True
        self.source = calibration.source
        self.image_size = calibration.image_size or image_size
        self._matrix = calibration.matrix
        self._inverse = _safe_inverse(calibration.matrix)
        self.quality = calibration.quality()
        if calibration.full_frame_suspected:
            logger.warning(
                "标定的像素点就是整幅画面四角（full_frame_suspected）; "
                "俯拍机位可能成立, 转播机位请确认是否点错了")

    def _init_ad_hoc(self, src_points: Sequence[Point], dst_points: Sequence[Point],
                     image_size: Optional[Tuple[int, int]], pitch: PitchSpec) -> None:
        self.calibration = None
        self.pitch = pitch
        self.calibrated = False
        self.source = "ad_hoc"
        self.image_size = image_size
        self._matrix = solve_homography(src_points, dst_points)
        self._inverse = _safe_inverse(self._matrix)
        self.quality = {
            "source": "ad_hoc",
            "points": len(src_points),
            "reprojection_error_m": None,
            "image_size": list(image_size) if image_size else None,
            "full_frame_suspected": None,
            "notes": "调用方直接给的两组点, 未经标定校验 —— 结果不具备场上物理意义",
        }
        logger.warning(
            "FieldHomography 由 src_points/dst_points 直接构造, calibrated=False; "
            "以'米'为单位的结论不可用, 请提供真实标定")

    @classmethod
    def from_calibration(cls, calibration: PitchCalibration,
                         image_size: Optional[Tuple[int, int]] = None) -> "FieldHomography":
        return cls(calibration=calibration, image_size=image_size)

    # ---------------------------------------------------------------- 投影

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    def pixel_to_meter(self, x: float, y: float) -> Point:
        """像素坐标 → 球场米坐标（俯视图）。

        标定错误时【不再静默返回原像素】—— 静默降级会让下游把像素当米用。
        """
        return apply_homography(self._matrix, x, y)

    def meter_to_pixel(self, gx: float, gy: float) -> Point:
        """球场米坐标 → 像素坐标（画参考线、反查用）。"""
        if self._inverse is None:
            raise CalibrationError("单应性矩阵不可逆, 无法反投影")
        return apply_homography(self._inverse, gx, gy)

    def distance_m(self, p1: Point, p2: Point) -> float:
        """两点在球场上的真实距离（米）。"""
        return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

    # ---------------------------------------------------------------- 区域

    def point_in_penalty_area(self, x: float, y: float, side: str = "left") -> bool:
        """点是否在罚球区内（俯视图坐标）。"""
        if not self.pitch.penalty_area_y_min <= y <= self.pitch.penalty_area_y_max:
            return False
        if side == "left":
            return 0.0 <= x <= self.pitch.penalty_area_depth
        return self.pitch.length - self.pitch.penalty_area_depth <= x <= self.pitch.length

    def point_in_goal_area(self, x: float, y: float, side: str = "left") -> bool:
        """点是否在球门区内。"""
        if not self.pitch.goal_area_y_min <= y <= self.pitch.goal_area_y_max:
            return False
        if side == "left":
            return 0.0 <= x <= self.pitch.goal_area_depth
        return self.pitch.length - self.pitch.goal_area_depth <= x <= self.pitch.length

    def in_goal_mouth(self, y: float, *, tolerance: float = 0.0) -> bool:
        """球的 y 是否落在两根门柱之间。"""
        return self.pitch.goal_y_min - tolerance <= y <= self.pitch.goal_y_max + tolerance

    # ---------------------------------------------------------------- 门线

    def ball_crossed_goal_line(self, x: float, y: float, side: Optional[str] = None,
                               *, tolerance: float = 0.0) -> bool:
        """整个球是否越过球门线【且在两门柱之间】。

        旧版的签名是 (x, goal_y_min, goal_y_max) 却把后两个参数直接丢掉、只判断
        x<0 或 x>105 —— 结果"离球门 30 米远的边线球"也会被判成过门线。这里修正为
        必须同时满足"越线"和"在门框内", 并且给 y 一个 tolerance（球半径量级）。

        Args:
            x, y: 球在俯视图上的米坐标（球心）
            side: 'left' / 'right' / None（None 表示两端都判）
            tolerance: 门柱内沿的容差（米）
        """
        if not self.in_goal_mouth(y, tolerance=tolerance):
            return False
        if side in (None, "left") and x < self.pitch.left_goal_line_x - tolerance:
            return True
        return bool(side in (None, "right")
                    and x > self.pitch.right_goal_line_x + tolerance)

    def crossed_touchline(self, y: float, *, tolerance: float = 0.0) -> bool:
        """球是否越过边线（y 方向）。"""
        return y < -tolerance or y > self.pitch.width + tolerance

    # ---------------------------------------------------------------- 报告

    def to_audit(self) -> Dict[str, Any]:
        """给审计/结果文件用的标定摘要。"""
        return {"calibrated": self.calibrated, **self.quality}


def _safe_inverse(matrix: np.ndarray) -> Optional[np.ndarray]:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        logger.warning("单应性矩阵奇异, 反投影不可用")
        return None