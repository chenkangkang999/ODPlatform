#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : ball_state.py
# @Project   : ODPlatform
# @Function  : 球状态判定 —— "球是否静止"（定位球的前置条件）

"""球静止判定。

★ 为什么不能只看单帧速度

    标定有误差, 逐帧的球心位置天然带抖动。若用相邻两帧的位移当作"速度", 一个**完全
    静止**的球也会算出非零速度, 于是"任意球摆好待发"会被判成"球还在动"。
    这里看的是【一个窗口内的总位移】: 时间够长, 抖动会互相抵消, 而真正在滚动的球
    会累积出明显位移。这是"用积分换抗噪"的标准做法。

★ 为什么必须区分"样本不足"与"静止"

    窗口里只有一两个观测点时, 总位移小是因为**样本太少**, 不是球静止。样本不足时
    本模块返回 `stationary=None`（未知）, 而不是 False 也不是 True ——
    球刚进入画面那几帧必须停在"未知", 不能顺势被当成定位球证据。

★ 阈值全部来自 RuleSettings, 本文件不出现任何字面量。
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional, Tuple

from od_platform.geometric.settings import RuleSettings

logger = logging.getLogger(__name__)

Point = Tuple[float, float]


@dataclass
class BallStationaryTracker:
    """跨帧维护"球是否静止"。

    Args:
        settings: 规则阈值（窗口长度、最少样本、位移上限都从这里取）

    用法::

        tracker = BallStationaryTracker(settings.rules)
        tracker.update(ball_pos_m, frame_index)      # 每帧喂一次
        if tracker.status()["stationary"]:           # 注意: 可能是 None(未知)
            ...
    """

    settings: RuleSettings
    _history: Deque[Tuple[int, Point]] = field(default_factory=deque, repr=False)

    # ---------------------------------------------------------------- 写入

    def update(self, position: Point, frame_index: int) -> None:
        """记录一帧的球位（米坐标）。"""
        window = self.settings.ball_stationary_window_frames
        self._history.append((int(frame_index), (float(position[0]), float(position[1]))))
        # 只保留窗口内的观测; 多留一点余量以便处理跳帧
        cutoff = int(frame_index) - window + 1
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()
        while len(self._history) > window:
            self._history.popleft()

    def reset(self) -> None:
        """清空历史（例如换镜头、球出界后重新摆球）。"""
        self._history.clear()

    # ---------------------------------------------------------------- 读取

    def _displacement_m(self, window: Tuple[Tuple[int, Point], ...]) -> float:
        """窗口内所有球位两两之间的最大距离（米）。

        取"两两最大"而不是"首尾距离": 球如果先往左再往右回到原处, 首尾距离会骗人,
        而两两最大如实反映它确实动过。
        """
        best = 0.0
        for i in range(len(window)):
            for j in range(i + 1, len(window)):
                (x1, y1), (x2, y2) = window[i][1], window[j][1]
                best = max(best, float(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5))
        return best

    def status(self) -> Dict[str, Any]:
        """当前的球静止判定。

        Returns:
            stationary: True=静止 / False=在动 / **None=未知（样本不足）**
            samples / displacement_m / threshold_m / window_frames / reason
        """
        window = tuple(self._history)
        settings = self.settings
        samples = len(window)
        if samples < settings.ball_stationary_min_samples:
            return {
                "stationary": None,
                "samples": samples,
                "displacement_m": None,
                "threshold_m": settings.ball_stationary_max_displacement_m,
                "window_frames": settings.ball_stationary_window_frames,
                "min_samples": settings.ball_stationary_min_samples,
                "reason": (f"观测样本不足（{samples} < "
                           f"{settings.ball_stationary_min_samples}）, 无法判断球是否静止"),
            }
        displacement = self._displacement_m(window)
        stationary = displacement <= settings.ball_stationary_max_displacement_m
        return {
            "stationary": bool(stationary),
            "samples": samples,
            "displacement_m": round(displacement, 3),
            "threshold_m": settings.ball_stationary_max_displacement_m,
            "window_frames": settings.ball_stationary_window_frames,
            "min_samples": settings.ball_stationary_min_samples,
            "first_frame": window[0][0],
            "last_frame": window[-1][0],
            "reason": (f"窗口内最大位移 {displacement:.3f} m "
                       f"{'≤' if stationary else '>'} "
                       f"{settings.ball_stationary_max_displacement_m:.3f} m"),
        }

    @property
    def is_stationary(self) -> Optional[bool]:
        """便捷属性; 语义与 status()["stationary"] 完全一致（含 None）。"""
        return bool(self.status()["stationary"]) if self.status()["stationary"] is not None else None