#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""规则引擎 —— 越位/冲撞/球出界/犯规区域判断。

所有阈值从配置读，不硬编码。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .homography import FieldHomography

logger = logging.getLogger(__name__)


@dataclass
class RuleEngine:
    """足球规则判断引擎。

    Args:
        homography: 场地标定对象
        config: 规则阈值配置（football.yaml 的 rules 节）
    """

    homography: FieldHomography
    config: dict[str, Any] = field(default_factory=dict)

    # 默认阈值（可被 __post_init__ 里的 config 覆盖）
    offside_margin: float = 1.0       # 越位容差（米）
    collision_dist: float = 2.0       # 冲撞距离阈值（米）
    ball_tol: float = 0.5             # 球归属距离（米）

    def __post_init__(self):
        self.offside_margin = float(self.config.get("offside_margin", 1.0))
        self.collision_dist = float(self.config.get("collision_distance_m", 2.0))
        self.ball_tol = float(self.config.get("ball_assignment_tolerance_m", 0.5))

    def check_offside(self, attacking_players: list[tuple[float, float]],
                      defending_players: list[tuple[float, float]],
                      ball_pos: tuple[float, float],
                      attacking_direction: str = "right") -> list[dict]:
        """越位判断。

        规则：进攻球员比倒数第二个防守球员更靠近球门，且比球更靠前。
        """
        if not defending_players:
            return []

        # 找倒数第二个防守球员（最靠近进攻方的那个）
        if attacking_direction == "right":
            defenders_sorted = sorted(defending_players, key=lambda p: p[0])
        else:
            defenders_sorted = sorted(defending_players, key=lambda p: -p[0])

        second_last_def = defenders_sorted[1] if len(defenders_sorted) >= 2 else defenders_sorted[0]

        offside_list = []
        for i, (ax, ay) in enumerate(attacking_players):
            if attacking_direction == "right":
                is_offside = (ax > second_last_def[0] + self.offside_margin and
                              ax > ball_pos[0])
            else:
                is_offside = (ax < second_last_def[0] - self.offside_margin and
                              ax < ball_pos[0])
            if is_offside:
                offside_list.append({"player_idx": i, "pos": (ax, ay)})
        return offside_list

    def check_collision(self, players: list[tuple[float, float]]) -> list[dict]:
        """冲撞判断：两名球员距离 < 阈值"""
        collisions = []
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                dx = players[i][0] - players[j][0]
                dy = players[i][1] - players[j][1]
                dist = (dx ** 2 + dy ** 2) ** 0.5
                if dist < self.collision_dist:
                    collisions.append({"p1": i, "p2": j, "dist_m": round(dist, 2)})
        return collisions

    def check_ball_out(self, ball_pos: tuple[float, float]) -> str | None:
        """球出界判断：返回 'goal_kick' | 'corner' | 'throw_in' | None"""
        x, y = ball_pos
        if x < 0 or x > 105:
            # 左右出界
            if 25 <= y <= 43:  # 球门宽度内
                return "goal_kick" if x < 0 else "corner"
            return "goal_kick"
        if y < 0 or y > 68:
            return "throw_in"
        return None
