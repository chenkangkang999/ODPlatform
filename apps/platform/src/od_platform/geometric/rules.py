#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""规则引擎 —— 越位 / 接触 / 出界 / 进球的【候选】判定。

★ 三个必须说清楚的口径（也是本项目答辩时的护身符）:

    1. 本引擎产出的是【候选】, 不是判罚。足球规则里"越位位置"不等于"越位犯规",
       "球员靠得近"不等于"冲撞犯规", "球过线"不等于"进球有效"。每一条候选都带
       decision_status / requires_evidence, 由人工复核台定性。
    2. 所有距离/坐标都是【米】, 前提是 homography 有真实标定。未标定时本引擎
       拒绝产出任何结果（返回空 + skip_reason），而不是拿像素当米糊弄。
    3. 阈值全部来自 RuleSettings（geometric/settings.py）, 场地尺寸来自 PitchSpec,
       本文件里不出现任何字面量。

历史版本的三处硬伤（本次修正）:
    · ball_crossed_goal_line(x, goal_y_min, goal_y_max) 把后两个参数丢掉,
      只判断 x<0/x>105 —— "离球门 30 米的边线球"也会被判过门线。已移到
      FieldHomography 并修正为"越线 且 在门框内"。
    · check_ball_out 不看最后触球方 —— 而这是角球/球门球的唯一判据。
      现在要求 last_touch_team, 缺它就返回 unknown 而不是猜。
    · check_offside 写了却从未被调用。现在由 geometric.analyzer 调用,
      并且把"触球时刻未知"显式标成 needs_evidence。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from od_platform.geometric.homography import FieldHomography
from od_platform.geometric.settings import RuleSettings

logger = logging.getLogger(__name__)

Point = Tuple[float, float]

# 每条候选都必须挂上"还缺什么证据" —— 这是本项目与"AI 直接给判罚"的分界线。
# 它们不是阈值, 而是规则语义的一部分（IFAB Law 11 / 12 / 10 的要求）。
OFFSIDE_REQUIRED_EVIDENCE = ("kick_moment", "active_involvement", "legal_body_part")
BALL_OUT_REQUIRED_EVIDENCE = ("last_touch_team",)
GOAL_REQUIRED_EVIDENCE = ("whole_ball_over_line", "no_offence_before_goal")

DECISION_PENDING = "pending_human_review"
DECISION_NEEDS_EVIDENCE = "needs_evidence"


@dataclass
class RuleEngine:
    """足球规则候选引擎。

    Args:
        homography: 场地标定对象（必须 calibrated=True 才会真正产出结果）
        settings: 规则阈值（默认取 RuleSettings 的出厂值）
        require_calibration: 未标定时是否拒绝产出（默认 True, 不要关掉）
    """

    homography: FieldHomography
    settings: Optional[RuleSettings] = None
    require_calibration: bool = True

    def __post_init__(self) -> None:
        if self.settings is None:            # 显式传 None 时回退默认, 不让调用方踩空
            self.settings = RuleSettings()
        self.pitch = self.homography.pitch
        self.usable = bool(self.homography.calibrated) or not self.require_calibration
        self.skip_reason: Optional[str] = None
        if not self.usable:
            self.skip_reason = "未提供球场标定（homography.calibrated=False），几何规则拒绝产出"
            logger.warning("%s —— 结果里不会出现 collision/ball_out/offside/goal 候选",
                           self.skip_reason)

    # ------------------------------------------------------------------ 越位

    def check_offside(
        self,
        attacking_players: Sequence[Point],
        defending_players: Sequence[Point],
        ball_pos: Point,
        attacking_direction: str = "right",
        *,
        kick_frame: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """越位【位置】候选（IFAB Law 11 的几何部分）。

        判据: 进攻球员在对方半场, 且比【球】和【倒数第二名防守球员】都更靠近对方球门线。
        注意手臂不计入合法部位, 而检测框是整个人 —— 所以框中心会高估, 需人工核对。

        Args:
            attacking_players: 进攻方球员的球场坐标（米）
            defending_players: 防守方球员的球场坐标（米）
            ball_pos: 球的球场坐标（米）
            attacking_direction: "right" 表示进攻方向朝球场右端
            kick_frame: 队友触球的帧号; 为 None 时标记为 needs_evidence
        """
        if not self.usable or not defending_players or not attacking_players:
            return []

        sign = 1.0 if attacking_direction == "right" else -1.0
        attack_goal_x = (self.pitch.right_goal_line_x if attacking_direction == "right"
                         else self.pitch.left_goal_line_x)
        own_half_boundary = self.pitch.centre_x

        # 倒数第二名防守球员 = 离本方球门线第二近的那一个（第一近的通常是门将）
        defenders_sorted = sorted(defending_players, key=lambda p: abs(p[0] - attack_goal_x))
        if len(defenders_sorted) >= 2:
            second_last, second_last_inferred = defenders_sorted[1], False
        else:
            second_last, second_last_inferred = defenders_sorted[0], True

        # 越位线: 取"球"与"倒数第二防守球员"中更靠近球门的那条。
        # 距离是"到球门线的距离", 所以要沿场地往回退, 不是往球门外面推。
        offside_line_x = attack_goal_x - sign * min(
            abs(attack_goal_x - second_last[0]), abs(attack_goal_x - ball_pos[0]))

        margin = self.settings.offside_margin_m
        candidates: List[Dict[str, Any]] = []
        for index, (ax, ay) in enumerate(attacking_players):
            if (ax - own_half_boundary) * sign <= 0:      # 还在本方半场: 不可能越位
                continue
            beyond_defender = (ax - second_last[0]) * sign > margin
            beyond_ball = (ax - ball_pos[0]) * sign > margin
            if beyond_defender and beyond_ball:
                candidates.append({
                    "player_idx": index,
                    "pos": (round(ax, 3), round(ay, 3)),
                    "offside_line_x": round(offside_line_x, 3),
                    "second_last_defender": (round(second_last[0], 3),
                                             round(second_last[1], 3)),
                    "ball_pos": (round(ball_pos[0], 3), round(ball_pos[1], 3)),
                    "attacking_direction": attacking_direction,
                    "offside_margin_m": margin,
                    "kick_frame": kick_frame,
                    "second_last_inferred": second_last_inferred,
                    "decision_status": (DECISION_PENDING if kick_frame is not None
                                        else DECISION_NEEDS_EVIDENCE),
                    "requires_evidence": list(OFFSIDE_REQUIRED_EVIDENCE),
                    "note": "越位位置不等于越位犯规; 需在触球时刻判断是否参与实际比赛",
                })
        return candidates

    # ------------------------------------------------------------------ 接触

    def check_collision(self, players: Sequence[Point]) -> List[Dict[str, Any]]:
        """接触【候选】: 两名球员的场上距离小于阈值。

        这是"值得回看的接触", 不是"犯规"。完整判定还需要动作、触球顺序、
        接触部位、强度与比赛语境（IFAB Law 12）。
        """
        if not self.usable:
            return []
        threshold = self.settings.collision_distance_m
        contacts: List[Dict[str, Any]] = []
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                distance = self.homography.distance_m(players[i], players[j])
                if distance < threshold:
                    contacts.append({
                        "p1": i, "p2": j, "dist_m": round(distance, 3),
                        "position": (round((players[i][0] + players[j][0]) / 2, 3),
                                     round((players[i][1] + players[j][1]) / 2, 3)),
                        "decision_status": DECISION_PENDING,
                        "requires_evidence": ["contact_frames", "action_type", "severity"],
                        "note": "近距离接触候选, 不能直接判为犯规",
                    })
        return contacts

    # ------------------------------------------------------------------ 出界

    def check_ball_out(
        self,
        ball_pos: Point,
        *,
        last_touch_team: Optional[str] = None,
        goal_defending_team: Optional[str] = None,
        side: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """球出界 → 重启方式候选。

        判据（IFAB Law 15/16/17）:
            越过边线            → 掷界外球, 判给【未最后触球】的一方
            越过门线且在门框外  → 最后触球方是进攻方 → 球门球; 是防守方 → 角球
            越过门线且在门框内  → 这是进球, 交给 check_goal（本函数不重复报）

        Args:
            last_touch_team: 最后触球的队伍标识（如 "A"/"B"）; 未知时返回 unknown
            goal_defending_team: 该端球门的防守方标识（如 "A"/"B"）
            side: 'left' / 'right', 不传则按 x 自动判断
        """
        if not self.usable:
            return None

        x, y = ball_pos
        tolerance = self.settings.effective_goal_line_tolerance_m

        if self.homography.in_goal_mouth(y, tolerance=tolerance) and (
                x < self.pitch.left_goal_line_x - tolerance
                or x > self.pitch.right_goal_line_x + tolerance):
            return None                      # 门框内过线 → 进球, 由 check_goal 负责

        over_endline = (x < -tolerance or x > self.pitch.length + tolerance)
        over_touchline = self.homography.crossed_touchline(y, tolerance=tolerance)
        if not over_endline and not over_touchline:
            return None

        if side is None and over_endline:
            side = "left" if x < 0 else "right"

        base: Dict[str, Any] = {
            "crossing_point": (round(x, 3), round(y, 3)),
            "side": side,
            "last_touch_team": last_touch_team,
            "decision_status": DECISION_PENDING,
        }

        if over_touchline:
            base.update({
                "restart": "throw_in",
                "awarded_to": (None if last_touch_team is None
                               else _other_team(last_touch_team)),
                "requires_evidence": list(BALL_OUT_REQUIRED_EVIDENCE),
                "note": "掷界外球判给未最后触球的一方",
            })
            if last_touch_team is None:
                base["decision_status"] = DECISION_NEEDS_EVIDENCE
            return base

        # 越过门线但不在门框内 → 球门球 / 角球
        if last_touch_team is None:
            base.update({
                "restart": "unknown",
                "awarded_to": None,
                "requires_evidence": list(BALL_OUT_REQUIRED_EVIDENCE),
                "decision_status": DECISION_NEEDS_EVIDENCE,
                "note": "角球与球门球的唯一判据是最后触球方, 当前未知 —— 不猜",
            })
            return base

        if goal_defending_team is not None and last_touch_team == goal_defending_team:
            restart, awarded_to = "corner", _other_team(goal_defending_team)
        else:
            restart, awarded_to = "goal_kick", goal_defending_team
        base.update({
            "restart": restart,
            "awarded_to": awarded_to,
            "requires_evidence": list(BALL_OUT_REQUIRED_EVIDENCE),
            "note": ("最后触球方为该端防守方 → 角球" if restart == "corner"
                     else "最后触球方为进攻方 → 球门球"),
        })
        return base

    # ------------------------------------------------------------------ 进球

    def check_goal(
        self,
        ball_pos: Point,
        *,
        last_touch_team: Optional[str] = None,
        side: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """进球【候选】: 整个球越过门线且在两门柱之间。

        仍然只是候选 —— 进球有效还要排除进球方之前的犯规/越位, 而球心越线
        不等于整个球越线（转播视角更给不出毫米级判定）。
        """
        if not self.usable:
            return None
        x, y = ball_pos
        tolerance = self.settings.effective_goal_line_tolerance_m
        if side is None:
            if x < self.pitch.left_goal_line_x - tolerance:
                side = "left"
            elif x > self.pitch.right_goal_line_x + tolerance:
                side = "right"
            else:
                return None
        if not self.homography.ball_crossed_goal_line(x, y, side, tolerance=tolerance):
            return None
        return {
            "side": side,
            "ball_pos": (round(x, 3), round(y, 3)),
            "goal_mouth_y": [self.pitch.goal_y_min, self.pitch.goal_y_max],
            "scoring_team": (None if last_touch_team is None else last_touch_team),
            "decision_status": DECISION_NEEDS_EVIDENCE,
            "requires_evidence": list(GOAL_REQUIRED_EVIDENCE),
            "note": "球心越线只是候选; 需确认整个球体越线且进球前无犯规",
        }


def _other_team(team: str) -> str:
    """A↔B; 未知标识原样返回, 不硬凑。"""
    if team == "A":
        return "B"
    if team == "B":
        return "A"
    return team