#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""规则引擎 —— 接触 / 出界 / 定位球（角球 / 点球 / 任意球）的【候选】判定。

★ 需求变更说明（本次）

    按新的需求清单, **越位与进球已移出范围** —— 原先的 check_offside / check_goal
    连同它们的"必需证据"常量一并删除。清单里的"角球"由两部分覆盖:

        · check_ball_out   → 球越过门线且最后触球方是该端防守方 = **该判角球**（判罚侧）
        · check_corner_kick → 球摆进了角球弧 = **正在踢角球**（执行侧）

    两者刻意分成两个函数、两个事件, 不互相包含: 判罚之后球还没摆好、或摆好了而出界
    已被前一帧报过, 都是常见情形。

    "任意球 / 点球"是本次新增的纯几何判据（球静止 + 对手退到 9.15 米 / 球摆在罚球点）,
    不依赖任何模型, 因此不需要重新训练就能工作。

★ 三个必须说清楚的口径（也是本项目答辩时的护身符）:

    1. 本引擎产出的是【候选】, 不是判罚。"球员靠得近"不等于"冲撞犯规",
       "球摆在罚球点上"也不等于"裁判吹了点球"。每一条候选都带
       decision_status / requires_evidence, 由人工复核台定性。
    2. 所有距离/坐标都是【米】, 前提是 homography 有真实标定。未标定时本引擎
       拒绝产出任何结果（返回空 + skip_reason），而不是拿像素当米糊弄。
    3. 阈值全部来自 RuleSettings（geometric/settings.py）, 场地尺寸来自 PitchSpec,
       本文件里不出现任何字面量。

历史版本修正过的一处硬伤（保留记录）:
    · check_ball_out 不看最后触球方 —— 而这是角球/球门球的唯一判据。
      现在要求 last_touch_team, 缺它就返回 unknown 而不是猜。
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
# 它们不是阈值, 而是规则语义的一部分（IFAB Law 12 / 13 / 14 / 17 的要求）。
BALL_OUT_REQUIRED_EVIDENCE = ("last_touch_team",)
CORNER_KICK_REQUIRED_EVIDENCE = ("restart_awarded_by_referee",)
PENALTY_KICK_REQUIRED_EVIDENCE = ("referee_award", "offence_type")
FREE_KICK_REQUIRED_EVIDENCE = ("referee_whistle", "offence_type")

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
            logger.warning("%s —— 结果里不会出现 collision / ball_out / corner_kick / "
                           "penalty_kick / free_kick 候选", self.skip_reason)

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
            越过门线且在门框内  → 这是进球; 进球不在本次需求清单内, 故不产出

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
            # 门框内过线 = 进球。进球不在本次需求清单内, 所以不产出事件;
            # 保留这个分支是为了不把它误报成"球门球 / 角球"。
            return None

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

    # ------------------------------------------------------------------ 定位球
    # 角球 / 点球 / 任意球: 判据全是纯几何的, 不需要任何模型、也不需要重新训练。

    def check_corner_kick(self, ball_pos: Point) -> Optional[Dict[str, Any]]:
        """角球【候选】—— 球摆进了角球弧（"正在踢角球"的直接几何证据）。

        ★ 与 check_ball_out 的分工（刻意不合并）:

            · 球越过门线、最后触球方是该端防守方 → 那是一次【判罚】
              (check_ball_out 的 restart="corner"), 回答"该判角球";
            · 球摆在角球弧里                    → 这是【正在踢角球】(本函数),
              回答"角球已经摆好、准备开出"。

          两件事都可能单独发生（判罚了但球还没摆好; 或摆好了而出界已被前一帧报过）,
          所以是两个函数、两个事件, 互不包含。
        """
        if not self.usable:
            return None
        x, y = ball_pos
        arc = self.homography.corner_arc_status(
            x, y, tolerance=self.settings.corner_arc_tolerance_m)
        if not arc["inside"]:
            return None
        return {
            "set_piece": "corner_kick",
            "side": _corner_side(arc["corner"]),
            "corner": arc["corner"],
            "ball_pos": (round(x, 3), round(y, 3)),
            "arc": arc,
            "decision_status": DECISION_PENDING,
            "requires_evidence": list(CORNER_KICK_REQUIRED_EVIDENCE),
            "note": "球在角球弧内 = 正在踢角球; 是否真判了角球需裁判确认",
        }

    def check_penalty_kick(self, ball_pos: Point, *,
                           side: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """点球【候选】—— 球摆在罚球点上。

        判据: 球到某一侧罚球点的距离 ≤ penalty_spot_tolerance_m。

        ★ 为什么不用"球在罚球区内"当判据: 球进罚球区是比赛里最平常的事, 拿它当
          判据会把大量正常进攻误报成点球。点球是"**球被摆到罚球点**"这个具体事件。
        """
        if not self.usable:
            return None
        x, y = ball_pos
        sides = [side] if side in ("left", "right") else ["left", "right"]
        for candidate in sides:
            spot = self.homography.penalty_spot_status(
                x, y, candidate, tolerance=self.settings.penalty_spot_tolerance_m)
            if not spot["on_spot"]:
                continue
            return {
                "set_piece": "penalty_kick",
                "side": candidate,
                "ball_pos": (round(x, 3), round(y, 3)),
                "spot": spot,
                "in_penalty_area": self.homography.point_in_penalty_area(x, y, candidate),
                "decision_status": DECISION_PENDING,
                "requires_evidence": list(PENALTY_KICK_REQUIRED_EVIDENCE),
                "note": "球摆在罚球点上 = 准备踢点球; 判罚本身需裁判确认",
            }
        return None

    def check_free_kick(self, ball_pos: Point, opponents: Sequence[Point], *,
                        stationary: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        """任意球【候选】—— 球静止 **且** 对手已退到 9.15 米外。

        判据（IFAB Law 13）:
            · 球是静止的 —— 由 geometric/ball_state.py 的 BallStationaryTracker 给出,
              注意它的"样本不足"是 **None**, 必须与 False 区分开;
            · 离球最近的对手 ≥ free_kick_wall_distance_m - wall_distance_tolerance_m。

        ★ 两条缺一不可: 只有"球静止"可能是球员在等哨或准备掷界外球; 只有"人墙退开"
          可能只是正常的防守站位。两者同时成立才是任意球待发的样子。
        """
        if not self.usable or stationary is None or not stationary:
            return None
        nearest = self.homography.min_distance_to(ball_pos, opponents)
        if nearest is None:
            return None                  # 一个对手都没看到 → 证据不足, 不硬判
        required = (self.settings.free_kick_wall_distance_m
                    - self.settings.wall_distance_tolerance_m)
        if nearest < required:
            return None
        x, y = ball_pos
        return {
            "set_piece": "free_kick",
            "ball_pos": (round(x, 3), round(y, 3)),
            "stationary": True,
            "nearest_opponent_m": round(nearest, 3),
            "required_wall_distance_m": self.settings.free_kick_wall_distance_m,
            "effective_required_m": round(required, 3),
            "decision_status": DECISION_PENDING,
            "requires_evidence": list(FREE_KICK_REQUIRED_EVIDENCE),
            "note": "球静止且对手已退够 9.15 米 = 任意球待发; 犯规类型需裁判确认",
        }


def _other_team(team: str) -> str:
    """A↔B; 未知标识原样返回, 不硬凑。"""
    if team == "A":
        return "B"
    if team == "B":
        return "A"
    return team


def _corner_side(corner_name: str) -> str:
    """从角点名取所在端（left / right）; 名字不认识时返回 unknown, 不猜。"""
    for side in ("left", "right"):
        if corner_name.endswith(side):
            return side
    return "unknown"