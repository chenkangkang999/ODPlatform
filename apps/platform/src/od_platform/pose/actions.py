#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : actions.py
# @Project   : ODPlatform
# @Function  : 姿态型动作的纯几何判据（铲球/高抬腿/肘击/拉人推人）

"""姿态型动作的几何判据。

★ 这一层为什么不做"分类"

    清单里的"铲球 / 高抬腿 / 肘击 / 拉人推人"都有明确的几何定义, 所以这里判的是
    几何量, 而不是把骨骼图喂给分类器猜。好处是**可解释**: 每个结论都能说清
    "哪个量、算出来多少、门槛多少、为什么算命中" —— 答辩现场会被追问的正是这个。

★ 本模块【不 import ultralytics】

    只吃 :class:`~od_platform.pose.estimator.PersonPose`。于是这一层可以完全脱离
    模型单测: 摆一组关键点坐标就能验证判据, 不需要 torch, 也不需要权重。

★ 归一化的两种尺度

    · **躯干长度**（肩中点→髋中点）: 与远近、分辨率无关, 是默认尺度;
    · **米**（需要场地标定）: 由调用方注入 ``distance_fn``（例如几何子系统里
      单应矩阵的 ``distance_m``）。只有"接触距离"这一项真正需要米 ——
      因为"手离对手多远算碰到"是个物理量, 用身体比例近似只能算候选。

    ★ 没有标定时, 结论会被标成【候选 candidate】而不是【判据 judgement】:
      身体比例近似不知道镜头远近, 两个像素挨着的人可能在真实场地里差着两米。

★ 关键点缺失时返回【证据不足】, 不猜

    任何一个必需关键点不可用（缺失或置信度不足）, 结论就是
    ``evidence_level=insufficient``、``hit=False``, 并列出缺了哪些点。
    绝不拿 0 或者噪声坐标去算出一个"看起来很像"的结论。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from od_platform.pose.estimator import (
    ANKLES,
    ARMS,
    HIPS,
    LEGS,
    SHOULDERS,
    SIDES,
    PersonPose,
)
from od_platform.pose.settings import MIN_TORSO_LENGTH_PX, PoseSettings

logger = logging.getLogger(__name__)

Point = Tuple[float, float]
# 两个像素点 → 米。由调用方注入（几何子系统的单应矩阵就是天然实现）。
DistanceFn = Callable[[Point, Point], float]

# ---- 动作名（对外统一口径, 不散落字符串）----
ACTION_HIGH_LEG = "high_leg"
ACTION_SLIDING_TACKLE = "sliding_tackle"
ACTION_ELBOW_STRIKE = "elbow_strike"
ACTION_PULLING_PUSHING = "pulling_pushing"

# ---- 证据等级 ----
# judgement:    判据成立（几何量本身与标定无关, 或已经用了米制距离）
# candidate:    有迹象, 但只用了身体比例近似 / 置信度偏低 → 只能当候选
# insufficient: 关键点不可用 → 证据不足, 不下结论
EVIDENCE_JUDGEMENT = "judgement"
EVIDENCE_CANDIDATE = "candidate"
EVIDENCE_INSUFFICIENT = "insufficient"

OP_GE = ">="
OP_LE = "<="
_SYMBOL = {OP_GE: "≥", OP_LE: "≤"}


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass
class ActionVerdict:
    """一个动作的判定结论（含"为什么"）。

    Attributes:
        action: 动作名（``ACTION_*`` 之一）
        hit: 是否命中
        confidence: 结论的可信度 [0,1] = 关键点平均置信度 × 通过判据的比例
        evidence_level: 见 ``EVIDENCE_*``
        reasons: 逐条判据的人话解释（含实算值与门槛）
        missing: 缺了哪些关键点（证据不足时才有内容）
        measurements: 判据量的数值（便于写进结果文件 / 复核）
    """

    action: str
    hit: bool
    confidence: float = 0.0
    evidence_level: str = EVIDENCE_JUDGEMENT
    reasons: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    measurements: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "hit": self.hit,
            "confidence": round(self.confidence, 4),
            "evidence_level": self.evidence_level,
            "reasons": list(self.reasons),
            "missing_keypoints": list(self.missing),
            "measurements": dict(self.measurements),
        }


class _VerdictBuilder:
    """判据累积器: 逐条记"算了什么、门槛多少、是否命中", 最后统一出结论。

    抽出来是为了让四个判据函数的写法一致, 并且**每条判据都自带解释** ——
    否则解释文字散在四处, 迟早有人漏写。
    """

    def __init__(self, action: str, thresholds: Any) -> None:
        self._action = action
        self._thresholds = thresholds
        self.reasons: List[str] = []
        self.missing: List[str] = []
        self.measurements: Dict[str, float] = {}
        self._hits: List[bool] = []
        self._confs: List[float] = []
        self._broken = False

    # ---------------------------------------------------------- 关键点
    def note_points(self, person: PersonPose, names: Sequence[str]) -> None:
        """记录参与判据的关键点置信度（用于算结论可信度）。"""
        for name in names:
            keypoint = person.get(name)
            if keypoint is not None:
                self._confs.append(keypoint.conf)

    def require(self, person: PersonPose, names: Sequence[str]) -> bool:
        """这些关键点必须全部可用; 否则本次判定标记为证据不足。"""
        absent = person.missing(names)
        if absent:
            self.missing.extend(absent)
            self._broken = True
            return False
        self.note_points(person, names)
        return True

    def mark_missing(self, names: Sequence[str]) -> None:
        """直接标记某些关键点缺失（用于"多选一"的情形, 例如两条腿取其一）。"""
        self.missing.extend(names)
        self._broken = True

    # ---------------------------------------------------------- 判据
    def check(self, label: str, value: float, threshold: float, *,
              op: str = OP_GE, unit: str = "", note: str = "") -> bool:
        """记一条"实算值 vs 门槛"的判据, 返回是否通过。"""
        if value is None or not math.isfinite(value):
            self.mark_missing([label])
            return False
        hit = value >= threshold if op == OP_GE else value <= threshold
        self.measurements[label] = round(float(value), 4)
        self.reasons.append(
            f"{label} = {value:.3f}{unit} {_SYMBOL[op]} 门槛 {threshold:.3f}{unit} "
            f"→ {'命中' if hit else '未命中'}" + (f"（{note}）" if note else ""))
        self._hits.append(hit)
        return hit

    # ---------------------------------------------------------- 结论
    def finish(self, *, level: str = EVIDENCE_JUDGEMENT) -> ActionVerdict:
        if self._broken:
            absent = sorted(set(self.missing))
            return ActionVerdict(
                action=self._action, hit=False, confidence=0.0,
                evidence_level=EVIDENCE_INSUFFICIENT,
                reasons=self.reasons + [f"证据不足: 缺少可用关键点 {', '.join(absent)}"],
                missing=absent, measurements=self.measurements)

        hit = bool(self._hits) and all(self._hits)
        confidence = mean(self._confs) * (sum(self._hits) / len(self._hits))
        # 命中但可信度不足 → 降级为候选（有迹象, 不足以作判据）
        if hit and confidence < self._thresholds.min_hit_confidence:
            level = EVIDENCE_CANDIDATE
            self.reasons.append(
                f"命中但可信度 {confidence:.3f} 低于门槛 "
                f"{self._thresholds.min_hit_confidence:.3f} → 降级为候选")
        return ActionVerdict(action=self._action, hit=hit, confidence=confidence,
                             evidence_level=level, reasons=self.reasons,
                             measurements=self.measurements)


# ============================================================
# 通用几何helper
# ============================================================

def angle_at(vertex: Point, first: Point, second: Point) -> Optional[float]:
    """在 ``vertex`` 处的夹角（度）。任一向量退化（零长）时返回 None。"""
    ax, ay = first[0] - vertex[0], first[1] - vertex[1]
    bx, by = second[0] - vertex[0], second[1] - vertex[1]
    na, nb = math.hypot(ax, ay), math.hypot(bx, by)
    if na <= 0.0 or nb <= 0.0:
        return None
    cosine = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
    return math.degrees(math.acos(cosine))


def _torso(person: PersonPose, builder: _VerdictBuilder) -> Optional[float]:
    """取归一化基准（躯干长度）; 太小则视为不可信。"""
    length = person.torso_length()
    if length is None or length < MIN_TORSO_LENGTH_PX:
        builder.mark_missing(["shoulders/hips(躯干长度不可用)"])
        return None
    return length


def _point_distance_m(distance_fn: Optional[DistanceFn], first: Point,
                      second: Point) -> Optional[float]:
    """有标定就用米制距离; 没有返回 None（由调用方决定降级策略）。"""
    if distance_fn is None:
        return None
    try:
        return float(distance_fn(first, second))
    except Exception as exc:  # noqa: BLE001 —— 标定实现异常不该炸整条判据
        logger.warning("米制距离计算失败, 本次按无标定处理: %s: %s", type(exc).__name__, exc)
        return None


# ============================================================
# 一、高抬腿  high_leg
# ============================================================

def judge_high_leg(person: PersonPose, *, settings: Optional[PoseSettings] = None,
                   distance_fn: Optional[DistanceFn] = None) -> ActionVerdict:
    """高抬腿（抬脚过高 / high boot）。

    判据: ``(髋_y - 踝_y) / 躯干长度 >= 门槛``（图像 y 向下为正）。
    正常站立时踝在髋下方很远, 该量约 -1.5; 抬到与髋同高约 0; 高于髋为正。
    两条腿取更抬得高的那条。

    ★ 已知边界: 滑铲时人躺下, 髋与踝的相对关系会失真, 也可能触发本判据。
      真实系统应把本结论与 :func:`judge_sliding_tackle` 一起看（本模块不下复合结论）。
    """
    settings = settings or PoseSettings()
    thresholds = settings.thresholds
    builder = _VerdictBuilder(ACTION_HIGH_LEG, thresholds)

    usable_legs = [side for side in SIDES if person.get(LEGS[side][2]) is not None]
    if not usable_legs:
        builder.mark_missing(list(ANKLES))
        return builder.finish()
    if not builder.require(person, HIPS):
        return builder.finish()
    torso = _torso(person, builder)
    if torso is None:
        return builder.finish()

    hip = person.hip_center()
    best: Optional[Tuple[str, float]] = None
    for side in usable_legs:
        ankle = person.point(LEGS[side][2])
        if ankle is None:
            continue
        builder.note_points(person, [LEGS[side][2]])
        ratio = (hip[1] - ankle[1]) / torso
        if best is None or ratio > best[1]:
            best = (side, ratio)

    builder.check("踝高于髋的比例", best[1],
                  thresholds.high_leg_ankle_above_hip_ratio,
                  unit=" 躯干长", note=f"{best[0]} 腿最高")
    return builder.finish()


# ============================================================
# 二、滑铲  sliding_tackle
# ============================================================

def judge_sliding_tackle(person: PersonPose, *, settings: Optional[PoseSettings] = None,
                         distance_fn: Optional[DistanceFn] = None) -> ActionVerdict:
    """滑铲（铲球）。

    三条判据同时成立才算命中（缺一条都不给结论）:
        1. ``髋部高度 / 自身高度 <= 门槛``  —— 髋沉下去了（站立约 0.45~0.50）
        2. ``躯干相对竖直的倾角 >= 门槛``  —— 躯干倒下去了（跑动 < 20°）
        3. ``膝关节伸展角 >= 门槛``        —— 有一条腿伸出去（伸直约 180°）
    腿取伸得更直的那条。
    """
    settings = settings or PoseSettings()
    thresholds = settings.thresholds
    builder = _VerdictBuilder(ACTION_SLIDING_TACKLE, thresholds)

    if not builder.require(person, (*SHOULDERS, *HIPS)):
        return builder.finish()

    usable_legs = [side for side in SIDES
                   if person.get(LEGS[side][1]) is not None
                   and person.get(LEGS[side][2]) is not None]
    if not usable_legs:
        builder.mark_missing(["knees/ankles"])
        return builder.finish()

    hip_ratio = person.hip_height_ratio()
    incline = person.torso_incline_deg()
    if hip_ratio is None or incline is None:
        builder.mark_missing(["hip/torso 几何量"])
        return builder.finish()

    best_knee: Optional[Tuple[str, float]] = None
    for side in usable_legs:
        hip, knee, ankle = (person.point(name) for name in LEGS[side])
        angle = angle_at(knee, hip, ankle)
        if angle is None:
            continue
        builder.note_points(person, list(LEGS[side]))
        if best_knee is None or angle > best_knee[1]:
            best_knee = (side, angle)
    if best_knee is None:
        builder.mark_missing(["knees/ankles"])
        return builder.finish()

    builder.check("髋部高度比", hip_ratio, thresholds.slide_hip_height_ratio,
                  op=OP_LE, note="髋沉到自身高度的比例以下")
    builder.check("躯干倾角", incline, thresholds.slide_torso_incline_deg,
                  unit="°", note="躯干倒向地面")
    builder.check("膝关节伸展角", best_knee[1], thresholds.slide_knee_extension_deg,
                  unit="°", note=f"{best_knee[0]} 膝伸得更直")
    return builder.finish()


# ============================================================
# 三、肘击  elbow_strike
# ============================================================

def judge_elbow_strike(person: PersonPose, others: Sequence[PersonPose], *,
                       settings: Optional[PoseSettings] = None,
                       distance_fn: Optional[DistanceFn] = None) -> ActionVerdict:
    """肘击。

    两条判据同时成立（同一条手臂上）:
        1. ``上臂(肩→肘) 与 躯干轴(肩→髋) 的夹角 >= 门槛``
           —— 即肩关节外展把肘抬起来。手臂自然下垂时约 10~20°。
           ★ 说明: "肘关节外展"解剖上不成立（肘只能屈伸）, 真正抬肘来自肩外展,
             所以这里量的是上臂相对躯干抬起的角度。
        2. ``肘 → 最近对手头部 的距离 / 自身躯干长度 <= 门槛``

    ``others`` 里没有可用对手时返回证据不足（肘击必须有个被肘的人）。
    """
    settings = settings or PoseSettings()
    thresholds = settings.thresholds
    builder = _VerdictBuilder(ACTION_ELBOW_STRIKE, thresholds)

    if not builder.require(person, (*SHOULDERS, *HIPS)):
        return builder.finish()
    torso = _torso(person, builder)
    if torso is None:
        return builder.finish()

    # 候选对手: 头部可用的其它人
    candidates = [other for other in others
                  if other is not person and other.head_center() is not None]
    if not candidates:
        builder.mark_missing(["others(head)"])
        builder.reasons.append("没有可用于比对的对手头部 —— 肘击必须有个被肘的人")
        return builder.finish()

    hip_center = person.hip_center()
    best: Optional[Tuple[str, float, float, float]] = None
    for side in SIDES:
        shoulder_name, elbow_name = ARMS[side][0], ARMS[side][1]
        if person.get(shoulder_name) is None or person.get(elbow_name) is None:
            continue
        shoulder, elbow = person.point(shoulder_name), person.point(elbow_name)
        reference_hip = person.point(ARMS[side][3]) or hip_center
        abduction = angle_at(shoulder, elbow, reference_hip)
        if abduction is None:
            continue
        builder.note_points(person, [shoulder_name, elbow_name])
        for other in candidates:
            head = other.head_center()
            ratio = math.dist(elbow, head) / torso
            if best is None or ratio < best[1]:
                best = (side, ratio, abduction, other)

    if best is None:
        builder.mark_missing(["elbows(无可用手臂)"])
        return builder.finish()

    side, ratio, abduction, other = best
    builder.check("肩关节外展角", abduction, thresholds.elbow_abduction_deg,
                  unit="°", note=f"{side} 臂抬起")
    builder.check("肘到最近对手头部距离", ratio, thresholds.elbow_head_distance_ratio,
                  op=OP_LE, unit=" 躯干长", note=f"对手 #{other.index}")
    return builder.finish()


# ============================================================
# 四、拉人推人  pulling_pushing
# ============================================================

def judge_pulling_pushing(person: PersonPose, others: Sequence[PersonPose], *,
                          settings: Optional[PoseSettings] = None,
                          distance_fn: Optional[DistanceFn] = None) -> ActionVerdict:
    """拉人 / 推人（手部对对手的抓拽与推搡）。

    三条判据同时成立（同一条手臂上）:
        1. ``手臂张开角 = 在肩处量 (肩-腕) 与躯干轴(肩-髋) 的夹角 >= 门槛``
           —— **这才是关键判据**。实测基准: 自然下垂约 10°, 伸手抓人/推人约 60~90°。
        2. ``肩-腕距离 / 躯干长度 >= 门槛`` —— 守卫: 手臂没有折叠
           （自然下垂的手臂也能过这条, 那是预期的: 它守的是"别是屈着的"）。
        3. 手部到对手躯干中心的距离:
             · 有标定 → ``<= pull_contact_distance_m``（米）→ 判据
             · 无标定 → ``<= pull_contact_distance_ratio``（躯干长）→ **候选**

    ★ 为什么第 1 条不能用"肩-腕距离比"代替: 自然下垂的手臂本来也是完全伸直的,
      那个比值是 1.08 —— 它区分不了"垂着"和"伸出去", 会把每个站着的人都判成在拉人。

    ★ 这是唯一真正需要米的判据: "手离对手多近算碰到"是物理量。
      没有标定时按身体比例近似会忽略镜头远近, 所以降级为候选。
    """
    settings = settings or PoseSettings()
    thresholds = settings.thresholds
    builder = _VerdictBuilder(ACTION_PULLING_PUSHING, thresholds)

    if not builder.require(person, (*SHOULDERS, *HIPS)):
        return builder.finish()
    torso = _torso(person, builder)
    if torso is None:
        return builder.finish()

    candidates = [other for other in others
                  if other is not person and other.torso_center() is not None]
    if not candidates:
        builder.mark_missing(["others(torso)"])
        builder.reasons.append("没有可用于比对的对手躯干 —— 拉人推人必须有个被拉/被推的人")
        return builder.finish()

    # ---- 判据 1+2: 选张开得最大的那条手臂（同分时取伸得更直的）----
    hip_center = person.hip_center()
    best: Optional[Tuple[str, float, float]] = None      # (侧, 张开角, 伸展度)
    for side in SIDES:
        shoulder_name, _, wrist_name, hip_name = ARMS[side]
        if person.get(shoulder_name) is None or person.get(wrist_name) is None:
            continue
        shoulder, wrist = person.point(shoulder_name), person.point(wrist_name)
        reference_hip = person.point(hip_name) or hip_center
        if reference_hip is None:
            continue
        abduction = angle_at(shoulder, wrist, reference_hip)
        if abduction is None:
            continue
        reach = math.dist(shoulder, wrist) / torso
        builder.note_points(person, [shoulder_name, wrist_name])
        if best is None or abduction > best[1] or (abduction == best[1] and reach > best[2]):
            best = (side, abduction, reach)
    if best is None:
        builder.mark_missing(["wrists(无可用手臂)"])
        return builder.finish()

    side, abduction, reach = best
    _, _, wrist_name, _ = ARMS[side]
    builder.check("手臂张开角", abduction, thresholds.pull_reach_abduction_deg,
                  unit="°", note=f"{side} 臂抬离身体")
    builder.check("手臂伸展度", reach, thresholds.pull_reach_extension_ratio,
                  unit=" 躯干长", note=f"{side} 臂未折叠")

    # ---- 判据 2: 接触距离（有标定用米, 无标定用躯干长并降级为候选）----
    wrist = person.point(wrist_name)
    nearest: Optional[Tuple[float, PersonPose, str]] = None
    use_meters = distance_fn is not None
    for other in candidates:
        target = other.torso_center()
        if target is None:
            continue
        if use_meters:
            value = _point_distance_m(distance_fn, wrist, target)
            if value is None:
                continue
        else:
            value = math.dist(wrist, target) / torso
        if nearest is None or value < nearest[0]:
            nearest = (value, other, "m" if use_meters else " 躯干长")
    if nearest is None:
        builder.mark_missing(["others(torso)"])
        return builder.finish()

    value, other, unit = nearest
    if use_meters and settings.allow_uncalibrated_candidate:
        builder.check("手到对手躯干距离", value, thresholds.pull_contact_distance_m,
                      op=OP_LE, unit="m", note=f"对手 #{other.index}（米制, 有标定）")
        level = EVIDENCE_JUDGEMENT
    else:
        # 无标定: 按身体比例近似 —— 只能当候选
        builder.check("手到对手躯干距离", value, thresholds.pull_contact_distance_ratio,
                      op=OP_LE, unit=unit,
                      note=f"对手 #{other.index}（按身体比例近似, 未标定）")
        level = EVIDENCE_CANDIDATE
        if not settings.allow_uncalibrated_candidate:
            builder.mark_missing(["calibration"])
            builder.reasons.append(
                "配置不允许无标定近似, 而本次没有可用的米制距离函数 → 不下结论")
            return builder.finish()
    return builder.finish(level=level)


# ============================================================
# 注册表: 让调用方按名字批量判, 不必自己写 if/elif 分派
# ============================================================
# 单人判据（(person, **kw)）
SINGLE_PERSON_JUDGES: Dict[str, Callable[..., ActionVerdict]] = {
    ACTION_HIGH_LEG: judge_high_leg,
    ACTION_SLIDING_TACKLE: judge_sliding_tackle,
}
# 双人判据（(person, others, **kw)）
PAIR_JUDGES: Dict[str, Callable[..., ActionVerdict]] = {
    ACTION_ELBOW_STRIKE: judge_elbow_strike,
    ACTION_PULLING_PUSHING: judge_pulling_pushing,
}
ACTION_JUDGES: Tuple[str, ...] = (
    ACTION_HIGH_LEG, ACTION_SLIDING_TACKLE, ACTION_ELBOW_STRIKE, ACTION_PULLING_PUSHING,
)


def judge_all(person: PersonPose, others: Sequence[PersonPose] = (), *,
              settings: Optional[PoseSettings] = None,
              distance_fn: Optional[DistanceFn] = None) -> List[ActionVerdict]:
    """对一个人跑全部四个判据（按 ``ACTION_JUDGES`` 顺序）。"""
    verdicts: List[ActionVerdict] = []
    for judge in SINGLE_PERSON_JUDGES.values():
        verdicts.append(judge(person, settings=settings, distance_fn=distance_fn))
    for judge in PAIR_JUDGES.values():
        verdicts.append(judge(person, others, settings=settings, distance_fn=distance_fn))
    return [v for v in verdicts if v.action in ACTION_JUDGES]
