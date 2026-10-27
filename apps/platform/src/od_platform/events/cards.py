#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : cards.py
# @Project   : ODPlatform / events
# @Function  : 红牌 / 黄牌事件的判定（含 HSV 颜色二次确认）

"""红牌 / 黄牌事件的判定。

★ 能力边界（同 ``od_platform.events`` 的说明, 这里再强调一次）

    本模块**不是**"已经能识别红黄牌" —— 它需要**一个能检出"牌"这个类别的模型**。
    当前仓库的足球权重只有 ``player`` / ``football`` 两类, 检不出牌, 所以
    **本模块在现有权重上不会产出任何事件**（检测结果里根本没有牌）。
    要跑起来: 先把数据集扩到含 牌 / 裁判 类别并训练。

★ 判定链路（每一步都给得出理由, 答辩会问"凭什么"）

    1. 从一帧检测里挑出"牌"—— 类别名走配置层解析, 逻辑里不写类名字面量;
    2. 形状与置信度过滤: 面积占比、宽高比、置信度 —— 不像牌的**直接不算事件**
       （注意与"证据不足"的区别: 这里说"这压根不像牌", 不是"像牌但没确认"）;
    3. 找出示者: 优先**裁判**（牌是裁判掏的）, 没有裁判才退到最近的球员;
       距离用【出示者框高的倍数】衡量 —— 于是完全不依赖标定, 也不受镜头远近影响;
       太远 → **不构成出示**, 直接不算事件;
    4. **HSV 颜色二次确认**: 对牌的裁剪区域算红/黄像素占比。
       模型可能把黄牌判成红牌, 所以:
         · 颜色与类别一致 → 确认;
         · 颜色与类别矛盾 / 区域里既不明显红也不明显黄 → **降级为"证据不足"**,
           置信度被压到 ``unconfirmed_confidence_ceiling`` 以下, 并写明原因;
         · 模型只给通用 ``card`` 类时, 由**颜色**决定是红牌还是黄牌。

★ ``frame`` 是 BGR（与 cv2.imread 一致）。不给 frame 也能跑, 但颜色确认无从做起,
  于是同类事件会**降级为"证据不足"**而不是偷偷信模型 —— 这是有意的。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from od_platform.events import (
    KIND_CARD,
    KIND_RED_CARD,
    KIND_SEVERITY,
    KIND_YELLOW_CARD,
    ROLE_GENERIC_CARD,
    ROLE_PLAYER,
    ROLE_RED_CARD,
    ROLE_REFEREE,
    ROLE_YELLOW_CARD,
    Detection,
    PenaltyEvent,
    cap_confidence,
    image_size_of,
    in_area_ratio_range,
    in_aspect_ratio_range,
    parse_detections,
)
from od_platform.events.settings import ColorCheckSettings, EventSettings

logger = logging.getLogger(__name__)

# 角色 → 事件类型（模型直接给了具体牌类时用它; 通用牌类由颜色决定）
_ROLE_TO_KIND: Mapping[str, str] = {
    ROLE_RED_CARD: KIND_RED_CARD,
    ROLE_YELLOW_CARD: KIND_YELLOW_CARD,
    ROLE_GENERIC_CARD: KIND_CARD,
}

# 颜色名（判据里给人看的字眼, 不是类别名）
_COLOR_NAMES: Mapping[str, str] = {"red": "红", "yellow": "黄"}

# 牌型的中文名（判据/摘要用）
_KIND_NAMES: Mapping[str, str] = {
    KIND_RED_CARD: "红牌",
    KIND_YELLOW_CARD: "黄牌",
    KIND_CARD: "牌(牌型未定)",
}


@dataclass(frozen=True)
class ColorVerdict:
    """颜色二次确认的结论。"""

    kind: str                                    # 最终认定的牌型
    confirmed: bool                              # 颜色是否支持这个牌型
    reasons: Tuple[str, ...] = ()
    evidence: Dict[str, Any] = field(default_factory=dict)


def _mask_ratio(cv2: Any, hsv: Any, ranges: Sequence[Sequence[int]], total: float) -> float:
    """HSV 图里落在给定各段范围内的像素占比。"""
    mask = None
    for low_high in ranges:
        h_min, s_min, v_min, h_max, s_max, v_max = (int(v) for v in low_high)
        piece = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))
        mask = piece if mask is None else cv2.bitwise_or(mask, piece)
    if mask is None or total <= 0:
        return 0.0
    return float(cv2.countNonZero(mask)) / total


def measure_color_ratios(frame: Any, bbox: Tuple[int, int, int, int],
                         settings: ColorCheckSettings) -> Optional[Dict[str, float]]:
    """算牌区域内红/黄像素的占比。

    Returns:
        ``{"red": r, "yellow": y, "crop_pixels": n}``;
        算不出来（没装 cv2 / 裁剪区域退化）时返回 None —— 调用方必须据此降级,
        **不能**把 None 当成 0.0（"没量"不等于"量出来是 0"）。
    """
    try:
        import cv2
    except ImportError:                       # pragma: no cover - 取决于环境
        return None
    if frame is None:
        return None
    shape = getattr(frame, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    height, width = int(shape[0]), int(shape[1])
    x1, y1, x2, y2 = bbox
    # 框可能越界（检测框常常贴边）, 先夹进画面再裁
    x1 = max(0, min(int(x1), width))
    x2 = max(0, min(int(x2), width))
    y1 = max(0, min(int(y1), height))
    y2 = max(0, min(int(y2), height))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    crop = frame[y1:y2, x1:x2]
    total = float(crop.shape[0] * crop.shape[1])
    if total < settings.min_crop_pixels:
        return None
    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    except Exception:                          # noqa: BLE001 —— 非法图像格式/通道数
        return None
    return {
        "red": _mask_ratio(cv2, hsv, settings.red_ranges, total),
        "yellow": _mask_ratio(cv2, hsv, settings.yellow_ranges, total),
        "crop_pixels": total,
    }


class CardEventDetector:
    """逐帧判定"是否有人出示了红/黄牌"。

    用法::

        detector = CardEventDetector(EventSettings.load())
        for index, (frame, dets) in enumerate(source):
            for event in detector.detect(dets, frame=frame, frame_index=index):
                logger.info(event.describe())

    切换视频/输入源时必须调 :meth:`reset`, 否则新视频开头会被上一段的冷却压住。

    Args:
        settings: 全部阈值; None 时用默认值（不读文件, 由调用方决定要不要 load）
        homography: 可选的场地单应; 本模块**不依赖**它（距离用相对尺度衡量）,
            保留该参数是为了将来要按米制距离复核时不必改签名
    """

    def __init__(self, settings: Optional[EventSettings] = None, *, homography: Any = None):
        self.settings = settings or EventSettings()
        self.homography = homography
        self._last_frame_of_kind: Dict[str, int] = {}
        self._tick = 0

    # ---------------------------------------------------------------- 状态
    def reset(self) -> None:
        """清空冷却记忆。"""
        self._last_frame_of_kind.clear()
        self._tick = 0

    def _next_tick(self, frame_index: Optional[int]) -> int:
        """冷却用的时间轴。

        调用方给了 ``frame_index`` 就用它（最准）; 没给就自己数调用次数 ——
        于是"不传帧号"也能有冷却, 不会变成每帧重复报同一个事件。
        """
        if frame_index is not None:
            return int(frame_index)
        self._tick += 1
        return self._tick

    def _cooldown_allows(self, kind: str, tick: int) -> bool:
        last = self._last_frame_of_kind.get(kind)
        return last is None or (tick - last) >= self.settings.cards.cooldown_frames

    # ---------------------------------------------------------------- 主流程
    def detect(self, raw_detections: Any, *, frame: Any = None,
               frame_index: Optional[int] = None) -> List[PenaltyEvent]:
        """判定当前帧的红/黄牌事件。

        Args:
            raw_detections: ``[{"cls","conf","xyxy"}]``（与 from_yolo_results 同格式）
            frame: BGR 图像; 给了才能做颜色二次确认, 不给则相关事件降级为证据不足
            frame_index: 帧号; 用于冷却。不给则按内部调用计数

        Returns:
            本帧确认或降级的事件（可能为空）
        """
        detections = parse_detections(raw_detections)
        if not detections:
            return []
        cards_cfg = self.settings.cards
        classes = self.settings.classes
        image_size = image_size_of(frame)
        tick = self._next_tick(frame_index)

        card_dets = [(det, classes.resolve_role(det.cls)) for det in detections]
        card_dets = [(det, role) for det, role in card_dets if role in _ROLE_TO_KIND]
        if not card_dets:
            return []

        anchors = self._usable_anchors(detections)
        events: List[PenaltyEvent] = []
        for det, role in card_dets:
            # --- 1) 置信度 / 形状过滤: 不通过就"不像牌", 直接不算事件
            reason = self._reject_reason(det, image_size)
            if reason:
                logger.debug("跳过疑似牌 %s: %s", det.cls, reason)
                continue

            # --- 2) 找"出示者"
            anchor = self._nearest_anchor(det, anchors)
            distance_px, distance_heights = self._anchor_distance(det, anchor)
            if anchor is not None and distance_heights > cards_cfg.max_show_distance_body_heights:
                # 太远 → 不构成"出示"（牌可能只是躺在草地上/远处别的牌子）
                logger.debug("牌距离出示者 %.2f 个身高, 超过阈值 %.2f, 不构成出示",
                             distance_heights, cards_cfg.max_show_distance_body_heights)
                continue
            if anchor is not None and cards_cfg.require_card_above_anchor \
                    and det.centre[1] >= anchor.centre[1]:
                logger.debug("配置要求牌位于出示者中心之上, 不满足, 不构成出示")
                continue

            # --- 3) 颜色二次确认
            verdict = self.evaluate_color(frame, det, role)

            reasons: List[str] = list(verdict.reasons)
            confirmed = verdict.confirmed
            if anchor is None:
                confirmed = False
                reasons.append("未检出可用的裁判/球员框, 无法确认是谁出示的牌")
            else:
                reasons.append(
                    f"距离出示者({anchor.cls}) {distance_heights:.2f} 个框高"
                    f"(阈值 {cards_cfg.max_show_distance_body_heights:g}), 判定为出示")
            reasons.append(f"{_KIND_NAMES.get(verdict.kind, verdict.kind)}"
                           f" 严重级 {KIND_SEVERITY.get(verdict.kind, 0)}")

            if not self._cooldown_allows(verdict.kind, tick):
                # 冷却期内不再重复报同一个事件（同一张牌会连续多帧被检到）
                continue

            ceiling = cards_cfg.unconfirmed_confidence_ceiling
            base_conf = det.conf if det.conf is not None else ceiling
            evidence = dict(verdict.evidence)
            evidence.update({
                "card_class": det.cls,
                "card_role": role,
                "detection_confidence": det.conf,
                "aspect_ratio": round(det.aspect_ratio(), 4),
                "distance_to_anchor_px": (round(distance_px, 2)
                                          if distance_px is not None else None),
                "anchor_height_px": (round(anchor.height, 2) if anchor else None),
                "distance_body_heights": (round(distance_heights, 4)
                                          if math.isfinite(distance_heights) else None),
                "severity": KIND_SEVERITY.get(verdict.kind, 0),
                "thresholds": {
                    "min_confidence": cards_cfg.min_confidence,
                    "area_ratio_range": [cards_cfg.min_area_ratio, cards_cfg.max_area_ratio],
                    "aspect_ratio_range": [cards_cfg.min_aspect_ratio,
                                           cards_cfg.max_aspect_ratio],
                    "max_show_distance_body_heights":
                        cards_cfg.max_show_distance_body_heights,
                    "color_min_pixel_ratio": cards_cfg.color.min_pixel_ratio,
                },
                "cooldown_frames": cards_cfg.cooldown_frames,
            })
            events.append(PenaltyEvent(
                kind=verdict.kind, confirmed=confirmed,
                confidence=cap_confidence(base_conf, confirmed, ceiling),
                bbox=det.bbox, frame_index=frame_index, reasons=tuple(reasons),
                evidence=evidence,
                anchor_role=(self.settings.classes.resolve_role(anchor.cls)
                             if anchor else ""),
                anchor_class=(anchor.cls if anchor else ""),
                anchor_bbox=(anchor.bbox if anchor else None),
            ))
            self._last_frame_of_kind[verdict.kind] = tick
        return events

    # ---------------------------------------------------------------- 颜色确认
    def evaluate_color(self, frame: Any, det: Detection, role: str) -> ColorVerdict:
        """对牌的裁剪区域做颜色二次确认, 返回最终牌型与是否确认。

        规则（答辩会被追问, 所以写清楚）:
          · 颜色二次确认关闭 → 以模型类别为准（并如实标注"已关闭"）;
          · 没给 frame → **降级**（"没量过"不能当成"量过了"）;
          · 区域里红/黄占比都低于 ``min_pixel_ratio`` → 颜色无法成立 →
            具体牌类**降级**; 通用牌类**无法定牌型**;
          · 颜色与模型类别矛盾 → 按 ``conflict_policy``:
            ``downgrade``（默认）降级为证据不足; ``trust_model`` 仍采信模型但记下矛盾;
          · 模型只给通用 ``card`` 类 → **由颜色决定**是红牌还是黄牌。
        """
        color_cfg = self.settings.cards.color
        claimed = _ROLE_TO_KIND.get(role, KIND_CARD)

        if not color_cfg.enabled:
            return ColorVerdict(claimed, True,
                                ("颜色二次确认已按配置关闭, 牌型以模型类别为准",),
                                {"color_check": "disabled"})

        if frame is None:
            return ColorVerdict(claimed, False,
                                ("未提供图像帧, 无法做颜色二次确认（不假装确定）",),
                                {"color_check": "no_frame"})

        ratios = measure_color_ratios(frame, det.bbox, color_cfg)
        if ratios is None:
            return ColorVerdict(claimed, False,
                                ("牌的裁剪区域无法度量颜色（区域退化或缺少 cv2）, "
                                 "颜色二次确认未完成",),
                                {"color_check": "unmeasurable"})

        red, yellow = ratios["red"], ratios["yellow"]
        threshold = color_cfg.min_pixel_ratio
        evidence: Dict[str, Any] = {
            "color_check": "ok",
            "red_pixel_ratio": round(red, 4),
            "yellow_pixel_ratio": round(yellow, 4),
            "color_min_pixel_ratio": threshold,
            "crop_pixels": ratios["crop_pixels"],
        }

        if max(red, yellow) < threshold:
            observed = None
            detail = (f"裁剪区域里既没有明显的红也没有明显的黄"
                      f"(红 {red:.1%} / 黄 {yellow:.1%}, 阈值 {threshold:.1%})")
        else:
            observed = "red" if red >= yellow else "yellow"
            detail = (f"裁剪区域以{_COLOR_NAMES[observed]}色为主"
                      f"(红 {red:.1%} / 黄 {yellow:.1%})")

        # --- 通用牌类: 颜色是唯一的牌型依据
        if claimed == KIND_CARD:
            if observed is None:
                return ColorVerdict(KIND_CARD, False,
                                    (f"模型只给了通用'牌'类, 而{detail}, 无法定牌型",),
                                    evidence)
            kind = KIND_RED_CARD if observed == "red" else KIND_YELLOW_CARD
            return ColorVerdict(kind, True,
                                (f"模型只给了通用'牌'类, 由颜色判定为"
                                 f"{_KIND_NAMES[kind]}（{detail}）",), evidence)

        # --- 具体牌类: 颜色用来复核
        wanted = "red" if claimed == KIND_RED_CARD else "yellow"
        if observed == wanted:
            return ColorVerdict(claimed, True,
                                (f"颜色二次确认通过: {detail}, 与模型类别"
                                 f"({_KIND_NAMES[claimed]})一致",), evidence)

        conflict = (f"颜色与类别{'矛盾' if observed else '不符'}: 模型判为"
                    f"{_KIND_NAMES[claimed]}, 但{detail}")
        if color_cfg.conflict_policy == "trust_model":
            return ColorVerdict(claimed, True,
                                (conflict + " —— 配置为 trust_model, 仍按模型类别采信",),
                                evidence)
        return ColorVerdict(claimed, False,
                            (conflict + " —— 以模型为准会误判, 故降级为证据不足",),
                            evidence)

    # ---------------------------------------------------------------- 内部工具
    def _reject_reason(self, det: Detection,
                       image_size: Optional[Tuple[int, int]]) -> str:
        """"这压根不像牌"的判断; 返回空串表示通过。

        与"证据不足"是两回事: 这里不合格的**根本不产生事件**（连候选都不是）。
        """
        cfg = self.settings.cards
        if det.conf is not None and det.conf < cfg.min_confidence:
            return f"置信度 {det.conf:.3f} 低于阈值 {cfg.min_confidence:g}"
        ratio_ok = in_area_ratio_range(det, cfg.min_area_ratio, cfg.max_area_ratio,
                                       image_size)
        if ratio_ok is False:
            return (f"面积占比 {det.normalised_area(image_size):.2e} 不在 "
                    f"[{cfg.min_area_ratio:g}, {cfg.max_area_ratio:g}] 内")
        if not in_aspect_ratio_range(det, cfg.min_aspect_ratio, cfg.max_aspect_ratio):
            return (f"宽高比 {det.aspect_ratio():.3f} 不在 "
                    f"[{cfg.min_aspect_ratio:g}, {cfg.max_aspect_ratio:g}] 内")
        return ""

    def _usable_anchors(self, detections: Sequence[Detection]) -> List[Detection]:
        """能当"出示者"的检测: 裁判或球员, 且框有高度（否则无法按相对尺度算距离）。"""
        classes = self.settings.classes
        anchors: List[Detection] = []
        for det in detections:
            role = classes.resolve_role(det.cls)
            if role in (ROLE_REFEREE, ROLE_PLAYER) and det.height > 0:
                anchors.append(det)
        return anchors

    def _nearest_anchor(self, det: Detection,
                        anchors: Sequence[Detection]) -> Optional[Detection]:
        """找离牌最近的出示者; 优先裁判（可配）。"""
        if not anchors:
            return None
        pool: Sequence[Detection] = anchors
        if self.settings.cards.prefer_referee:
            referees = [a for a in anchors
                        if self.settings.classes.resolve_role(a.cls) == ROLE_REFEREE]
            if referees:
                pool = referees
        return min(pool, key=lambda a: _centre_distance(det, a))

    @staticmethod
    def _anchor_distance(det: Detection,
                         anchor: Optional[Detection]) -> Tuple[Optional[float], float]:
        """(像素距离, 以出示者框高为单位的距离)。没有出示者时距离记为无穷大。"""
        if anchor is None or anchor.height <= 0:
            return None, math.inf
        distance = _centre_distance(det, anchor)
        return distance, distance / anchor.height


def _centre_distance(a: Detection, b: Detection) -> float:
    ax, ay = a.centre
    bx, by = b.centre
    return math.hypot(ax - bx, ay - by)