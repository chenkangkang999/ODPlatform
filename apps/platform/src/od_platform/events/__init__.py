#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : __init__.py
# @Project   : ODPlatform / events
# @Function  : 判罚事件（红牌 / 黄牌 / 换人）—— 共用模型与入口

"""``od_platform.events`` —— 判罚事件子系统（红牌 / 黄牌 / 换人）。

★ 能力边界（先看这一段, 别把这里当成"已支持红黄牌识别"）

    红牌/黄牌本质上要【把"牌"当物体检测】——裁判手里的一块红/黄矩形;
    换人要靠【第四官员举着的换人牌】。所以这三项**强依赖一个能检出这些类别的模型**。

    而本仓库当前的足球权重**只有** ``player`` / ``football`` 两类, **检不出牌**。
    因此本模块交付的是【判定逻辑 + 一处配置化的类别名映射】:

      · 类别名不写死, 全部走 :class:`~od_platform.events.settings.ClassNameSettings`
        （支持别名列表、大小写不敏感, 模型叫 ``red_card``/``红牌``/``card_red`` 都认）;
      · **没训练出这些类别时, 本模块一帧也不会产出事件** —— 这是设计行为不是 bug
        （检测结果里根本没有牌, 自然无从判定）。

    要真正跑起来: 先把数据集扩到包含 牌 / 裁判 / 换人牌 这些类别并训练。

★ 三类事件怎么判（判据都可解释, 答辩会问"凭什么")

    红 / 黄牌（:mod:`~od_platform.events.cards`）
        检出"牌" → 找最近的**裁判**（找不到就退到最近的球员）当"出示者"
        → 牌到出示者的距离必须小于【出示者框高的 N 倍】才算"出示"
        → 再对牌的裁剪区域做 **HSV 颜色二次确认**（防模型把黄牌判成红牌）
        → 颜色与类别矛盾时**降级为"证据不足"**, 不直接信模型。

    换人（:mod:`~od_platform.events.substitution`）
        检出换人牌 → 有【可信标定】时把牌的接地点映射成场地坐标,
        判断是否在边线附近; **没有标定时降级为候选**, 明确写"未标定",
        置信度也被压住 —— 不假装确定。

★ 为什么共用模型放在这个包文件里

    本子系统被限定只能新建 4 个模块文件, 而 ``cards`` 与 ``substitution`` 必须
    产出同一种事件对象。放在这里, 两个子模块 ``from od_platform.events import ...``
    即可, 且**不形成循环导入** —— ``__init__`` 先把共用模型定义完, 再在文件末尾
    导入子模块（Python 允许, 因为此时包模块已在 sys.modules 里且所需名字已绑定）。

★ 输入格式

    检测结果与 :meth:`od_platform.visualization.BeautifyVisualizer.from_yolo_results`
    的输出**同格式**, 所以那一路的产物可以原样喂进来::

        [{"cls": "red_card", "conf": 0.88, "xyxy": [x1, y1, x2, y2]}, ...]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from od_platform.events.settings import (   # noqa: F401  (对外再导出配置层)
    KIND_CARD,
    KIND_RED_CARD,
    KIND_SEVERITY,
    KIND_SUBSTITUTION,
    KIND_YELLOW_CARD,
    ROLE_GENERIC_CARD,
    ROLE_PLAYER,
    ROLE_RED_CARD,
    ROLE_REFEREE,
    ROLE_SUBSTITUTION_BOARD,
    ROLE_YELLOW_CARD,
    CardSettings,
    ClassNameSettings,
    ColorCheckSettings,
    EventSettings,
    SubstitutionSettings,
    default_config_path,
)

# 检测框坐标只取前 4 个数, 与 from_yolo_results 的约定一致
_BBOX_LENGTH = 4


@dataclass(frozen=True)
class Detection:
    """一条检测结果。

    与 :meth:`BeautifyVisualizer.from_yolo_results` 的输出一一对应, 但把
    "算面积/宽高比/中心"这些反复要用的小学运算收在一处。

    ``conf`` 用 Optional 是刻意的: ``from_yolo_results`` 在没有置信度数组时
    **不会**写 ``conf`` 字段。把缺失当成 0.0 会让"没有置信度"变成"置信度极低"
    而被阈值滤掉 —— **"不知道"不等于"0"**, 所以这里保留 None, 由判据决定
    是否过滤（见各 detector 的实现）。
    """

    cls: str
    x1: float
    y1: float
    x2: float
    y2: float
    conf: Optional[float] = None

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centre(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """取整后的框（画图/裁剪用）。"""
        return (int(round(self.x1)), int(round(self.y1)),
                int(round(self.x2)), int(round(self.y2)))

    @property
    def bottom_centre(self) -> Tuple[float, float]:
        """框的底边中点 —— 判断"接地位置"的通用约定。

        物体站在地面/贴着边线时, 接地的是它的**下边缘中点**而不是框中心;
        拿中心去换场地坐标会系统性地偏向场地内侧（透视越远越明显）。
        """
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def aspect_ratio(self) -> float:
        """宽 / 高; 高度为 0 时返回 0.0（调用方按"非法比例"处理）。"""
        height = self.height
        return self.width / height if height > 0 else 0.0

    def normalised_area(self, image_size: Optional[Tuple[int, int]]) -> Optional[float]:
        """面积占画面的比例; 不知道画面尺寸时返回 None。"""
        if not image_size:
            return None
        width, height = image_size
        if width <= 0 or height <= 0:
            return None
        return self.area / float(width * height)

    def to_dict(self) -> Dict[str, Any]:
        return {"cls": self.cls, "conf": self.conf, "xyxy": [self.x1, self.y1, self.x2, self.y2]}


def parse_detections(raw: Any) -> List[Detection]:
    """把 ``[{"cls","conf","xyxy"}]`` 解析成 :class:`Detection` 列表。

    宽容但不装懂:
      · 缺 ``conf`` → None（不编 0.0, 见 Detection 的说明）;
      · ``xyxy`` 不足 4 个数或转不成数 → **跳过该条**（宁可少判, 不可乱判）;
      · 缺 ``cls`` → 空字符串（后续 resolve_role 认不出, 自然被忽略）。
    """
    detections: List[Detection] = []
    for item in raw or ():
        if isinstance(item, Detection):
            detections.append(item)
            continue
        if not isinstance(item, Mapping):
            continue
        coords = _as_floats(item.get("xyxy"))
        if coords is None:
            continue
        raw_conf = item.get("conf")
        try:
            conf = None if raw_conf is None else float(raw_conf)
        except (TypeError, ValueError):
            conf = None
        detections.append(Detection(cls=str(item.get("cls") or ""), conf=conf,
                                    x1=coords[0], y1=coords[1],
                                    x2=coords[2], y2=coords[3]))
    return detections


def _as_floats(values: Any) -> Optional[Tuple[float, float, float, float]]:
    """取前 4 个数并转 float; 数量不够或转不动就返回 None。"""
    if values is None:
        return None
    try:
        items = list(values)
    except TypeError:
        return None
    if len(items) < _BBOX_LENGTH:
        return None
    try:
        return (float(items[0]), float(items[1]), float(items[2]), float(items[3]))
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class PenaltyEvent:
    """一次判罚判定结果（把"凭什么这么判"一并带上）。

    ★ ``confirmed`` 的含义（答辩必问, 也是本模块最核心的一条设计）

        True  = 判据齐备, 可以当成"确实发生了";
        False = **证据不足** —— 颜色与类别矛盾 / 没有可信标定 / 找不到出示者……
                此时仍然返回事件（否则调用方不知道"这里有个可疑目标"）,
                但 ``confidence`` 已被压到配置的
                ``unconfirmed_confidence_ceiling`` 以下, 汇总时不会被当成确定结论。
                ``reasons`` 里写清了**为什么降级**, ``evidence`` 里给了原始数字。
    """

    kind: str
    confirmed: bool
    confidence: float
    bbox: Tuple[int, int, int, int]
    reasons: Tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)
    frame_index: Optional[int] = None
    anchor_role: str = ""
    anchor_class: str = ""
    anchor_bbox: Optional[Tuple[int, int, int, int]] = None

    @property
    def severity(self) -> int:
        """严重级: 红牌(2) >= 黄牌(1) > 换人(0)。见 settings.KIND_SEVERITY。"""
        return KIND_SEVERITY.get(self.kind, 0)

    @property
    def is_card(self) -> bool:
        return self.kind in (KIND_RED_CARD, KIND_YELLOW_CARD, KIND_CARD)

    def describe(self) -> str:
        """一行中文摘要（日志/报告用）。"""
        state = "确认" if self.confirmed else "证据不足"
        head = f"[{state}] {self.kind} 置信度 {self.confidence:.2f}"
        if self.anchor_class:
            head += f" ← {self.anchor_class}"
        return head + (":" + "; ".join(self.reasons) if self.reasons else "")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "confirmed": self.confirmed,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
            "frame_index": self.frame_index,
            "anchor_role": self.anchor_role,
            "anchor_class": self.anchor_class,
            "anchor_bbox": list(self.anchor_bbox) if self.anchor_bbox else None,
            "reasons": list(self.reasons),
            "evidence": dict(self.evidence),
        }


def cap_confidence(confidence: float, confirmed: bool, ceiling: float) -> float:
    """未确认的事件把置信度压到上限以下 —— 数值上也不假装确定。"""
    if confirmed:
        return float(confidence)
    return float(min(confidence, ceiling))


def image_size_of(frame: Any) -> Optional[Tuple[int, int]]:
    """从 BGR 图像取 (宽, 高); 取不到返回 None（按"不知道画面尺寸"处理）。"""
    shape = getattr(frame, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    height, width = int(shape[0]), int(shape[1])
    return (width, height) if width > 0 and height > 0 else None


def in_area_ratio_range(det: Detection, low: float, high: float,
                        image_size: Optional[Tuple[int, int]]) -> Optional[bool]:
    """面积占比是否落在 [low, high] 内; 不知道画面尺寸时返回 None（不据此否决）。"""
    ratio = det.normalised_area(image_size)
    if ratio is None:
        return None
    return low <= ratio <= high


def in_aspect_ratio_range(det: Detection, low: float, high: float) -> bool:
    return low <= det.aspect_ratio() <= high


# ---------------------------------------------------------------- 子模块入口
# 放在文件末尾是【必须的】: cards / substitution 会 `from od_platform.events import
# Detection, PenaltyEvent`, 此时它们必须已经绑定在上面 —— 否则循环导入会拿到
# 半初始化的模块对象。放在末尾即可保证这个顺序。
from od_platform.events.cards import CardEventDetector          # noqa: E402
from od_platform.events.substitution import SubstitutionDetector  # noqa: E402

__all__ = [
    "CardEventDetector",
    "CardSettings",
    "ClassNameSettings",
    "ColorCheckSettings",
    "Detection",
    "EventSettings",
    "KIND_CARD",
    "KIND_RED_CARD",
    "KIND_SUBSTITUTION",
    "KIND_YELLOW_CARD",
    "KIND_SEVERITY",
    "PenaltyEvent",
    "ROLE_GENERIC_CARD",
    "ROLE_PLAYER",
    "ROLE_RED_CARD",
    "ROLE_REFEREE",
    "ROLE_SUBSTITUTION_BOARD",
    "ROLE_YELLOW_CARD",
    "SubstitutionDetector",
    "SubstitutionSettings",
    "cap_confidence",
    "default_config_path",
    "image_size_of",
    "in_area_ratio_range",
    "in_aspect_ratio_range",
    "parse_detections",
]