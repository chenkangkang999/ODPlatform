#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : settings.py
# @Project   : ODPlatform / events
# @Function  : 判罚事件（红牌/黄牌/换人）的可调参数与类别名映射

"""判罚事件的配置模型 —— 所有可调数字与类别名的唯一声明处。

★ 能力边界（必须如实说明, 别在文档/答辩里写成"已支持"）

    红牌/黄牌本质上要【把"牌"当物体检测】（裁判手里的一块红/黄矩形）,
    换人要靠【第四官员举着的换人牌】。也就是说这三项**强依赖一个能检出这些类别的
    模型**, 而本仓库当前的足球权重只有 ``player`` / ``football`` 两类, **检不出牌**。

    所以本模块提供的是【判定逻辑 + 一处配置化的类别名映射】, **不是红黄牌识别**。
    要真正跑起来, 必须先把数据集扩到包含 牌 / 裁判 / 换人牌 这些类别并训练。
    没训练时: 检测结果里永远不会出现牌, 于是本模块**一帧也不会产出事件** ——
    这是设计行为, 不是 bug。

★ 为什么类别名必须来自配置（而不是写死在逻辑里）

    模型训练时的类名不受本仓库控制: 可能叫 ``red_card`` / ``红牌`` / ``card_red``,
    甚至只给一个通用的 ``card``。所以:
      · 每个角色都配【别名列表】, 匹配**大小写不敏感**、两端空白自动忽略;
      · ``cards.py`` / ``substitution.py`` 里不出现任何类名字面量,
        一律走 :meth:`ClassNameSettings.resolve_role`;
      · 若模型只给通用 ``card`` 类, 则由**颜色**决定红牌还是黄牌
        （见 :class:`ColorCheckSettings`）。

events.yaml 里对应的节::

    classes:       类别名映射（含别名列表）
    cards:         红/黄牌的判定阈值、颜色二次确认、冷却
    substitution:  换人的判定阈值与冷却
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from od_platform.common import paths
from od_platform.common.config_utils import build_settings, read_section, report_unknown_keys

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "events.yaml"

# ============================================================
# 事件类型（本模块的内部词汇, 不是模型类名）
# ============================================================
KIND_RED_CARD = "red_card"
KIND_YELLOW_CARD = "yellow_card"
KIND_CARD = "card"                    # 通用"牌"且颜色无法判定 → 只报候选
KIND_SUBSTITUTION = "substitution"

# 严重级: 红牌 >= 黄牌（答辩会问"红黄牌怎么区分对待"）
KIND_SEVERITY: Mapping[str, int] = {
    KIND_RED_CARD: 2,
    KIND_YELLOW_CARD: 1,
    KIND_CARD: 1,
    KIND_SUBSTITUTION: 0,
}

# ============================================================
# 角色名（resolve_role 的返回值）
# ============================================================
ROLE_RED_CARD = "red_card"
ROLE_YELLOW_CARD = "yellow_card"
ROLE_GENERIC_CARD = "generic_card"
ROLE_REFEREE = "referee"
ROLE_PLAYER = "player"
ROLE_SUBSTITUTION_BOARD = "substitution_board"

# ============================================================
# 类别名默认值 —— 只在这里声明一次, 逻辑代码里不出现类名字面量
# ============================================================
DEFAULT_RED_CARD_CLASS = "red_card"
DEFAULT_YELLOW_CARD_CLASS = "yellow_card"
DEFAULT_GENERIC_CARD_CLASS = "card"
DEFAULT_REFEREE_CLASS = "referee"
DEFAULT_PLAYER_CLASS = "player"
DEFAULT_SUBSTITUTION_BOARD_CLASS = "substitution_board"

# 别名列表: 名字里带主名和常见变体, 也认中文
DEFAULT_RED_CARD_ALIASES: Tuple[str, ...] = (
    DEFAULT_RED_CARD_CLASS, "redcard", "card_red", "red", "红牌",
)
DEFAULT_YELLOW_CARD_ALIASES: Tuple[str, ...] = (
    DEFAULT_YELLOW_CARD_CLASS, "yellowcard", "card_yellow", "yellow", "黄牌",
)
DEFAULT_GENERIC_CARD_ALIASES: Tuple[str, ...] = (
    DEFAULT_GENERIC_CARD_CLASS, "cards", "牌", "卡片",
)
DEFAULT_REFEREE_ALIASES: Tuple[str, ...] = (
    DEFAULT_REFEREE_CLASS, "ref", "referee_person", "裁判", "主裁判",
)
DEFAULT_PLAYER_ALIASES: Tuple[str, ...] = (
    DEFAULT_PLAYER_CLASS, "person", "players", "球员", "人",
)
DEFAULT_SUBSTITUTION_BOARD_ALIASES: Tuple[str, ...] = (
    DEFAULT_SUBSTITUTION_BOARD_CLASS, "substitution", "sub_board", "subboard",
    "fourth_official_board", "换人牌", "换人",
)

# ============================================================
# 颜色范围（OpenCV 的 HSV 里 H 是 0..179, 不是 0..360）
# ============================================================
# 每段是 (h_min, s_min, v_min, h_max, s_max, v_max)
# 红色跨越 H 的 0/179 两端, 所以必须拆成两段 —— 只写一段会漏掉洋红侧的红色
DEFAULT_RED_HSV_RANGES: Tuple[Tuple[int, int, int, int, int, int], ...] = (
    (0, 70, 60, 10, 255, 255),
    (170, 70, 60, 179, 255, 255),
)
DEFAULT_YELLOW_HSV_RANGES: Tuple[Tuple[int, int, int, int, int, int], ...] = (
    (20, 70, 70, 35, 255, 255),
)

# 颜色矛盾时的处置策略
COLOR_POLICY_DOWNGRADE = "downgrade"      # 降级为"证据不足"（默认, 不盲信模型）
COLOR_POLICY_TRUST_MODEL = "trust_model"  # 仍以模型类别为准, 但把矛盾记进判据
COLOR_POLICIES: Tuple[str, ...] = (COLOR_POLICY_DOWNGRADE, COLOR_POLICY_TRUST_MODEL)

CONF_MIN = 0.0
CONF_MAX = 1.0


def _check_ratio_pair(name: str, low: float, high: float) -> None:
    if low < 0:
        raise ValueError(f"{name} 的下限不能为负, 收到 {low}")
    if low > high:
        raise ValueError(f"{name} 的下限不能大于上限: {low} > {high}")


@dataclass(frozen=True)
class ClassNameSettings:
    """类别名映射 —— 模型类名 → 本模块的角色。

    每个角色给一组别名, 匹配时**大小写不敏感**、自动去掉两端空白。
    """

    red_card_aliases: Tuple[str, ...] = DEFAULT_RED_CARD_ALIASES
    yellow_card_aliases: Tuple[str, ...] = DEFAULT_YELLOW_CARD_ALIASES
    generic_card_aliases: Tuple[str, ...] = DEFAULT_GENERIC_CARD_ALIASES
    referee_aliases: Tuple[str, ...] = DEFAULT_REFEREE_ALIASES
    player_aliases: Tuple[str, ...] = DEFAULT_PLAYER_ALIASES
    substitution_board_aliases: Tuple[str, ...] = DEFAULT_SUBSTITUTION_BOARD_ALIASES

    def __post_init__(self) -> None:
        # 配置从 yaml 来的时候, 别名可能是 list; 也可能被写成单个字符串。
        # 统一成"已规范化的小写元组"并把索引缓存下来, 匹配时就不必反复 lower()。
        for role in _KNOWN_ROLES:
            field_name = f"{role}_aliases"
            normalized = _normalize_aliases(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, normalized)
            object.__setattr__(self, f"_{role}_index", frozenset(normalized))

    # ---------------------------------------------------------------- 匹配
    def resolve_role(self, cls_name: Any) -> str:
        """把模型给的一个类名解析成角色; 认不出返回空串。

        解析顺序有意为之: **具体的牌类优先于通用牌类** —— 否则一个叫 ``card``
        的通用类和 ``red_card`` 同时存在时, 通用类会把红牌先吃掉。
        """
        key = _normalize_one(cls_name)
        if not key:
            return ""
        for role in (ROLE_RED_CARD, ROLE_YELLOW_CARD, ROLE_REFEREE,
                     ROLE_PLAYER, ROLE_SUBSTITUTION_BOARD, ROLE_GENERIC_CARD):
            if key in getattr(self, f"_{_alias_field(role)}_index"):
                return role
        return ""

    def aliases_of(self, role: str) -> Tuple[str, ...]:
        """取某个角色的别名（审计/报错信息里用, 免得只有一串看不懂的规则）。"""
        return getattr(self, f"{_alias_field(role)}_aliases")

    def to_audit(self) -> Dict[str, Any]:
        return {
            "red_card_aliases": list(self.red_card_aliases),
            "yellow_card_aliases": list(self.yellow_card_aliases),
            "generic_card_aliases": list(self.generic_card_aliases),
            "referee_aliases": list(self.referee_aliases),
            "player_aliases": list(self.player_aliases),
            "substitution_board_aliases": list(self.substitution_board_aliases),
        }


# 角色的合法取值。角色常量与别名字段名【一一对应】是有意的设计 —— 少一层映射表,
# 但必须在这里显式校验, 否则 getattr 拼错字段名只会变成 AttributeError 而不是清晰报错。
_KNOWN_ROLES: Tuple[str, ...] = (
    ROLE_RED_CARD, ROLE_YELLOW_CARD, ROLE_GENERIC_CARD,
    ROLE_REFEREE, ROLE_PLAYER, ROLE_SUBSTITUTION_BOARD,
)


def _alias_field(role: str) -> str:
    """角色常量 → 别名字段名前缀。"""
    if role not in _KNOWN_ROLES:
        raise ValueError(f"未知角色: {role!r}; 合法取值: {_KNOWN_ROLES}")
    return role


def _normalize_one(value: Any) -> str:
    return str(value).strip().lower() if value is not None else ""


def _normalize_aliases(raw: Any, where: str) -> Tuple[str, ...]:
    """把别名配置统一成小写元组; 空配置直接报错（不然整个模块静默失效）。"""
    if raw is None:
        values: List[Any] = []
    elif isinstance(raw, str):
        values = [raw]                       # 容忍只写一个名字
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        raise ValueError(f"{where} 必须是字符串或字符串列表, 收到 {type(raw).__name__}")
    cleaned = tuple(dict.fromkeys(_normalize_one(v) for v in values if _normalize_one(v)))
    if not cleaned:
        raise ValueError(f"{where} 不能为空 —— 否则该角色永远匹配不上, 模块静默失效")
    return cleaned


@dataclass(frozen=True)
class ColorCheckSettings:
    """对牌的裁剪区域做颜色二次确认（防模型把黄牌判成红牌）。"""

    enabled: bool = True                        # 关掉则完全以模型类别为准
    conflict_policy: str = COLOR_POLICY_DOWNGRADE   # 矛盾时的处置
    red_ranges: Tuple[Tuple[int, int, int, int, int, int], ...] = DEFAULT_RED_HSV_RANGES
    yellow_ranges: Tuple[Tuple[int, int, int, int, int, int], ...] = DEFAULT_YELLOW_HSV_RANGES
    min_pixel_ratio: float = 0.15               # 颜色像素占比达到此值才算"颜色成立"
    min_crop_pixels: int = 4                    # 裁剪区域小于此值就不判颜色（无意义）

    def __post_init__(self) -> None:
        if self.conflict_policy not in COLOR_POLICIES:
            raise ValueError(f"conflict_policy 只能是 {COLOR_POLICIES}, "
                             f"收到 {self.conflict_policy!r}")
        if not CONF_MIN <= self.min_pixel_ratio <= CONF_MAX:
            raise ValueError(f"min_pixel_ratio 必须在 [{CONF_MIN}, {CONF_MAX}] 内, "
                             f"收到 {self.min_pixel_ratio}")
        if self.min_crop_pixels < 1:
            raise ValueError(f"min_crop_pixels 必须 >= 1, 收到 {self.min_crop_pixels}")
        for name in ("red_ranges", "yellow_ranges"):
            normalized = _normalize_hsv_ranges(getattr(self, name), name)
            object.__setattr__(self, name, normalized)

    def to_audit(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "conflict_policy": self.conflict_policy,
            "min_pixel_ratio": self.min_pixel_ratio,
            "red_ranges": [list(r) for r in self.red_ranges],
            "yellow_ranges": [list(r) for r in self.yellow_ranges],
        }


def _normalize_hsv_ranges(raw: Any, where: str) -> Tuple[Tuple[int, int, int, int, int, int], ...]:
    """把 HSV 段配置统一成 6 元整数元组; 段数/长度不对就报错。"""
    if isinstance(raw, (list, tuple)) and raw and not isinstance(raw[0], (list, tuple)):
        raw = [raw]                          # 容忍只写一段 (0,70,60,10,255,255)
    ranges: List[Tuple[int, int, int, int, int, int]] = []
    for index, item in enumerate(raw or ()):
        if not isinstance(item, (list, tuple)) or len(item) != 6:
            raise ValueError(f"{where}[{index}] 必须是 6 个数 "
                             f"(h_min,s_min,v_min,h_max,s_max,v_max), 收到 {item!r}")
        values = tuple(int(v) for v in item)
        if values[0] > values[3] or values[1] > values[4] or values[2] > values[5]:
            raise ValueError(f"{where}[{index}] 的下限不能大于上限: {values}")
        ranges.append(values)                # type: ignore[arg-type]
    if not ranges:
        raise ValueError(f"{where} 不能为空 —— 否则该颜色永远判不出来")
    return tuple(ranges)


@dataclass(frozen=True)
class CardSettings:
    """红/黄牌的判定阈值。"""

    min_confidence: float = 0.3          # 牌本身的置信度下限
    min_area_ratio: float = 1e-5         # 牌面积占画面比的下限（1080p 上约 21px, 再小是噪点）
    max_area_ratio: float = 0.05         # 上限（再大就不是"举着的一张牌"）
    min_aspect_ratio: float = 0.35       # 宽/高 下限（牌是矩形, 极端细长的是旗杆之类）
    max_aspect_ratio: float = 3.0        # 宽/高 上限
    # 牌到"出示者"的距离上限, 单位是【出示者框的高度】——
    # 用相对尺度而不是像素/米, 于是完全不依赖标定, 也不受镜头远近影响。
    # 取值依据: 举过头顶的牌离出示者中心约 0.6~0.8 个身高, 侧举约 0.5,
    # 1.2 留了姿态变化的余量; 而"牌躺在草地上"约 0.5 —— **单靠距离区分不了这两种**,
    # 需要时把 require_card_above_anchor 打开（牌必须在出示者中心之上）。
    max_show_distance_body_heights: float = 1.2
    prefer_referee: bool = True          # 优先把裁判当出示者（牌是裁判掏的）
    require_card_above_anchor: bool = False  # 是否要求牌位于出示者中心之上
    cooldown_frames: int = 30            # 同类判罚在该帧数内只报一次
    unconfirmed_confidence_ceiling: float = 0.4   # 未确认事件的置信度上限（不假装确定）
    color: ColorCheckSettings = field(default_factory=ColorCheckSettings)

    def __post_init__(self) -> None:
        if not CONF_MIN <= self.min_confidence <= CONF_MAX:
            raise ValueError(f"min_confidence 必须在 [{CONF_MIN}, {CONF_MAX}] 内, "
                             f"收到 {self.min_confidence}")
        _check_ratio_pair("card 面积比", self.min_area_ratio, self.max_area_ratio)
        _check_ratio_pair("card 宽高比", self.min_aspect_ratio, self.max_aspect_ratio)
        if self.max_show_distance_body_heights <= 0:
            raise ValueError("max_show_distance_body_heights 必须为正数")
        if self.cooldown_frames < 0:
            raise ValueError(f"cooldown_frames 不能为负, 收到 {self.cooldown_frames}")
        if self.max_area_ratio <= 0:
            raise ValueError("max_area_ratio 必须为正数")
        if not CONF_MIN <= self.unconfirmed_confidence_ceiling <= CONF_MAX:
            raise ValueError("unconfirmed_confidence_ceiling 必须在 "
                             f"[{CONF_MIN}, {CONF_MAX}] 内, "
                             f"收到 {self.unconfirmed_confidence_ceiling}")

    def to_audit(self) -> Dict[str, Any]:
        return {
            "min_confidence": self.min_confidence,
            "area_ratio_range": [self.min_area_ratio, self.max_area_ratio],
            "aspect_ratio_range": [self.min_aspect_ratio, self.max_aspect_ratio],
            "max_show_distance_body_heights": self.max_show_distance_body_heights,
            "prefer_referee": self.prefer_referee,
            "require_card_above_anchor": self.require_card_above_anchor,
            "cooldown_frames": self.cooldown_frames,
            "unconfirmed_confidence_ceiling": self.unconfirmed_confidence_ceiling,
            "color": self.color.to_audit(),
        }


@dataclass(frozen=True)
class SubstitutionSettings:
    """换人（第四官员举牌）的判定阈值。"""

    min_confidence: float = 0.3          # 换人牌本身的置信度下限
    min_area_ratio: float = 1e-5
    max_area_ratio: float = 0.08         # 换人牌比红黄牌大, 上限放宽
    touchline_margin_m: float = 3.0      # 距边线多少米之内算"场边"
    require_touchline_proximity: bool = True   # 标定可用时, 太远就不构成事件
    require_calibration: bool = True     # 没有【可信】标定时是否降级为候选
    allow_uncalibrated_candidates: bool = True  # 无标定时仍产出候选（标注未确认）
    cooldown_frames: int = 30
    unconfirmed_confidence_ceiling: float = 0.4

    def __post_init__(self) -> None:
        if not CONF_MIN <= self.min_confidence <= CONF_MAX:
            raise ValueError(f"min_confidence 必须在 [{CONF_MIN}, {CONF_MAX}] 内, "
                             f"收到 {self.min_confidence}")
        _check_ratio_pair("换人牌面积比", self.min_area_ratio, self.max_area_ratio)
        if self.touchline_margin_m <= 0:
            raise ValueError("touchline_margin_m 必须为正数")
        if self.cooldown_frames < 0:
            raise ValueError(f"cooldown_frames 不能为负, 收到 {self.cooldown_frames}")
        if not CONF_MIN <= self.unconfirmed_confidence_ceiling <= CONF_MAX:
            raise ValueError("unconfirmed_confidence_ceiling 必须在 "
                             f"[{CONF_MIN}, {CONF_MAX}] 内, "
                             f"收到 {self.unconfirmed_confidence_ceiling}")

    def to_audit(self) -> Dict[str, Any]:
        return {
            "min_confidence": self.min_confidence,
            "area_ratio_range": [self.min_area_ratio, self.max_area_ratio],
            "touchline_margin_m": self.touchline_margin_m,
            "require_touchline_proximity": self.require_touchline_proximity,
            "require_calibration": self.require_calibration,
            "allow_uncalibrated_candidates": self.allow_uncalibrated_candidates,
            "cooldown_frames": self.cooldown_frames,
            "unconfirmed_confidence_ceiling": self.unconfirmed_confidence_ceiling,
        }


@dataclass(frozen=True)
class EventSettings:
    """判罚事件的完整配置。"""

    classes: ClassNameSettings = field(default_factory=ClassNameSettings)
    cards: CardSettings = field(default_factory=CardSettings)
    substitution: SubstitutionSettings = field(default_factory=SubstitutionSettings)

    @classmethod
    def from_config(cls, config: Any, *, where: str = CONFIG_FILENAME) -> "EventSettings":
        """从 events.yaml 的内容构造（顶层是 classes / cards / substitution 三节）。"""
        if not isinstance(config, Mapping):
            return cls()
        cards_raw = dict(read_section(config, "cards"))
        # color 是 cards 里的子节, 要单独摘出来构造 —— build_settings 不做嵌套
        color_raw = cards_raw.pop("color", None)
        color_values = dict(color_raw) if isinstance(color_raw, Mapping) else {}
        # 便利键: 有些配置习惯把颜色开关直接放在 cards 一级, 别静默丢掉它
        if "color_check_enabled" in cards_raw:
            color_values["enabled"] = cards_raw.pop("color_check_enabled")

        return cls(
            classes=_build_section(ClassNameSettings, read_section(config, "classes"),
                                   f"{where} 的 classes 节"),
            # color 是嵌套子节, 先按平字段构造 cards 再 replace 进去
            # （build_settings 只做"一层字段", 不做嵌套）
            cards=replace(
                _build_section(CardSettings, cards_raw, f"{where} 的 cards 节"),
                color=build_settings(ColorCheckSettings, color_values)),
            substitution=_build_section(SubstitutionSettings,
                                        read_section(config, "substitution"),
                                        f"{where} 的 substitution 节"),
        )

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "EventSettings":
        """从 yaml 文件加载; 默认 ``configs/runtime/events.yaml``。

        文件不存在时**全部走默认值**（而不是报错）—— 判罚事件是可选能力,
        不配也能跑, 只是用的都是默认阈值。
        """
        path = Path(path) if path else default_config_path()
        if not path.is_file():
            logger.info("未找到判罚事件配置 %s, 全部采用默认值", path)
            return cls()
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} 顶层不是 mapping: {type(data).__name__}")
        logger.info("已加载判罚事件配置: %s", path)
        return cls.from_config(data, where=str(path))

    def to_audit(self) -> Dict[str, Any]:
        """给结果文件/答辩用的参数快照（说明"这次是按什么规则判的"）。"""
        return {
            "classes": self.classes.to_audit(),
            "cards": self.cards.to_audit(),
            "substitution": self.substitution.to_audit(),
        }


def default_config_path() -> Path:
    """events.yaml 的默认位置（仓库相对, 由 paths 统一给出, 不写盘符）。"""
    return paths.RUNTIME_CONFIGS_DIR / CONFIG_FILENAME


def _build_section(cls, values: Any, label: str, **extra):
    """用一节配置构造 dataclass, 顺便把不认识的键报出来（拼错键名不该静默生效一半）。"""
    unknown: List[str] = []
    settings = build_settings(cls, values if isinstance(values, Mapping) else {},
                              unknown_sink=unknown, **extra)
    report_unknown_keys(label, unknown)
    return settings