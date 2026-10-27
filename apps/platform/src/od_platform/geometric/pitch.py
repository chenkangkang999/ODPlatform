#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""标准足球场几何模型 —— 全场唯一的场地尺寸真相源。

★ 关于"这里为什么有一堆数字":

    它们是 IFAB《Laws of the Game》Law 1 规定的球场尺寸。**没有任何一个数字
    写在业务逻辑里** —— 规则引擎、投影、分析器一律通过 PitchSpec 的属性取值,
    所以改场地尺寸只需要改一处（配置或这里的默认值）, 全链路跟着变。

    也就是说: 这里是【默认值的唯一声明处】, 不是硬编码散落点。
    演播室的非标准场地、五人制场地, 都可以用 football.yaml 的 pitch: 节覆盖。

坐标系约定（全场统一）:
    原点 (0, 0) = 球场左上角（俯视图、以"防守方在左"为基准）
    x 轴 = 沿球场长度方向, 0 → length 米
    y 轴 = 沿球场宽度方向, 0 → width  米
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Tuple

Point = Tuple[float, float]

# 配置键的别名（兼容历史配置里的 field_length / field_width 写法）
_CONFIG_ALIASES: Dict[str, str] = {
    "field_length": "length",
    "field_width": "width",
}

# 四个角点的名字 —— 角球弧的圆心就在这四点上。定义成常量而不是各处拼字符串,
# 保证 homography 的"最近角点"查询与 landmarks 那张表永远指同一批点。
CORNER_NAMES: Tuple[str, ...] = (
    "corner_top_left",
    "corner_top_right",
    "corner_bottom_left",
    "corner_bottom_right",
)


@dataclass(frozen=True)
class PitchSpec:
    """一块球场的尺寸规格（单位统一为米）。

    所有"由基本尺寸推导出来的坐标"都是 property —— 不另外存一份,
    避免出现"改了大小的定义、忘了改禁区坐标"这种不一致。
    """

    length: float = 105.0                 # 场地长度（国际比赛推荐）
    width: float = 68.0                   # 场地宽度
    goal_width: float = 7.32              # 两门柱内沿之间
    goal_height: float = 2.44             # 横梁下沿高度（录像几何用得到）
    penalty_area_depth: float = 16.5      # 罚球区纵深
    penalty_area_width: float = 40.32     # 罚球区宽度
    goal_area_depth: float = 5.5          # 球门区纵深
    goal_area_width: float = 18.32        # 球门区宽度
    penalty_spot_distance: float = 11.0   # 罚球点到门线
    centre_circle_radius: float = 9.15    # 中圈半径
    corner_arc_radius: float = 1.0        # 角球弧半径（IFAB: 1 米）

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"球场尺寸 {field.name} 必须为正数, 收到 {value!r}")

    # ---------------------------------------------------------------- 派生坐标

    @property
    def left_goal_line_x(self) -> float:
        return 0.0

    @property
    def right_goal_line_x(self) -> float:
        return self.length

    @property
    def centre_x(self) -> float:
        return self.length / 2

    @property
    def centre_y(self) -> float:
        return self.width / 2

    @property
    def goal_y_min(self) -> float:
        return self.centre_y - self.goal_width / 2

    @property
    def goal_y_max(self) -> float:
        return self.centre_y + self.goal_width / 2

    @property
    def penalty_area_y_min(self) -> float:
        return self.centre_y - self.penalty_area_width / 2

    @property
    def penalty_area_y_max(self) -> float:
        return self.centre_y + self.penalty_area_width / 2

    @property
    def goal_area_y_min(self) -> float:
        return self.centre_y - self.goal_area_width / 2

    @property
    def goal_area_y_max(self) -> float:
        return self.centre_y + self.goal_area_width / 2

    # ---------------------------------------------------------------- 查询

    def within(self, x: float, y: float, *, margin: float = 0.0) -> bool:
        """点是否落在球场范围内（margin 可放宽, 用于容忍标定误差）。"""
        return (-margin <= x <= self.length + margin
                and -margin <= y <= self.width + margin)

    def landmark_names(self) -> Tuple[str, ...]:
        """所有可用的具名关键点（人工标定时填的就是这些名字）。"""
        return tuple(self.landmarks())

    def landmark(self, name: str) -> Point:
        """按名字取场上坐标; 名字写错当场报错, 不静默给 (0, 0)。"""
        try:
            return self.landmarks()[name]
        except KeyError:
            raise KeyError(
                f"未知的场地关键点 {name!r}; 可用: {sorted(self.landmarks())}") from None

    def corner_points(self) -> Dict[str, Point]:
        """四个角点 —— 角球弧的圆心就在这四个点上。

        单独开一个方法是因为"球是不是在角球弧内"必须知道圆心在哪,
        而调用方不该自己去 landmarks() 里按名字挑。
        """
        table = self.landmarks()
        return {name: table[name] for name in CORNER_NAMES}

    def penalty_spot(self, side: str) -> Point:
        """某一侧的罚球点（点球就是从这里踢的）。"""
        return self.landmark(f"penalty_spot_{side}")

    def landmarks(self) -> Dict[str, Point]:
        """具名关键点表 —— 人工标定时的"公共词汇表"。

        只列真实画面里常见、可肉眼定位的点; 需要其它的自己加。
        """
        length, width = self.length, self.width
        centre_x, centre_y = self.centre_x, self.centre_y
        table: Dict[str, Point] = {
            # 四个角与中线
            "corner_top_left": (0.0, 0.0),
            "corner_top_right": (length, 0.0),
            "corner_bottom_left": (0.0, width),
            "corner_bottom_right": (length, width),
            "halfway_top": (centre_x, 0.0),
            "halfway_bottom": (centre_x, width),
            # 中圈
            "centre_spot": (centre_x, centre_y),
            "centre_circle_left": (centre_x - self.centre_circle_radius, centre_y),
            "centre_circle_right": (centre_x + self.centre_circle_radius, centre_y),
            # 球门（门柱 + 门线中点）
            "goal_left_post_top": (self.left_goal_line_x, self.goal_y_min),
            "goal_left_post_bottom": (self.left_goal_line_x, self.goal_y_max),
            "goal_left_centre": (self.left_goal_line_x, centre_y),
            "goal_right_post_top": (self.right_goal_line_x, self.goal_y_min),
            "goal_right_post_bottom": (self.right_goal_line_x, self.goal_y_max),
            "goal_right_centre": (self.right_goal_line_x, centre_y),
        }
        # 左右两侧对称的线: 左门线在 x=0, 所以"从中心往回退"用的是 -1 号方向
        for side, sign in (("left", -1.0), ("right", 1.0)):
            zone_x = centre_x + sign * (centre_x - self.penalty_area_depth)
            goal_zone_x = centre_x + sign * (centre_x - self.goal_area_depth)
            spot_x = centre_x + sign * (centre_x - self.penalty_spot_distance)
            table[f"penalty_area_{side}_top"] = (zone_x, self.penalty_area_y_min)
            table[f"penalty_area_{side}_bottom"] = (zone_x, self.penalty_area_y_max)
            table[f"goal_area_{side}_top"] = (goal_zone_x, self.goal_area_y_min)
            table[f"goal_area_{side}_bottom"] = (goal_zone_x, self.goal_area_y_max)
            table[f"penalty_spot_{side}"] = (spot_x, centre_y)
        # 角球弧: 圆心就是角点, 半径 corner_arc_radius（IFAB 1 米）。
        # 除了角点本身, 把弧与两条边线的交点也列出来 —— 它们在画面里是能肉眼
        # 定位的实线端点, 而角点常常被摄像机切出画面外。
        radius = self.corner_arc_radius
        table.update({
            "corner_arc_top_left_goal_line": (radius, 0.0),
            "corner_arc_top_left_touchline": (0.0, radius),
            "corner_arc_top_right_goal_line": (length - radius, 0.0),
            "corner_arc_top_right_touchline": (length, radius),
            "corner_arc_bottom_left_goal_line": (radius, width),
            "corner_arc_bottom_left_touchline": (0.0, width - radius),
            "corner_arc_bottom_right_goal_line": (length - radius, width),
            "corner_arc_bottom_right_touchline": (length, width - radius),
        })
        return table

    # ---------------------------------------------------------------- 配置

    @classmethod
    def from_config(cls, config: Any) -> "PitchSpec":
        """从 config 的 pitch 节构造; 未给的字段保持默认。

        不是字典（或为空）时原样返回默认规格 —— 配置缺失不该让几何层挂掉。
        """
        if not isinstance(config, dict):
            return cls()
        known = {field.name for field in fields(cls)}
        overrides: Dict[str, float] = {}
        for raw_key, value in config.items():
            key = _CONFIG_ALIASES.get(raw_key, raw_key)
            if key in known:
                overrides[key] = float(value)
        return cls(**overrides)


# 出厂默认规格（IFAB 国际比赛标准场地）—— 全项目唯一的默认声明处
DEFAULT_PITCH = PitchSpec()


def within_pitch(x: float, y: float, *, margin: float = 0.0,
                 pitch: PitchSpec = DEFAULT_PITCH) -> bool:
    """兼容函数: 按默认规格判断点是否在球场内。优先用 PitchSpec.within。"""
    return pitch.within(x, y, margin=margin)