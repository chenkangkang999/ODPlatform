"""类别平衡报告:转换后、划分前,让数据集作者看见类别分布。
只读:不碰任何文件。稀有类 = 在最小 split 里的期望样本量低于阈值的类。
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

from od_platform.common.constants import RARE_WARN_MIN_EXPECTED
from od_platform.common.string_utils import format_table_row, format_table_separator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassStat:
    class_id: int
    name: str
    n_images: int        # 出现在多少张图
    n_instances: int     # 共多少个实例


@dataclass(frozen=True)
class BalanceReport:
    total_images: int
    min_split_rate: float
    stats: List[ClassStat] = field(default_factory=list)
    rare: List[str] = field(default_factory=list)      # 稀有类名列表


def analyze_balance(
    per_image_classes: Dict[str, List[str]],
    names: List[str],
    min_split_rate: float,
    warn_min_expected: float = RARE_WARN_MIN_EXPECTED,
) -> BalanceReport:
    """per_image_classes: stem → 该图出现的类名列表(可重复,重复=多实例)。"""
    img_count: Counter = Counter()
    inst_count: Counter = Counter()
    for classes in per_image_classes.values():
        for c in set(classes):
            img_count[c] += 1
        for c in classes:
            inst_count[c] += 1
    stats = [ClassStat(i, n, img_count.get(n, 0), inst_count.get(n, 0))
             for i, n in enumerate(names)]
    stats.sort(key=lambda s: s.n_images)
    rare = [s.name for s in stats
            if 0 < s.n_images and s.n_images * min_split_rate < warn_min_expected]
    return BalanceReport(total_images=len(per_image_classes),
                         min_split_rate=min_split_rate, stats=stats, rare=rare)


def format_report(report: BalanceReport) -> List[str]:
    """渲染为文本行(复用 D1 的表格工具)。"""
    widths = [4, 12, 8, 8, 8, 12]
    lines = [format_table_separator(widths),
             format_table_row(["id", "name", "images", "inst", "期望", "备注"], widths),
             format_table_separator(widths)]
    for s in report.stats:
        expected = s.n_images * report.min_split_rate
        note = "稀有类!" if s.name in report.rare else ""
        lines.append(format_table_row(
            [s.class_id, s.name, s.n_images, s.n_instances,
             f"{expected:.1f}", note], widths))
    lines.append(format_table_separator(widths))
    return lines


def warn_if_rare(report: BalanceReport) -> None:
    """稀有类只预警,不拦截 —— 补不补数据是作者的决定。"""
    for name in report.rare:
        logger.warning(
            "稀有类 %r: 在最小 split 的期望样本量低于 %.1f,建议补充数据或检查划分",
            name, report.min_split_rate)
