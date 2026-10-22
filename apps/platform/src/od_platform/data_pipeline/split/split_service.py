"""划分服务层:校验依赖(requires_labels)→ 组装 SplitOptions → 调策略函数。"""
from __future__ import annotations

from typing import Dict, List, Optional

from od_platform.common.constants import (
    DEFAULT_RANDOM_STATE, DEFAULT_SPLIT_STRATEGY, DEFAULT_TRAIN_RATE, DEFAULT_VAL_RATE,
)
from od_platform.data_pipeline.split.registry import SplitOptions, get_strategy


def split_dataset(
    stems: List[str],
    train_rate: float = DEFAULT_TRAIN_RATE,
    val_rate: float = DEFAULT_VAL_RATE,
    random_state: int = DEFAULT_RANDOM_STATE,
    *,
    strategy: str = DEFAULT_SPLIT_STRATEGY,
    labels_per_image: Optional[Dict[str, List[str]]] = None,
    group_per_image: Optional[Dict[str, str]] = None,
) -> Dict[str, List[str]]:
    entry = get_strategy(strategy)
    if entry.requires_labels and labels_per_image is None:
        raise ValueError(
            f"划分策略 {strategy!r} 需要 labels_per_image,但未提供。"
            f"要么改用 random,要么由编排层传入标签信息。")
    options = SplitOptions(
        train_rate=train_rate, val_rate=val_rate, random_state=random_state,
        labels_per_image=labels_per_image, group_per_image=group_per_image)
    return entry.func(stems, options)
