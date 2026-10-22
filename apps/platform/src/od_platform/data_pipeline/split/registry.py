"""划分策略注册表:策略名 → 划分函数 + 依赖声明(requires_labels)。

与 convert 注册表同一模式:模块级 dict + 装饰器自注册 + 懒加载自动发现。
新增策略 = 在 strategies/ 下加一个文件。
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from od_platform.common.constants import (
    DEFAULT_RANDOM_STATE, DEFAULT_TRAIN_RATE, DEFAULT_VAL_RATE,
)

StrategyFunc = Callable[[List[str], "SplitOptions"], Dict[str, List[str]]]


@dataclass
class SplitOptions:
    """传给所有划分策略的横切选项(参数包)。"""
    train_rate: float = DEFAULT_TRAIN_RATE
    val_rate: float = DEFAULT_VAL_RATE
    random_state: int = DEFAULT_RANDOM_STATE
    labels_per_image: Optional[Dict[str, List[str]]] = None   # 分层策略需要
    group_per_image: Optional[Dict[str, str]] = None          # 分组策略需要(防泄漏)


@dataclass(frozen=True)
class StrategyEntry:
    func: StrategyFunc
    requires_labels: bool = False


_STRATEGIES: Dict[str, StrategyEntry] = {}
_LAZY_INITIALIZED = False


def register_strategy(name: str, *, requires_labels: bool = False):
    """装饰器:注册划分策略。同名覆盖给 warning(开发期允许迭代)。"""
    def deco(func: StrategyFunc) -> StrategyFunc:
        if name in _STRATEGIES:
            warnings.warn(f"划分策略重复注册,将被覆盖: {name!r}", stacklevel=2)
        _STRATEGIES[name] = StrategyEntry(func=func, requires_labels=requires_labels)
        return func
    return deco


def _lazy_init() -> None:
    global _LAZY_INITIALIZED
    if _LAZY_INITIALIZED:
        return
    from od_platform.common.registry_utils import import_submodules
    from od_platform.data_pipeline.split import strategies
    import_submodules(strategies)
    _LAZY_INITIALIZED = True


def get_strategy(name: str) -> StrategyEntry:
    _lazy_init()
    try:
        return _STRATEGIES[name]
    except KeyError:
        raise ValueError(
            f"未知划分策略: {name!r}。可用: {sorted(_STRATEGIES)}") from None


def list_strategies() -> Dict[str, StrategyEntry]:
    _lazy_init()
    return dict(_STRATEGIES)
