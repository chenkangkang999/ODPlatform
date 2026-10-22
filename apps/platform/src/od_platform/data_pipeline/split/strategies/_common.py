"""划分策略的公共小件:比例校验、三方计数、带种子洗牌。内部工具,不注册。"""
import random
from typing import Dict, List, Sequence, Tuple

from od_platform.common.constants import RATE_EPSILON


def validate_rates(train_rate: float, val_rate: float) -> float:
    """校验比例并返回 test_rate。浮点求和,容差 RATE_EPSILON。"""
    test_rate = 1.0 - train_rate - val_rate
    if test_rate < -RATE_EPSILON:
        raise ValueError(
            f"比例不合法: train={train_rate} + val={val_rate} > 1")
    return max(0.0, test_rate)


def three_way_counts(n: int, train_rate: float, val_rate: float
                     ) -> Tuple[int, int, int]:
    """n 个样本按比例切成 (n_train, n_val, n_test);test 吃舍入余量,保证总和=n。"""
    n_train = min(n, max(0, round(n * train_rate)))
    n_val = min(n - n_train, max(0, round(n * val_rate)))
    return n_train, n_val, n - n_train - n_val


def seeded_shuffled(seq: Sequence[str], rng: random.Random) -> List[str]:
    """用调用方给的 rng 洗牌(种子由调用方掌控),返回新列表。"""
    out = list(seq)
    rng.shuffle(out)
    return out
