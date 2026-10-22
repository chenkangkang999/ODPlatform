"""多标签分层划分(Sechidis 2011 简化实现):让稀有类尽量落到每个划分。

算法:按"最稀有标签"排序样本,贪心分配给当前配额缺口最大的划分;
同分时用该划分已收该类样本数决胜,再同分交给 rng(种子可控)。
零第三方依赖。requires_labels=True —— 没有 labels_per_image 直接拒跑。
"""
import random
from collections import Counter
from typing import Dict, List, Set

from od_platform.common.constants import SplitStrategy
from od_platform.data_pipeline.split.registry import SplitOptions, register_strategy
from od_platform.data_pipeline.split.strategies._common import validate_rates

_EPS = 1e-9


def _argmax_tiebreak(primary: Dict[str, float], secondary: Dict[str, float],
                     rng: random.Random) -> str:
    """按 primary 取最大;同分按 secondary 取最大;再同分 rng 随机。"""
    best_p = max(primary.values())
    cands = [k for k, v in primary.items() if abs(v - best_p) < _EPS]
    if len(cands) > 1:
        best_s = max(secondary[k] for k in cands)
        cands = [k for k in cands if abs(secondary[k] - best_s) < _EPS]
    return rng.choice(sorted(cands)) if len(cands) > 1 else cands[0]


def iterative_stratify(sample_labels: List[Set[str]], proportions: List[float],
                       random_state: int) -> List[int]:
    """返回每个样本的 fold 下标(0..len(proportions)-1)。

    贪心迭代:按"样本最稀有标签的出现次数"升序处理;每张图分给它最稀有标签
    配额缺口最大的 fold(同分按该 fold 已收该类样本数决胜,再同分 rng 随机)。
    无标签样本按 fold 总缺口填。
    """
    rng = random.Random(random_state)
    n_folds = len(proportions)
    label_count = Counter(lbl for labels in sample_labels for lbl in labels)
    n_samples = len(sample_labels)
    fold_counts = [0] * n_folds
    fold_label_counts = [Counter() for _ in range(n_folds)]
    fold_of: List[int] = [0] * n_samples

    def rarest(labels: Set[str]) -> float:
        return min((label_count[l] for l in labels), default=float("inf"))

    order = sorted(range(n_samples),
                   key=lambda i: (rarest(sample_labels[i]), i))
    for i in order:
        labels = sample_labels[i]
        if labels:
            # 样本携带的标签中最稀有者:按该标签在各 fold 的期望配额缺口分配
            # (期望配额 = 该标签总数 × fold 比例 —— 缺口是相对"标签配额"算的)
            rare = min(labels, key=lambda l: (label_count[l], str(l)))
            deficits = {str(f): label_count[rare] * proportions[f] - fold_label_counts[f][rare]
                        for f in range(n_folds)}
            secondary = {str(f): -fold_label_counts[f][rare] for f in range(n_folds)}
        else:
            # 无标签样本:按 fold 样本量总缺口填
            deficits = {str(f): n_samples * proportions[f] - fold_counts[f]
                        for f in range(n_folds)}
            secondary = {str(f): -fold_counts[f] for f in range(n_folds)}
        chosen = int(_argmax_tiebreak(deficits, secondary, rng))
        fold_of[i] = chosen
        fold_counts[chosen] += 1
        for l in labels:
            fold_label_counts[chosen][l] += 1
    return fold_of


@register_strategy(SplitStrategy.STRATIFIED_MULTILABEL, requires_labels=True)
def stratified_multilabel_split(stems: List[str],
                                options: SplitOptions) -> Dict[str, List[str]]:
    test_rate = validate_rates(options.train_rate, options.val_rate)
    if options.labels_per_image is None:
        raise ValueError("stratified_multilabel 需要 labels_per_image")
    ordered = sorted(stems)
    proportions = [options.train_rate, options.val_rate, test_rate]
    sample_labels = [set(options.labels_per_image.get(s, [])) for s in ordered]
    fold_of = iterative_stratify(sample_labels, proportions, options.random_state)

    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    names = ["train", "val", "test"]
    for i, stem in enumerate(ordered):
        splits[names[fold_of[i]]].append(stem)
    return splits
