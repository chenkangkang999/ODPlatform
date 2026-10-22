"""check: split_uniqueness —— 一个样本只能属于一个 split。泄漏是硬 ERROR,没有分级。"""
from __future__ import annotations

from collections import defaultdict
from typing import List

from od_platform.common.constants import DETAILS_PREVIEW_LIMIT
from od_platform.data_validation.registry import (
    CheckContext, CheckResult, CheckSeverity, check,
)

_NAME = "split_uniqueness"


@check(_NAME)
def validate_split_uniqueness(ctx: CheckContext) -> CheckResult:
    snap = ctx.snapshot
    splits_of_stem = defaultdict(list)
    for split in snap.splits:
        for img in snap.images_per_split[split]:
            splits_of_stem[img.stem].append(split)
    leaked: List[dict] = [{"stem": stem, "splits": sorted(ss)}
                          for stem, ss in sorted(splits_of_stem.items()) if len(ss) > 1]
    if not leaked:
        return CheckResult(_NAME, CheckSeverity.PASS,
                           f"全部 {len(splits_of_stem)} 个样本分属唯一 split", {})
    return CheckResult(_NAME, CheckSeverity.ERROR,
                       f"{len(leaked)} 个样本泄漏到多个 split — 数据泄漏,评估不可信",
                       {"leaked_preview": leaked[:DETAILS_PREVIEW_LIMIT]})
