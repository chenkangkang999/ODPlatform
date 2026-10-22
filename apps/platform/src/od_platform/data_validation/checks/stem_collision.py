"""check: stem_collision —— 同 split 内同名多扩展名(a.jpg+a.png → 抢同一个 a.txt)。"""
from __future__ import annotations

from collections import defaultdict
from typing import List

from od_platform.common.constants import DETAILS_PREVIEW_LIMIT
from od_platform.data_validation.registry import (
    CheckContext, CheckResult, CheckSeverity, check,
)

_NAME = "stem_collision"


@check(_NAME)
def validate_stem_collision(ctx: CheckContext) -> CheckResult:
    snap = ctx.snapshot
    collisions: List[dict] = []
    for split in snap.splits:
        by_stem = defaultdict(list)
        for img in snap.images_per_split[split]:
            by_stem[img.stem].append(img.name)
        for stem, names in sorted(by_stem.items()):
            if len(names) > 1:
                collisions.append({"split": split, "stem": stem, "files": sorted(names)})
    if not collisions:
        return CheckResult(_NAME, CheckSeverity.PASS, "无同名多扩展名冲突", {})
    return CheckResult(_NAME, CheckSeverity.WARNING,
                       f"{len(collisions)} 个同名多扩展名冲突 — 多张图映射到同一标签,需人工裁决",
                       {"collisions_preview": collisions[:DETAILS_PREVIEW_LIMIT]})
