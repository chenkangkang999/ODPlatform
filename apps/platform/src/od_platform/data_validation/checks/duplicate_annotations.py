"""check: duplicate_annotations —— 完全重复的标签行(规范化后逐行比对)。
完全重复行虚增该类权重 → WARNING。近似重复(高 IoU)不在本课范围(details 注明)。"""
from __future__ import annotations

from collections import Counter
from typing import List

from od_platform.common.constants import DETAILS_PREVIEW_LIMIT
from od_platform.data_validation.registry import (
    CheckContext, CheckResult, CheckSeverity, check,
)

_NAME = "duplicate_annotations"


@check(_NAME)
def validate_duplicate_annotations(ctx: CheckContext) -> CheckResult:
    snap = ctx.snapshot
    dup_rows = 0
    files_with_dup = 0
    previews: List[dict] = []
    for split in snap.splits:
        for lbl in snap.label_files_per_split[split]:
            try:
                lines = [l.strip() for l in lbl.read_text(encoding="utf-8").splitlines()
                         if l.strip()]
            except OSError:
                continue
            counts = Counter(lines)
            dups = {l: n for l, n in counts.items() if n > 1}
            if dups:
                files_with_dup += 1
                dup_rows += sum(n - 1 for n in dups.values())
                if len(previews) < DETAILS_PREVIEW_LIMIT:
                    previews.append({"split": split, "label": str(lbl), "dup": dups})
    details = {"dup_rows": dup_rows, "files_with_dup": files_with_dup,
               "previews": previews,
               "note": "近似重复(高 IoU)未在本课范围,如需请另做 embedding/IoU 方案"}
    if dup_rows == 0:
        return CheckResult(_NAME, CheckSeverity.PASS, "无完全重复的标注行", details)
    return CheckResult(_NAME, CheckSeverity.WARNING,
                       f"{dup_rows} 行完全重复(分布在 {files_with_dup} 个文件)— 会虚增权重",
                       details)
