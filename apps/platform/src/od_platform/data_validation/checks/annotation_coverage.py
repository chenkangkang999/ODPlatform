"""check: annotation_coverage —— 无标注图占比。零额外 I/O:直接消费 snapshot 的 SplitStats。
≥30% WARNING / ≥5% INFO / 否则 PASS。"""
from __future__ import annotations

from od_platform.common.constants import UNLABELED_INFO_RATIO, UNLABELED_WARN_RATIO
from od_platform.data_validation.registry import (
    CheckContext, CheckResult, CheckSeverity, check,
)

_NAME = "annotation_coverage"


@check(_NAME)
def validate_annotation_coverage(ctx: CheckContext) -> CheckResult:
    snap = ctx.snapshot
    total_img = sum(st.image_count for st in snap.stats_per_split.values())
    if total_img == 0:
        return CheckResult(_NAME, CheckSeverity.INFO, "数据集没有图像", {})
    unlabeled = sum(st.image_count - st.annotated_count
                    for st in snap.stats_per_split.values())
    ratio = unlabeled / total_img
    details = {"total_images": total_img, "unlabeled": unlabeled,
               "unlabeled_ratio": round(ratio, 4),
               "thresholds": {"warn": UNLABELED_WARN_RATIO, "info": UNLABELED_INFO_RATIO},
               "per_split": {s: {"images": st.image_count, "annotated": st.annotated_count}
                             for s, st in snap.stats_per_split.items()}}
    if ratio >= UNLABELED_WARN_RATIO:
        return CheckResult(_NAME, CheckSeverity.WARNING,
                           f"{unlabeled}/{total_img} 张图无有效标注({ratio:.1%})— 占比偏高",
                           details)
    if ratio >= UNLABELED_INFO_RATIO:
        return CheckResult(_NAME, CheckSeverity.INFO,
                           f"{unlabeled}/{total_img} 张图无有效标注({ratio:.1%})", details)
    return CheckResult(_NAME, CheckSeverity.PASS,
                       f"全部 {total_img} 张图均有有效标注", details)
