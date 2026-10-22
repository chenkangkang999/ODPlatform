"""check: class_balance —— 类别失衡(长尾)量化提醒。只提醒,不开药方。
失衡比 = 最多类实例数/最少类实例数(只看非零类);≥50 WARNING / ≥10 INFO。"""
from __future__ import annotations

from typing import Any, Dict, List

from od_platform.common.constants import (
    CLASS_IMBALANCE_INFO_RATIO, STATS_MAX_IMBALANCE_RATIO,
)
from od_platform.data_validation.registry import (
    CheckContext, CheckResult, CheckSeverity, check,
)

_NAME = "class_balance"


@check(_NAME)
def validate_class_balance(ctx: CheckContext) -> CheckResult:
    snap = ctx.snapshot
    findings: List[Dict[str, Any]] = []
    for split, st in snap.stats_per_split.items():
        nonzero = {c: n for c, n in st.class_instances.items() if n > 0}
        if len(nonzero) < 2:
            continue
        max_c = max(nonzero, key=nonzero.get)
        min_c = min(nonzero, key=nonzero.get)
        ratio = nonzero[max_c] / nonzero[min_c]
        if ratio >= STATS_MAX_IMBALANCE_RATIO:
            sev = CheckSeverity.WARNING
        elif ratio >= CLASS_IMBALANCE_INFO_RATIO:
            sev = CheckSeverity.INFO
        else:
            continue
        findings.append({"split": split, "ratio": round(ratio, 1), "severity": sev,
                         "message": f"{split} 失衡比 {ratio:.1f}x — 长尾提醒,"
                                    f"训练时建议关注 class_weights / 重采样"})
    if not findings:
        return CheckResult(_NAME, CheckSeverity.PASS, "各类别实例数未见显著失衡", {})
    severity = max((f["severity"] for f in findings), key=CheckSeverity.rank)
    summary = next(f["message"] for f in findings if f["severity"] == severity)
    return CheckResult(_NAME, severity, summary, {"findings": findings})
