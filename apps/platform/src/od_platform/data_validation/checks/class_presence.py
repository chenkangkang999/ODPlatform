"""check: class_presence —— 每个声明的类别,在该出现的划分里出现了吗。"""
from __future__ import annotations

from typing import Any, Dict, List

from od_platform.data_validation.registry import (
    CheckContext, CheckResult, CheckSeverity, check,
)

_NAME = "class_presence"


def _count(snap, split: str, cid: int) -> int:
    st = snap.stats_per_split.get(split)
    return st.class_instances.get(cid, 0) if st else 0


@check(_NAME)
def validate_class_presence(ctx: CheckContext) -> CheckResult:
    snap = ctx.snapshot
    nc = snap.nc
    if not isinstance(nc, int):
        return CheckResult(_NAME, CheckSeverity.PASS,
                           "nc 不可用,跳过类别出现性检查(见 yaml_schema)",
                           {"reason": "nc_unavailable"})
    findings: List[Dict[str, Any]] = []
    for cid in range(nc):
        name = snap.class_names[cid] if cid < len(snap.class_names) else str(cid)
        n_tr = _count(snap, "train", cid)
        n_va = _count(snap, "val", cid)
        n_te = _count(snap, "test", cid)
        if n_tr + n_va + n_te == 0:
            findings.append({"kind": "declared_unused", "severity": CheckSeverity.INFO,
                             "class": name,
                             "message": f"类别 {name!r} 全集 0 实例 — 类别表与标注脱节(死类别)"})
        elif "train" in snap.splits and n_tr == 0:
            findings.append({"kind": "missing_in_train", "severity": CheckSeverity.WARNING,
                             "class": name,
                             "message": f"类别 {name!r} 在 val/test 出现 {n_va + n_te} 次"
                                        f"但 train 为 0 — 模型无法学习,评估必然失真"})
        elif "val" in snap.splits and n_tr > 0 and n_va == 0:
            findings.append({"kind": "missing_in_val", "severity": CheckSeverity.WARNING,
                             "class": name,
                             "message": f"类别 {name!r} train 有 {n_tr} 但 val 为 0"
                                        f" — 该类训练效果不可评估"})
    if not findings:
        return CheckResult(_NAME, CheckSeverity.PASS,
                           f"{nc} 个类别在各划分均有出现", {"nc": nc})
    severity = max((f["severity"] for f in findings), key=CheckSeverity.rank)
    summary = next(f["message"] for f in findings if f["severity"] == severity)
    return CheckResult(_NAME, severity, summary, {"findings": findings})
