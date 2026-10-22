"""check: split_presence —— yaml 声明的划分真的可用吗(兑付 snapshot 的 scan_warnings)。
train 缺失 ERROR;未声明 val WARNING;未声明 test INFO;split 图数 <30 INFO。整体取最严。"""
from __future__ import annotations

from typing import Any, Dict, List

from od_platform.common.constants import SPLIT_MIN_IMAGES
from od_platform.data_validation.registry import (
    CheckContext, CheckResult, CheckSeverity, check,
)

_NAME = "split_presence"


@check(_NAME)
def validate_split_presence(ctx: CheckContext) -> CheckResult:
    snap = ctx.snapshot
    findings: List[Dict[str, Any]] = []
    for w in snap.scan_warnings:
        if "split" in w:
            findings.append({"severity": CheckSeverity.WARNING, "message": w})
    declared = [s for s in ("train", "val", "test") if s in snap.yaml_data]
    if "train" not in snap.splits:
        findings.append({"severity": CheckSeverity.ERROR,
                         "message": "train 划分不可用 — 无训练数据,流程必须终止"})
    if "val" not in declared:
        findings.append({"severity": CheckSeverity.WARNING,
                         "message": "yaml 未声明 val 划分 — 训练将盲飞(无验证集)"})
    if "test" not in declared:
        findings.append({"severity": CheckSeverity.INFO,
                         "message": "yaml 未声明 test 划分 — 无最终诚实评估"})
    for s in snap.splits:
        n = len(snap.images_per_split[s])
        if n < SPLIT_MIN_IMAGES:
            findings.append({"severity": CheckSeverity.INFO,
                             "message": f"{s} 划分共 {n} 张,低于 {SPLIT_MIN_IMAGES},"
                                        f"评估方差偏大(提示)"})
    if not findings:
        return CheckResult(_NAME, CheckSeverity.PASS,
                           "train/val/test 三划分齐备且规模足够", {})
    severity = max((f["severity"] for f in findings), key=CheckSeverity.rank)
    summary = next(f["message"] for f in findings if f["severity"] == severity)
    return CheckResult(_NAME, severity, summary, {"findings": findings})
