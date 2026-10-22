"""质检服务层:跑全部 check 并聚合。
唯一的 except Exception 护栏在这里:单 check 崩溃降级为 ERROR 结果,不拖垮整场质检。"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from od_platform.common import paths
from od_platform.data_validation.registry import (
    CheckContext, CheckEntry, CheckResult, CheckSeverity, get_all_checks,
)
from od_platform.data_validation.report import ValidationReport
from od_platform.data_validation.snapshot import build_snapshot


def _safe_run_one(entry: CheckEntry, ctx: CheckContext) -> CheckResult:
    try:
        return entry.func(ctx)
    except Exception as e:                          # noqa: BLE001 —— 护栏,必须宽
        return CheckResult(entry.name, CheckSeverity.ERROR,
                           f"check 自身崩溃: {type(e).__name__}: {e}",
                           {"exception": repr(e)})


def run_all_checks(ctx: CheckContext) -> List[CheckResult]:
    return [_safe_run_one(e, ctx) for e in get_all_checks()]


def validate_dataset(yaml_path, task_type: Optional[str] = None,
                     run_id: Optional[str] = None,
                     write_report: bool = True) -> ValidationReport:
    """端到端:快照 → 全量 check → 报告(logger + json/csv)。"""
    t0 = time.perf_counter()
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    yaml_path = Path(yaml_path)
    snapshot = build_snapshot(yaml_path, task_type=task_type)
    results = run_all_checks(CheckContext(yaml_path=yaml_path, snapshot=snapshot))
    report = ValidationReport(run_id=run_id, snapshot=snapshot, results=results,
                              duration_seconds=time.perf_counter() - t0)
    report.render_to_logger()
    if write_report:
        out_dir = paths.validation_run_dir(run_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        report.write_json(out_dir / "report.json")
        report.write_results_csv(out_dir / "results.csv")
    return report
