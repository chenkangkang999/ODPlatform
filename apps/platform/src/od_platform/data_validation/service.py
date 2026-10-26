"""质检服务层:跑全部 check 并聚合。
唯一的 except Exception 护栏在这里:单 check 崩溃降级为 ERROR 结果,不拖垮整场质检。"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from od_platform.common import paths, stamp
from od_platform.common.lineage import compute_data_fingerprint
from od_platform.data_validation import VALIDATOR_VERSION
from od_platform.data_validation.registry import (
    CheckContext, CheckEntry, CheckResult, CheckSeverity, get_all_checks,
)
from od_platform.data_validation.report import ValidationReport
from od_platform.data_validation.compare import compare_with_history
from od_platform.data_validation.snapshot import build_snapshot
from od_platform.data_validation.checks._lineage import load_manifest_from_yaml

logger = logging.getLogger(__name__)


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
    # 历史对比要在渲染/落盘之前算好: 这样日志和报告里都带上结论,
    # 而且此时本次的 report.json 还没写下去, 不可能把自己当成"上一次"。
    # 没有历史(首次质检)就如实说一句, 不静默跳过。
    if write_report:
        try:
            comparison = compare_with_history(yaml_path, report.to_dict(),
                                              exclude_run_id=run_id)
        except Exception as e:  # noqa: BLE001 —— 历史对比失败不该影响本次质检
            logger.warning("历史对比失败, 跳过: %s", e)
            comparison = None
        if comparison is not None:
            report.comparison = comparison
        else:
            logger.info("没有同数据集的历史质检报告, 本次为首次质检")
    report.render_to_logger()
    if write_report:
        out_dir = paths.validation_run_dir(run_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        report.write_json(out_dir / "report.json")
        report.write_results_csv(out_dir / "results.csv")

    # D6 增量: 盖 validate 章(我这一站验过这批数据)
    try:
        manifest, _err, _odp = load_manifest_from_yaml(snapshot.yaml_data)
        if manifest is not None:
            data_fp = compute_data_fingerprint(manifest.samples)
            stamp.write_stamp(
                stage_file=stamp.STAMP_VALIDATE, data_fingerprint=data_fp,
                run_id=run_id, tool_version=VALIDATOR_VERSION,
                payload={"overall_severity": report.overall_severity,
                         "passed": report.overall_severity == "PASS"},
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("盖章失败(不影响质检结果): %s", e)

    return report
