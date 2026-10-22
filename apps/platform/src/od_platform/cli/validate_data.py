"""odp-validate:数据质检 CLI(只读,不改任何文件)。
退出码:ERROR→2 / WARNING→1 / PASS·INFO→0。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from od_platform.common import paths
from od_platform.common.constants import Task
from od_platform.common.logging_utils import get_logger
from od_platform.data_validation.registry import CheckSeverity
from od_platform.data_validation.service import validate_dataset

_EXIT_CODE = {CheckSeverity.ERROR: 2, CheckSeverity.WARNING: 1,
              CheckSeverity.INFO: 0, CheckSeverity.PASS: 0}

logger = get_logger(base_path=paths.LOGGING_DIR, log_type="validate")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="odp-validate",
                                description="数据集质检(只读):15 项 check,按最严聚合成退出码")
    p.add_argument("yaml_path", help="数据集 yaml(如 configs/datasets/demo.yaml)")
    p.add_argument("--task", default=None, choices=Task.all(),
                   help="不传则取 yaml 里 odp_meta.task")
    p.add_argument("--run-id", default=None)
    p.add_argument("--no-report", action="store_true", help="不写 json/csv 报告")
    args = p.parse_args(argv)
    report = validate_dataset(Path(args.yaml_path), task_type=args.task,
                              run_id=args.run_id, write_report=not args.no_report)
    return _EXIT_CODE.get(report.overall_severity, 2)


if __name__ == "__main__":
    sys.exit(main())
