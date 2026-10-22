"""odp-transform:数据流水线 CLI。
退出码:0=成功;1=数据/参数错误(脏数据、覆盖率不足等);2=用法错误(argparse)。
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from od_platform.common import paths
from od_platform.common.constants import (
    AnnotationFormat, DEFAULT_RANDOM_STATE, DEFAULT_SPLIT_STRATEGY,
    DEFAULT_TRAIN_RATE, DEFAULT_VAL_RATE, SplitStrategy, Task,
)
from od_platform.common.logging_utils import get_logger
from od_platform.common.run_context import RunContext
from od_platform.data_pipeline.orchestrator import DatasetPipeline

EXIT_OK = 0
EXIT_DATA_ERR = 1
EXIT_USAGE = 2

logger = get_logger(base_path=paths.LOGGING_DIR, log_type="transform")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="odp-transform",
        description="把原始数据集转换为可训练的 YOLO 成品(转换+划分+物化+契约)")
    p.add_argument("--dataset", required=True, help="数据集名(data/raw/<name>)")
    p.add_argument("--format", required=True, choices=AnnotationFormat.all())
    p.add_argument("--task", default=Task.DETECT, choices=Task.all())
    p.add_argument("--train-rate", type=float, default=DEFAULT_TRAIN_RATE)
    p.add_argument("--val-rate", type=float, default=DEFAULT_VAL_RATE)
    p.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    p.add_argument("--split-strategy", default=DEFAULT_SPLIT_STRATEGY,
                   choices=SplitStrategy.all())
    p.add_argument("--classes", nargs="*", default=None,
                   help="类别白名单(顺序即 class_id);不传=按数据自带;传空=全排除")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with RunContext("data_pipeline") as run:
            result = DatasetPipeline(
                args.dataset, args.format, task=args.task,
                train_rate=args.train_rate, val_rate=args.val_rate,
                classes=args.classes, random_state=args.seed,
                split_strategy=args.split_strategy, run=run,
            ).run()
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", e)
        return EXIT_DATA_ERR
    logger.info("yaml: %s", result["yaml"])
    logger.info("manifest: %s", result["manifest_path"])
    logger.info("fingerprint: %s", result["contract_fingerprint"])
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
