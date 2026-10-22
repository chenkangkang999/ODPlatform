"""odp-train: 解析命令行; 全程唯一一次挂 logging handler; 串起 D1-D6; 返回退出码。"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Dict

from od_platform.common import paths, refs
from od_platform.common.logging_utils import get_logger
from od_platform.common.naming import run_stem
from od_platform.common.paths import RUNTIME_CONFIGS_DIR
from od_platform.common.run_context import RunContext
from od_platform.runtime_config import build_train_config
from od_platform.training.service import train_yolo

_CLI_SWITCHES = {"config", "no_archive", "skip_validate"}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="odp-train", description="ODPlatform 训练命令")
    p.add_argument("--config", default="train", help="运行配置引用: 'train'/'train.yaml'/路径")
    p.add_argument("--no-archive", action="store_true", help="不归档 best.pt")
    p.add_argument("--skip-validate", action="store_true",
                   help="跳过数据身份核验(明知故犯)")
    p.add_argument("--data", default=None, help="数据集引用: 'demo'/路径")
    p.add_argument("--model", default=None, help="模型引用: 'yolov8n'/'yolov8n.pt'/路径")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=float, default=None, help="int 或 float(<1)")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--device", default=None, help="'0'/'0,1'/'cpu'")
    p.add_argument("--experiment-name", dest="experiment_name", default=None,
                   help="给人看的实验名(进日志与审计)")
    return p


def _collect_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    return {k: v for k, v in vars(args).items()
            if v is not None and k not in _CLI_SWITCHES}


def main() -> int:
    args = _build_parser().parse_args()
    cli_overrides = _collect_cli_overrides(args)

    cfg_ref = args.config
    yaml_path = cfg_ref if ("/" in cfg_ref or "\\" in cfg_ref) \
        else RUNTIME_CONFIGS_DIR / (cfg_ref if cfg_ref.endswith(".yaml") else f"{cfg_ref}.yaml")
    config, merger = build_train_config(yaml_path=str(yaml_path),
                                        cli_args=cli_overrides or None)

    with RunContext("training") as run:
        data_yaml = refs.resolve_dataset_yaml(config.data)
        model_ref = getattr(config, "model", None) or "yolov8n.pt"
        stem = run_stem(stage="train", run_id=run.run_id,
                        dataset=data_yaml.stem, model=model_ref)

        logger = get_logger(paths.LOGGING_DIR, log_type="train", run_id=run.run_id,
                            dataset_name=data_yaml.stem, model_name=model_ref)
        logger.info("启动 odp-train | CLI 显式给出: %s", sorted(cli_overrides))

        result = train_yolo(config=config, data_yaml=data_yaml, run=run, stem=stem,
                            merger=merger, archive=not args.no_archive,
                            require_validated=not args.skip_validate)

    log = logging.getLogger(__name__)
    if result.success:
        log.info("训练完成 | run_id=%s | 权重: %s", result.run_id, result.best_weight)
    else:
        log.error("训练未完成 | run_id=%s | %s", result.run_id, result.message)
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
