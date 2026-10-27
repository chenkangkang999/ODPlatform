#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : evaluate_model.py
# @Project   : ODPlatform
# @Function  : odp-eval —— 独立评估已训好的检测模型, 产出完整指标报告

"""``odp-eval``: 独立评估一个已训好的检测模型。

★ 存在的意义

    "训练完看 mAP50 / 每类 AP / IoU / 置信度"这件事, 以训练日志为载体是**不可复现**的:
    日志会滚、格式会变、想重看只能把训练重跑一遍。而论文和答辩里的数字必须能随时
    重跑出来, 并且**每次跑的是同一件事** —— 所以评估要有独立入口、独立配置、独立报告。

用法::

    odp-eval --weights models/trained/football_yolo/weights/best.pt \\
             --data data/raw/football_det/data.yaml --split val

    # 额外出一套"排除小目标 ball 之后"的平均口径（ball 只有几百个框, 会拖低按类平均）
    odp-eval --weights ... --data ... --exclude-classes ball

退出码: 0 成功 / 1 用法或参数错误 / 2 评估失败（找不到权重或数据集、模型加载失败）。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from od_platform.common import paths
from od_platform.common.config_utils import override_settings, parse_batch_arg
from od_platform.common.logging_utils import get_logger
from od_platform.evaluation.service import evaluate_model
from od_platform.evaluation.settings import EvalSettings, default_config_path

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_FAILED = 2

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odp-eval",
        description="独立评估已训好的检测模型: mAP50 / mAP50-95 / 每类 AP·P·R / "
                    "置信度扫描 / 推荐阈值, 报告写入 runs/<task>_eval/<run_id>/")
    parser.add_argument("--weights", required=True,
                        help="权重文件（如 models/trained/xxx/weights/best.pt）")
    parser.add_argument("--data", required=True,
                        help="数据集 yaml（Roboflow 导出的也可以, 相对路径会自动解析）")
    parser.add_argument("--yaml", default=None,
                        help=f"评估配置; 默认 {default_config_path()}")
    parser.add_argument("--split", default=None, choices=("val", "test", "train"),
                        help="评估哪个 split（默认取配置里的）")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", default=None, help="批大小; -1=自动")
    parser.add_argument("--device", default=None, help="cpu | 0 | 0,1")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-det", type=int, default=None)
    parser.add_argument("--conf", type=float, default=None,
                        help="val 用的置信度下限; 越低 P-R 曲线越完整（默认 0.001）")
    parser.add_argument("--iou", type=float, default=None, help="NMS 的 IoU 阈值")
    parser.add_argument("--classes", nargs="+", default=None, metavar="NAME",
                        help="额外出一套「只算这些类」的平均口径")
    parser.add_argument("--exclude-classes", nargs="+", default=None, metavar="NAME",
                        help="额外出一套「排除这些类」的平均口径（如 ball）")
    parser.add_argument("--conf-sweep", dest="conf_sweep", action="store_true",
                        default=None, help="做置信度扫描（默认按配置）")
    parser.add_argument("--no-conf-sweep", dest="conf_sweep", action="store_false",
                        default=None, help="跳过置信度扫描（省 N 倍 val 时间）")
    parser.add_argument("--limit", type=int, default=None,
                        help="只评估前 N 张; 仅供链路冒烟, 指标不代表整个 split")
    parser.add_argument("--run-id", default=None, help="报告 id; 默认时间戳")
    parser.add_argument("--output-dir", default=None, help="报告目录; 默认进 runs/")
    parser.add_argument("--no-report", action="store_true", help="只打印, 不写 json/csv")
    parser.add_argument("--log-level", default="INFO", choices=_LOG_LEVELS)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse 自己处理不了的参数错误会直接 sys.exit(2)。本 CLI 的退出码约定是
        # 1=用法/参数错误、2=评估失败, 所以这里把 argparse 的 2 映射成 1,
        # 免得"参数写错"和"评估跑挂"在脚本里无法区分。（--help 的 0 原样透传。）
        return EXIT_OK if not exc.code else EXIT_USAGE

    logger = get_logger(base_path=paths.LOGGING_DIR, log_type="eval",
                        log_level=getattr(logging, args.log_level))

    try:
        settings = EvalSettings.load(Path(args.yaml) if args.yaml else None)
    except (ValueError, OSError) as exc:
        print(f"评估配置有问题: {exc}", file=sys.stderr)
        return EXIT_USAGE

    # CLI 覆盖: 只认非 None 的项, 类型按字段转换并重新走一遍校验
    overrides = {
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": None if args.batch is None else parse_batch_arg(args.batch),
        "device": args.device,
        "workers": args.workers,
        "max_det": args.max_det,
        "conf": args.conf,
        "iou": args.iou,
        "conf_sweep": args.conf_sweep,
        "limit": args.limit,
    }
    try:
        settings = override_settings(settings, overrides)
    except ValueError as exc:
        print(f"参数有问题: {exc}", file=sys.stderr)
        return EXIT_USAGE

    logger.info("odp-eval 启动: weights=%s data=%s split=%s",
                args.weights, args.data, settings.split)

    try:
        report = evaluate_model(
            args.weights, args.data, settings=settings,
            output_dir=args.output_dir, run_id=args.run_id,
            exclude_classes=tuple(args.exclude_classes or ()),
            include_classes=tuple(args.classes or ()),
            write_report=not args.no_report)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        # 这三类都是"用户能看懂并自己修"的失败: 路径不对、权重坏了、口径冲突
        logger.error("评估失败: %s", exc)
        print(f"评估失败: {exc}", file=sys.stderr)
        return EXIT_FAILED

    print(report.summary_line())
    if report.warnings:
        print(f"注意: 有 {len(report.warnings)} 条警告, 详见报告与日志")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
