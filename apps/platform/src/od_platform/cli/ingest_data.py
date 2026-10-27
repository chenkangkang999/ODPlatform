#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : ingest_data.py
# @Project   : ODPlatform
# @Function  : odp-ingest —— 把各种原始布局整理成 D3 能吃的标准原始目录

"""``odp-ingest``: 数据接入。

把"从各处下载来的各种原始布局"整理成 ::

    data/raw/<name>/images/*.jpg
    data/raw/<name>/labels/*.txt

也就是 ``odp-transform --dataset <name>`` 唯一认得的形态。

★ 默认是**预演**: 只扫描并打印计划（布局、样本数、每类框数、越界框数、
  图片体检、重名冲突）, **一个字节都不写**。真的要落地必须显式 ``--apply``。
  这是本仓库对"会产生新数据"的操作的统一态度。

★ 只读源: 本命令**绝不修改** ``data/raw/`` 下已有的任何字节。落地完成后会把
  "源标签内容哈希前后一致"写进报告 —— 让"源没被改"是个**可验证的结论**,
  而不是一句设计承诺。

退出码: 0=成功; 1=用法错误; 2=接入失败（源不合法 / 重名冲突 / 目标已存在）。
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

from od_platform.common import paths
from od_platform.common.logging_utils import get_logger
from od_platform.data_pipeline.ingest import (
    CONFLICT_POLICIES,
    IngestError,
    IngestSettings,
    apply_plan,
    scan,
)

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_FAILED = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odp-ingest",
        description="把各种原始布局整理成 D3 能吃的标准原始目录（默认只预演, 不落地）")
    parser.add_argument("--source", required=True,
                        help="源目录（原始下载数据的根）")
    parser.add_argument("--name", required=True,
                        help="数据集名; 目标是 data/raw/<name>")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="类别名, 顺序即 class_id（不传则从源里的 data.yaml 读）")
    parser.add_argument("--apply", action="store_true",
                        help="真的落地（不加则只预演）")
    parser.add_argument("--force", action="store_true",
                        help="目标目录已存在且非空时也继续")
    parser.add_argument("--conflict-policy", choices=list(CONFLICT_POLICIES), default=None,
                        help="拍平后的重名冲突怎么办（默认 abort: 直接拒绝）")
    parser.add_argument("--copy", action="store_true",
                        help="强制复制而不是硬链接（跨卷或想留独立副本时用）")
    parser.add_argument("--no-clamp", action="store_true",
                        help="不把越界框裁回 [0,1]（默认会裁）")
    parser.add_argument("--no-probe", action="store_true",
                        help="跳过图片解码探针（大图片集上可以省时间）")
    parser.add_argument("--config", default=None,
                        help="接入配置 yaml 路径（默认 configs/runtime/ingest.yaml）")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def _settings_from_args(args: argparse.Namespace) -> IngestSettings:
    """加载配置, 再让命令行覆盖它（CLI 优先级最高, 与仓库其它命令一致）。"""
    settings = IngestSettings.load(Path(args.config) if args.config else None)
    overrides = {}
    if args.copy:
        overrides["prefer_hardlink"] = False
    if args.no_clamp:
        overrides["clamp_boxes"] = False
    if args.no_probe:
        overrides["probe_images"] = False
    if args.conflict_policy:
        overrides["conflict_policy"] = args.conflict_policy
    return replace(settings, **overrides) if overrides else settings


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logger = get_logger(base_path=paths.LOGGING_DIR, log_type="ingest")

    try:
        settings = _settings_from_args(args)
    except ValueError as exc:
        print(f"配置不合法: {exc}", file=sys.stderr)
        return EXIT_USAGE

    source = Path(args.source)
    target = paths.raw_dataset_root(args.name)

    print(f"接入: {source}")
    print(f"  →   {target}")
    print(f"布局探测规则: {settings.layouts.split_dirnames} / "
          f"{settings.layouts.images_dirname}+{settings.layouts.labels_dirname}")
    print()

    try:
        plan = scan(source, target, args.name, settings, classes=args.classes)
    except IngestError as exc:
        print(f"❌ 接入失败:\n{exc}", file=sys.stderr)
        logger.error("odp-ingest 扫描失败: %s", exc)
        return EXIT_FAILED

    for line in plan.render_lines():
        print(line)

    if plan.class_names is None:
        print("\n⚠ 没找到类名。odp-transform 需要 --classes 指定, "
              "否则数据集配置里的 names 会是空的。")

    if not args.apply:
        print("\n[预演] 没有写入任何文件。确认无误后加 --apply 落地。")
        return EXIT_OK

    if target.exists() and any(target.iterdir()) and not args.force:
        print(f"\n❌ 目标目录已存在且非空: {target}\n"
              "   为避免与既有数据集混在一起, 默认拒绝。确认要覆盖就加 --force。",
              file=sys.stderr)
        return EXIT_FAILED

    print()
    try:
        result = apply_plan(plan, settings, force=args.force)
    except (FileExistsError, OSError) as exc:
        print(f"❌ 落地失败: {exc}", file=sys.stderr)
        logger.error("odp-ingest 落地失败: %s", exc)
        return EXIT_FAILED

    for line in result.render_lines():
        print(line)
    if result.report_path is not None:
        print(f"体检报告: {result.report_path}")
    print()
    print("下一步:")
    print(f"  odp-transform --dataset {args.name} --format yolo"
          + (f" --classes {' '.join(plan.class_names)}" if plan.class_names else ""))
    print(f"  odp-validate {paths.dataset_yaml_path(args.name)}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())