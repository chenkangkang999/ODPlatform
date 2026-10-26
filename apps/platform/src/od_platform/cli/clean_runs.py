#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : clean_runs.py
# @Project   : ODPlatform
# @Function  : odp-clean —— 按保留策略清理 runs/ 下的历史现场

"""odp-clean: 清理 ``runs/`` 下的历史运行现场。

★ 默认是**预演**: 只打印"打算删哪些、能释放多少", 不动盘; 真的要删必须显式
  ``--apply``。这是本仓库对破坏性操作的统一态度（``odp-reset`` 同理）。

★ ``runs/lineage/`` 永远豁免: 那是 D3/D4/D6 盖的章, 删了数据身份核验就对不上账,
  而且无法重建。它不靠"默认参数恰好没选到", 而是硬编码在保留策略里。

退出码: 0=成功; 1=用法/策略参数错误。
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from od_platform.common import paths
from od_platform.common.logging_utils import get_logger
from od_platform.common.retention import (
    DEFAULT_KEEP_LATEST,
    DEFAULT_MAX_AGE_DAYS,
    PROTECTED_SUBDIRS,
    RetentionPolicy,
    apply_cleanup,
    plan_cleanup,
    summarize_runs,
)

EXIT_OK = 0
EXIT_USAGE = 1

_MB = 1048576.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odp-clean",
        description="按保留策略清理 runs/ 下的历史运行现场（默认只预演, 不删除）")
    parser.add_argument("--apply", action="store_true",
                        help="真的执行删除（不加则只预演）")
    parser.add_argument("--keep-latest", type=int, default=DEFAULT_KEEP_LATEST,
                        help=f"每个子目录至少保留最新的 N 个（默认 {DEFAULT_KEEP_LATEST}）")
    parser.add_argument("--max-age-days", type=float, default=DEFAULT_MAX_AGE_DAYS,
                        help=f"这个天数内的一律不动（默认 {DEFAULT_MAX_AGE_DAYS:g}）")
    parser.add_argument("--protect", nargs="*", default=None,
                        help="额外豁免的子目录名（空格分隔）")
    parser.add_argument("--summary", action="store_true",
                        help="只打印现状概览, 不计算清理计划")
    return parser


def _print_summary(summary: dict) -> None:
    print(f"运行现场根目录: {summary['runs_dir']}")
    print(f"当前总占用: {summary['total_bytes'] / _MB:.1f} MB")
    for name, info in summary["groups"].items():
        mark = "   [豁免]" if info["protected"] else ""
        print(f"  {name:<18} {info['subdirs']:>4} 个现场  "
              f"{info['size_bytes'] / _MB:>9.1f} MB{mark}")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        policy = RetentionPolicy(keep_latest_per_group=args.keep_latest,
                                 max_age_days=args.max_age_days,
                                 extra_protected=tuple(args.protect or ()))
    except ValueError as exc:
        print(f"策略参数无效: {exc}", file=sys.stderr)
        return EXIT_USAGE

    _print_summary(summarize_runs())
    if args.summary:
        return EXIT_OK

    print(f"\n豁免（永不清理）: {', '.join(PROTECTED_SUBDIRS)}")
    print(f"策略: 每组保留最新 {policy.keep_latest_per_group} 个, "
          f"且 {policy.max_age_days:g} 天内的不动\n")

    plan = plan_cleanup(policy=policy)
    for line in plan.render_lines():
        print(line)

    if not args.apply:
        print("\n[预演] 没有删除任何东西。确认无误后加 --apply 执行。")
        return EXIT_OK

    logger = get_logger(base_path=paths.LOGGING_DIR, log_type="clean")
    logger.info("odp-clean 执行: policy=%s plan=%d 个",
                policy, len(plan.candidates))
    result = apply_cleanup(plan)
    logger.info("清理完成: %s", result)
    print(f"\n已清理 {result['removed']} 个现场, "
          f"释放 {result['freed_bytes'] / _MB:.1f} MB"
          + (f", 跳过 {result['skipped']} 个（见日志）" if result["skipped"] else ""))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())