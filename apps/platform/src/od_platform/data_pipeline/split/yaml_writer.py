"""写 Ultralytics 数据集 yaml:path/train/val/test/names/nc + odp_meta 审计块。

odp_meta 是"产品合格证":记录这份成品由谁、用什么配方产出,契约指纹与 manifest 路径
都在里面 —— D4 质检凭它找到 manifest 对账。yaml 最后写:物化成功才盖章。

**path 必须可移植**:这个 yaml 进 git,所以 ``path`` 写【相对仓库根】的 POSIX 路径
(如 ``data/processed/demo``),绝不写死盘符 —— 路径规则与解析统一在
:mod:`od_platform.common.dataset_yaml`,别在这里自己拼。
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import yaml

from od_platform.common.dataset_yaml import to_repo_relative
from od_platform.common.lineage import SplitManifest


def write_dataset_yaml(
    yaml_path: Path,
    *,
    dataset_root: Path,
    classes: List[str],
    manifest: SplitManifest,
    dataset_name: str,
    source_format: str,
    task: str,
    manifest_ref: str,
) -> None:
    doc = {
        "path": to_repo_relative(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(classes)},
        "nc": len(classes),
        "odp_meta": {
            "schema_version": 1,
            "dataset_name": dataset_name,
            "source_format": source_format,
            "task": task,
            "tool_version": manifest.tool_version,
            "created_at": manifest.created_at,
            "contract_fingerprint": manifest.contract_fingerprint,
            "manifest_path": manifest_ref,
            "split": {
                "strategy": manifest.strategy,
                "seed": manifest.seed,
                "rates": {
                    "train": round(manifest.ratios[0], 6),
                    "val": round(manifest.ratios[1], 6),
                    "test": round(manifest.ratios[2], 6),
                },
                "counts": {
                    s: len(manifest.stems_of(s)) for s in ("train", "val", "test")
                },
            },
        },
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
