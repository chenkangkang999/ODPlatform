"""数据流转卡(traveler / 构建溯源)。
每一站在中立公共地留下自己那一联"章"。
故意不提供"读回旧章合并"的接口 —— 从 API 层面堵死转述。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from od_platform.common import paths

logger = logging.getLogger(__name__)

STAMP_TRANSFORM = "transform.stamp.json"
STAMP_VALIDATE = "validate.stamp.json"
STAMP_TRAIN = "train.stamp.json"

LINEAGE_DIR = paths.RUNS_DIR / "lineage"


def lineage_dir_of(data_fingerprint: str) -> Path:
    return LINEAGE_DIR / data_fingerprint


def write_stamp(
    *,
    stage_file: str,
    data_fingerprint: str,
    run_id: str,
    tool_version: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Path:
    d = lineage_dir_of(data_fingerprint)
    d.mkdir(parents=True, exist_ok=True)
    stamp: Dict[str, Any] = {
        "stage": stage_file.split(".")[0],
        "data_fingerprint": data_fingerprint,
        "run_id": run_id,
        "tool_version": tool_version,
    }
    if payload:
        stamp.update(payload)
    path = d / stage_file
    path.write_text(json.dumps(stamp, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("已盖章: %s (data_fingerprint=%s)", path, data_fingerprint[:16])
    return path


def read_stamp(data_fingerprint: str, stage_file: str) -> Optional[Dict[str, Any]]:
    path = lineage_dir_of(data_fingerprint) / stage_file
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("章损坏, 读取失败 %s: %s", path, e)
        return None
