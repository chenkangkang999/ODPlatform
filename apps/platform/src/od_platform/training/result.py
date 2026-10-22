"""训练结果指标。TrainMetrics 纯数据(frozen) + log_train_metrics 纯日志。"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from od_platform.common.constants import Task
from od_platform.common.string_utils import pad_to_width

_METRIC_FIELDS_BY_TASK: Dict[str, List[Tuple[str, str]]] = {
    Task.DETECT: [
        ("metrics/precision(B)", "Precision(B)"), ("metrics/recall(B)", "Recall(B)"),
        ("metrics/mAP50(B)", "mAP50(B)"), ("metrics/mAP50-95(B)", "mAP50-95(B)"),
    ],
    Task.SEGMENT: [
        ("metrics/precision(B)", "Precision(B)"), ("metrics/recall(B)", "Recall(B)"),
        ("metrics/mAP50(B)", "mAP50(B)"), ("metrics/mAP50-95(B)", "mAP50-95(B)"),
        ("metrics/precision(M)", "Precision(M)"), ("metrics/recall(M)", "Recall(M)"),
        ("metrics/mAP50(M)", "mAP50(M)"), ("metrics/mAP50-95(M)", "mAP50-95(M)"),
    ],
}


def _safe_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class TrainMetrics:
    """一次训练结果的结构化快照(纯数据)。"""
    run_id:          str
    task:            str
    save_dir:        Path
    speed_ms:        Dict[str, float]
    overall:         Dict[str, float]
    class_map_50_95: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_yolo_results(cls, results: Any, *, run_id: str,
                          model_trainer: Any = None) -> "TrainMetrics":
        save_dir_raw = getattr(results, "save_dir", None)
        if save_dir_raw is None and model_trainer is not None:
            save_dir_raw = getattr(model_trainer, "save_dir", None)
        save_dir = Path(save_dir_raw) if save_dir_raw is not None else Path("unknown")

        speed_raw = getattr(results, "speed", {}) or {}
        speed_ms: Dict[str, float] = {k: _safe_float(speed_raw.get(k))
                                      for k in ("preprocess", "inference", "loss", "postprocess")}
        valid = [v for v in speed_ms.values() if not math.isnan(v)]
        speed_ms["total"] = sum(valid) if valid else math.nan

        overall: Dict[str, float] = {"fitness": _safe_float(getattr(results, "fitness", None))}
        for k, v in (getattr(results, "results_dict", {}) or {}).items():
            overall[str(k)] = _safe_float(v)

        class_map: Dict[str, float] = {}
        names = getattr(results, "names", {}) or {}
        maps = getattr(results, "maps", None)
        size = int(getattr(maps, "size", 0) or 0)
        if names and size:
            for idx, class_name in names.items():
                if int(idx) < size:
                    class_map[str(class_name)] = _safe_float(maps[idx])

        return cls(run_id=run_id, task=getattr(results, "task", "unknown"),
                   save_dir=save_dir, speed_ms=speed_ms, overall=overall,
                   class_map_50_95=class_map)

    def to_dict(self) -> Dict[str, Any]:
        def _clean(d: Dict[str, float]) -> Dict[str, Any]:
            return {k: (None if isinstance(v, float) and math.isnan(v) else v)
                    for k, v in d.items()}
        return {"run_id": self.run_id, "task": self.task, "save_dir": str(self.save_dir),
                "speed_ms": _clean(self.speed_ms), "overall": _clean(self.overall),
                "class_map_50_95": _clean(self.class_map_50_95)}


def log_train_metrics(metrics: TrainMetrics, *, logger: Optional[logging.Logger] = None,
                      key_width: int = 20, width: int = 60) -> None:
    log = logger or logging.getLogger(__name__)
    log.info("=" * width)
    log.info(f"训练结果 ({metrics.task})".center(width))
    log.info("=" * width)
    log.info(f"{pad_to_width('run_id', key_width)}: {metrics.run_id}")
    log.info(f"{pad_to_width('现场目录', key_width)}: {metrics.save_dir}")

    log.info("处理速度 (ms/image)".center(width))
    log.info("-" * width)
    for disp, key in [("预处理", "preprocess"), ("推理", "inference"),
                      ("损失计算", "loss"), ("后处理", "postprocess"), ("总计", "total")]:
        log.info(f"{pad_to_width(disp, key_width)}: {metrics.speed_ms.get(key, math.nan):.3f} ms")

    log.info("整体评估指标".center(width))
    log.info("-" * width)
    log.info(f"{pad_to_width('Fitness', key_width)}: {metrics.overall.get('fitness', math.nan):.4f}")
    fields = _METRIC_FIELDS_BY_TASK.get(metrics.task, [])
    if fields:
        for raw_key, disp in fields:
            log.info(f"{pad_to_width(disp, key_width)}: {metrics.overall.get(raw_key, math.nan):.4f}")
    else:
        log.info(f"(task='{metrics.task}' 不在已知表, 打印 results_dict 全量)")
        for k, v in metrics.overall.items():
            if k != "fitness":
                log.info(f"{pad_to_width(k, key_width)}: {v:.4f}")

    if metrics.class_map_50_95:
        kind = "Mask" if metrics.task == Task.SEGMENT else "Box"
        log.info(f"类别级 mAP@0.5:0.95 ({kind})".center(width))
        log.info("-" * width)
        valid = {k: v for k, v in metrics.class_map_50_95.items() if not math.isnan(v)}
        if valid:
            for name, v in sorted(valid.items(), key=lambda kv: kv[1], reverse=True):
                log.info(f"{pad_to_width(name, key_width)}: {v:.4f}")
        else:
            log.warning("类别 mAP 全为 NaN, 跳过打印")
    log.info("=" * width)
