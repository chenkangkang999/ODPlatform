"""训练编排器。只做四件事: 顺序、验章(确认数据身份)、记录、归档。"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from od_platform.common import refs, stamp
from od_platform.common.dataset_yaml import write_resolved_copy
from od_platform.common.paths import PRETRAINED_MODELS_DIR
from od_platform.common.run_context import RunContext
from od_platform.data_validation.identity import DataIdentity, reconcile_data_identity
from od_platform.runtime_config.report import log_config_report
from od_platform.training.archive import archive_best_weight
from od_platform.training.result import TrainMetrics, log_train_metrics

logger = logging.getLogger(__name__)

TRAINER_VERSION = "odp-train/0.4"

# 运行期数据集副本名。ultralytics 把相对 path 解析成 DATASETS_DIR/path, 与 ODP 的
# "相对仓库根"约定不同(见 od_platform.common.dataset_yaml), 所以开训前先落一份
# path 已解析成绝对路径的副本, 只喂给 ultralytics; 源 yaml 保持可移植。
DATASET_RESOLVED_NAME = "dataset_resolved.yaml"


@dataclass(frozen=True)
class TrainResult:
    success:     bool
    run_id:      str
    save_dir:    Path | None = None
    best_weight: Path | None = None
    metrics:     TrainMetrics | None = None
    identity:    DataIdentity | None = None
    message:     str = ""


def _resolve_model_arg(model: str) -> str:
    p = refs.resolve_ref(model, base_dir=PRETRAINED_MODELS_DIR, default_suffix=".pt")
    if p.exists():
        logger.info("模型(解析): %s", p)
        return str(p)
    logger.info("本地无 %s, 交给 ultralytics 自动下载: %s", p.name, model)
    return model


def _run_training(model_arg: str, config: Any, data_yaml: Path, run: RunContext) -> Any:
    from ultralytics import YOLO
    model = YOLO(model_arg)
    kwargs = config.to_ultralytics_kwargs()
    kwargs.pop("data", None)
    kwargs.pop("model", None)
    kwargs.update(project=str(run.run_dir.parent), name=run.run_id, exist_ok=True)
    return model.train(data=str(data_yaml), **kwargs)


def _write_audit(run: RunContext, config: Any, merger: Any | None, data_yaml: Path,
                 stem: str, metrics: TrainMetrics,
                 identity: DataIdentity | None,
                 resolved_yaml: Path | None = None) -> Path:
    audit = {
        "run_id": run.run_id, "created_at": run.created_at,
        "artifacts": {"run_dir": str(run.run_dir), "stem": stem},
        "experiment_name": getattr(config, "experiment_name", None),
        "data_yaml": str(data_yaml),
        "data_yaml_resolved": str(resolved_yaml) if resolved_yaml else None,
        "data_identity": identity.to_dict() if identity is not None else None,
        "config": config.to_audit_snapshot(),
        "metrics": metrics.to_dict(),
    }
    if merger is not None:
        audit["config_sources"] = merger.to_audit_log()
    path = run.artifact_path("odp_audit.json")
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("审计快照已落盘: %s", path)
    return path


def train_yolo(*, config: Any, data_yaml: Path, run: RunContext, stem: str,
               merger: Any | None = None, archive: bool = True,
               require_validated: bool = True) -> TrainResult:
    logger.info("=" * 60)
    logger.info("训练开始 | run_id=%s", run.run_id)
    logger.info("现场目录: %s | 产物 stem: %s", run.run_dir, stem)
    logger.info("实验名: %s | 数据集: %s",
                getattr(config, "experiment_name", None) or "(未命名)", data_yaml)
    logger.info("=" * 60)

    if merger is not None:
        log_config_report(config, merger, logger=logger)

    identity = reconcile_data_identity(data_yaml, require_validated=require_validated)
    if not identity.ok:
        logger.error("数据身份核验未通过, 拒绝开训: %s", identity.reason)
        return TrainResult(success=False, run_id=run.run_id, identity=identity,
                           message=f"数据身份核验未通过: {identity.reason}")

    model_ref = getattr(config, "model", None) or "yolov8n.pt"
    logger.info("模型(声明): %s", model_ref)
    model_arg = _resolve_model_arg(model_ref)

    resolved_yaml: Path | None = None
    try:
        resolved_yaml = write_resolved_copy(
            data_yaml, run.artifact_path(DATASET_RESOLVED_NAME))
        results = _run_training(model_arg, config, resolved_yaml, run)
    except KeyboardInterrupt:
        logger.warning("训练被手动中断 (Ctrl-C) | run_id=%s | 现场保留: %s",
                       run.run_id, run.run_dir)
        return TrainResult(success=False, run_id=run.run_id, save_dir=run.run_dir,
                           identity=identity, message="训练被手动中断")
    except Exception as e:
        logger.exception("训练失败: %s", e)
        return TrainResult(success=False, run_id=run.run_id, save_dir=run.run_dir,
                           identity=identity, message=f"训练异常: {e}")

    metrics = TrainMetrics.from_yolo_results(results, run_id=run.run_id)
    log_train_metrics(metrics, logger=logger)

    if identity.data_fingerprint:
        stamp.write_stamp(
            stage_file=stamp.STAMP_TRAIN, data_fingerprint=identity.data_fingerprint,
            run_id=run.run_id, tool_version=TRAINER_VERSION,
            payload={"stem": stem,
                     "experiment_name": getattr(config, "experiment_name", None),
                     "reconciled": identity.reconciled},
        )
    _write_audit(run, config, merger, data_yaml, stem, metrics, identity,
                 resolved_yaml=resolved_yaml)

    best = archive_best_weight(run.run_dir, stem) if archive else None
    logger.info("训练完成 | run_id=%s | 现场: %s", run.run_id, run.run_dir)
    return TrainResult(success=True, run_id=run.run_id, save_dir=run.run_dir,
                       best_weight=best, metrics=metrics, identity=identity)
