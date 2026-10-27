#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : service.py
# @Project   : ODPlatform
# @Function  : 评估服务（端到端串联: 视图 → 模型 → val → 指标 → 报告）

"""``odp-eval`` 的服务层。

流程（每一步都刻意留了可替换的缝, 便于离线测试）:

    数据集 yaml ──► 只读安全的数据集视图（硬链接, 不碰 data/raw）
                        │
    权重 ──► _load_model ┤──► model.val(conf=0.001)  ──► 每类指标 + P-R 曲线
                        │
                        └──► 逐个阈值再 val 一次     ──► 置信度扫描 → 推荐阈值
                                                          │
                                              如实警告 ──► EvaluationReport（三种出口）

★ 两个刻意的设计

    1. **模型可注入**（``model=``）。测试塞一个"有 ``.val()`` 就行的鸭子类型对象"
       即可离线跑完整条链路 —— 不需要权重、不需要 GPU、不需要网络。
       评估逻辑本身也因此可以被测得很细（阈值推荐、均值口径、警告触发）。
    2. **置信度扫描是"多跑几次 val", 不是从曲线推**。框架给的 P-R 曲线横轴是召回率,
       没有 "conf → R" 的对应关系, 推不出来。代价是 N 倍 val 时间 ——
       所以既可配置、也可用 ``--no-conf-sweep`` 关掉, 并且文档里如实写明。
"""
from __future__ import annotations

import logging
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from od_platform.common import paths
from od_platform.common.progress import ProgressReporter
from od_platform.evaluation import metrics as metrics_mod
from od_platform.evaluation.dataset import DatasetView, build_dataset_view
from od_platform.evaluation.report import (
    CONF_SWEEP_CSV_FILENAME,
    JSON_FILENAME,
    PER_CLASS_CSV_FILENAME,
    EvaluationReport,
)
from od_platform.evaluation.settings import EvalSettings

logger = logging.getLogger(__name__)

RUN_DIR_SUFFIX = "_eval"          # runs/<task_label>_eval/<run_id>/
DATASET_VIEW_DIRNAME = "_dataset_view"


# ---------------------------------------------------------------- 路径显示

def _display_path(value: Any) -> str:
    """尽量给**仓库相对路径**: 报告更可读, 也便于跨机器对比。

    只影响报告里的字符串; 传给框架的必须是真实路径（调用处用的是 view.yaml_path）。
    """
    path = Path(str(value))
    try:
        return path.resolve().relative_to(paths.ROOT_DIR.resolve()).as_posix()
    except (ValueError, OSError):
        return str(path)


def default_run_dir(run_id: str, settings: EvalSettings) -> Path:
    """默认报告目录: ``runs/<task_label>_eval/<run_id>/``。"""
    return paths.RUNS_DIR / f"{settings.task_label}{RUN_DIR_SUFFIX}" / run_id


# ---------------------------------------------------------------- 模型

def load_model(weights: Path) -> Any:
    """加载 ultralytics 模型; 失败时报出**具体**原因, 不抛一个空泛的异常。"""
    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"权重文件不存在: {weights}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:                       # pragma: no cover - 取决于环境
        raise RuntimeError(f"评估需要 ultralytics, 但导入失败: {exc}") from None
    try:
        model = YOLO(str(weights))
    except Exception as exc:                         # noqa: BLE001 —— 权重损坏/格式不对都在这
        raise RuntimeError(
            f"权重加载失败: {weights}（{type(exc).__name__}: {exc}）") from None
    logger.info("已加载权重: %s", weights)
    return model


# ---------------------------------------------------------------- val

def run_val(model: Any, view: DatasetView, settings: EvalSettings, *,
            conf: Optional[float] = None, split: Optional[str] = None) -> Dict[str, Any]:
    """跑一次 val, 并把框架结果抽成朴素的 dict。

    抽成 dict 是为了让上层完全不知道 ultralytics 的对象长什么样 ——
    测试只要伪造一个"有 .box 的结果对象"就能跑完整条链路。
    """
    kwargs: Dict[str, Any] = {
        "data": str(view.yaml_path),
        "split": split or settings.split,
        "imgsz": settings.imgsz,
        "batch": settings.batch,
        "device": settings.device,
        "workers": settings.workers,
        "max_det": settings.max_det,
        "conf": settings.conf if conf is None else conf,
        "iou": settings.iou,
        "half": settings.half,
        "plots": settings.plots,
        "verbose": settings.verbose,
    }
    result = model.val(**kwargs)
    return extract_result(result)


def extract_result(result: Any) -> Dict[str, Any]:
    """把 ``val()`` 的返回值抽成朴素 dict（只取需要的字段, 不碰私有属性）。"""
    box = getattr(result, "box", None)
    if box is None:
        raise RuntimeError("val() 的返回值里没有 box 指标（不是检测任务的输出?）")

    def _floats(name: str) -> List[float]:
        value = getattr(box, name, None)
        if value is None:
            return []
        try:
            return [float(v) for v in value]
        except TypeError:                            # 标量
            return [float(value)]

    def _ints(value: Any) -> List[int]:
        # ★ 不能写成 ``value or []``: 框架这里给的是 numpy 数组,
        #   而 numpy 数组的布尔求值会抛 "truth value ... is ambiguous"。
        if value is None:
            return []
        return [int(v) for v in value]

    curves = getattr(box, "curves_results", None)
    speed = getattr(result, "speed", None)
    return {
        "map50": _scalar(getattr(box, "map50", None)),
        "map50_95": _scalar(getattr(box, "map", None)),
        "map75": _scalar(getattr(box, "map75", None)),
        "mean_precision": _scalar(getattr(box, "mp", None)),
        "mean_recall": _scalar(getattr(box, "mr", None)),
        "ap_class_index": _ints(getattr(box, "ap_class_index", None)),
        "ap50": _floats("ap50"),
        "ap": _floats("ap"),
        "precision": _floats("p"),
        "recall": _floats("r"),
        "curves": list(curves) if curves is not None else [],
        "speed": dict(speed) if speed else {},
    }


def _scalar(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- 端到端

def evaluate_model(weights: Any, data_yaml: Any, *,
                   settings: Optional[EvalSettings] = None,
                   output_dir: Optional[Any] = None,
                   split: Optional[str] = None,
                   exclude_classes: Sequence[str] = (),
                   include_classes: Sequence[str] = (),
                   run_id: Optional[str] = None,
                   write_report: bool = True,
                   model: Any = None,
                   view_root: Optional[Any] = None) -> EvaluationReport:
    """评估一个已训好的检测模型, 产出完整指标报告。

    Args:
        weights: 权重文件
        data_yaml: 数据集 yaml（Roboflow 导出的也可以, 相对路径会被自动解析）
        settings: 评估参数; None 时从 ``configs/runtime/eval.yaml`` 加载
        output_dir: 报告目录; None 时用 ``runs/<task_label>_eval/<run_id>/``
        split: 覆盖 settings.split
        exclude_classes: 额外出一套"排除这些类"的平均口径
        include_classes: 额外出一套"只算这些类"的平均口径
        run_id: 报告 id; None 时用时间戳
        write_report: 是否落盘 json/csv
        model: 注入的模型（测试用）; None 时按 weights 加载
        view_root: 数据集视图位置; None 时放在报告目录下的 ``_dataset_view/``

    Returns:
        :class:`EvaluationReport`

    Raises:
        FileNotFoundError: 权重或数据集找不到、split 路径解析不出来
        RuntimeError: 模型加载失败、val 结果结构不对
    """
    settings = settings or EvalSettings.load()
    if split:
        settings = replace(settings, split=split)
    if exclude_classes and include_classes:
        raise ValueError("exclude_classes 与 include_classes 不能同时给 —— 口径必须唯一")

    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    weights_path = Path(weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")

    out_dir = Path(output_dir) if output_dir else default_run_dir(run_id, settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    # ---- 1) 数据集视图: 绝不把 data/raw 直接交给框架（它会往里写 labels.cache）----
    view = build_dataset_view(
        Path(data_yaml), split=settings.split,
        view_root=Path(view_root) if view_root else out_dir / DATASET_VIEW_DIRNAME,
        limit=settings.limit)
    _warn_unknown_classes(view, exclude_classes, include_classes)

    # ---- 2) 模型 ----
    net = model if model is not None else load_model(weights_path)

    # ---- 3) 主评估（低阈值, P-R 曲线才完整）----
    logger.info("开始评估: split=%s imgsz=%s conf=%s device=%s",
                settings.split, settings.imgsz, settings.conf, settings.device)
    raw = run_val(net, view, settings, conf=settings.conf)

    rows = metrics_mod.per_class_rows(
        view.class_names, ap_class_index=raw["ap_class_index"],
        ap50=raw["ap50"], ap=raw["ap"], precision=raw["precision"],
        recall=raw["recall"], instances=view.instances_per_class)

    # ---- 4) 多套平均口径 ----
    stratified = [_round_block(metrics_mod.mean_ap(rows), settings)]
    if exclude_classes:
        stratified.append(_round_block(
            metrics_mod.mean_ap(rows, exclude=exclude_classes), settings))
    if include_classes:
        stratified.append(_round_block(
            metrics_mod.mean_ap(rows, include=include_classes), settings))
    for block in stratified:
        for key in ("AP50", "AP50_95"):
            block[key] = settings.round(block.get(key))
        for key in ("precision", "recall"):
            block[key] = settings.round(block.get(key))

    # ---- 5) 置信度扫描 ----
    sweep: List[Dict[str, Any]] = []
    recommended: Dict[str, Any] = {}
    if settings.conf_sweep:
        observations = _conf_sweep_observations(net, view, settings, base=raw)
        sweep = [_round_sweep(row, settings)
                 for row in metrics_mod.conf_sweep_rows(settings.conf_thresholds,
                                                        observations)]
        recommended = metrics_mod.recommend_threshold(
            sweep, target_precision=settings.target_precision, decimals=settings.decimals)

    # ---- 6) 报告 ----
    report = EvaluationReport(
        run_id=run_id,
        weights=_display_path(weights_path),
        data_yaml=_display_path(view.yaml_path),
        split=settings.split,
        overall={
            "map50": settings.round(raw["map50"]),
            "map50_95": settings.round(raw["map50_95"]),
            "map75": settings.round(raw["map75"]),
            "precision": settings.round(raw["mean_precision"]),
            "recall": settings.round(raw["mean_recall"]),
            # 这两项是"框架自己算的"全类平均, 与 stratified 的第一块口径一致
            # （不排除任何类时应当相等, 有一条测试专门守这个一致性）
            "note": "map50/map50_95 为框架口径（在'有真值的类'上取平均）",
        },
        per_class=[_round_row(row, settings) for row in rows],
        stratified=stratified,
        conf_sweep=sweep,
        recommended=recommended,
        warnings=metrics_mod.build_warnings(view, rows, settings),
        dataset=view.to_dict(),
        settings_audit=settings.to_audit(),
        speed_ms={key: round(float(value), 3) for key, value in raw["speed"].items()},
        pr_curves=(metrics_mod.pr_curve_payload(raw["curves"], view.class_names,
                                                points=settings.pr_curve_points)
                   if settings.include_pr_curves else []),
        duration_seconds=time.perf_counter() - started,
    )
    report.render_to_logger()

    if write_report:
        report.write_json(out_dir / JSON_FILENAME)
        report.write_per_class_csv(out_dir / PER_CLASS_CSV_FILENAME)
        report.write_conf_sweep_csv(out_dir / CONF_SWEEP_CSV_FILENAME)
        logger.info("报告已写入: %s（report.json / metrics.csv / conf_sweep.csv）", out_dir)

    return report


def _conf_sweep_observations(net: Any, view: DatasetView, settings: EvalSettings,
                             *, base: Dict[str, Any]) -> List[Dict[str, Any]]:
    """逐个阈值跑 val, 收集 P/R。

    ★ 第一个阈值如果是 settings.conf 就复用主评估的结果 —— 同一次计算没必要跑两遍。
    """
    thresholds = list(settings.conf_thresholds)
    progress = ProgressReporter("置信度扫描", len(thresholds),
                                logger=logger, unit="个阈值")
    observations: List[Dict[str, Any]] = []
    for index, threshold in enumerate(thresholds, 1):
        if base is not None and abs(float(threshold) - float(settings.conf)) < 1e-12:
            observed = base
        else:
            observed = run_val(net, view, settings, conf=float(threshold))
        observations.append({"precision": observed.get("mean_precision"),
                             "recall": observed.get("mean_recall")})
        progress.update(index)
    progress.finish(done=len(thresholds))
    return observations


def _round_row(row: Dict[str, Any], settings: EvalSettings) -> Dict[str, Any]:
    rounded = dict(row)
    for key in ("AP50", "AP50_95", "precision", "recall"):
        rounded[key] = settings.round(row.get(key))
    return rounded


def _round_block(block: Dict[str, Any], settings: EvalSettings) -> Dict[str, Any]:
    rounded = dict(block)
    for key in ("AP50", "AP50_95", "precision", "recall"):
        rounded[key] = settings.round(block.get(key))
    return rounded


def _round_sweep(row: Dict[str, Any], settings: EvalSettings) -> Dict[str, Any]:
    rounded = dict(row)
    for key in ("precision", "recall", "f1"):
        rounded[key] = settings.round(row.get(key))
    return rounded


def _warn_unknown_classes(view: DatasetView, exclude: Sequence[str],
                          include: Sequence[str]) -> None:
    """拼错的类名会让"分层口径"静默地什么也没排除 —— 这里明确报出来。"""
    known = set(view.class_names)
    for label, names in (("--exclude-classes", exclude), ("--classes", include)):
        unknown = [name for name in names if name not in known]
        if unknown:
            logger.warning("%s 里有数据集之外的类名（不会起作用）: %s；数据集里是 %s",
                           label, unknown, list(view.class_names))
