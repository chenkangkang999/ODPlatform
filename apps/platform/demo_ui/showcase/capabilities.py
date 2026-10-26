#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : capabilities.py
# @Project   : ODPlatform / demo_ui
# @Function  : 如实汇总算法侧能力与参数（全部从配置层与训练产物读出）

"""把"算法现在到底能做什么、参数是多少"如实汇总出来。

★ 这一页的价值全在**诚实**。答辩现场会被追问的正是"训了多少轮""SlowFast 到底有没有
  ""场地标定有没有做"—— 所以这里一条都不写死在模板里, 全部从真实来源读:

    · 检测/跟踪参数  ← od_platform.detection.settings.DetectionSettings（football.yaml）
    · 动作识别       ← od_platform.action_recognition.settings.ActionRecognitionSettings
    · 场地/标定/规则 ← od_platform.geometric.settings.GeometrySettings
    · 标定到底有没有 ← od_platform.geometric.calibration（磁盘实况）
    · 训了多少轮/指标← models/trained/<run>/results.csv 与 args.yaml（训练产物）

  这个模块**不 import Django**, 便于单测直接调用。
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from od_platform.action_recognition.settings import (
    PLANNED_FAMILIES,
    SUPPORTED_FAMILIES,
    ActionRecognitionSettings,
)
from od_platform.common import paths
from od_platform.common.config_utils import read_section
from od_platform.common.football_config import load_config
from od_platform.detection.settings import DetectionSettings
from od_platform.geometric.calibration import default_calibration_dir
from od_platform.geometric.settings import GeometrySettings

from showcase import demo_config

logger = logging.getLogger(__name__)


# ============================================================
# 训练产物: 训了多少轮、指标多少（不写死数字）
# ============================================================

def training_summary(run_dir: Optional[Path] = None) -> Dict[str, Any]:
    """从训练产物里如实读出轮数与指标。

    取值方式是**按列名**取而不是按列序, 这样 ultralytics 增删列也不会读错。

    Args:
        run_dir: 训练产物目录; None 时用 demo_config 里的默认 run

    Returns:
        含 epochs_done / planned_epochs / metrics / is_smoke 等; 产物缺失时
        available=False 并给出原因（而不是编一组好看的数字）。
    """
    run_dir = Path(run_dir) if run_dir else demo_config.trained_run_dir()
    # 权重就在这个 run 目录里（默认 run 与 demo_config 的默认权重是同一处）
    weights = run_dir / "weights" / demo_config.WEIGHTS_FILENAME
    summary: Dict[str, Any] = {
        "available": False,
        "run_dir": str(run_dir),
        "weights": str(weights),
        "weights_exists": weights.is_file(),
    }

    results_path = run_dir / demo_config.TRAIN_RESULTS_FILENAME
    if not results_path.is_file():
        summary["reason"] = f"没有训练记录文件: {results_path.name}"
        return summary

    try:
        with results_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        summary["reason"] = f"训练记录读不出来: {exc}"
        return summary
    if not rows:
        summary["reason"] = "训练记录里没有任何轮次"
        return summary

    last = rows[-1]

    def _number(key: str) -> Optional[float]:
        """按列名取一个数; 取不到就返回 None（不编 0）。"""
        raw = last.get(key)
        if raw in (None, ""):
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    epochs_done = _number("epoch")
    summary.update({
        "available": True,
        "epochs_done": int(epochs_done) if epochs_done is not None else None,
        "precision": _number("metrics/precision(B)"),
        "recall": _number("metrics/recall(B)"),
        "map50": _number("metrics/mAP50(B)"),
        "map50_95": _number("metrics/mAP50-95(B)"),
        "seconds": _number("time"),
    })

    args_path = run_dir / demo_config.TRAIN_ARGS_FILENAME
    planned = imgsz = batch = device = model = dataset = None
    if args_path.is_file():
        try:
            import yaml

            args = yaml.safe_load(args_path.read_text(encoding="utf-8")) or {}
            planned = args.get("epochs")
            imgsz = args.get("imgsz")
            batch = args.get("batch")
            device = args.get("device")
            model = args.get("model")
            dataset = args.get("data")
        except Exception as exc:  # noqa: BLE001 —— 读不到就当没记录, 不该让页面挂
            logger.warning("训练 args 读不出来: %s", exc)
    summary.update({"planned_epochs": planned, "imgsz": imgsz, "batch": batch,
                    "device": device, "base_model": model, "trained_dataset": dataset})

    # 是否够得上"可用模型": 轮数太少就如实标成冒烟级
    done = summary.get("epochs_done")
    summary["is_smoke"] = done is not None and done <= demo_config.SMOKE_EPOCH_THRESHOLD
    return summary


# ============================================================
# 场地标定: 到底有没有（决定几何事件产不产出）
# ============================================================

def calibration_status(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """报告场地标定的真实状态。

    ★ 这是"越位/出界/进球/接触能不能产出"的唯一开关: 没有标定就没有"米",
      几何事件一律不产出（这是设计行为, 不是 bug）。所以演示页必须如实显示,
      而不是含糊地说"支持场地标定"。

    检查三处（与 PitchCalibration.discover 的候选顺序一致）:
      1. football.yaml 的 homography.points      —— 直接内联的标定点
      2. football.yaml 的 homography.calibration_file
      3. configs/runtime/calibrations/*.json     —— 生成好的标定文件目录
    """
    config = load_config() if config is None else config
    homography = read_section(config, "homography")
    inline_points = list(homography.get("points") or [])
    configured_file = homography.get("calibration_file")
    directory = default_calibration_dir()
    files: List[str] = sorted(p.name for p in directory.glob("*.json")) if directory.is_dir() else []
    configured_exists = bool(configured_file) and (
        paths.RUNTIME_CONFIGS_DIR / str(configured_file)).is_file()

    calibrated = bool(inline_points) or configured_exists or bool(files)
    if calibrated:
        if inline_points:
            reason = f"配置里内联了 {len(inline_points)} 个标定点"
        elif configured_exists:
            reason = f"配置指定了标定文件: {configured_file}"
        else:
            reason = f"标定目录里有 {len(files)} 个标定文件"
    else:
        reason = ("没有任何标定: 配置里既没内联标定点、也没指定标定文件, "
                  f"且 {directory.name}/ 目录下没有 *.json —— 几何事件不会产出")

    return {
        "calibrated": calibrated,
        "reason": reason,
        "inline_points": len(inline_points),
        "configured_file": configured_file,
        "configured_file_exists": configured_exists,
        "directory": str(directory),
        "files": files,
    }


# ============================================================
# 汇总
# ============================================================

def build_report(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """汇总四层能力的真实状态与参数, 供总览页渲染。"""
    config = load_config() if config is None else config
    detection = DetectionSettings.from_app_config(config)
    action = ActionRecognitionSettings.from_app_config(config)
    geometry = GeometrySettings.from_config(config)

    weights = demo_config.trained_weights_path()
    return {
        "config_loaded": bool(config),
        "detection": {
            "parameters": detection.to_audit(),
            "weights": str(weights),
            "weights_exists": weights.is_file(),
            "weights_megabytes": round(weights.stat().st_size / (1024 * 1024), 2) if weights.is_file() else None,
            "training": training_summary(),
        },
        "tracking": {
            "backend": detection.tracker,
            # 注意区分: 对外产出 track_id 的是 ultralytics 的 BoT-SORT, 参数在
            # configs/runtime/botsort.yaml; 几何层内部另有一个跟踪器（football.yaml
            # 的 tracking 节）。这里把两者都摆出来, 免得被问到时说不清。
            "tracker_config": detection.resolved_tracker(),
            "external_tracker_file": str(paths.RUNTIME_CONFIGS_DIR / detection.tracker),
            "internal_tracker_section": read_section(config, "tracking"),
        },
        "geometry": {
            "parameters": geometry.to_audit(),
            "calibration": calibration_status(config),
        },
        "action": {
            **action.to_audit(),
            # 已实现/仅预留的模型族直接从设置的常量来 —— 免得页面上把预留位
            # 说成"已支持"。slowfast 属于 PLANNED_FAMILIES, 代码里没有实现。
            "supported_families": list(SUPPORTED_FAMILIES),
            "planned_families": list(PLANNED_FAMILIES),
        },
    }