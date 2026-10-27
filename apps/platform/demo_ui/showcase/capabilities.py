#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : capabilities.py
# @Project   : ODPlatform / demo_ui
# @Function  : 如实汇总算法侧能力与参数（全部从配置层与训练产物读出）

"""把"算法现在到底能做什么、参数是多少"如实汇总出来。

★ 这一页的价值全在**诚实**。答辩现场会被追问的正是"训了多少轮""场地标定有没有做"
  "动作识别到底加载的是什么模型"—— 所以这里一条都不写死在模板里, 全部从真实来源读:

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
from od_platform.events import (
    KIND_RED_CARD,
    KIND_SEVERITY,
    KIND_SUBSTITUTION,
    KIND_YELLOW_CARD,
    EventSettings,
)
from od_platform.geometric.calibration import default_calibration_dir
from od_platform.geometric.settings import GeometrySettings
from od_platform.pose import (
    ACTION_JUDGES,
    EVIDENCE_CANDIDATE,
    EVIDENCE_INSUFFICIENT,
    EVIDENCE_JUDGEMENT,
    PoseSettings,
)

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

    ★ 这是"接触/出界/角球/点球/任意球能不能产出"的唯一开关: 没有标定就没有"米",
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
# 姿态估计（od_platform.pose）—— 铲球 / 高抬腿 / 肘击 / 拉人推人
# ============================================================

def pose_summary() -> Dict[str, Any]:
    """姿态估计子系统的真实状态与参数。

    ★ 它和"动作识别"卡片是**两件不同的事**, 答辩时别混:

        · 姿态估计 → 用**骨骼几何**判"姿态型动作"（铲球 / 高抬腿 / 肘击 / 拉人推人）;
        · 动作识别 → 用**视频分类模型**判"语义型动作"（假摔 / 推搡 …）。

      "高抬腿"就是"脚踝高于髋部" —— 这是几何判据, 拿分类器去猜反而更容易错。

    ★ 权重拿不到就如实报不可用: 加载失败时 estimate() 会优雅降级为空列表
      （不抛异常）, 页面上不能假装有结果。
    """
    try:
        settings = PoseSettings.load()
    except (ValueError, TypeError) as exc:
        return {"available": False, "actions": [], "reason": f"姿态配置不合法: {exc}"}

    weights = Path(str(settings.model.resolved_weights()))
    exists = weights.is_file()
    return {
        "available": exists,
        "weights": str(weights),
        "weights_exists": exists,
        "weights_megabytes": (round(weights.stat().st_size / (1024 * 1024), 2)
                              if exists else None),
        "parameters": settings.to_audit(),
        "actions": list(ACTION_JUDGES),
        "evidence_levels": [EVIDENCE_JUDGEMENT, EVIDENCE_CANDIDATE, EVIDENCE_INSUFFICIENT],
        "reason": ("姿态权重就绪, 可用骨骼几何判定姿态型动作" if exists else
                   f"找不到姿态权重 {weights} —— 判定会全部返回"
                   f"「{EVIDENCE_INSUFFICIENT}」(证据不足), 不会瞎猜"),
    }


# ============================================================
# 判罚事件（od_platform.events）—— 红牌 / 黄牌 / 换人
# ============================================================

def events_summary() -> Dict[str, Any]:
    """判罚事件子系统的真实状态。

    ★★ 必须如实说清的一件事: 这三项**依赖一个能检出「牌」的模型**。

       本仓库现有足球权重只有 player / football 两类, **检不出牌** —— 所以本子系统
       **目前一帧也不会产出事件**。要工作必须先扩类训练, 这是硬前提, 调参解决不了。

    ★ 类别名映射是**配置项而非代码**: 模型训练时的类名可能叫 red_card / card_red /
       红牌 …, 换模型时只改 configs/runtime/events.yaml, 不用动代码。这里读出来展示。
    """
    try:
        settings = EventSettings.load()
    except (ValueError, TypeError) as exc:
        return {"available": False, "reason": f"事件配置不合法: {exc}"}

    aliases = settings.classes
    expected = sorted(set(aliases.red_card_aliases) | set(aliases.yellow_card_aliases))
    # 现有模型认识哪些类: 取自本演示的展示映射（即 player / football 两个）
    known = sorted(demo_config.LABEL_MAPPING)
    missing = [name for name in expected if name not in known]
    return {
        # available 恒为 False, 且写成与配置无关的常量 —— 它反映的是事实
        # （现有权重里没有"牌"这个类）, 而不是"某个参数没调好"。
        "available": False,
        "parameters": settings.to_audit(),
        "event_kinds": [KIND_RED_CARD, KIND_YELLOW_CARD, KIND_SUBSTITUTION],
        "severity": dict(KIND_SEVERITY),
        "expected_card_classes": expected,
        "known_classes": known,
        "missing_card_classes": missing,
        "reason": (f"判罚事件需要能检出「牌」的模型; 现有权重只有 {'/'.join(known)} 两类, "
                   f"检不出 {', '.join(missing[:3])} 等 → 本子系统目前不产出任何事件"),
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
            # 说成"已支持"。slowfast 已按需求清单删除, PLANNED_FAMILIES 现在只剩 custom。
            "supported_families": list(SUPPORTED_FAMILIES),
            "planned_families": list(PLANNED_FAMILIES),
        },
        # 两个新子系统: 姿态估计（骨骼几何）与判罚事件（红黄牌/换人）。
        # 都走"从设置类读 + 如实报状态"的同一条路, 不在模板里写死任何结论。
        "pose": pose_summary(),
        "events": events_summary(),
    }