#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : report.py
# @Project   : ODPlatform
# @Function  : 评估报告（纯数据 + 三种出口: logger / json / csv）

"""评估报告 —— 纯数据 + 三种出口, 报告本身不含任何判断逻辑。

组织方式刻意与 D4 的 ``data_validation/report.py`` 保持一致: 同一个项目里"看报告"
的习惯应该是一样的, 而不是质检一套、评估又一套。

出口:
    · :meth:`EvaluationReport.render_to_logger` —— 人看的摘要（含每类一行 + 推荐阈值 + 警告）
    · :meth:`EvaluationReport.write_json`        —— 机器读的完整结构（含 P-R 曲线点集）
    · :meth:`EvaluationReport.write_per_class_csv` / :meth:`write_conf_sweep_csv`
      —— Excel 直接打开（utf-8-sig, 中文不乱码）

★ 报告里为什么要有 ``warnings`` 这一块

    指标越小越该解释清楚。``mAP50=0.31`` 单独看像是"模型很差", 但如果同时写着
    "验证集只有 49 张图、ball 类只有 327 个框", 结论就完全不同。
    这些警告由 :func:`evaluation.metrics.build_warnings` 生成, 报告只负责如实呈现。
"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

JSON_FILENAME = "report.json"
PER_CLASS_CSV_FILENAME = "metrics.csv"
CONF_SWEEP_CSV_FILENAME = "conf_sweep.csv"

_CSV_ENCODING = "utf-8-sig"          # 带 BOM: Excel 直接打开中文不乱码
_SEPARATOR_WIDTH = 64


@dataclass
class EvaluationReport:
    """一次模型评估的全部事实。"""

    run_id: str
    weights: str
    data_yaml: str
    split: str
    overall: Dict[str, Any] = field(default_factory=dict)
    per_class: List[Dict[str, Any]] = field(default_factory=list)
    stratified: List[Dict[str, Any]] = field(default_factory=list)
    conf_sweep: List[Dict[str, Any]] = field(default_factory=list)
    recommended: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    dataset: Dict[str, Any] = field(default_factory=dict)
    settings_audit: Dict[str, Any] = field(default_factory=dict)
    speed_ms: Dict[str, Any] = field(default_factory=dict)
    pr_curves: List[Dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0

    # ---------------------------------------------------------------- 取值

    @property
    def map50(self) -> Optional[float]:
        return self.overall.get("map50")

    @property
    def map50_95(self) -> Optional[float]:
        return self.overall.get("map50_95")

    def summary_line(self) -> str:
        """一句话结论（日志与 CLI 都用它, 免得两处各写一种格式）。"""
        def _fmt(value: Any) -> str:
            return "—" if value is None else f"{float(value):.4f}"
        return (f"mAP50={_fmt(self.map50)}  mAP50-95={_fmt(self.map50_95)}  "
                f"验证集 {self.dataset.get('n_images', '?')} 张 / "
                f"{self.dataset.get('total_instances', '?')} 个真值框")

    # ---------------------------------------------------------------- 出口

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "weights": self.weights,
            "data_yaml": self.data_yaml,
            "split": self.split,
            "duration_seconds": round(self.duration_seconds, 3),
            "overall": self.overall,
            "per_class": self.per_class,
            "stratified": self.stratified,
            "conf_sweep": self.conf_sweep,
            "recommended_threshold": self.recommended,
            "warnings": self.warnings,
            "dataset": self.dataset,
            "settings": self.settings_audit,
            "speed_ms": self.speed_ms,
            "pr_curves": self.pr_curves,
        }

    def write_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8")

    def write_per_class_csv(self, path: Path) -> None:
        """每类一行 —— 报告里最该被拿去做表的那张。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding=_CSV_ENCODING, newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["run_id", "class_id", "class_name", "evaluated",
                             "instances", "AP50", "AP50_95", "precision", "recall"])
            for row in self.per_class:
                writer.writerow([
                    self.run_id, row.get("class_id"), row.get("class_name"),
                    row.get("evaluated"), row.get("instances"),
                    _csv_number(row.get("AP50")), _csv_number(row.get("AP50_95")),
                    _csv_number(row.get("precision")), _csv_number(row.get("recall")),
                ])

    def write_conf_sweep_csv(self, path: Path) -> None:
        """每个置信度阈值一行。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding=_CSV_ENCODING, newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["run_id", "conf", "precision", "recall", "f1"])
            for row in self.conf_sweep:
                writer.writerow([self.run_id, row.get("conf"),
                                 _csv_number(row.get("precision")),
                                 _csv_number(row.get("recall")),
                                 _csv_number(row.get("f1"))])

    def render_to_logger(self, log: Optional[logging.Logger] = None) -> None:
        """人看的摘要。"""
        log = log or logger
        log.info("=" * _SEPARATOR_WIDTH)
        log.info("模型评估报告 run_id=%s  split=%s  (%.2fs)",
                 self.run_id, self.split, self.duration_seconds)
        log.info("=" * _SEPARATOR_WIDTH)
        log.info("权重:   %s", self.weights)
        log.info("数据集: %s", self.data_yaml)
        log.info("规模:   %s 张图 / %s 个真值框 / %s 类",
                 self.dataset.get("n_images", "?"),
                 self.dataset.get("total_instances", "?"),
                 self.dataset.get("n_classes", "?"))
        log.info("-" * _SEPARATOR_WIDTH)
        log.info("%s", self.summary_line())

        for block in self.stratified:
            log.info("口径「%s」(%d 类): AP50=%s  AP50-95=%s  P=%s  R=%s",
                     block.get("label"), block.get("n_classes", 0),
                     _fmt(block.get("AP50")), _fmt(block.get("AP50_95")),
                     _fmt(block.get("precision")), _fmt(block.get("recall")))

        if self.per_class:
            log.info("-" * _SEPARATOR_WIDTH)
            log.info("%-18s %10s %10s %10s %10s %10s",
                     "class", "instances", "AP50", "AP50-95", "P", "R")
            for row in self.per_class:
                log.info("%-18s %10s %10s %10s %10s %10s",
                         str(row.get("class_name")), row.get("instances"),
                         _fmt(row.get("AP50")), _fmt(row.get("AP50_95")),
                         _fmt(row.get("precision")), _fmt(row.get("recall")))

        if self.recommended:
            log.info("-" * _SEPARATOR_WIDTH)
            log.info("推荐置信度阈值: %s (conf=%s  P=%s  R=%s  F1=%s)",
                     self.recommended.get("status"),
                     self.recommended.get("conf"),
                     _fmt(self.recommended.get("precision")),
                     _fmt(self.recommended.get("recall")),
                     _fmt(self.recommended.get("f1")))
            log.info("  依据: %s", self.recommended.get("reason"))

        if self.conf_sweep:
            log.info("-" * _SEPARATOR_WIDTH)
            log.info("置信度扫描:")
            log.info("  %8s %10s %10s %10s", "conf", "P", "R", "F1")
            for row in self.conf_sweep:
                log.info("  %8s %10s %10s %10s", row.get("conf"),
                         _fmt(row.get("precision")), _fmt(row.get("recall")),
                         _fmt(row.get("f1")))

        if self.warnings:
            log.info("-" * _SEPARATOR_WIDTH)
            log.info("必须说在前的警告 (%d 条):", len(self.warnings))
            for text in self.warnings:
                log.info("  · %s", text)


def _fmt(value: Any) -> str:
    """None 显示成破折号 —— "没有值"和"值是 0"必须看起来不一样。"""
    if value is None:
        return "—"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _csv_number(value: Any) -> Any:
    """CSV 里的 None 留空(Excel 里是空单元格), 不写成 0。"""
    return "" if value is None else value
