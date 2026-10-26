#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : compare.py
# @Project   : ODPlatform
# @Function  : 质检报告历史对比（本次 vs 上次: 新增了什么、修好了什么）

"""质检报告历史对比。

★ 解决什么问题

    单次报告只回答"现在有没有问题", 回答不了**"这一次比上一次好了还是坏了"**。
    而修数据的人最想知道的就是这两件事: 我刚修的那个问题真的消失了吗? 有没有
    顺手引入新的? 此前每次质检都是孤立的一跑, 这些信息只能靠人肉 diff 两份 json。

★ 设计取舍

    · **对比口径用 severity 的变化**, 不逐项解析各 check 的 details ——
      details 的结构每个 check 都不一样（``n_out`` / ``n_bad`` / ``missing`` /
      ``n_changed`` / ``issue_kinds`` ...）, 强行统一会退化成一堆脆弱特例;
      而 severity 是所有 check 的共同语言。
    · **历史报告靠 ``yaml_path`` 精确匹配**（报告里没记数据集名）。同一份数据换个
      路径会被当成不同数据集 —— 这是有意的: 路径变了, 数据就不是同一批。
    · INFO 不算"问题": 与 registry 的定义保持一致（``passed = severity in
      (PASS, INFO)``）。把 INFO 当问题会让每份不均衡数据集的报告都常年飘红。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from od_platform.common import paths
from od_platform.data_validation.registry import CheckSeverity

logger = logging.getLogger(__name__)

REPORT_FILENAME = "report.json"
# "算作问题"的严重级: INFO 是提示不是问题（与 registry.passed 同一口径）
PROBLEM_SEVERITIES = (CheckSeverity.WARNING, CheckSeverity.ERROR)


def severity_rank(severity: Optional[str]) -> int:
    """严重级排序; 未知/缺失按最低处理（不炸）。"""
    try:
        return CheckSeverity.rank(severity)
    except KeyError:
        return CheckSeverity.rank(CheckSeverity.PASS)


def is_problem(severity: Optional[str]) -> bool:
    return severity in PROBLEM_SEVERITIES


@dataclass(frozen=True)
class CheckChange:
    """单个检查项在两份报告之间的变化。"""
    name: str
    previous: Optional[str]
    current: Optional[str]
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "previous": self.previous,
                "current": self.current, "summary": self.summary}


@dataclass(frozen=True)
class Comparison:
    """两份质检报告的对比结论。"""
    previous_run_id: str
    current_run_id: str
    previous_overall: str
    current_overall: str
    newly_failing: List[CheckChange] = field(default_factory=list)
    fixed: List[CheckChange] = field(default_factory=list)
    still_failing: List[CheckChange] = field(default_factory=list)
    added_checks: List[str] = field(default_factory=list)
    removed_checks: List[str] = field(default_factory=list)

    @property
    def improved(self) -> bool:
        """整体变好: 分数下降, 或者修好的比新出的多。"""
        before = severity_rank(self.previous_overall)
        after = severity_rank(self.current_overall)
        if after != before:
            return after < before
        return len(self.fixed) > len(self.newly_failing)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "previous_run_id": self.previous_run_id,
            "current_run_id": self.current_run_id,
            "previous_overall": self.previous_overall,
            "current_overall": self.current_overall,
            "improved": self.improved,
            "newly_failing": [c.to_dict() for c in self.newly_failing],
            "fixed": [c.to_dict() for c in self.fixed],
            "still_failing": [c.to_dict() for c in self.still_failing],
            "added_checks": self.added_checks,
            "removed_checks": self.removed_checks,
        }

    def render_lines(self) -> List[str]:
        """给人看的几行结论（CLI 直接打这些）。"""
        lines = [
            f"与上次对比(run {self.previous_run_id} → {self.current_run_id}): "
            f"总评 {self.previous_overall} → {self.current_overall} "
            f"({'改善' if self.improved else '未改善'})"
        ]
        if self.fixed:
            lines.append(f"  已修复 {len(self.fixed)} 项: "
                         + ", ".join(f"{c.name}({c.previous}→{c.current})" for c in self.fixed))
        if self.newly_failing:
            lines.append(f"  ⚠ 新出现问题 {len(self.newly_failing)} 项: "
                         + ", ".join(f"{c.name}({c.previous}→{c.current})"
                                     for c in self.newly_failing))
        if self.still_failing:
            lines.append(f"  仍然存在 {len(self.still_failing)} 项: "
                         + ", ".join(c.name for c in self.still_failing))
        if self.added_checks:
            lines.append(f"  新增检查项: {', '.join(self.added_checks)}")
        if self.removed_checks:
            lines.append(f"  移除检查项: {', '.join(self.removed_checks)}")
        if not (self.fixed or self.newly_failing or self.still_failing):
            lines.append("  两次报告的问题集合没有变化")
        return lines


# ---------------------------------------------------------------- 历史定位

def load_report(path: Path) -> Optional[Dict[str, Any]]:
    """读一份 report.json; 读不了或结构不对返回 None（历史对比失败不该影响本次质检）。"""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("历史报告读取失败, 跳过对比: %s (%s)", path, exc)
        return None
    return data if isinstance(data, dict) else None


def find_previous_report(yaml_path: Path, *, exclude_run_id: Optional[str] = None,
                         runs_dir: Optional[Path] = None) -> Optional[Path]:
    """找同一份数据集（同一 yaml）上一次的质检报告。

    Args:
        yaml_path: 本次质检的数据集 yaml
        exclude_run_id: 本次的 run_id（绝不能把自己当成上一次）
        runs_dir: 报告根目录; None 时用 paths.VALIDATION_RUNS_DIR

    Returns:
        上一份 report.json 的路径; 没有历史返回 None
    """
    runs_dir = Path(runs_dir) if runs_dir else paths.VALIDATION_RUNS_DIR
    if not runs_dir.is_dir():
        return None
    target = Path(yaml_path).resolve()
    # run_id 形如 20261026-195456, 字典序即时间序
    for run_dir in sorted((d for d in runs_dir.iterdir() if d.is_dir()), reverse=True):
        if exclude_run_id and run_dir.name == exclude_run_id:
            continue
        candidate = run_dir / REPORT_FILENAME
        if not candidate.is_file():
            continue
        data = load_report(candidate)
        if not data:
            continue
        recorded = data.get("yaml_path")
        if not recorded:
            continue
        try:
            if Path(recorded).resolve() == target:
                return candidate
        except OSError:                      # 路径在当前机器上无法解析
            continue
    return None


# ---------------------------------------------------------------- 对比

def _by_name(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    results = report.get("results")
    if not isinstance(results, list):
        return {}
    return {str(r.get("name")): r for r in results if isinstance(r, dict) and r.get("name")}


def compare_reports(previous: Dict[str, Any], current: Dict[str, Any]) -> Comparison:
    """对比两份报告, 给出新增/已修复/仍然存在的问题。"""
    before, after = _by_name(previous), _by_name(current)
    newly_failing: List[CheckChange] = []
    fixed: List[CheckChange] = []
    still_failing: List[CheckChange] = []

    for name, current_result in sorted(after.items()):
        current_sev = current_result.get("severity")
        summary = str(current_result.get("summary", ""))
        previous_result = before.get(name)
        if previous_result is None:
            # 新加进来的检查项: 如果它一上来就报警, 那是"新发现的问题"
            if is_problem(current_sev):
                newly_failing.append(CheckChange(name, None, current_sev, summary))
            continue
        previous_sev = previous_result.get("severity")
        if severity_rank(current_sev) > severity_rank(previous_sev):
            newly_failing.append(CheckChange(name, previous_sev, current_sev, summary))
        elif severity_rank(current_sev) < severity_rank(previous_sev) and is_problem(previous_sev):
            fixed.append(CheckChange(name, previous_sev, current_sev, summary))
        elif is_problem(current_sev):
            still_failing.append(CheckChange(name, previous_sev, current_sev, summary))

    return Comparison(
        previous_run_id=str(previous.get("run_id", "?")),
        current_run_id=str(current.get("run_id", "?")),
        previous_overall=str(previous.get("overall_severity", CheckSeverity.PASS)),
        current_overall=str(current.get("overall_severity", CheckSeverity.PASS)),
        newly_failing=newly_failing, fixed=fixed, still_failing=still_failing,
        added_checks=sorted(set(after) - set(before)),
        removed_checks=sorted(set(before) - set(after)),
    )


def compare_with_history(yaml_path: Path, current_report: Dict[str, Any], *,
                         exclude_run_id: Optional[str] = None,
                         runs_dir: Optional[Path] = None) -> Optional[Comparison]:
    """端到端: 找到同一数据集的上一次报告并对比; 没有历史返回 None。"""
    previous_path = find_previous_report(
        yaml_path, exclude_run_id=exclude_run_id or current_report.get("run_id"),
        runs_dir=runs_dir)
    if previous_path is None:
        return None
    previous = load_report(previous_path)
    if previous is None:
        return None
    return compare_reports(previous, current_report)