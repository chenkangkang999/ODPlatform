#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : retention.py
# @Project   : ODPlatform
# @Function  : 运行现场的保留策略（runs/ 的清理规则）

"""runs/ 的保留策略。

★ 问题

    每次跑流水线都会在 ``runs/`` 下留一个现场目录: 推理输出（图片/视频）、质检
    报告、划分 manifest、训练现场…… 长期累积能吃掉几十 GB, 而此前**没有任何保留
    策略** —— 只有 ``odp-reset`` 那种"整目录白名单清空"的粗暴手段, 没人敢在有
    真实产物时用它。

★ 关于 lineage: 这一条最重要

    ``runs/lineage/`` 存的是 D3/D4/D6 盖的**章**——数据出身链的凭据。
    ``odp-train`` 的身份核验靠它三方对账。**删了它, 身份核验就再也对不上账**,
    而且无法重建（重跑 D3 会换 manifest 指纹）。
    所以它被硬性排除在清理范围之外, 不是靠"默认参数没选到它"。

★ 策略（可解释、可预测, 不做"猜你想删哪个"）

    在每个受管子目录里保留:
      · 最新的 ``keep_latest_per_group`` 个, 以及
      · 任何修改时间在 ``max_age_days`` 之内的;
    其余才列入清理计划。**两个条件都满足才删** —— 宁可留着。

    于是"跑得多"不会把最近的产物删掉, "跑得久"也不会把最新的删掉。
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from od_platform.common import paths

logger = logging.getLogger(__name__)

# 永远不清理的子目录: 出身链的章（删了身份核验就断了, 且无法重建）
PROTECTED_SUBDIRS: Tuple[str, ...] = ("lineage",)

DEFAULT_KEEP_LATEST: int = 5         # 每组至少留最新的几个
DEFAULT_MAX_AGE_DAYS: float = 30.0   # 这个天数内的东西一律不动
SECONDS_PER_DAY: float = 86400.0


@dataclass(frozen=True)
class RetentionPolicy:
    """保留策略。默认值偏保守: 宁可留着, 也不要误删还在用的现场。"""

    keep_latest_per_group: int = DEFAULT_KEEP_LATEST
    max_age_days: float = DEFAULT_MAX_AGE_DAYS
    extra_protected: Tuple[str, ...] = ()      # 额外豁免的子目录名

    def __post_init__(self) -> None:
        if self.keep_latest_per_group < 1:
            raise ValueError(f"keep_latest_per_group 必须 >= 1, 收到 {self.keep_latest_per_group}")
        if self.max_age_days < 0:
            raise ValueError(f"max_age_days 不能为负, 收到 {self.max_age_days}")

    @property
    def protected(self) -> Tuple[str, ...]:
        return tuple(PROTECTED_SUBDIRS) + tuple(self.extra_protected)

    def max_age_seconds(self) -> float:
        return self.max_age_days * SECONDS_PER_DAY


@dataclass(frozen=True)
class CleanupCandidate:
    """一个可清理的现场目录。"""
    path: Path
    group: str
    size_bytes: int
    age_days: float
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return {"path": str(self.path), "group": self.group,
                "size_bytes": self.size_bytes, "age_days": round(self.age_days, 2),
                "reason": self.reason}


@dataclass
class CleanupPlan:
    """清理计划（还没动手）。"""
    candidates: List[CleanupCandidate] = field(default_factory=list)
    kept: List[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(c.size_bytes for c in self.candidates)

    @property
    def is_empty(self) -> bool:
        return not self.candidates

    def render_lines(self) -> List[str]:
        if self.is_empty:
            return ["没有需要清理的现场目录（策略内全部保留）"]
        lines = [f"计划清理 {len(self.candidates)} 个现场, "
                 f"可释放 {self.total_bytes / 1048576:.1f} MB:"]
        for candidate in sorted(self.candidates, key=lambda c: -c.size_bytes):
            lines.append(f"  [{candidate.group}] {candidate.path.name}  "
                         f"{candidate.size_bytes / 1048576:.1f} MB  "
                         f"{candidate.age_days:.1f} 天  ({candidate.reason})")
        return lines


# ---------------------------------------------------------------- 内部

def _dir_size(path: Path) -> int:
    """目录占用的字节数; 读不到的项跳过（不算错）。"""
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file() and not entry.is_symlink():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def _managed_groups(runs_dir: Path, policy: RetentionPolicy) -> List[Path]:
    """受管的子目录（排除豁免项）。"""
    if not runs_dir.is_dir():
        return []
    protected = set(policy.protected)
    return sorted(d for d in runs_dir.iterdir()
                  if d.is_dir() and not d.is_symlink() and d.name not in protected)


def _assert_within(sandbox: Path, target: Path) -> None:
    """围栏: target 必须严格位于 sandbox 之内（且不等于 sandbox 本身）。"""
    sb, tg = sandbox.resolve(), target.resolve()
    if tg == sb:
        raise ValueError(f"拒绝操作沙盒根目录本身: {tg}")
    try:
        tg.relative_to(sb)
    except ValueError:
        raise ValueError(f"拒绝操作沙盒外路径: {tg}(沙盒: {sb})") from None


# ---------------------------------------------------------------- 对外

def plan_cleanup(runs_dir: Optional[Path] = None, policy: Optional[RetentionPolicy] = None,
                 *, now: Optional[float] = None) -> CleanupPlan:
    """算出要清理哪些现场（只算不动）。

    Args:
        runs_dir: 运行现场根目录; None 时用 paths.RUNS_DIR
        policy: 保留策略; None 时用默认策略
        now: 当前时间戳（测试注入用）
    """
    runs_dir = Path(runs_dir) if runs_dir else paths.RUNS_DIR
    policy = policy or RetentionPolicy()
    now = time.time() if now is None else now

    plan = CleanupPlan()
    for group_dir in _managed_groups(runs_dir, policy):
        entries = sorted((d for d in group_dir.iterdir() if d.is_dir() and not d.is_symlink()),
                         key=lambda d: d.stat().st_mtime, reverse=True)
        for index, entry in enumerate(entries):
            age_seconds = max(0.0, now - entry.stat().st_mtime)
            keep_by_count = index < policy.keep_latest_per_group
            keep_by_age = age_seconds < policy.max_age_seconds()
            if keep_by_count or keep_by_age:
                plan.kept.append(f"{group_dir.name}/{entry.name}")
                continue
            reasons = []
            if not keep_by_count:
                reasons.append(f"不是最新 {policy.keep_latest_per_group} 个")
            if not keep_by_age:
                reasons.append(f"超过 {policy.max_age_days:g} 天")
            plan.candidates.append(CleanupCandidate(
                path=entry, group=group_dir.name, size_bytes=_dir_size(entry),
                age_days=age_seconds / SECONDS_PER_DAY, reason=" 且 ".join(reasons)))
    return plan


def apply_cleanup(plan: CleanupPlan, *, runs_dir: Optional[Path] = None) -> Dict[str, int]:
    """执行清理计划。返回 {"removed": n, "freed_bytes": n, "skipped": n}。

    每个候选都先过围栏校验; 不合规的跳过并告警, 不抛异常打断整场清理。
    """
    runs_dir = Path(runs_dir) if runs_dir else paths.RUNS_DIR
    removed = skipped = freed = 0
    for candidate in plan.candidates:
        try:
            _assert_within(runs_dir, candidate.path)
            if candidate.path.is_symlink():
                raise ValueError("是符号链接, 拒绝递归删除")
            shutil.rmtree(candidate.path)
        except (OSError, ValueError) as exc:
            logger.warning("跳过 %s: %s", candidate.path, exc)
            skipped += 1
            continue
        removed += 1
        freed += candidate.size_bytes
        logger.info("已清理 %s/%s（%.1f MB）", candidate.group,
                    candidate.path.name, candidate.size_bytes / 1048576)
    return {"removed": removed, "freed_bytes": freed, "skipped": skipped}


def summarize_runs(runs_dir: Optional[Path] = None) -> Dict[str, object]:
    """给 CLI 用的现状概览: 各组目录数 + 总占用。"""
    runs_dir = Path(runs_dir) if runs_dir else paths.RUNS_DIR
    groups: Dict[str, Dict[str, object]] = {}
    total = 0
    if runs_dir.is_dir():
        for entry in sorted(runs_dir.iterdir()):
            if not entry.is_dir() or entry.is_symlink():
                continue
            size = _dir_size(entry)
            total += size
            groups[entry.name] = {
                "subdirs": sum(1 for d in entry.iterdir() if d.is_dir()),
                "size_bytes": size,
                "protected": entry.name in PROTECTED_SUBDIRS,
            }
    return {"runs_dir": str(runs_dir), "total_bytes": total, "groups": groups}