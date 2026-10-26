#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""runs/ 保留策略（D5 2.1 第四条）的测试。

守三件事:
    · lineage 目录**永不**被列入清理计划（删了身份核验就断了, 且无法重建）;
    · "最新的 N 个"和"N 天内"两个条件都满足才删 —— 宁可留着;
    · 执行阶段有围栏: 沙盒外/符号链接一律拒绝, 且失败不打断整场清理。
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from od_platform.common.retention import (
    PROTECTED_SUBDIRS,
    CleanupPlan,
    RetentionPolicy,
    apply_cleanup,
    plan_cleanup,
    summarize_runs,
)

DAY = 86400.0


def make_run(runs_dir: Path, group: str, name: str, *, age_days: float,
             size_bytes: int = 1024, now: float) -> Path:
    """造一个现场目录, 并把 mtime 设成 age_days 天前。"""
    path = runs_dir / group / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "artifact.bin").write_bytes(b"x" * size_bytes)
    stamp = now - age_days * DAY
    import os
    os.utime(path, (stamp, stamp))
    os.utime(path / "artifact.bin", (stamp, stamp))
    return path


@pytest.fixture()
def now():
    return time.time()


# ---------------------------------------------------------------- 策略本身

def test_policy_rejects_impossible_values():
    with pytest.raises(ValueError):
        RetentionPolicy(keep_latest_per_group=0)
    with pytest.raises(ValueError):
        RetentionPolicy(max_age_days=-1)


def test_lineage_is_protected_and_extra_protection_is_additive():
    assert "lineage" in PROTECTED_SUBDIRS
    assert "lineage" in RetentionPolicy().protected
    assert "my_keep" in RetentionPolicy(extra_protected=("my_keep",)).protected


# ---------------------------------------------------------------- 计划

def test_recent_runs_are_never_planned_for_cleanup(tmp_path, now):
    """不管有多少个, 只要都在 max_age_days 内就一个都不删。"""
    for index in range(12):
        make_run(tmp_path, "detect_infer", f"run{index:02d}", age_days=1.0, now=now)
    plan = plan_cleanup(tmp_path, RetentionPolicy(keep_latest_per_group=2, max_age_days=30),
                        now=now)
    assert plan.is_empty
    assert len(plan.kept) == 12


def test_latest_n_are_kept_even_when_old(tmp_path, now):
    """跑得久也不行: 最新的 N 个必须在。

    注意 age_days 越大越旧: run00 最新(100 天), run05 最旧(105 天)。
    """
    for index in range(6):
        make_run(tmp_path, "detect_infer", f"run{index:02d}",
                 age_days=100.0 + index, now=now)
    plan = plan_cleanup(tmp_path, RetentionPolicy(keep_latest_per_group=3, max_age_days=30),
                        now=now)
    removed_names = {c.path.name for c in plan.candidates}
    assert removed_names == {"run03", "run04", "run05"}      # 最旧的三个
    # plan.kept 里存的是"组/名"的相对路径, 不是裸名字
    assert {"detect_infer/run00", "detect_infer/run01"} <= set(plan.kept)


def test_lineage_is_never_a_candidate_even_when_old(tmp_path, now):
    """这是这一项最重要的一条: 出身链的章必须活着。"""
    for index in range(20):
        make_run(tmp_path, "lineage", f"stamp{index:02d}", age_days=500.0, now=now)
    plan = plan_cleanup(tmp_path, RetentionPolicy(keep_latest_per_group=1, max_age_days=0),
                        now=now)
    assert plan.is_empty
    assert not any(c.group == "lineage" for c in plan.candidates)


def test_plan_groups_are_independent(tmp_path, now):
    """每组各自算"最新 N 个", 不会被别的组挤掉。"""
    for index in range(4):
        make_run(tmp_path, "detect_infer", f"r{index}", age_days=100.0, now=now)
        make_run(tmp_path, "data_validation", f"v{index}", age_days=100.0, now=now)
    plan = plan_cleanup(tmp_path, RetentionPolicy(keep_latest_per_group=2, max_age_days=0),
                        now=now)
    per_group: dict = {}
    for candidate in plan.candidates:
        per_group.setdefault(candidate.group, []).append(candidate.path.name)
    assert len(per_group["detect_infer"]) == 2
    assert len(per_group["data_validation"]) == 2


def test_candidate_reports_size_and_reason(tmp_path, now):
    make_run(tmp_path, "detect_infer", "old", age_days=100.0, size_bytes=2048, now=now)
    make_run(tmp_path, "detect_infer", "new", age_days=1.0, size_bytes=1024, now=now)
    plan = plan_cleanup(tmp_path, RetentionPolicy(keep_latest_per_group=1, max_age_days=30),
                        now=now)
    candidate = plan.candidates[0]
    assert candidate.path.name == "old"
    assert candidate.size_bytes == 2048
    assert candidate.age_days > 90
    assert "不是最新" in candidate.reason and "超过" in candidate.reason
    assert plan.total_bytes == 2048
    assert "可释放" in "\n".join(plan.render_lines())


def test_missing_runs_dir_is_not_an_error(tmp_path):
    plan = plan_cleanup(tmp_path / "nope", RetentionPolicy(), now=time.time())
    assert plan.is_empty
    assert "没有需要清理" in plan.render_lines()[0]


# ---------------------------------------------------------------- 执行

def test_apply_removes_only_the_planned_directories(tmp_path, now):
    old = make_run(tmp_path, "detect_infer", "old", age_days=100.0, now=now)
    fresh = make_run(tmp_path, "detect_infer", "fresh", age_days=1.0, now=now)
    plan = plan_cleanup(tmp_path, RetentionPolicy(keep_latest_per_group=1, max_age_days=30),
                        now=now)

    result = apply_cleanup(plan, runs_dir=tmp_path)
    assert result["removed"] == 1 and result["freed_bytes"] > 0 and result["skipped"] == 0
    assert not old.exists()
    assert fresh.is_dir()


def test_apply_refuses_paths_outside_the_runs_dir(tmp_path, now):
    """围栏: 计划里塞进沙盒外的路径时必须拒绝, 而不是照删。"""
    outside = tmp_path / "outside" / "precious"
    outside.mkdir(parents=True)
    (outside / "keep.txt").write_text("别删我", encoding="utf-8")
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    from od_platform.common.retention import CleanupCandidate
    plan = CleanupPlan(candidates=[CleanupCandidate(
        path=outside, group="evil", size_bytes=1, age_days=999, reason="测试")])
    result = apply_cleanup(plan, runs_dir=runs_dir)

    assert result["removed"] == 0 and result["skipped"] == 1
    assert outside.is_dir() and (outside / "keep.txt").is_file()


def test_apply_refuses_symlinks(tmp_path, now):
    """符号链接一律拒绝递归删除（链接一跟, 源就没了）。"""
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "data.txt").write_text("真实数据", encoding="utf-8")
    runs_dir = tmp_path / "runs" / "detect_infer"
    runs_dir.mkdir(parents=True)
    link = runs_dir / "linked"
    try:
        link.symlink_to(victim, target_is_directory=True)
    except OSError:
        pytest.skip("当前环境不允许创建符号链接")

    plan = plan_cleanup(tmp_path / "runs",
                        RetentionPolicy(keep_latest_per_group=1, max_age_days=0), now=now)
    assert all(not c.path.is_symlink() for c in plan.candidates)
    result = apply_cleanup(plan, runs_dir=tmp_path / "runs")
    assert victim.is_dir() and (victim / "data.txt").is_file()
    assert result["removed"] == 0


# ---------------------------------------------------------------- 概览

def test_summarize_runs_marks_protected_groups(tmp_path, now):
    make_run(tmp_path, "lineage", "stamp", age_days=1.0, now=now)
    make_run(tmp_path, "detect_infer", "run", age_days=1.0, size_bytes=4096, now=now)
    summary = summarize_runs(tmp_path)

    assert summary["groups"]["lineage"]["protected"] is True
    assert summary["groups"]["detect_infer"]["protected"] is False
    assert summary["groups"]["detect_infer"]["subdirs"] == 1
    assert summary["total_bytes"] >= 4096