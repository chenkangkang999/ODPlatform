#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""质检报告历史对比（D4 1.2 第三条）的测试。

守三件事:
    · "新出现的问题"与"已修复的问题"分得清（这是这一项存在的全部意义）;
    · INFO 不算问题（与 registry.passed 同口径, 否则每份不均衡数据集都常年飘红）;
    · 历史定位不能认错数据集、也不能把自己当成上一次。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from od_platform.data_validation.compare import (
    REPORT_FILENAME,
    compare_reports,
    compare_with_history,
    find_previous_report,
    is_problem,
    severity_rank,
)
from od_platform.data_validation.registry import CheckSeverity


def make_report(run_id: str, yaml_path: Path, overall: str, results) -> dict:
    return {
        "run_id": run_id,
        "yaml_path": str(yaml_path),
        "overall_severity": overall,
        "results": [{"name": name, "severity": sev, "summary": f"{name} 结论", "details": {}}
                    for name, sev in results],
    }


def write_report(runs_dir: Path, run_id: str, payload: dict) -> Path:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / REPORT_FILENAME
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------- 口径

def test_info_is_not_treated_as_a_problem():
    assert is_problem(CheckSeverity.WARNING) and is_problem(CheckSeverity.ERROR)
    assert not is_problem(CheckSeverity.INFO)
    assert not is_problem(CheckSeverity.PASS)


def test_unknown_severity_ranks_lowest_without_exploding():
    assert severity_rank(None) == severity_rank(CheckSeverity.PASS)
    assert severity_rank("NOT_A_SEVERITY") == severity_rank(CheckSeverity.PASS)
    assert severity_rank(CheckSeverity.ERROR) > severity_rank(CheckSeverity.WARNING)


# ---------------------------------------------------------------- 对比

def test_compare_classifies_fixed_newly_failing_and_still_failing(tmp_path):
    yaml_path = tmp_path / "data.yaml"
    previous = make_report("20260101-000001", yaml_path, CheckSeverity.WARNING, [
        ("bbox_within_image", CheckSeverity.WARNING),     # → 后来修好了
        ("pair_existence", CheckSeverity.ERROR),          # → 一直没修
        ("class_balance", CheckSeverity.PASS),            # → 后来变差了
    ])
    current = make_report("20260102-000002", yaml_path, CheckSeverity.ERROR, [
        ("bbox_within_image", CheckSeverity.PASS),
        ("pair_existence", CheckSeverity.ERROR),
        ("class_balance", CheckSeverity.WARNING),
    ])
    result = compare_reports(previous, current)

    assert [c.name for c in result.fixed] == ["bbox_within_image"]
    assert [c.name for c in result.newly_failing] == ["class_balance"]
    assert [c.name for c in result.still_failing] == ["pair_existence"]
    assert result.previous_overall == CheckSeverity.WARNING
    assert result.current_overall == CheckSeverity.ERROR
    assert result.improved is False


def test_compare_reports_improvement_when_severity_drops(tmp_path):
    yaml_path = tmp_path / "data.yaml"
    previous = make_report("20260101-000001", yaml_path, CheckSeverity.WARNING,
                           [("bbox_within_image", CheckSeverity.WARNING)])
    current = make_report("20260102-000002", yaml_path, CheckSeverity.INFO,
                          [("bbox_within_image", CheckSeverity.PASS)])
    result = compare_reports(previous, current)
    assert result.improved is True
    assert [c.name for c in result.fixed] == ["bbox_within_image"]


def test_added_check_that_warns_counts_as_a_newly_found_problem(tmp_path):
    """新加进来的检查项一上来就报警 —— 那正是"新发现的问题"。"""
    yaml_path = tmp_path / "data.yaml"
    previous = make_report("20260101-000001", yaml_path, CheckSeverity.PASS, [])
    current = make_report("20260102-000002", yaml_path, CheckSeverity.WARNING,
                          [("image_decodable", CheckSeverity.WARNING),
                           ("yaml_schema", CheckSeverity.PASS)])
    result = compare_reports(previous, current)
    assert result.added_checks == ["image_decodable", "yaml_schema"]
    assert [c.name for c in result.newly_failing] == ["image_decodable"]


def test_removed_checks_are_reported(tmp_path):
    yaml_path = tmp_path / "data.yaml"
    previous = make_report("20260101-000001", yaml_path, CheckSeverity.PASS,
                           [("legacy_check", CheckSeverity.PASS)])
    current = make_report("20260102-000002", yaml_path, CheckSeverity.PASS, [])
    result = compare_reports(previous, current)
    assert result.removed_checks == ["legacy_check"]
    assert result.fixed == [] and result.newly_failing == []


def test_render_lines_mention_the_important_parts(tmp_path):
    yaml_path = tmp_path / "data.yaml"
    previous = make_report("20260101-000001", yaml_path, CheckSeverity.WARNING,
                           [("bbox_within_image", CheckSeverity.WARNING)])
    current = make_report("20260102-000002", yaml_path, CheckSeverity.INFO,
                          [("bbox_within_image", CheckSeverity.PASS)])
    text = "\n".join(compare_reports(previous, current).render_lines())
    assert "20260101-000001" in text and "20260102-000002" in text
    assert "已修复" in text and "bbox_within_image" in text


# ---------------------------------------------------------------- 历史定位

def test_find_previous_report_picks_the_newest_for_the_same_yaml(tmp_path):
    runs_dir = tmp_path / "runs"
    yaml_path = tmp_path / "data.yaml"
    write_report(runs_dir, "20260101-000001", make_report("20260101-000001", yaml_path, "PASS", []))
    newest = write_report(runs_dir, "20260103-000003",
                          make_report("20260103-000003", yaml_path, "PASS", []))

    found = find_previous_report(yaml_path, runs_dir=runs_dir)
    assert found == newest


def test_find_previous_report_never_returns_the_current_run(tmp_path):
    runs_dir = tmp_path / "runs"
    yaml_path = tmp_path / "data.yaml"
    write_report(runs_dir, "20260101-000001", make_report("20260101-000001", yaml_path, "PASS", []))
    write_report(runs_dir, "20260102-000002", make_report("20260102-000002", yaml_path, "PASS", []))

    found = find_previous_report(yaml_path, exclude_run_id="20260102-000002", runs_dir=runs_dir)
    assert found is not None and found.parent.name == "20260101-000001"


def test_find_previous_report_ignores_other_datasets(tmp_path):
    runs_dir = tmp_path / "runs"
    write_report(runs_dir, "20260101-000001",
                 make_report("20260101-000001", tmp_path / "other.yaml", "PASS", []))
    assert find_previous_report(tmp_path / "data.yaml", runs_dir=runs_dir) is None


def test_find_previous_report_skips_broken_files(tmp_path):
    runs_dir = tmp_path / "runs"
    yaml_path = tmp_path / "data.yaml"
    broken = runs_dir / "20260102-000002"
    broken.mkdir(parents=True)
    (broken / REPORT_FILENAME).write_text("{ 这不是 json", encoding="utf-8")
    good = write_report(runs_dir, "20260101-000001",
                        make_report("20260101-000001", yaml_path, "PASS", []))

    assert find_previous_report(yaml_path, runs_dir=runs_dir) == good


def test_no_history_directory_returns_none(tmp_path):
    assert find_previous_report(tmp_path / "data.yaml", runs_dir=tmp_path / "missing") is None


# ---------------------------------------------------------------- 端到端

def test_compare_with_history_end_to_end(tmp_path):
    runs_dir = tmp_path / "runs"
    yaml_path = tmp_path / "data.yaml"
    write_report(runs_dir, "20260101-000001", make_report(
        "20260101-000001", yaml_path, CheckSeverity.WARNING,
        [("bbox_within_image", CheckSeverity.WARNING)]))
    current = make_report("20260102-000002", yaml_path, CheckSeverity.INFO,
                          [("bbox_within_image", CheckSeverity.PASS)])

    comparison = compare_with_history(yaml_path, current, runs_dir=runs_dir)
    assert comparison is not None
    assert comparison.previous_run_id == "20260101-000001"
    assert [c.name for c in comparison.fixed] == ["bbox_within_image"]


def test_compare_with_history_returns_none_on_first_run(tmp_path):
    runs_dir = tmp_path / "runs"
    yaml_path = tmp_path / "data.yaml"
    current = make_report("20260102-000002", yaml_path, CheckSeverity.PASS, [])
    assert compare_with_history(yaml_path, current, runs_dir=runs_dir) is None


def test_comparison_serialises_for_the_report_json(tmp_path):
    yaml_path = tmp_path / "data.yaml"
    previous = make_report("20260101-000001", yaml_path, "WARNING",
                           [("bbox_within_image", "WARNING")])
    current = make_report("20260102-000002", yaml_path, "PASS",
                          [("bbox_within_image", "PASS")])
    payload = compare_reports(previous, current).to_dict()
    assert payload["improved"] is True
    assert payload["fixed"][0]["name"] == "bbox_within_image"
    json.dumps(payload)                       # 必须可序列化（要写进 report.json）