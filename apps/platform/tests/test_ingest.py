#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""``odp-ingest`` 数据接入的测试。

★ 这一组里最重要的是 :func:`test_write_text_fresh_does_not_touch_a_hardlinked_source`
  和 :func:`test_source_labels_are_byte_identical_after_clamping` —— 它们守的是
  "**只读源**"这条铁律: 硬链接拍平后目标与源是同一个 inode, 往目标写内容会
  **就地截断源文件**。今天真实踩过这个坑, 所以要有测试钉住。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from PIL import Image

from od_platform.data_pipeline.ingest import (
    CONFLICT_ABORT,
    CONFLICT_PREFIX,
    LAYOUT_FLAT,
    LAYOUT_IMAGES_ONLY,
    LAYOUT_LEGACY,
    LAYOUT_PRESPLIT,
    IngestError,
    IngestSettings,
    LayoutSettings,
    apply_plan,
    box_out_of_range,
    clamp_label_text,
    collect_samples,
    detect_layout,
    find_class_names,
    find_conflicts,
    format_box,
    place_file,
    resolve_name,
    scan,
)
from od_platform.data_pipeline.ingest.runner import (
    _write_text_fresh,
    verify_sources_untouched,
)

# 一个"轻微越界"的框: 中心 0.999500、宽 0.003 → 右边到 1.001, 越界 0.001
OUT_OF_RANGE_LINE = "0 0.999500 0.500000 0.003000 0.200000"
IN_RANGE_LINE = "0 0.500000 0.500000 0.200000 0.200000"


# ============================================================
# 夹具
# ============================================================

def make_image(path: Path, size=(64, 64), color="green") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def write_label(path: Path, lines) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    path.write_text(text + "\n" if text else "", encoding="utf-8")
    return path


def write_yaml(path: Path, names) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "nc: %d\nnames: [%s]\n" % (len(names), ", ".join(f"'{n}'" for n in names))
    path.write_text(body, encoding="utf-8")
    return path


def make_flat_source(root: Path, *, lines=None, names=("ball", "player")) -> Path:
    image = make_image(root / "images" / "a.jpg")
    write_label(root / "labels" / "a.txt", lines if lines is not None else [IN_RANGE_LINE])
    write_yaml(root / "data.yaml", names)
    assert image.is_file()
    return root


def make_presplit_source(root: Path, names=("ball", "goalkeeper", "player", "referee")) -> Path:
    for split, count in (("train", 2), ("valid", 1), ("test", 1)):
        for index in range(count):
            make_image(root / split / "images" / f"{split}{index}.jpg")
            write_label(root / split / "labels" / f"{split}{index}.txt", [IN_RANGE_LINE])
    write_yaml(root / "data.yaml", names)
    return root


def make_legacy_source(root: Path, names=("ball", "player")) -> Path:
    for split in ("train", "valid"):
        make_image(root / "data" / split / f"{split}_a.jpg")
        write_label(root / "data" / split / f"{split}_a.txt", [IN_RANGE_LINE])
    write_yaml(root / "data.yaml", names)
    return root


def default_settings(**overrides) -> IngestSettings:
    return IngestSettings(**overrides) if overrides else IngestSettings()


def run_ingest(source: Path, target: Path, *, classes=None, settings=None, force=False):
    settings = settings or default_settings()
    plan = scan(source, target, target.name, settings, classes=classes)
    result = apply_plan(plan, settings, force=force)
    return plan, result


# ============================================================
# 布局探测
# ============================================================

def test_detects_flat_layout(tmp_path):
    assert detect_layout(make_flat_source(tmp_path / "src"), LayoutSettings()) == LAYOUT_FLAT


def test_detects_presplit_layout(tmp_path):
    assert detect_layout(make_presplit_source(tmp_path / "src"),
                         LayoutSettings()) == LAYOUT_PRESPLIT


def test_detects_legacy_layout(tmp_path):
    assert detect_layout(make_legacy_source(tmp_path / "src"),
                         LayoutSettings()) == LAYOUT_LEGACY


def test_presplit_samples_keep_their_split_names(tmp_path):
    plan = scan(make_presplit_source(tmp_path / "src"), tmp_path / "out", "out",
                default_settings())
    assert plan.splits == {"train": 2, "valid": 1, "test": 1}


def test_images_only_source_is_rejected_with_a_usable_reason(tmp_path):
    """只有图没标签 → 明确报错, 而不是静默产出一个空数据集。"""
    source = tmp_path / "src"
    make_image(source / "images" / "a.jpg")
    assert detect_layout(source, LayoutSettings()) == LAYOUT_IMAGES_ONLY
    with pytest.raises(IngestError, match="只有图片、没有标签"):
        scan(source, tmp_path / "out", "out", default_settings())


def test_unknown_layout_is_rejected(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "readme.md").write_text("啥也没有", encoding="utf-8")
    with pytest.raises(IngestError, match="认不出源目录的布局"):
        scan(source, tmp_path / "out", "out", default_settings())


def test_missing_source_is_rejected(tmp_path):
    with pytest.raises(IngestError, match="源目录不存在"):
        scan(tmp_path / "nope", tmp_path / "out", "out", default_settings())


# ============================================================
# 类名
# ============================================================

def test_class_names_come_from_data_yaml_in_order(tmp_path):
    source = make_presplit_source(tmp_path / "src",
                                  names=("ball", "goalkeeper", "player", "referee"))
    plan = scan(source, tmp_path / "out", "out", default_settings())
    assert plan.class_names == ["ball", "goalkeeper", "player", "referee"]
    assert "data.yaml" in plan.class_source


def test_class_names_from_dict_form_are_sorted_by_id(tmp_path):
    source = make_flat_source(tmp_path / "src", names=("ball", "player"))
    (source / "data.yaml").write_text("nc: 2\nnames:\n  1: player\n  0: ball\n",
                                      encoding="utf-8")
    names, _source = find_class_names(source, LayoutSettings())
    assert names == ["ball", "player"]


def test_missing_class_names_are_reported_not_invented(tmp_path):
    """★ 找不到类名时如实报告, 绝不编一个 —— 编出来的类名会让整份数据集静默错位。"""
    source = make_flat_source(tmp_path / "src")
    (source / "data.yaml").unlink()
    names, source_note = find_class_names(source, LayoutSettings())
    assert names is None
    assert "未找到类名" in source_note


def test_cli_classes_override_the_file(tmp_path):
    source = make_flat_source(tmp_path / "src", names=("ball", "player"))
    plan = scan(source, tmp_path / "out", "out", default_settings(), classes=["x", "y"])
    assert plan.class_names == ["x", "y"]
    assert "--classes" in plan.class_source


def test_class_id_outside_the_class_list_is_rejected(tmp_path):
    """★ 标签的 class_id 超出类别表 → 直接报错（静默放过会让整份数据集类别错位）。"""
    source = make_flat_source(tmp_path / "src", lines=["3 0.5 0.5 0.2 0.2"])
    with pytest.raises(IngestError, match="超出类别表"):
        scan(source, tmp_path / "out", "out", default_settings(), classes=["ball", "player"])


# ============================================================
# 重名冲突
# ============================================================

def make_conflicting_source(root: Path) -> Path:
    """train 与 test 里各有一个同名文件 —— 拍平后会撞车。"""
    for split in ("train", "test"):
        make_image(root / split / "images" / "same.jpg")
        write_label(root / split / "labels" / "same.txt", [IN_RANGE_LINE])
    write_yaml(root / "data.yaml", ("ball", "player"))
    return root


def test_duplicate_names_abort_by_default(tmp_path):
    source = make_conflicting_source(tmp_path / "src")
    with pytest.raises(IngestError, match="重名冲突"):
        scan(source, tmp_path / "out", "out", default_settings())


def test_conflicts_are_detected_before_any_writing(tmp_path):
    source = make_conflicting_source(tmp_path / "src")
    samples = collect_samples(source, LayoutSettings(), LAYOUT_PRESPLIT)
    conflicts = find_conflicts(samples)
    assert [c.name for c in conflicts] == ["same.jpg", "same.txt"]


def test_prefix_policy_disambiguates_and_keeps_pairs(tmp_path):
    """★ prefix 策略下: 重名的图片与标签都改名, 而且**配对关系不能断**。"""
    source = make_conflicting_source(tmp_path / "src")
    settings = default_settings(conflict_policy=CONFLICT_PREFIX)
    target = tmp_path / "out"
    _plan, result = run_ingest(source, target, settings=settings)
    images = sorted(p.name for p in (target / "images").iterdir())
    labels = sorted(p.name for p in (target / "labels").iterdir())
    assert len(images) == 2 and len(labels) == 2
    assert len(set(images)) == 2, "改名后仍然撞车就失去了 prefix 的意义"
    # 每个标签都要能找到同名的图片
    assert {Path(n).stem for n in labels} == {Path(n).stem for n in images}
    assert result.labels_linked + result.labels_written == 2


def test_resolve_name_only_renames_actual_conflicts(tmp_path):
    settings = default_settings(conflict_policy=CONFLICT_PREFIX)
    assert resolve_name("ok.jpg", tmp_path, ["same.jpg"], settings) == "ok.jpg"
    renamed = resolve_name("same.jpg", tmp_path / "train", ["same.jpg"], settings)
    assert renamed != "same.jpg" and renamed.endswith(".jpg")
    # 同一个来源目录 → 同样的名字（可复现）
    assert renamed == resolve_name("same.jpg", tmp_path / "train", ["same.jpg"], settings)


# ============================================================
# 越界框裁剪
# ============================================================

def test_box_out_of_range_uses_edges_not_just_centre():
    """★ 判据是四条边: 中心在画面内、但框很宽的, 一样是越界的。"""
    assert box_out_of_range(0.5, 0.5, 0.2, 0.2, tolerance=1e-6) is False
    # 中心 0.9 在 [0,1] 内, 但宽 0.9 → 右边到 1.35, 明显越界
    assert box_out_of_range(0.9, 0.5, 0.9, 0.2, tolerance=1e-6) is True
    assert box_out_of_range(0.9995, 0.5, 0.003, 0.2, tolerance=1e-6) is True


def test_format_box_clamps_and_is_optional():
    settings = IngestSettings()
    clamped = format_box(0, 0.9995, 0.5, 0.003, 0.2, settings)
    assert clamped is not None
    _cls, cx, _cy, w, _h = clamped.split()
    assert float(cx) + float(w) / 2.0 <= 1.0 + 1e-6      # 右边界收回来了
    # 已经在范围内的框: 值不该被改动
    unchanged = format_box(0, 0.5, 0.5, 0.2, 0.2, settings)
    assert unchanged == "0 0.500000 0.500000 0.200000 0.200000"


def test_degenerate_boxes_are_dropped():
    settings = IngestSettings()
    # 整框贴在右边界外 → 夹完宽度为 0 → 丢掉
    assert format_box(0, 1.5, 0.5, 0.1, 0.2, settings) is None
    text, clamped, dropped, malformed = clamp_label_text(OUT_OF_RANGE_LINE + "\n",
                                                        settings)
    assert clamped == 1 and dropped == 0 and malformed == 0
    assert text.strip().split()[0] == "0"


def test_clamped_boxes_stay_within_bounds_after_rounding():
    """★ 舍入不能再造出新的越界（D4 的容差是 1e-6）。"""
    settings = IngestSettings()
    for line in ("0 0.999500 0.500000 0.003000 0.2",     # 右侧越界
                 "0 0.000500 0.500000 0.003000 0.2",     # 左侧越界
                 "0 0.500000 0.000500 0.2 0.003000",     # 上侧越界
                 "0 0.500000 0.999500 0.2 0.003000"):    # 下侧越界
        formatted = format_box(*([0] + [float(v) for v in line.split()[1:]]), settings)
        assert formatted is not None
        _cls, cx, cy, w, h = formatted.split()
        cx, cy, w, h = float(cx), float(cy), float(w), float(h)
        assert cx - w / 2.0 >= -1e-6 and cx + w / 2.0 <= 1.0 + 1e-6
        assert cy - h / 2.0 >= -1e-6 and cy + h / 2.0 <= 1.0 + 1e-6


def test_clamp_can_be_turned_off(tmp_path):
    source = make_flat_source(tmp_path / "src", lines=[OUT_OF_RANGE_LINE])
    settings = default_settings(clamp_boxes=False)
    target = tmp_path / "out"
    _plan, result = run_ingest(source, target, settings=settings)
    assert result.clamped_boxes == 0
    # 关掉裁剪时应当**硬链接**原文, 一个字节都不改
    assert (target / "labels" / "a.txt").read_text(encoding="utf-8").strip() \
        == OUT_OF_RANGE_LINE
    assert result.labels_linked == 1


def test_scan_counts_out_of_range_boxes(tmp_path):
    source = make_flat_source(tmp_path / "src",
                              lines=[OUT_OF_RANGE_LINE, IN_RANGE_LINE])
    plan = scan(source, tmp_path / "out", "out", default_settings())
    assert plan.label_stats.out_of_range_boxes == 1
    assert plan.label_stats.total_boxes == 2


# ============================================================
# 标签体检
# ============================================================

def test_empty_label_files_are_counted_as_negatives(tmp_path):
    source = make_flat_source(tmp_path / "src", lines=[])
    plan = scan(source, tmp_path / "out", "out", default_settings())
    assert plan.label_stats.empty_label_files == 1
    assert plan.label_stats.total_boxes == 0


def test_malformed_lines_are_counted_and_left_alone(tmp_path):
    """坏行要能被统计出来, 而且**原样保留** —— 修标签不是本命令的职责。"""
    source = make_flat_source(tmp_path / "src", lines=["0 0.5 0.5", "这不是标签"])
    plan = scan(source, tmp_path / "out", "out", default_settings())
    assert plan.label_stats.malformed_lines == 2
    target = tmp_path / "out"
    _plan, _result = run_ingest(source, target)
    kept = (target / "labels" / "a.txt").read_text(encoding="utf-8")
    assert "这不是标签" in kept


def test_images_without_labels_are_counted(tmp_path):
    source = make_flat_source(tmp_path / "src")
    make_image(source / "images" / "orphan.jpg")          # 有图无标签
    plan = scan(source, tmp_path / "out", "out", default_settings())
    assert plan.label_stats.images_without_label == 1


def test_per_class_box_and_image_counts(tmp_path):
    source = make_flat_source(tmp_path / "src",
                              lines=["0 0.2 0.2 0.1 0.1", "0 0.3 0.3 0.1 0.1",
                                     "1 0.6 0.6 0.2 0.2"])
    plan = scan(source, tmp_path / "out", "out", default_settings())
    assert plan.label_stats.boxes_per_class == {0: 2, 1: 1}
    assert plan.label_stats.images_per_class == {0: 1, 1: 1}


# ============================================================
# 图片体检
# ============================================================

def test_probe_detects_undecodable_image(tmp_path):
    source = make_flat_source(tmp_path / "src")
    broken = source / "images" / "broken.jpg"
    broken.write_bytes(b"\xff\xd8\xff\xe0" + b"garbage" * 20)     # 截断的 JPEG
    plan = scan(source, tmp_path / "out", "out", default_settings())
    assert plan.image_health.undecodable == 1
    assert "broken.jpg" in plan.image_health.undecodable_preview[0]


def test_probe_records_resolution_distribution(tmp_path):
    source = make_flat_source(tmp_path / "src")            # 默认 64x64
    make_image(source / "images" / "bigger.jpg", size=(100, 100))
    plan = scan(source, tmp_path / "out", "out", default_settings())
    assert plan.image_health.probed == 2
    # 64 与 100 都小于 320 → 两张都算"过小"
    assert plan.image_health.small == 2
    assert plan.image_health.min_sides == [64, 100]
    assert plan.image_health.median_min_side == 100
    assert plan.image_health.to_dict(320)["min_min_side_px"] == 64


def test_large_images_are_not_counted_as_small(tmp_path):
    source = make_flat_source(tmp_path / "src")
    make_image(source / "images" / "hd.jpg", size=(1280, 720))
    plan = scan(source, tmp_path / "out", "out", default_settings())
    assert plan.image_health.small == 1                   # 只有默认那张 64px 的小图
    assert plan.image_health.median_min_side == 720


def test_probe_can_be_skipped(tmp_path):
    source = make_flat_source(tmp_path / "src")
    plan = scan(source, tmp_path / "out", "out",
                default_settings(probe_images=False))
    assert plan.image_health.probed == 0
    assert plan.image_health.min_sides == []


def test_probe_works_on_non_ascii_paths(tmp_path):
    """★ 中文路径: OpenCV 的 imread 会静默返回 None, PIL 没有这个问题。

    所以探针用 PIL —— 否则"路径有中文"会被误报成"图片损坏"。
    """
    source = tmp_path / "足球数据集"
    make_flat_source(source, names=("ball", "player"))
    plan = scan(source, tmp_path / "out", "out", default_settings())
    assert plan.image_health.probed == 1
    assert plan.image_health.undecodable == 0


# ============================================================
# 落地与安全（本组最重要的部分）
# ============================================================

def test_dry_run_writes_nothing(tmp_path):
    """★ 预演就是预演: 一个字节都不写。"""
    source = make_flat_source(tmp_path / "src")
    target = tmp_path / "out"
    scan(source, target, "out", default_settings())          # 只扫描
    assert not target.exists()


def test_apply_creates_the_flat_target_layout(tmp_path):
    source = make_presplit_source(tmp_path / "src")
    target = tmp_path / "out"
    _plan, result = run_ingest(source, target)
    assert (target / "images").is_dir() and (target / "labels").is_dir()
    assert len(list((target / "images").iterdir())) == 4
    assert len(list((target / "labels").iterdir())) == 4
    assert result.total_placed > 0


def test_target_non_empty_is_refused_without_force(tmp_path):
    source = make_flat_source(tmp_path / "src")
    target = tmp_path / "out"
    target.mkdir()
    (target / "existing.txt").write_text("已存在", encoding="utf-8")
    plan = scan(source, target, "out", default_settings())
    with pytest.raises(FileExistsError, match="已存在且非空"):
        apply_plan(plan, default_settings(), force=False)


def test_force_allows_writing_into_a_non_empty_target(tmp_path):
    source = make_flat_source(tmp_path / "src")
    target = tmp_path / "out"
    target.mkdir()
    (target / "existing.txt").write_text("已存在", encoding="utf-8")
    plan = scan(source, target, "out", default_settings())
    result = apply_plan(plan, default_settings(), force=True)
    assert result.total_placed > 0
    assert (target / "existing.txt").is_file(), "force 不该删除既有文件"


def test_hardlink_is_preferred(tmp_path):
    source = make_flat_source(tmp_path / "src")
    target = tmp_path / "out"
    _plan, result = run_ingest(source, target)
    assert result.linked > 0
    assert result.copied == 0
    # 硬链接: 与源同一个 inode
    source_inode = (source / "images" / "a.jpg").stat().st_ino
    target_inode = (target / "images" / "a.jpg").stat().st_ino
    assert source_inode == target_inode


def test_falls_back_to_copy_when_hardlink_fails(tmp_path, monkeypatch):
    source = make_flat_source(tmp_path / "src")
    target = tmp_path / "out"

    def refuse(*args, **kwargs):
        raise OSError("跨卷 / 不支持硬链接（模拟）")

    monkeypatch.setattr(os, "link", refuse)
    _plan, result = run_ingest(source, target)
    assert result.linked == 0
    assert result.copied > 0
    # 复制出来的是独立 inode
    assert (source / "images" / "a.jpg").stat().st_ino != \
        (target / "images" / "a.jpg").stat().st_ino
    assert (target / "images" / "a.jpg").is_file()


def test_place_file_replaces_an_existing_target(tmp_path):
    source_file = tmp_path / "src.txt"
    source_file.write_text("新内容", encoding="utf-8")
    target_file = tmp_path / "dst.txt"
    target_file.write_text("旧内容", encoding="utf-8")
    place_file(source_file, target_file, default_settings())
    assert target_file.read_text(encoding="utf-8") == "新内容"


def test_write_text_fresh_does_not_touch_a_hardlinked_source(tmp_path):
    """★★ 本组最重要的一条: 硬链接目标的存在不会让源文件被改写。

    这是 :func:`_write_text_fresh` 存在的全部理由 —— 它必须先 unlink 再写,
    否则会就地截断与源共享的那个 inode。
    """
    source = tmp_path / "src.txt"
    source.write_text("原始内容", encoding="utf-8")
    target = tmp_path / "dst.txt"
    os.link(source, target)
    assert target.stat().st_ino == source.stat().st_ino      # 确实是同一个 inode

    _write_text_fresh(target, "新内容")

    assert source.read_text(encoding="utf-8") == "原始内容", "★ 源文件被改写了"
    assert target.read_text(encoding="utf-8") == "新内容"
    assert target.stat().st_ino != source.stat().st_ino      # 目标换成了新 inode


def test_naive_write_would_have_corrupted_the_source(tmp_path):
    """反面证明: **直接** write_text 会改写源 —— 这就是为什么必须先 unlink。

    这条测试故意制造破坏, 用来钉住"硬链接目标不能就地写"这个前提本身;
    哪天平台/文件系统的语义变了, 它会先失败, 提醒我们规则需要重新审视。
    """
    source = tmp_path / "src.txt"
    source.write_text("原始内容", encoding="utf-8")
    target = tmp_path / "dst.txt"
    os.link(source, target)

    target.write_text("新内容", encoding="utf-8")            # 故意的"错误做法"

    assert source.read_text(encoding="utf-8") == "新内容"     # 源被毁了


def test_source_labels_are_byte_identical_after_clamping(tmp_path):
    """★★ 端到端守"只读源": 裁剪越界框会写新标签, 但**源标签必须一个字节都不变**。"""
    source = make_flat_source(tmp_path / "src", lines=[OUT_OF_RANGE_LINE])
    source_label = source / "labels" / "a.txt"
    before = source_label.read_bytes()
    inode_before = source_label.stat().st_ino

    target = tmp_path / "out"
    _plan, result = run_ingest(source, target)

    assert result.clamped_boxes == 1
    assert result.labels_written == 1
    assert source_label.read_bytes() == before, "★ 源标签被改写了"
    assert source_label.stat().st_ino == inode_before
    # 目标那边确实被裁过了
    kept = (target / "labels" / "a.txt").read_text(encoding="utf-8").strip()
    assert kept != OUT_OF_RANGE_LINE
    _cls, cx, _cy, w, _h = kept.split()
    assert float(cx) + float(w) / 2.0 <= 1.0 + 1e-6


def test_source_guard_reports_zero_changes_even_when_clamping(tmp_path):
    """源指纹复查要作为**可验证的结论**写进结果, 不是一句设计承诺。"""
    source = make_flat_source(tmp_path / "src", lines=[OUT_OF_RANGE_LINE])
    target = tmp_path / "out"
    _plan, result = run_ingest(source, target)
    assert result.source_guard["checked"] == 1
    assert result.source_guard["changed"] == 0
    assert result.source_guard["algorithm"] == "sha1"


def test_source_guard_detects_a_changed_source(tmp_path):
    """指纹复查本身要有效: 源真被改了就必须报出来。"""
    source = make_flat_source(tmp_path / "src")
    label = source / "labels" / "a.txt"
    guard = verify_sources_untouched({str(label): "deadbeef"})
    assert guard["changed"] == 1
    guard_ok = verify_sources_untouched({})
    assert guard_ok["checked"] == 0 and guard_ok["changed"] == 0


def test_source_images_are_never_modified(tmp_path):
    """图片只被硬链接, 从不写入 —— 内容与 inode 都必须保持不变。"""
    source = make_flat_source(tmp_path / "src")
    image = source / "images" / "a.jpg"
    before = image.read_bytes()
    inode = image.stat().st_ino
    target = tmp_path / "out"
    run_ingest(source, target)
    assert image.read_bytes() == before
    assert image.stat().st_ino == inode


# ============================================================
# 报告与配置
# ============================================================

def test_report_is_written_with_the_health_summary(tmp_path):
    source = make_flat_source(tmp_path / "src", lines=[OUT_OF_RANGE_LINE,
                                                       IN_RANGE_LINE])
    target = tmp_path / "out"
    _plan, result = run_ingest(source, target)

    assert result.report_path is not None and result.report_path.is_file()
    payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert payload["dataset_name"] == "out"
    assert payload["layout"] == LAYOUT_FLAT
    assert payload["class_names"] == ["ball", "player"]
    assert payload["labels"]["out_of_range_boxes"] == 1
    assert payload["labels"]["total_boxes"] == 2
    assert payload["result"]["clamped_boxes"] == 1
    assert payload["result"]["source_guard"]["changed"] == 0
    assert payload["images"]["probed"] == 1
    assert payload["settings"]["clamp_boxes"] is True


def test_report_filename_comes_from_settings(tmp_path):
    source = make_flat_source(tmp_path / "src")
    settings = IngestSettings(report_filename="custom_report.json")
    target = tmp_path / "out"
    _plan, result = run_ingest(source, target, settings=settings)
    assert result.report_path is not None
    assert result.report_path.name == "custom_report.json"


# ============================================================
# 设置层
# ============================================================

def test_settings_reject_bad_values():
    with pytest.raises(ValueError, match="conflict_policy"):
        IngestSettings(conflict_policy="whatever")
    with pytest.raises(ValueError, match="coordinate_precision"):
        IngestSettings(coordinate_precision=0)
    with pytest.raises(ValueError, match="box_edge_tolerance"):
        IngestSettings(box_edge_tolerance=-1)
    with pytest.raises(ValueError, match="small_image_side_px"):
        IngestSettings(small_image_side_px=0)
    with pytest.raises(ValueError, match="report_filename"):
        IngestSettings(report_filename="")
    with pytest.raises(ValueError, match="split_dirnames"):
        LayoutSettings(split_dirnames=())
    with pytest.raises(ValueError, match="image_suffixes"):
        LayoutSettings(image_suffixes=())
    with pytest.raises(ValueError, match="label_suffix"):
        LayoutSettings(label_suffix="txt")


def test_yaml_overrides_both_sections(tmp_path):
    config = {
        "layouts": {"images_dirname": "imgs", "split_dirnames": ["train", "test"]},
        "ingest": {"clamp_boxes": False, "prefer_hardlink": False, "preview_limit": 3},
    }
    settings = IngestSettings.from_config(config)
    assert settings.layouts.images_dirname == "imgs"
    assert settings.layouts.split_dirnames == ("train", "test")
    assert settings.clamp_boxes is False
    assert settings.prefer_hardlink is False
    assert settings.preview_limit == 3


def test_custom_layout_names_are_actually_used(tmp_path):
    """★ 换一种目录命名（新导出工具）只改配置就能接 —— 这正是不写死的意义。"""
    source = tmp_path / "src"
    make_image(source / "imgs" / "a.jpg")
    write_label(source / "anns" / "a.txt", [IN_RANGE_LINE])
    settings = IngestSettings.from_config(
        {"layouts": {"images_dirname": "imgs", "labels_dirname": "anns"}})
    plan = scan(source, tmp_path / "out", "out", settings)
    assert plan.layout == LAYOUT_FLAT
    assert len(plan.samples) == 1


def test_repo_ingest_yaml_matches_the_code_defaults():
    """★ 契约测试: 仓库里那份 ingest.yaml 的取值必须等于 dataclass 默认值。

    两份漂移的话, "改 yaml 就能调"这句话就不成立了 —— 而且很难发现。
    """
    settings = IngestSettings.load()
    assert settings.conflict_policy == CONFLICT_ABORT
    assert settings.clamp_boxes is True
    assert settings.probe_images is True
    assert settings.prefer_hardlink is True
    assert settings.coordinate_precision == 6
    assert settings.box_edge_tolerance == 1e-6
    assert settings.small_image_side_px == 320
    assert settings.layouts.images_dirname == "images"
    assert settings.layouts.labels_dirname == "labels"
    assert settings.layouts.split_dirnames == ("train", "valid", "val", "test")


def test_audit_snapshot_lists_the_effective_values():
    audit = IngestSettings().to_audit()
    assert audit["clamp_boxes"] is True
    assert audit["layouts"]["images_dirname"] == "images"
    assert audit["conflict_policy"] == CONFLICT_ABORT


# ============================================================
# 计划摘要
# ============================================================

def test_plan_render_lines_mention_the_key_numbers(tmp_path):
    source = make_presplit_source(tmp_path / "src")
    plan = scan(source, tmp_path / "out", "out", default_settings())
    text = "\n".join(plan.render_lines())
    assert LAYOUT_PRESPLIT in text
    assert "train=2" in text             # 来源集合分布
    # 夹具里的框 class_id=0, 而默认类名表是 (ball, goalkeeper, player, referee) → ball
    assert "ball=4" in text
    assert "每类图片" in text
    assert str(tmp_path / "out") in text


def test_plan_class_name_lookup_falls_back_to_id(tmp_path):
    source = make_flat_source(tmp_path / "src")
    plan = scan(source, tmp_path / "out", "out", default_settings(),
                classes=["only_one"])
    assert plan.class_name_of(0) == "only_one"
    assert plan.class_name_of(7) == "<class_id=7>"