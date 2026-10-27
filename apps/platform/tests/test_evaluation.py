#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""`odp-eval` 评估子系统的测试。

全部**离线**跑: 不下载任何东西、不碰 GPU、不需要真权重 ——
链条上的每一段都留了可注入的缝（假模型 / 假指标 / 合成数据集）。

分五块:
    1. 设置层           —— 默认值、非法值、yaml 覆盖、示例配置与代码默认值一致
    2. 数据集视图       —— 路径解析（含 Roboflow 的 ../ 坑）、真值计数、不写原始数据
    3. 指标整理（纯函数）—— 每类一行、多套平均口径、置信度扫描、推荐阈值、如实警告
    4. 报告三种出口     —— json / csv / logger
    5. 服务与 CLI       —— 端到端（假模型）、失败路径与退出码
"""
from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path

import numpy as np
import pytest

from od_platform.cli import evaluate_model as cli_mod
from od_platform.evaluation import metrics as M
from od_platform.evaluation.dataset import (
    DatasetView,
    build_dataset_view,
    class_names_from,
    count_instances,
    load_dataset_yaml,
    reconcile_class_count,
    resolve_split_dir,
)
from od_platform.evaluation.report import EvaluationReport
from od_platform.evaluation.service import evaluate_model, extract_result
from od_platform.evaluation.settings import EvalSettings, default_config_path

# ============================================================
# 合成数据集
# ============================================================

YAML_LIST_FORM = """\
path: .
train: images
val: images
nc: 3
names: ['ball', 'player', 'referee']
"""

YAML_ROBOFLOW_FORM = """\
train: ../train/images
val: ../valid/images
nc: 3
names: ['ball', 'player', 'referee']
"""

YAML_WITH_TEST_KEY = """\
path: .
train: images
val: images
test: images
nc: 3
names: ['ball', 'player', 'referee']
"""


def make_dataset(root: Path, *, images: int = 3, yaml_text: str = YAML_LIST_FORM,
                 layout: str = "flat") -> Path:
    """造一个最小可评估数据集。

    Args:
        layout: "flat"   —— root/{images,labels}（本仓库生成的形态）
                "roboflow" —— root/{train,valid}/{images,labels}（Roboflow 导出形态）
    """
    root.mkdir(parents=True, exist_ok=True)
    if layout == "flat":
        splits = {"": root}
    else:
        splits = {"train": root / "train", "valid": root / "valid"}
    for sub in splits.values():
        (sub / "images").mkdir(parents=True, exist_ok=True)
        (sub / "labels").mkdir(parents=True, exist_ok=True)
    for sub in splits.values():
        for index in range(images):
            (sub / "images" / f"img{index}.jpg").write_bytes(b"fake-jpeg-bytes")
            # ball 1 个 / player 2 个 / referee 0 个 —— 故意造出"某类没有真值"
            (sub / "labels" / f"img{index}.txt").write_text(
                "0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.2 0.2\n1 0.7 0.7 0.2 0.2\n",
                encoding="utf-8")
    (root / "data.yaml").write_text(yaml_text, encoding="utf-8")
    return root / "data.yaml"


# ============================================================
# 假模型 / 假指标（鸭子类型, 离线）
# ============================================================

class FakeBox:
    def __init__(self, *, ap_class_index, ap50, ap, p, r,
                 map50, map50_95, mp, mr, map75=0.0):
        self.ap_class_index = np.asarray(ap_class_index)
        self.ap50 = np.asarray(ap50, dtype=float)
        self.ap = np.asarray(ap, dtype=float)
        self.p = np.asarray(p, dtype=float)
        self.r = np.asarray(r, dtype=float)
        self.map50 = map50
        self.map = map50_95
        self.map75 = map75
        self.mp = mp
        self.mr = mr
        # (召回率, 每类精度) —— 与 ultralytics 的 curves_results 同构
        recall = np.linspace(0, 1, 1000)
        self.curves_results = [(recall,
                                np.tile(1.0 - recall, (len(ap50), 1)),
                                "Recall", "Precision")]


class FakeResult:
    def __init__(self, box, speed=None):
        self.box = box
        self.speed = speed if speed is not None else {
            "preprocess": 1.5, "inference": 60.0, "postprocess": 3.0}


class FakeModel:
    """只认 ``.val(**kwargs)`` 的假模型; 指标随 conf 变化, 好验证推荐阈值逻辑。"""

    def __init__(self):
        self.calls = []

    def val(self, **kwargs):
        conf = float(kwargs.get("conf", 0.001))
        self.calls.append(kwargs)
        precision = min(0.99, 0.5 + conf)
        recall = max(0.0, 0.9 - conf)
        # class 1(ball) 的 AP 故意压得很低 —— 用来验证"排除小类"的口径
        box = FakeBox(ap_class_index=[0, 1], ap50=[0.20, 0.90],
                      ap=[0.10, 0.70], p=[precision, precision],
                      r=[recall, recall], map50=0.55, map50_95=0.40,
                      mp=precision, mr=recall)
        return FakeResult(box)


def fake_view(n_images: int = 3, instances=(1, 2, 0), copied: int = 0,
              limited_to: int = 0) -> DatasetView:
    return DatasetView(
        root=Path("view"), yaml_path=Path("view/data.yaml"),
        images_dir=Path("view/images"), labels_dir=Path("view/labels"),
        split="val", class_names=("ball", "player", "referee"),
        n_images=n_images, instances_per_class=tuple(instances),
        source_yaml=Path("src/data.yaml"), source_split_dir=Path("src/images"),
        copied_files=copied, limited_to=limited_to)


# ============================================================
# 1. 设置层
# ============================================================

def test_defaults_are_valid_and_auditable():
    settings = EvalSettings()
    audit = settings.to_audit()
    assert settings.split == "val"
    assert settings.device == "cpu"                 # ★ 默认不去抢 GPU
    assert settings.conf < 0.01                     # ★ val 用极低阈值
    assert audit["conf"] == settings.conf
    assert audit["conf_thresholds"] == list(settings.conf_thresholds)


def test_example_yaml_matches_code_defaults():
    """★ 契约测试: 示例配置与 dataclass 默认值必须一致, 防止两边漂移。"""
    path = default_config_path()
    assert path.is_file(), f"示例配置不在: {path}"
    assert EvalSettings.load(path) == EvalSettings()


@pytest.mark.parametrize("kwargs, match", [
    ({"split": "validation"}, "split"),
    ({"imgsz": 100}, "imgsz"),
    ({"imgsz": 0}, "imgsz"),
    ({"batch": 0}, "batch"),
    ({"workers": -1}, "workers"),
    ({"max_det": 0}, "max_det"),
    ({"conf": 0}, "conf"),
    ({"conf": 1.5}, "conf"),
    ({"iou": -0.1}, "iou"),
    ({"target_precision": 0}, "target_precision"),
    ({"conf_thresholds": ()}, "conf_thresholds"),
    ({"conf_thresholds": (0.5, 0.5)}, "conf_thresholds"),
    ({"conf_thresholds": (0.5, 0.2)}, "conf_thresholds"),
    ({"conf_thresholds": (0.0, 0.5)}, "conf_thresholds"),
    ({"min_images_for_reliable_map": 0}, "min_images"),
    ({"min_instances_for_reliable_ap": 0}, "min_instances"),
    ({"imbalance_ratio_warn": 1.0}, "imbalance_ratio_warn"),
    ({"decimals": -1}, "decimals"),
    ({"pr_curve_points": 1}, "pr_curve_points"),
    ({"limit": -1}, "limit"),
    ({"task_label": " "}, "task_label"),
])
def test_settings_reject_bad_values(kwargs, match):
    with pytest.raises(ValueError, match=match):
        EvalSettings(**kwargs)


def test_settings_from_config_overrides_and_aliases():
    settings = EvalSettings.from_config({
        "evaluation": {"size": 320, "device": "0", "thresholds": (0.3, 0.6),
                       "min_images": 7}})
    assert settings.imgsz == 320                    # 别名 size → imgsz
    assert settings.device == "0"
    assert settings.conf_thresholds == (0.3, 0.6)
    assert settings.min_images_for_reliable_map == 7


def test_unknown_config_keys_are_reported(caplog):
    with caplog.at_level(logging.WARNING):
        EvalSettings.from_config({"evaluation": {"nope": 1}})
    assert any("无法识别的配置键" in record.message for record in caplog.records)


def test_settings_round_keeps_none_as_none():
    assert EvalSettings(decimals=2).round(0.12345) == 0.12
    assert EvalSettings(decimals=2).round(None) is None       # None 不能变成 0
    assert EvalSettings(decimals=2).round("abc") == "abc"


def test_settings_load_missing_file_uses_defaults(tmp_path):
    assert EvalSettings.load(tmp_path / "nope.yaml") == EvalSettings()


# ============================================================
# 2. 数据集视图
# ============================================================

def test_class_names_from_both_yaml_styles():
    assert class_names_from({"names": ["a", "b"]}) == ("a", "b")
    assert class_names_from({"names": {1: "b", 0: "a"}}) == ("a", "b")   # 按 id 排序
    assert class_names_from({}) == ()


def test_reconcile_class_count_catches_self_contradiction():
    assert reconcile_class_count({"nc": 2, "names": ["a", "b"]}, ["a", "b"]) == 2
    assert reconcile_class_count({"names": ["a"]}, ["a"]) == 1
    with pytest.raises(ValueError, match="自相矛盾"):
        reconcile_class_count({"nc": 5, "names": ["a", "b"]}, ["a", "b"])
    with pytest.raises(ValueError, match="无法确定类别数"):
        reconcile_class_count({}, [])


def test_resolve_split_dir_handles_roboflow_dotdot(tmp_path):
    """★ 回归测试: Roboflow 的 ``../valid/images`` 必须能解析到 yaml 旁边的 valid/。"""
    root = make_dataset(tmp_path / "ds", layout="roboflow",
                        yaml_text=YAML_ROBOFLOW_FORM)
    data = load_dataset_yaml(root)
    resolved, notes = resolve_split_dir(data["val"], bases=[root.parent])
    assert resolved is not None
    assert resolved == (root.parent / "valid" / "images").resolve()
    assert any("解析为" in note for note in notes)


def test_resolve_split_dir_reports_every_candidate_on_failure(tmp_path):
    resolved, notes = resolve_split_dir("../nope/images", bases=[tmp_path])
    assert resolved is None
    assert any("都不存在" in note for note in notes)


def test_count_instances_counts_per_class(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.txt").write_text("0 1 1 1 1\n1 1 1 1 1\n", encoding="utf-8")
    (labels / "b.txt").write_text("1 1 1 1 1\n", encoding="utf-8")
    # 坏行与越界类号都不该让计数炸掉
    (labels / "c.txt").write_text("bad-line\n9 1 1 1 1\n\n", encoding="utf-8")
    assert count_instances(labels, 3) == (1, 2, 0)


def test_build_dataset_view_creates_normalized_yaml_and_counts(tmp_path):
    source = make_dataset(tmp_path / "ds")
    view = build_dataset_view(source, split="val", view_root=tmp_path / "view")

    assert view.n_images == 3
    assert view.instances_per_class == (3, 6, 0)      # ball 1/图, player 2/图
    assert view.total_instances == 9
    # ultralytics 强制要求 train 与 val 两个键都在
    text = view.yaml_path.read_text(encoding="utf-8")
    assert "train: images" in text and "val: images" in text
    assert "nc: 3" in text
    # ★ 视图 yaml 里必须是相对路径 —— 有盘符的话视图就搬不走了。
    # 用 os.path.splitdrive 判断, 而不是在源码里写一个盘符例子（那会被 CI 守卫拦下）。
    for line in text.splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        assert not os.path.splitdrive(value.strip())[0], \
            f"视图 yaml 的 {key} 里出现了盘符路径: {value}"
    assert view.images_dir.is_dir() and view.labels_dir.is_dir()


def test_build_dataset_view_hardlinks_instead_of_copying(tmp_path):
    """硬链接是零额外字节的关键; 同一卷上应当一个字节都不复制。"""
    source = make_dataset(tmp_path / "ds")
    view = build_dataset_view(source, split="val", view_root=tmp_path / "view")
    assert view.copied_files == 0
    # 硬链接 = 同一个 inode
    created = view.images_dir / "img0.jpg"
    assert created.stat().st_ino == (source.parent / "images" / "img0.jpg").stat().st_ino


def test_build_dataset_view_limit_marks_the_report(tmp_path):
    source = make_dataset(tmp_path / "ds", images=5)
    view = build_dataset_view(source, split="val", view_root=tmp_path / "view", limit=2)
    assert view.n_images == 2
    assert view.limited_to == 2


def test_build_dataset_view_writes_a_test_key_when_asked(tmp_path):
    source = make_dataset(tmp_path / "ds", yaml_text=YAML_WITH_TEST_KEY)
    view = build_dataset_view(source, split="test", view_root=tmp_path / "view")
    text = view.yaml_path.read_text(encoding="utf-8")
    assert "test: images" in text


def test_build_dataset_view_never_touches_the_source(tmp_path):
    """★ 原始数据只读: 视图建完后源目录必须一个字节都没变（框架会往里写 labels.cache）。"""
    source = make_dataset(tmp_path / "ds")
    before = sorted(p.name for p in source.parent.rglob("*"))
    build_dataset_view(source, split="val", view_root=tmp_path / "view")
    after = sorted(p.name for p in source.parent.rglob("*"))
    assert before == after
    assert not (source.parent / "labels.cache").exists()


def test_build_dataset_view_errors_when_split_missing(tmp_path):
    source = make_dataset(tmp_path / "ds",
                          yaml_text="train: images\nnc: 1\nnames: ['a']\n")
    with pytest.raises(FileNotFoundError, match="解析不出来"):
        build_dataset_view(source, split="val", view_root=tmp_path / "view")


def test_build_dataset_view_errors_when_yaml_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="不存在"):
        build_dataset_view(tmp_path / "nope.yaml", split="val",
                           view_root=tmp_path / "view")


def test_view_to_dict_is_serialisable():
    payload = fake_view().to_dict()
    assert payload["n_classes"] == 3
    assert payload["instances_per_class"] == {"ball": 1, "player": 2, "referee": 0}
    json.dumps(payload)


# ============================================================
# 3. 指标整理（纯函数）
# ============================================================

def test_per_class_rows_marks_missing_truth_as_not_evaluated():
    """★ 没有真值的类必须标成"无真值", 不能补 0 —— 补 0 会把"没评测"说成"很差"。"""
    rows = M.per_class_rows(("ball", "player", "referee"),
                            ap_class_index=[0, 1], ap50=[0.2, 0.9], ap=[0.1, 0.7],
                            precision=[0.8, 0.9], recall=[0.5, 0.6],
                            instances=(10, 20, 0))
    assert [row["evaluated"] for row in rows] == [True, True, False]
    assert rows[2]["AP50"] is None and rows[2]["AP50_95"] is None
    assert "没有真值" in rows[2]["note"]
    assert rows[0]["instances"] == 10


def test_mean_ap_full_and_stratified_views():
    rows = M.per_class_rows(("ball", "player", "referee"),
                            ap_class_index=[0, 1, 2], ap50=[0.2, 0.9, 0.7],
                            ap=[0.1, 0.8, 0.6], precision=[0.8, 0.9, 0.85],
                            recall=[0.5, 0.6, 0.55], instances=(10, 200, 100))
    full = M.mean_ap(rows)
    assert full["n_classes"] == 3
    assert full["AP50"] == pytest.approx((0.2 + 0.9 + 0.7) / 3)

    without_ball = M.mean_ap(rows, exclude=["ball"])
    assert without_ball["n_classes"] == 2
    assert without_ball["AP50"] == pytest.approx((0.9 + 0.7) / 2)
    assert "排除 ball" in without_ball["label"]

    only_player = M.mean_ap(rows, include=["player"])
    assert only_player["AP50"] == pytest.approx(0.9)


def test_mean_ap_ignores_classes_without_ground_truth():
    rows = M.per_class_rows(("ball", "player", "referee"),
                            ap_class_index=[0, 1], ap50=[0.2, 0.9], ap=[0.1, 0.8],
                            precision=[0.8, 0.9], recall=[0.5, 0.6],
                            instances=(10, 20, 0))
    assert M.mean_ap(rows)["n_classes"] == 2
    assert M.mean_ap(rows)["AP50"] == pytest.approx((0.2 + 0.9) / 2)


def test_mean_ap_rejects_two_views_at_once():
    with pytest.raises(ValueError, match="口径必须唯一"):
        M.mean_ap([], exclude=["a"], include=["b"])


def test_mean_ap_on_empty_selection_returns_none_not_zero():
    block = M.mean_ap([], exclude=["ball"])
    assert block["n_classes"] == 0
    assert block["AP50"] is None and block["AP50_95"] is None


def test_f1_of_handles_missing_and_zero():
    assert M.f1_of(1.0, 1.0) == pytest.approx(1.0)
    assert M.f1_of(0.5, 0.5) == pytest.approx(0.5)
    assert M.f1_of(None, 0.5) is None
    assert M.f1_of(0.0, 0.0) is None            # 分母为 0 不给结论


def test_conf_sweep_rows_computes_f1_per_threshold():
    rows = M.conf_sweep_rows([0.1, 0.5],
                             [{"precision": 0.5, "recall": 0.5},
                              {"precision": 0.9, "recall": 0.3}])
    assert [row["conf"] for row in rows] == [0.1, 0.5]
    assert rows[0]["f1"] == pytest.approx(0.5)
    assert rows[1]["f1"] == pytest.approx(2 * 0.9 * 0.3 / 1.2)


def test_recommend_threshold_picks_max_recall_above_target_precision():
    """★ 规则: 精度达标的前提下召回率最大 —— 不是"取最大 F1", 也不是"取最高 conf"。"""
    rows = M.conf_sweep_rows([0.1, 0.3, 0.5, 0.7],
                             [{"precision": 0.60, "recall": 0.80},
                              {"precision": 0.92, "recall": 0.60},
                              {"precision": 0.95, "recall": 0.45},
                              {"precision": 0.98, "recall": 0.20}])
    result = M.recommend_threshold(rows, target_precision=0.90)
    assert result["status"] == "达到目标精度"
    assert result["conf"] == 0.3                 # 达标者里 R 最大的
    assert result["recall"] == pytest.approx(0.60)
    assert "召回率最大" in result["reason"]


def test_recommend_threshold_prefers_higher_precision_on_recall_tie():
    rows = M.conf_sweep_rows([0.2, 0.4],
                             [{"precision": 0.91, "recall": 0.5},
                              {"precision": 0.95, "recall": 0.5}])
    assert M.recommend_threshold(rows, target_precision=0.90)["conf"] == 0.4


def test_recommend_threshold_is_honest_when_target_is_unreachable():
    """★ 达不到目标精度时不许硬推荐 —— 要给"不可达"并说明参考值。"""
    rows = M.conf_sweep_rows([0.1, 0.5],
                             [{"precision": 0.40, "recall": 0.90},
                              {"precision": 0.60, "recall": 0.40}])
    result = M.recommend_threshold(rows, target_precision=0.95)
    assert result["status"] == "目标精度不可达"
    # 两行的 F1: conf=0.1 → 2*.4*.9/1.3 ≈ 0.554; conf=0.5 → 2*.6*.4/1.0 = 0.48
    # 所以参考值取 conf=0.1（F1 更高）
    assert result["conf"] == 0.1
    assert "没有任何阈值" in result["reason"]


def test_recommend_threshold_without_observations():
    assert M.recommend_threshold([], target_precision=0.9)["status"] == "无可用观测"


def test_pr_curve_payload_downsamples_and_names_series():
    curves = [(np.linspace(0, 1, 1000), np.tile(np.linspace(1, 0, 1000), (2, 1)),
               "Recall", "Precision")]
    payload = M.pr_curve_payload(curves, ["ball", "player"], points=10)
    assert len(payload) == 1
    assert len(payload[0]["recall"]) == 10
    assert set(payload[0]["precision_by_class"]) == {"ball", "player"}


def test_pr_curve_payload_survives_garbage():
    """曲线是附加信息: 解析不了就跳过, 不该把整份报告拖垮。"""
    assert M.pr_curve_payload([("bad", "worse")], ["a"], points=5) == []
    assert M.pr_curve_payload([], ["a"], points=5) == []


# ---------------------------------------------------------------- 警告

def test_warns_on_small_validation_set():
    rows = M.per_class_rows(("ball",), ap_class_index=[0], ap50=[0.9], ap=[0.8],
                            precision=[0.9], recall=[0.9], instances=(500,))
    warnings = M.build_warnings(fake_view(n_images=49, instances=(1, 2, 0)), rows,
                                EvalSettings())
    assert any("只有 49 张图" in text for text in warnings)
    assert any("统计抖动" in text for text in warnings)


def test_warns_on_class_with_too_few_instances():
    rows = M.per_class_rows(("ball", "player"), ap_class_index=[0, 1],
                            ap50=[0.2, 0.9], ap=[0.1, 0.8],
                            precision=[0.5, 0.9], recall=[0.5, 0.9],
                            instances=(120, 40))
    warnings = M.build_warnings(fake_view(n_images=500), rows, EvalSettings())
    assert any("player=40 框" in text for text in warnings)


def test_warns_on_severe_class_imbalance():
    rows = M.per_class_rows(("ball", "player"), ap_class_index=[0, 1],
                            ap50=[0.2, 0.9], ap=[0.1, 0.8],
                            precision=[0.5, 0.9], recall=[0.5, 0.9],
                            instances=(327, 7427))
    warnings = M.build_warnings(fake_view(n_images=500), rows, EvalSettings())
    assert any("严重不平衡" in text and "22.7" in text for text in warnings)


def test_warns_when_some_classes_have_no_ground_truth():
    rows = M.per_class_rows(("ball", "player", "referee"), ap_class_index=[0, 1],
                            ap50=[0.2, 0.9], ap=[0.1, 0.8],
                            precision=[0.5, 0.9], recall=[0.5, 0.9],
                            instances=(200, 300, 0))
    warnings = M.build_warnings(fake_view(n_images=500), rows, EvalSettings())
    assert any("没有真值" in text and "referee" in text for text in warnings)


def test_warns_when_limit_or_copy_happened():
    rows = M.per_class_rows(("ball",), ap_class_index=[0], ap50=[0.9], ap=[0.8],
                            precision=[0.9], recall=[0.9], instances=(500,))
    warnings = M.build_warnings(fake_view(n_images=500, copied=7, limited_to=12),
                               rows, EvalSettings())
    assert any("不代表整个 split" in text for text in warnings)
    assert any("未能硬链接" in text for text in warnings)


def test_no_warnings_on_a_healthy_run():
    rows = M.per_class_rows(("a", "b"), ap_class_index=[0, 1], ap50=[0.9, 0.88],
                            ap=[0.8, 0.7], precision=[0.9, 0.9], recall=[0.9, 0.9],
                            instances=(500, 400))
    assert M.build_warnings(fake_view(n_images=500), rows, EvalSettings()) == []


# ============================================================
# 4. 报告的三种出口
# ============================================================

def sample_report() -> EvaluationReport:
    rows = M.per_class_rows(("ball", "player", "referee"),
                            ap_class_index=[0, 1], ap50=[0.2, 0.9], ap=[0.1, 0.8],
                            precision=[0.8, 0.95], recall=[0.5, 0.7],
                            instances=(327, 7427, 0))
    return EvaluationReport(
        run_id="20260101-000000", weights="models/x/best.pt",
        data_yaml="runs/x/data.yaml", split="val",
        overall={"map50": 0.55, "map50_95": 0.40},
        per_class=rows,
        stratified=[M.mean_ap(rows), M.mean_ap(rows, exclude=["ball"])],
        conf_sweep=M.conf_sweep_rows([0.1, 0.3],
                                     [{"precision": 0.6, "recall": 0.8},
                                      {"precision": 0.95, "recall": 0.5}]),
        recommended=M.recommend_threshold(
            M.conf_sweep_rows([0.1, 0.3], [{"precision": 0.6, "recall": 0.8},
                                           {"precision": 0.95, "recall": 0.5}]),
            target_precision=0.9),
        warnings=["验证集只有 49 张图: 统计抖动可能大于结论本身"],
        dataset=fake_view(n_images=49).to_dict(), settings_audit=EvalSettings().to_audit(),
        speed_ms={"inference": 60.0}, duration_seconds=1.234)


def test_report_json_contains_every_block(tmp_path):
    path = tmp_path / "report.json"
    sample_report().write_json(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("overall", "per_class", "stratified", "conf_sweep",
                "recommended_threshold", "warnings", "dataset", "settings"):
        assert key in data, f"报告缺 {key}"
    assert data["stratified"][1]["label"].startswith("排除 ball")


def test_report_per_class_csv_keeps_none_empty(tmp_path):
    path = tmp_path / "metrics.csv"
    sample_report().write_per_class_csv(path)
    text = path.read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(text.splitlines()))
    assert len(rows) == 3
    referee = [row for row in rows if row["class_name"] == "referee"][0]
    assert referee["evaluated"] == "False"
    assert referee["AP50"] == ""               # ★ None 留空, 不写成 0
    assert rows[0]["instances"] == "327"


def test_report_conf_sweep_csv(tmp_path):
    path = tmp_path / "conf_sweep.csv"
    sample_report().write_conf_sweep_csv(path)
    rows = list(csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines()))
    assert [row["conf"] for row in rows] == ["0.1", "0.3"]
    assert float(rows[1]["f1"]) > 0


def test_report_logger_renders_the_important_parts(caplog):
    with caplog.at_level(logging.INFO):
        sample_report().render_to_logger(logging.getLogger("test-report"))
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "模型评估报告" in text
    assert "mAP50=0.5500" in text
    assert "排除 ball" in text
    assert "推荐置信度阈值" in text
    assert "必须说在前的警告" in text
    assert "—" in text                          # 没有真值的类显示破折号


def test_report_summary_line_handles_missing_numbers():
    report = EvaluationReport(run_id="r", weights="w", data_yaml="d", split="val")
    assert "mAP50=—" in report.summary_line()


# ============================================================
# 5. 服务与 CLI
# ============================================================

def test_extract_result_pulls_primitives():
    raw = extract_result(FakeResult(FakeBox(
        ap_class_index=[0, 1], ap50=[0.2, 0.9], ap=[0.1, 0.8],
        p=[0.7, 0.9], r=[0.4, 0.6], map50=0.55, map50_95=0.4, mp=0.8, mr=0.5)))
    assert raw["map50"] == 0.55
    assert raw["ap_class_index"] == [0, 1]
    assert raw["ap50"] == [0.2, 0.9]
    assert raw["speed"]["inference"] == 60.0
    assert len(raw["curves"]) == 1


def test_extract_result_rejects_a_non_detection_result():
    class NoBox:
        pass
    with pytest.raises(RuntimeError, match="没有 box"):
        extract_result(NoBox())


@pytest.fixture()
def weights_file(tmp_path):
    path = tmp_path / "best.pt"
    path.write_bytes(b"not-a-real-checkpoint")
    return path


def test_evaluate_model_end_to_end_with_injected_model(tmp_path, weights_file):
    """★ 整条链路（视图 → 假模型 → 指标 → 报告）离线跑通。"""
    source = make_dataset(tmp_path / "ds", images=4)
    model = FakeModel()
    out = tmp_path / "out"

    report = evaluate_model(weights_file, source, settings=EvalSettings(),
                            output_dir=out, run_id="test-run", model=model)

    assert report.run_id == "test-run"
    assert report.overall["map50"] == 0.55
    assert [row["class_name"] for row in report.per_class] == ["ball", "player", "referee"]
    assert report.per_class[2]["evaluated"] is False
    assert len(report.stratified) == 1                 # 没给 exclude 就只出全类平均
    assert len(report.conf_sweep) == len(EvalSettings().conf_thresholds)
    assert report.recommended["status"] == "达到目标精度"
    assert report.dataset["n_images"] == 4
    assert report.warnings                              # 4 张图 + 类别不全 → 必有警告
    # 三种出口都落了盘
    assert (out / "report.json").is_file()
    assert (out / "metrics.csv").is_file()
    assert (out / "conf_sweep.csv").is_file()
    assert (out / "_dataset_view" / "data.yaml").is_file()


def test_evaluate_model_reuses_base_run_for_the_matching_threshold(tmp_path, weights_file):
    """★ 扫描里如果某个阈值正好等于主评估的 conf, 就复用结果 —— 同一次计算不该跑两遍。"""
    source = make_dataset(tmp_path / "ds")
    model = FakeModel()
    settings = EvalSettings(conf_thresholds=(0.001, 0.3))
    evaluate_model(weights_file, source, settings=settings,
                   output_dir=tmp_path / "out", model=model)
    assert len(model.calls) == 2            # 主评估 1 次 + 扫描里非复用的 1 次


def test_evaluate_model_can_skip_the_conf_sweep(tmp_path, weights_file):
    source = make_dataset(tmp_path / "ds")
    model = FakeModel()
    report = evaluate_model(weights_file, source,
                            settings=EvalSettings(conf_sweep=False),
                            output_dir=tmp_path / "out", model=model)
    assert report.conf_sweep == [] and report.recommended == {}
    assert len(model.calls) == 1


def test_evaluate_model_adds_the_exclude_view(tmp_path, weights_file):
    source = make_dataset(tmp_path / "ds")
    report = evaluate_model(weights_file, source, settings=EvalSettings(),
                            output_dir=tmp_path / "out", model=FakeModel(),
                            exclude_classes=["ball"])
    labels = [block["label"] for block in report.stratified]
    assert any("全类平均" in label for label in labels)
    assert any("排除 ball" in label for label in labels)
    full = report.stratified[0]["AP50"]
    without = report.stratified[1]["AP50"]
    assert without > full          # 排掉低分小类后均值必须上升


def test_evaluate_model_warns_about_unknown_class_names(tmp_path, weights_file, caplog):
    source = make_dataset(tmp_path / "ds")
    with caplog.at_level(logging.WARNING):
        evaluate_model(weights_file, source, settings=EvalSettings(conf_sweep=False),
                       output_dir=tmp_path / "out", model=FakeModel(),
                       exclude_classes=["striker"])
    assert any("不会起作用" in record.message for record in caplog.records)


def test_evaluate_model_rejects_two_class_views(tmp_path, weights_file):
    source = make_dataset(tmp_path / "ds")
    with pytest.raises(ValueError, match="口径必须唯一"):
        evaluate_model(weights_file, source, output_dir=tmp_path / "out",
                       model=FakeModel(), exclude_classes=["ball"],
                       include_classes=["player"])


def test_evaluate_model_reports_missing_weights_clearly(tmp_path):
    source = make_dataset(tmp_path / "ds")
    with pytest.raises(FileNotFoundError, match="权重文件不存在"):
        evaluate_model(tmp_path / "nope.pt", source, output_dir=tmp_path / "out",
                       model=FakeModel())


def test_evaluate_model_reports_bad_dataset_clearly(tmp_path, weights_file):
    with pytest.raises(FileNotFoundError, match="不存在"):
        evaluate_model(weights_file, tmp_path / "nope.yaml",
                       output_dir=tmp_path / "out", model=FakeModel())


def test_default_run_dir_uses_the_task_label():
    from od_platform.common import paths
    from od_platform.evaluation.service import default_run_dir
    run_dir = default_run_dir("abc", EvalSettings(task_label="detect"))
    assert run_dir.parent.name == "detect_eval"
    assert run_dir.parent.parent == paths.RUNS_DIR


# ---------------------------------------------------------------- CLI

def test_cli_rejects_bad_arguments(weights_file, tmp_path):
    source = make_dataset(tmp_path / "ds")
    assert cli_mod.main(["--weights", str(weights_file), "--data", str(source),
                         "--split", "nope"]) == cli_mod.EXIT_USAGE


def test_cli_rejects_bad_settings_file(weights_file, tmp_path):
    source = make_dataset(tmp_path / "ds")
    bad = tmp_path / "bad.yaml"
    bad.write_text("evaluation:\n  imgsz: 100\n", encoding="utf-8")
    assert cli_mod.main(["--weights", str(weights_file), "--data", str(source),
                         "--yaml", str(bad)]) == cli_mod.EXIT_USAGE


def test_cli_returns_failed_code_when_weights_missing(tmp_path):
    source = make_dataset(tmp_path / "ds")
    code = cli_mod.main(["--weights", str(tmp_path / "nope.pt"), "--data", str(source),
                         "--no-conf-sweep", "--no-report"])
    assert code == cli_mod.EXIT_FAILED


def test_cli_success_path_with_patched_evaluation(monkeypatch, tmp_path, capsys):
    """CLI 的参数解析/覆盖/退出码 —— 用替身把真正的评估换掉, 保持离线。"""
    captured = {}

    def fake_evaluate(weights, data, **kwargs):
        captured.update(kwargs)
        captured["weights"], captured["data"] = weights, data
        return sample_report()

    monkeypatch.setattr(cli_mod, "evaluate_model", fake_evaluate)
    code = cli_mod.main(["--weights", "w.pt", "--data", "d.yaml",
                         "--split", "test", "--imgsz", "320", "--batch", "4",
                         "--exclude-classes", "ball", "--no-report"])
    assert code == cli_mod.EXIT_OK
    assert captured["settings"].split == "test"          # CLI 覆盖生效
    assert captured["settings"].imgsz == 320
    assert captured["settings"].batch == 4
    assert captured["exclude_classes"] == ("ball",)
    assert captured["write_report"] is False
    assert "mAP50=" in capsys.readouterr().out


def test_cli_batch_argument_keeps_integers_integers(monkeypatch):
    """★ 回归: argparse 若把 batch 写成 float, ``--batch 4`` 会变成 4.0 而被配置校验拒掉。"""
    captured = {}

    def fake_evaluate(weights, data, **kwargs):
        captured.update(kwargs)
        return sample_report()

    monkeypatch.setattr(cli_mod, "evaluate_model", fake_evaluate)
    assert cli_mod.main(["--weights", "w.pt", "--data", "d.yaml",
                         "--batch", "4"]) == cli_mod.EXIT_OK
    assert captured["settings"].batch == 4
    assert isinstance(captured["settings"].batch, int)
