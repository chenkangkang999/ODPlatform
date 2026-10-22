"""data_pipeline 回归测试:七个测试守住本课全部关键语义。"""
import pytest

from od_platform.common import paths
from od_platform.common.constants import AnnotationFormat, SplitStrategy, Task
from od_platform.data_pipeline.convert.registry import list_converters
from od_platform.data_pipeline.orchestrator import DatasetPipeline
from od_platform.data_pipeline.report import analyze_balance
from od_platform.data_pipeline.split.registry import list_strategies
from od_platform.data_pipeline.split.split_service import split_dataset


def test_registries_populated():
    """懒加载自动发现:首次查询时注册表已装满。"""
    conv = list_converters()
    assert set(conv) >= {AnnotationFormat.PASCAL_VOC, AnnotationFormat.COCO,
                         AnnotationFormat.YOLO}
    assert conv[AnnotationFormat.COCO].supports(Task.SEGMENT)
    assert not conv[AnnotationFormat.PASCAL_VOC].supports(Task.SEGMENT)
    strats = list_strategies()
    assert set(strats) >= {SplitStrategy.RANDOM, SplitStrategy.STRATIFIED_MULTILABEL}
    assert strats[SplitStrategy.STRATIFIED_MULTILABEL].requires_labels


def test_split_loses_no_image():
    """划分守恒:一个不少、一个不重复。"""
    stems = [f"s{i:03d}" for i in range(63)]
    splits = split_dataset(stems)
    total = splits["train"] + splits["val"] + splits["test"]
    assert sorted(total) == sorted(stems)


def test_random_reproducible():
    """同种子同结果(划分可复现是契约指纹的前提)。"""
    stems = [f"s{i:03d}" for i in range(63)]
    assert split_dataset(stems, random_state=42) == split_dataset(stems, random_state=42)


def test_stratified_requires_labels():
    """分层策略没有 labels_per_image 直接拒跑。"""
    stems = [f"s{i}" for i in range(10)]
    with pytest.raises(ValueError, match="labels_per_image"):
        split_dataset(stems, strategy=SplitStrategy.STRATIFIED_MULTILABEL,
                      labels_per_image=None)


def test_stratified_rescues_rare_class():
    """12 张稀有类图混在 63 张里:分层划分保证每个 split 都拿到稀有类。"""
    stems = [f"c_{i:03d}" for i in range(51)] + [f"rare_{i:03d}" for i in range(12)]
    labels = {s: ["head"] for s in stems}
    for i in range(12):
        labels[f"rare_{i:03d}"] = ["head", "反光衣"]
    splits = split_dataset(stems, 0.7, 0.15, 4,
                           strategy=SplitStrategy.STRATIFIED_MULTILABEL,
                           labels_per_image=labels)
    rare_in = {sp: sum(1 for s in ss if "反光衣" in labels[s])
               for sp, ss in splits.items()}
    assert all(n >= 1 for n in rare_in.values()), rare_in


def test_report_flags_rare():
    """平衡报告抓稀有类:反光衣(12/63,最小 split 期望 1.2 < 3)被标稀有。"""
    per_image = {f"c_{i:03d}": ["head"] for i in range(51)}
    for i in range(12):
        per_image[f"rare_{i:03d}"] = ["head", "反光衣"]
    report = analyze_balance(per_image, ["head", "反光衣"], min_split_rate=0.1)
    assert "反光衣" in report.rare
    assert "head" not in report.rare


def test_coverage_gate_rejects_broken():
    """覆盖率闸门:10 图 1 标注(10%)低于硬阈值 50%,直接拒跑。"""
    if not paths.raw_dataset_root("broken").exists():
        pytest.skip("缺少 data/raw/broken,先跑 scripts/demo_data/make_broken_demo.py")
    pipe = DatasetPipeline("broken", AnnotationFormat.PASCAL_VOC)
    with pytest.raises(ValueError, match="覆盖率"):
        pipe.run()
