"""D4 测试夹具:全部基于 D3 真实产物的【拷贝】,绝不改 data/processed 原样。"""
from pathlib import Path
import shutil

import pytest
import yaml

from od_platform.common import paths
from od_platform.data_pipeline.orchestrator import DatasetPipeline


def _ensure_demo_yaml() -> Path:
    """确保 D3 已产出 demo;没跑过就现场跑一遍。"""
    yaml_path = paths.dataset_yaml_path("demo")
    if not yaml_path.exists():
        DatasetPipeline("demo", "pascal_voc",
                        split_strategy="stratified_multilabel").run()
    return yaml_path


@pytest.fixture(scope="session")
def demo_yaml() -> Path:
    return _ensure_demo_yaml()


def _make_variant(tmp_path: Path, name: str, mutate=None, mutate_yaml=None) -> Path:
    """拷贝 processed/demo 到 tmp_path/name,动手脚,写指向拷贝的 yaml(odp_meta 原样)。"""
    src_yaml = _ensure_demo_yaml()
    dst_root = tmp_path / name
    shutil.copytree(paths.dataset_processed_dir("demo"), dst_root)
    if mutate:
        mutate(dst_root)
    doc = yaml.safe_load(src_yaml.read_text(encoding="utf-8"))
    doc["path"] = dst_root.as_posix()
    if mutate_yaml:
        mutate_yaml(doc)
    yp = tmp_path / f"{name}.yaml"
    yp.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False),
                  encoding="utf-8")
    return yp


def _first_label(root: Path, split: str) -> Path:
    return sorted((root / "labels" / split).glob("*.txt"))[0]


# ---- 上篇夹具 ----

@pytest.fixture
def demo_yaml_missing_labels(tmp_path) -> Path:
    """删掉 train 里每 5 个标签中的 1 个(≈16% → WARNING)。"""
    def mutate(root: Path):
        for i, lbl in enumerate(sorted((root / "labels" / "train").glob("*.txt"))):
            if i % 5 == 0:
                lbl.unlink()
    return _make_variant(tmp_path, "missing_labels", mutate)


@pytest.fixture
def demo_yaml_wrong_fingerprint(tmp_path) -> Path:
    """yaml 指纹清零(张冠李戴)→ fingerprint_match ERROR。"""
    return _make_variant(tmp_path, "wrong_fp",
                         mutate_yaml=lambda doc: doc["odp_meta"].update(
                             contract_fingerprint="0" * 64))


@pytest.fixture
def demo_yaml_tampered_label(tmp_path) -> Path:
    """改动一个标签的内容 → manifest_lineage 抓 changed。"""
    def mutate(root: Path):
        lbl = _first_label(root, "train")
        lbl.write_text(lbl.read_text(encoding="utf-8") + "1 0.50 0.50 0.20 0.20\n",
                       encoding="utf-8")
    return _make_variant(tmp_path, "tampered_label", mutate)


# ---- 下篇夹具 ----

@pytest.fixture
def demo_yaml_bbox_overflow(tmp_path) -> Path:
    """加一个出界框(右缘 1.08)→ bbox_within_image 抓(WARNING,占比 <10%)。"""
    def mutate(root: Path):
        lbl = _first_label(root, "train")
        lbl.write_text(lbl.read_text(encoding="utf-8") + "0 0.93 0.50 0.30 0.20\n",
                       encoding="utf-8")
    return _make_variant(tmp_path, "bbox_overflow", mutate)


@pytest.fixture
def demo_yaml_bad_field_count(tmp_path) -> Path:
    """一行只有 4 列 → label_format ERROR;bbox_within_image 跳过不重复报。"""
    def mutate(root: Path):
        lbl = _first_label(root, "train")
        lbl.write_text(lbl.read_text(encoding="utf-8") + "0 0.5 0.5 0.2\n",
                       encoding="utf-8")
    return _make_variant(tmp_path, "bad_field_count", mutate)


@pytest.fixture
def demo_yaml_leaked_sample(tmp_path) -> Path:
    """把一张 train 图复制进 val → split_uniqueness 硬 ERROR。"""
    def mutate(root: Path):
        img = sorted((root / "images" / "train").glob("*"))[0]
        shutil.copy2(img, root / "images" / "val" / img.name)
    return _make_variant(tmp_path, "leaked_sample", mutate)


@pytest.fixture
def demo_yaml_class_missing_train(tmp_path) -> Path:
    """train 里删光 反光衣(class 1)、并保证 val 里有一条 → missing_in_train WARNING。"""
    def mutate(root: Path):
        for lbl in (root / "labels" / "train").glob("*.txt"):
            kept = [l for l in lbl.read_text(encoding="utf-8").splitlines()
                    if l.strip() and not l.startswith("1 ")]
            lbl.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        val_lbl = _first_label(root, "val")
        val_lbl.write_text(val_lbl.read_text(encoding="utf-8") + "1 0.50 0.50 0.20 0.20\n",
                           encoding="utf-8")
    return _make_variant(tmp_path, "class_missing_train", mutate)


@pytest.fixture
def demo_yaml_degenerate_and_extreme(tmp_path) -> Path:
    """w=0.0005 的框:既是退化框、也算极端宽高比 —— if-chain 只报 zero/tiny 一个。"""
    def mutate(root: Path):
        lbl = _first_label(root, "train")
        lbl.write_text(lbl.read_text(encoding="utf-8") + "0 0.50 0.50 0.0005 0.20\n",
                       encoding="utf-8")
    return _make_variant(tmp_path, "degenerate_and_extreme", mutate)
