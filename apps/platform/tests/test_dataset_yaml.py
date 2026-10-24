"""dataset_yaml —— 数据集路径"可移植性"的单元测试(写端 / 读端 / 运行期副本)。

对应决策见 docs/architecture/ADR-002-dataset-yaml-portable-path.md:
签入 git 的 yaml 里 path 必须相对仓库根, 绝不写死盘符。
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from od_platform.common import paths
from od_platform.common.dataset_yaml import (
    resolve_dataset_root,
    to_repo_relative,
    write_resolved_copy,
)
from od_platform.data_pipeline.split.yaml_writer import write_dataset_yaml


def _stub_manifest() -> SimpleNamespace:
    """只提供 yaml_writer 需要的字段, 避免为一个纯写入测试去造完整 manifest。"""
    return SimpleNamespace(
        tool_version="odp-transform/test",
        created_at="2026-01-01T00:00:00",
        contract_fingerprint="f" * 64,
        strategy="random",
        seed=42,
        ratios=(0.8, 0.1, 0.1),
        stems_of=lambda split: (),
    )


def test_to_repo_relative_is_portable():
    rel = to_repo_relative(paths.PROCESSED_DATA_DIR / "demo")
    assert rel == "data/processed/demo"
    assert not Path(rel).is_absolute()


def test_to_repo_relative_falls_back_outside_repo(tmp_path):
    outside = tmp_path / "ds"
    outside.mkdir()
    assert Path(to_repo_relative(outside)).is_absolute()


def test_resolve_absolute_path_unchanged(tmp_path):
    p = tmp_path / "ds"
    assert resolve_dataset_root(tmp_path / "x.yaml", p.as_posix()) == p


def test_resolve_relative_to_yaml_dir_first(tmp_path):
    root = tmp_path / "ds"
    root.mkdir()
    yaml_path = tmp_path / "demo.yaml"
    yaml_path.write_text("path: ds\n", encoding="utf-8")
    assert resolve_dataset_root(yaml_path, "ds") == root.resolve()


def test_resolve_falls_back_to_repo_root(tmp_path):
    yaml_path = tmp_path / "demo.yaml"
    yaml_path.write_text("path: data/processed/demo\n", encoding="utf-8")
    assert resolve_dataset_root(yaml_path, "data/processed/demo") \
        == (paths.ROOT_DIR / "data/processed/demo").resolve()


def test_empty_path_returns_none(tmp_path):
    assert resolve_dataset_root(tmp_path / "x.yaml", None) is None
    assert resolve_dataset_root(tmp_path / "x.yaml", "") is None


def test_written_yaml_is_portable(tmp_path):
    yaml_path = tmp_path / "demo.yaml"
    write_dataset_yaml(
        yaml_path, dataset_root=paths.PROCESSED_DATA_DIR / "demo",
        classes=["划痕", "污渍"], manifest=_stub_manifest(), dataset_name="demo",
        source_format="pascal_voc", task="detect",
        manifest_ref="runs/data_pipeline/test/manifest.json")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert data["path"] == "data/processed/demo"
    assert not Path(data["path"]).is_absolute()


def test_resolved_copy_is_absolute_and_keeps_other_fields(tmp_path):
    src = tmp_path / "demo.yaml"
    src.write_text(yaml.safe_dump(
        {"path": "data/processed/demo", "nc": 2, "names": {0: "划痕", 1: "污渍"}},
        allow_unicode=True), encoding="utf-8")
    dest = tmp_path / "现场" / "dataset_resolved.yaml"

    write_resolved_copy(src, dest)

    data = yaml.safe_load(dest.read_text(encoding="utf-8"))
    assert Path(data["path"]).is_absolute()
    assert Path(data["path"]) == (paths.ROOT_DIR / "data/processed/demo").resolve()
    assert data["nc"] == 2
    assert data["names"] == {0: "划痕", 1: "污渍"}
    # 源 yaml 保持可移植, 不被就地改写
    assert yaml.safe_load(src.read_text(encoding="utf-8"))["path"] == "data/processed/demo"
