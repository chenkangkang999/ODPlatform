"""A tracked YAML must not conceal missing ignored demo artifacts."""
import importlib.util
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def bootstrap(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("demo_bootstrap_regression", Path(__file__).with_name("conftest.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yaml_path = tmp_path / "demo.yaml"
    root = tmp_path / "processed"
    manifest = tmp_path / "manifest.json"
    calls = []

    def write_config():
        yaml_path.write_text(yaml.safe_dump({"path": root.as_posix(),
            **{split: "images/" + split for split in ("train", "val", "test")},
            "odp_meta": {"manifest_path": manifest.as_posix()}}), encoding="utf-8")

    def materialize():
        write_config()
        manifest.write_text("{}", encoding="utf-8")
        for split in ("train", "val", "test"):
            images, labels = root / "images" / split, root / "labels" / split
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)
            (images / "sample.jpg").write_bytes(b"fixture-presence-only")
            (labels / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    class Pipeline:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

        def run(self):
            materialize()

    monkeypatch.setattr(module.paths, "dataset_yaml_path", lambda _: yaml_path)
    monkeypatch.setattr(module, "DatasetPipeline", Pipeline)
    return module, yaml_path, root, manifest, calls, write_config, materialize


def test_tracked_yaml_without_processed_data_is_rebuilt(bootstrap):
    module, yaml_path, _, _, calls, write_config, _ = bootstrap
    write_config()
    assert module._ensure_demo_yaml() == yaml_path
    assert len(calls) == 1


def test_missing_manifest_is_rebuilt(bootstrap):
    module, _, _, manifest, calls, _, materialize = bootstrap
    materialize()
    manifest.unlink()
    module._ensure_demo_yaml()
    assert len(calls) == 1 and manifest.is_file()


def test_incomplete_split_is_rebuilt(bootstrap):
    module, _, root, _, calls, _, materialize = bootstrap
    materialize()
    (root / "labels/val/sample.txt").unlink()
    module._ensure_demo_yaml()
    assert len(calls) == 1


def test_complete_cached_data_is_reused(bootstrap):
    module, yaml_path, _, _, calls, _, materialize = bootstrap
    materialize()
    assert module._ensure_demo_yaml() == yaml_path
    assert calls == []


def test_pipeline_failure_is_not_swallowed(bootstrap, monkeypatch):
    module, _, _, _, _, _, _ = bootstrap

    class FailingPipeline:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            raise ValueError("conversion failed")

    monkeypatch.setattr(module, "DatasetPipeline", FailingPipeline)
    with pytest.raises(ValueError, match="conversion failed"):
        module._ensure_demo_yaml()


def test_pipeline_that_produces_no_artifacts_does_not_pass(bootstrap, monkeypatch):
    module, _, _, _, _, _, _ = bootstrap

    class EmptyPipeline:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            pass

    monkeypatch.setattr(module, "DatasetPipeline", EmptyPipeline)
    with pytest.raises(RuntimeError, match="manifest"):
        module._ensure_demo_yaml()
