"""Optional model scripts import safely and accept project configuration."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from od_platform.common import paths


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), paths.ROOT_DIR / "scripts/demo_data" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_downloader_import_does_not_start_network(capsys):
    load("download_videomae.py")
    assert capsys.readouterr().out == ""


def test_downloader_uses_selected_repo_directory_and_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://example.invalid")
    module = load("download_videomae.py")
    seen = {}

    def download(**kwargs):
        seen.update(kwargs)
        return kwargs["local_dir"]

    output = tmp_path / "weights"
    assert module.main(["--repo", "sample/model", "--output", str(output), "--endpoint", "https://huggingface.co"], downloader=download) == 0
    assert seen["repo_id"] == "sample/model" and Path(seen["local_dir"]) == output
    assert os.environ["HF_ENDPOINT"] == "https://huggingface.co"


def test_quick_train_preserves_existing_run_and_reports_actual_save_dir(tmp_path):
    module = load("train_football_quick.py")
    data = tmp_path / "data.yaml"
    data.write_text("names: [player, football]\n", encoding="utf-8")
    output = tmp_path / "chosen-run2"
    (output / "weights").mkdir(parents=True)
    (output / "weights/best.pt").write_bytes(b"mock trainer artifact, not a trained model")
    captured = {}

    class Model:
        trainer = SimpleNamespace(save_dir=output)

        def train(self, **kwargs):
            captured.update(kwargs)

    assert module.main(["--data", str(data), "--device", "cpu", "--epochs", "1"], model_factory=lambda _: Model()) == 0
    assert captured["device"] == "cpu" and captured["epochs"] == 1
    assert captured["exist_ok"] is False


def test_quick_train_missing_data_fails_before_loading_weights(tmp_path):
    module = load("train_football_quick.py")
    with pytest.raises(SystemExit) as failure:
        module.main(["--data", str(tmp_path / "missing.yaml")], model_factory=lambda _: pytest.fail("weights must not load"))
    assert failure.value.code == 2
