"""End-to-end layout, category order and safe archive handling for the ZIP tool."""
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest
import yaml
from PIL import Image

from od_platform.common import dataset_yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/demo_data/coco_to_yolo.py"
spec = importlib.util.spec_from_file_location("coco_to_yolo_script", SCRIPT)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def make_zip(root, split="train", *, filenames=("first/frame.jpg", "second/frame.jpg"), missing=False, bbox=None):
    root.mkdir(parents=True, exist_ok=True)
    encoded = io.BytesIO()
    Image.new("RGB", (80, 40), "green").save(encoded, format="JPEG")
    images = [{"id": index, "file_name": name, "width": 80, "height": 40} for index, name in enumerate(filenames)]
    annotations = [{"id": index, "image_id": index, "category_id": 4, "bbox": bbox or [10, 10, 20, 20]}
                   for index in range(len(images))]
    coco = {"images": images, "annotations": annotations, "categories": [{"id": 4, "name": "player"}, {"id": 7, "name": "football"}]}
    with zipfile.ZipFile(root / f"{split}.zip", "w") as archive:
        archive.writestr("_annotations.coco.json", json.dumps(coco))
        if not missing:
            for filename in filenames:
                archive.writestr(filename, encoded.getvalue())


def test_real_cli_exports_matching_yaml_paths_and_preserves_nested_names(tmp_path):
    source, output = tmp_path / "input", tmp_path / "converted"
    for split in ("train", "valid", "test"):
        make_zip(source, split)
    assert converter.main(["--input", str(source), "--output", str(output), "--classes", "football", "player"]) == 0
    config = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert config["names"] == {0: "football", 1: "player"}
    assert Path(config["path"]) == output.resolve()
    for key, split in (("train", "train"), ("val", "valid"), ("test", "test")):
        assert (output / config[key]).is_dir()
        for parent in ("first", "second"):
            with Image.open(output / "images" / split / parent / "frame.jpg") as image:
                assert image.size == (80, 40)
            label = output / "labels" / split / parent / "frame.txt"
            assert label.read_text(encoding="utf-8") == "1 0.250000 0.500000 0.250000 0.500000"


def test_existing_output_is_untouched(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("important", encoding="utf-8")
    with pytest.raises(ValueError, match="empty directory"):
        converter.convert_dataset(tmp_path / "not-needed", output, splits=("train",))
    assert sentinel.read_text(encoding="utf-8") == "important"
    assert list(output.iterdir()) == [sentinel]


def test_output_inside_workspace_writes_a_portable_yaml(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_yaml, "ROOT_DIR", tmp_path)
    make_zip(tmp_path / "input")
    output = tmp_path / "data/processed/football"
    converter.convert_dataset(tmp_path / "input", output, splits=("train",))
    config = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert config["path"] == "data/processed/football"
    assert (tmp_path / config["path"] / config["train"]).is_dir()


@pytest.mark.parametrize("name", ["../escape.jpg", "/escape.jpg", "..\\escape.jpg", "C" + ":/escape.jpg"])
def test_unsafe_archive_filenames_are_rejected_before_writing(tmp_path, name):
    make_zip(tmp_path / "input", filenames=(name,))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Unsafe image filename"):
        converter.convert_dataset(tmp_path / "input", output, splits=("train",))
    assert not output.exists()
    assert not (tmp_path / "escape.jpg").exists()


def test_missing_archive_image_is_an_error_not_a_silent_drop(tmp_path, capsys):
    make_zip(tmp_path / "input", missing=True)
    output = tmp_path / "output"
    assert converter.main(["--input", str(tmp_path / "input"), "--output", str(output), "--splits", "train"]) == 1
    assert "image is missing" in capsys.readouterr().err
    assert not output.exists()


def test_nonfinite_bbox_does_not_publish_partial_dataset(tmp_path):
    make_zip(tmp_path / "input", bbox=[10, 10, float("nan"), 20])
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="finite number"):
        converter.convert_dataset(tmp_path / "input", output, splits=("train",))
    assert not output.exists()


def test_same_stem_different_extensions_cannot_overwrite_labels(tmp_path):
    make_zip(tmp_path / "input", filenames=("frame.jpg", "frame.png"))
    with pytest.raises(ValueError, match="Duplicate image/label"):
        converter.convert_dataset(tmp_path / "input", tmp_path / "output", splits=("train",))


def test_selected_split_and_val_alias_are_configurable(tmp_path):
    make_zip(tmp_path / "input", "val")
    output = tmp_path / "output"
    assert converter.convert_dataset(tmp_path / "input", output, splits=("val",)) == {"val": 2}
    config = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert config["val"] == "images/val"
    assert "test" not in config
