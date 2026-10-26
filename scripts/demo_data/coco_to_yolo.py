#!/usr/bin/env python
"""Convert split ZIPs containing _annotations.coco.json to a YOLO dataset.

Paths and category order are configurable. Existing nonempty output is never
deleted, and archive filenames are validated before any data is materialized.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

import yaml

from od_platform.common.dataset_yaml import to_repo_relative
from od_platform.common.paths import RAW_DATA_DIR

DEFAULT_INPUT = RAW_DATA_DIR / "soccer" / "football-det" / "data"
DEFAULT_OUTPUT = RAW_DATA_DIR / "soccer" / "football-yolo"
DEFAULT_CLASSES = ("player", "football")
DEFAULT_SPLITS = ("train", "valid", "test")


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def coco_to_yolo(bbox, img_w, img_h):
    """COCO [x,y,w,h] pixels -> YOLO normalized centre/width/height."""
    if not isinstance(bbox, list | tuple) or len(bbox) != 4:
        raise ValueError("COCO bbox must contain x, y, width, height")
    img_w, img_h = _number(img_w, "image width"), _number(img_h, "image height")
    x, y, w, h = [_number(value, "bbox coordinate") for value in bbox]
    if img_w <= 0 or img_h <= 0 or w <= 0 or h <= 0:
        raise ValueError("Image and bbox dimensions must be positive")
    return f"{(x + w / 2) / img_w:.6f} {(y + h / 2) / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}"


def _image_path(filename):
    if not isinstance(filename, str) or not filename or "\0" in filename:
        raise ValueError("COCO file_name must be nonempty text")
    path = PurePosixPath(filename.replace("\\", "/"))
    if path.is_absolute() or PureWindowsPath(filename).drive or ".." in path.parts or path == PurePosixPath("."):
        raise ValueError(f"Unsafe image filename: {filename!r}")
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        raise ValueError(f"Unsupported image filename: {filename!r}")
    return path


def _plan_split(archive, classes):
    with zipfile.ZipFile(archive) as bundle:
        try:
            coco = json.loads(bundle.read("_annotations.coco.json"))
        except KeyError as exc:
            raise ValueError(f"{archive.name}: missing _annotations.coco.json") from exc
        try:
            categories = {item["id"]: item["name"] for item in coco["categories"]}
            images, annotations = coco["images"], coco["annotations"]
            if not all(isinstance(value, list) for value in (images, annotations, coco["categories"])):
                raise ValueError("images/categories/annotations must be lists")
            image_ids = {image["id"] for image in images}
            if len(image_ids) != len(images):
                raise ValueError("Duplicate COCO image id")
            by_image = {}
            for annotation in annotations:
                if annotation["image_id"] not in image_ids:
                    raise ValueError("Annotation references an unknown image")
                category = categories[annotation["category_id"]]
                if category not in classes:
                    raise ValueError(f"Category {category!r} is absent from --classes")
                by_image.setdefault(annotation["image_id"], []).append(annotation)
            planned, image_names, label_names = [], set(), set()
            for image in images:
                relative = _image_path(image["file_name"])
                label = relative.with_suffix(".txt")
                if relative in image_names or label in label_names:
                    raise ValueError(f"Duplicate image/label destination: {relative}")
                image_names.add(relative)
                label_names.add(label)
                try:
                    info = bundle.getinfo(relative.as_posix())
                except KeyError as exc:
                    raise ValueError(f"{archive.name}: image is missing: {relative}") from exc
                if info.is_dir() or info.file_size == 0:
                    raise ValueError(f"{archive.name}: image is empty: {relative}")
                for key in ("width", "height"):
                    if _number(image[key], key) <= 0:
                        raise ValueError("Image dimensions must be positive")
                lines = [f"{classes.index(categories[a['category_id']])} "
                         f"{coco_to_yolo(a['bbox'], image['width'], image['height'])}"
                         for a in by_image.get(image["id"], [])]
                planned.append((relative, label, "\n".join(lines)))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{archive.name}: invalid COCO fields: {exc}") from exc
    return planned


def convert_dataset(input_dir, output_dir, *, classes=DEFAULT_CLASSES, splits=DEFAULT_SPLITS):
    if Path(output_dir).is_symlink():
        raise ValueError("Output must not be a symbolic link")
    source, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    classes, splits = tuple(classes), tuple(splits)
    if not classes or any(not isinstance(name, str) or not name.strip() for name in classes) or len(set(classes)) != len(classes):
        raise ValueError("--classes must contain distinct nonempty category names")
    if not splits or len(set(splits)) != len(splits) or any(split not in {"train", "valid", "val", "test"} for split in splits):
        raise ValueError("--splits must be distinct train/valid/val/test names")
    if "valid" in splits and "val" in splits:
        raise ValueError("Use valid or val, not both")
    if output.exists() and (not output.is_dir() or output.is_symlink() or any(output.iterdir())):
        raise ValueError(f"Output is not an empty directory; choose a new path: {output}")
    plans = {split: _plan_split(source / f"{split}.zip", classes) for split in splits}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        for split, examples in plans.items():
            with zipfile.ZipFile(source / f"{split}.zip") as archive:
                for relative, label, content in examples:
                    destination = staging / "images" / split / relative.as_posix()
                    annotation = staging / "labels" / split / label.as_posix()
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    annotation.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(relative.as_posix()) as src, destination.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    annotation.write_text(content, encoding="utf-8")
        config = {"path": to_repo_relative(output), "names": dict(enumerate(classes))}
        for split in splits:
            config["val" if split == "valid" else split] = f"images/{split}"
            (staging / "images" / split).mkdir(parents=True, exist_ok=True)
            (staging / "labels" / split).mkdir(parents=True, exist_ok=True)
        (staging / "data.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        if output.exists():
            output.rmdir()  # Empty directory only; never recursively erase user data.
        staging.rename(output)
    return {split: len(examples) for split, examples in plans.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Directory with split.zip files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="New or empty output directory")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, help="Output category names in class-id order")
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS, choices=("train", "valid", "val", "test"))
    args = parser.parse_args(argv)
    try:
        counts = convert_dataset(args.input, args.output, classes=args.classes, splits=args.splits)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"Conversion failed: {exc}", file=sys.stderr)
        return 1
    for split, count in counts.items():
        print(f"{split}: {count} images")
    print(f"data.yaml: {args.output.resolve() / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
