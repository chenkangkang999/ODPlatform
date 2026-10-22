"""COCO → YOLO。单 json:images/annotations/categories 三表,以 image_id 关联。
能力:detect + segment(annotations 里自带 segmentation 多边形)。
"""
import json
from collections import defaultdict
from pathlib import Path
from typing import List

from od_platform.common.constants import AnnotationFormat, Task
from od_platform.data_pipeline.convert.converters._geometry import (
    normalize_box, normalize_polygon,
)
from od_platform.data_pipeline.convert.registry import ConvertOptions, register_converter


@register_converter(AnnotationFormat.COCO, supported_tasks=(Task.DETECT, Task.SEGMENT))
def convert_coco(input_dir: Path, output_labels_dir: Path,
                 options: ConvertOptions) -> List[str]:
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"在 {input_dir} 下没有找到任何 .json 标注文件")
    data = json.loads(json_files[0].read_text(encoding="utf-8"))

    cat = {c["id"]: c["name"] for c in data["categories"]}
    if options.classes is None:
        names = [cat[cid] for cid in sorted(cat)]       # 按 category_id 排序
    else:
        names = list(options.classes)
    class_to_idx = {n: i for i, n in enumerate(names)}

    img_info = {im["id"]: im for im in data["images"]}
    per_image = defaultdict(list)                       # image_id → [ann]
    for a in data["annotations"]:
        per_image[a["image_id"]].append(a)

    seg = options.task == Task.SEGMENT
    for img_id, im in img_info.items():
        W, H = float(im["width"]), float(im["height"])
        lines: List[str] = []
        for a in per_image.get(img_id, []):
            name = cat[a["category_id"]]
            if name not in class_to_idx:
                continue
            idx = class_to_idx[name]
            if seg:
                for poly in a.get("segmentation", []):
                    pts = normalize_polygon(poly, W, H)
                    lines.append(f"{idx} " + " ".join(f"{v:.6f}" for v in pts))
            else:
                x, y, w, h = a["bbox"]                  # COCO bbox: 左上 + 宽高
                cx, cy, bw, bh = normalize_box(x, y, x + w, y + h, W, H)
                lines.append(f"{idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        stem = Path(im["file_name"]).stem
        (output_labels_dir / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return names
