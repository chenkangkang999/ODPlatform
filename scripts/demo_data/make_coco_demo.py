"""生成 COCO 风格演示数据集 data/raw/demococo/(8 张图)。
每张图有 bbox(检测)+ segmentation 多边形(分割),前 2 张额外含稀有类 反光衣。
"""
import json
from od_platform.common.paths import RAW_DATA_DIR

ROOT = RAW_DATA_DIR / "demococo"
IMG = ROOT / "images"
ANN = ROOT / "annotations"
IMG.mkdir(parents=True, exist_ok=True)
ANN.mkdir(parents=True, exist_ok=True)

images, anns = [], []
for i in range(8):
    stem = f"co_{i:03d}"
    (IMG / f"{stem}.jpg").write_bytes(b"")
    images.append({"id": i, "file_name": f"{stem}.jpg", "width": 640, "height": 480})
    anns.append({"id": len(anns), "image_id": i, "category_id": 1,
                 "bbox": [40 + i, 50, 120, 140],
                 "segmentation": [[40 + i, 50, 160 + i, 50, 160 + i, 190, 40 + i, 190]]})
for i in range(2):                                      # co_000, co_001 各加一个 反光衣
    anns.append({"id": len(anns), "image_id": i, "category_id": 2,
                 "bbox": [60, 150, 180, 260],
                 "segmentation": [[60, 150, 240, 150, 240, 410, 60, 410]]})

coco = {"images": images, "annotations": anns,
        "categories": [{"id": 1, "name": "head"}, {"id": 2, "name": "反光衣"}]}
(ANN / "instances.json").write_text(
    json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"COCO 演示数据就绪: {ROOT}  (8 图 / {len(anns)} 标注, 含分割多边形)")
