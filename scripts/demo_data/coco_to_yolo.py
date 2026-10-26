#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""COCO -> YOLO 格式转换脚本 (足球检测数据集)"""
import json
import os
import zipfile
import shutil
from pathlib import Path

BASE = Path(r"D:\ODPlatform\data\raw\soccer\football-det\data")
OUT = Path(r"D:\ODPlatform\data\raw\soccer\football-yolo")

# COCO类别: player=0, football=1
CLASSES = ['player', 'football']

def coco_to_yolo(bbox, img_w, img_h):
    """COCO bbox [x,y,w,h] (绝对像素) -> YOLO [cx,cy,w,h] (归一化)"""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return f"{cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"

def process_split(split):
    zip_path = BASE / f"{split}.zip"
    out_dir = OUT / split
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as z:
        # 读COCO标注
        with z.open("_annotations.coco.json") as f:
            coco = json.load(f)

        # image_id -> {file_name, width, height}
        imgs = {im["id"]: im for im in coco["images"]}
        # category_id -> name
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        # image_id -> [annotations]
        anns = {}
        for a in coco["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)

        count = 0
        for img_id, im in imgs.items():
            fname = im["file_name"]
            # 解压图片
            try:
                data = z.read(fname)
            except KeyError:
                continue
            (img_dir / fname).write_bytes(data)

            # 写YOLO标注
            lines = []
            for a in anns.get(img_id, []):
                cat_name = cats[a["category_id"]]
                cls_id = CLASSES.index(cat_name)
                yolo_line = coco_to_yolo(a["bbox"], im["width"], im["height"])
                lines.append(f"{cls_id} {yolo_line}")

            lbl_name = Path(fname).stem + ".txt"
            (lbl_dir / lbl_name).write_text("\n".join(lines), encoding="utf-8")
            count += 1

    print(f"{split}: {count} 张图 -> {out_dir}")

def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ["train", "valid", "test"]:
        process_split(split)

    # 写data.yaml
    yaml = f"""# 足球检测数据集 (YOLO格式)
path: {OUT.as_posix()}
train: images/train
val: images/valid
test: images/test

names:
  0: player
  1: football
"""
    (OUT / "data.yaml").write_text(yaml, encoding="utf-8")
    print(f"\ndata.yaml 已写入: {OUT / 'data.yaml'}")
    print("完成!")

if __name__ == "__main__":
    main()
