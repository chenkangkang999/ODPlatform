#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""快速训练足球检测模型（跑通用，非最终模型）。

数据集: football-yolo (player / football 两类)
模型: yolo11n (最小最快)
输出: D:/ODPlatform/models/trained/football_yolo/
"""
from pathlib import Path
from ultralytics import YOLO

DATA_YAML = r"D:\ODPlatform\data\raw\soccer\football-yolo\data.yaml"
PROJECT = r"D:\ODPlatform\models\trained"
NAME = "football_yolo"

def main():
    model = YOLO(r"D:\ODPlatform\models\pretrained\yolo11n.pt")
    model.train(
        data=DATA_YAML,
        epochs=3,
        imgsz=640,
        batch=16,
        device=0,            # GPU
        workers=0,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        verbose=True,
    )
    best = Path(PROJECT) / NAME / "weights" / "best.pt"
    print(f"训练完成，最佳权重: {best}")
    print(f"存在: {best.is_file()}")

if __name__ == "__main__":
    main()
