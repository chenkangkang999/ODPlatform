"""生成 YOLO 风格演示数据集 data/raw/demoyolo/(6 张图)。
YOLO 格式不含类名:转换时必须显式提供类别顺序(--classes)。
"""
from od_platform.common.paths import RAW_DATA_DIR

ROOT = RAW_DATA_DIR / "demoyolo"
IMG = ROOT / "images"
LBL = ROOT / "labels"
IMG.mkdir(parents=True, exist_ok=True)
LBL.mkdir(parents=True, exist_ok=True)

for i in range(6):
    stem = f"y_{i:03d}"
    (IMG / f"{stem}.jpg").write_bytes(b"")
    lines = [f"0 0.4{i} 0.35 0.20 0.30"]
    if i < 2:                                           # y_000, y_001 各加一个 class 1
        lines.append(f"1 0.5{i} 0.60 0.25 0.40")
    (LBL / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"YOLO 演示数据就绪: {ROOT}  (6 图, 5 列格式)")
