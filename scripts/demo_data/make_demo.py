"""生成 VOC 风格演示数据集 data/raw/demo/(63 张图)。
三个刻意设计:
  1. 12 张图(约19%)含稀有类 污渍 —— 驱动类别平衡报告和分层划分
  2. classes.txt 故意乱序 —— 类名→class_id 由 classes.txt 决定,不由 xml 出现顺序
  3. c_050.xml 里埋了一个越界框(xmax=700 > 640)—— 留给 D4 质检抓,本课照常处理
幂等:重跑覆盖同名文件。
"""
import numpy as np
import cv2
from od_platform.common.paths import RAW_DATA_DIR

ROOT = RAW_DATA_DIR / "demo"
IMG = ROOT / "images"
ANN = ROOT / "annotations"
IMG.mkdir(parents=True, exist_ok=True)
ANN.mkdir(parents=True, exist_ok=True)

(ROOT / "classes.txt").write_text("划痕\n污渍\n缺角\n变形\n", encoding="utf-8")  # 故意乱序

XML = ("<annotation><size><width>640</width><height>480</height></size>\n"
       "{objs}</annotation>\n")
OBJ = ('  <object><name>{n}</name><bndbox>'
       "<xmin>{a}</xmin><ymin>{b}</ymin><xmax>{c}</xmax><ymax>{d}</ymax>"
       "</bndbox></object>\n")


def _make_jpg(path, seed):
    """生成 640x480 随机噪声图(可被 cv2/ultralytics 读取)."""
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


for i in range(51):
    _make_jpg(IMG / f"c_{i:03d}.jpg", seed=i)
    objs = OBJ.format(n="划痕", a=40 + i % 100, b=60, c=160 + i % 100, d=200)
    (ANN / f"c_{i:03d}.xml").write_text(XML.format(objs=objs), encoding="utf-8")

# 埋雷:c_050 的框故意越界(xmax=700>640),留给 D4 的 bbox_within_image 抓
(ANN / "c_050.xml").write_text(
    XML.format(objs=OBJ.format(n="划痕", a=500, b=60, c=700, d=200)),
    encoding="utf-8")

for i in range(12):                                     # 稀有类:污渍
    x = 30 + (i * 11) % 260
    objs = (OBJ.format(n="划痕", a=40 + i, b=60, c=160 + i, d=200)
            + OBJ.format(n="污渍", a=x, b=260, c=x + 100, d=420))
    (ANN / f"rare_{i:03d}.xml").write_text(XML.format(objs=objs), encoding="utf-8")
    _make_jpg(IMG / f"rare_{i:03d}.jpg", seed=1000 + i)

print(f"演示数据就绪: {ROOT}  (63 图 / 75 标注, 其中 12 张含稀有类污渍)")
