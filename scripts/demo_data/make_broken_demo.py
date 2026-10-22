"""生成残缺演示数据集 data/raw/broken/:10 张图,只有 1 个标注(模拟标注导出中断)。
用途:验证 odp-transform 的覆盖率前置闸门 —— 应直接拒绝(退出码 1)。
"""
from od_platform.common.paths import RAW_DATA_DIR

ROOT = RAW_DATA_DIR / "broken"
IMG = ROOT / "images"
ANN = ROOT / "annotations"
IMG.mkdir(parents=True, exist_ok=True)
ANN.mkdir(parents=True, exist_ok=True)

XML = ("<annotation><size><width>640</width><height>480</height></size>\n"
       '  <object><name>head</name><bndbox>'
       "<xmin>40</xmin><ymin>60</ymin><xmax>160</xmax><ymax>200</ymax></bndbox></object>\n"
       "</annotation>\n")

for i in range(10):
    (IMG / f"b_{i:03d}.jpg").write_bytes(b"")
(ANN / "b_000.xml").write_text(XML, encoding="utf-8")   # 只有 1 个标注:覆盖率 10%

print(f"残缺演示集就绪: 图 10 张, 标注 1 个 -> {ROOT}")
