"""YOLO → YOLO(直通 + 校验)。格式已合规,做的是:
  1. 要求显式 classes(文件里没有类名,顺序即 class_id,不能猜)
  2. 逐行检查列数与 task 路由一致(detect=5 列;segment=奇数且≥7)
检查的是"路由一致性",不是内容质量 —— 内容好坏留给 D4 质检。
"""
from pathlib import Path
from typing import List

from od_platform.common.constants import AnnotationFormat, Task
from od_platform.data_pipeline.convert.registry import ConvertOptions, register_converter


@register_converter(AnnotationFormat.YOLO, supported_tasks=(Task.DETECT, Task.SEGMENT))
def convert_yolo(input_dir: Path, output_labels_dir: Path,
                 options: ConvertOptions) -> List[str]:
    if not options.classes:
        raise ValueError(
            "yolo 格式不含类名,必须通过 options.classes 显式提供类别顺序。")
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    seg = options.task == Task.SEGMENT

    for tf in sorted(input_dir.glob("*.txt")):
        out_lines: List[str] = []
        for ln, raw in enumerate(tf.read_text(encoding="utf-8").splitlines(), 1):
            raw = raw.strip()
            if not raw:
                continue
            n = len(raw.split())
            ok = (n >= 7 and n % 2 == 1) if seg else (n == 5)
            if not ok:
                raise ValueError(
                    f"{tf.name}:{ln} 列数 {n} 与 task={options.task} 不符"
                    f"(detect 需 5 列,segment 需奇数且≥7 列)")
            out_lines.append(raw)
        (output_labels_dir / tf.name).write_text(
            "\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    return list(options.classes)
