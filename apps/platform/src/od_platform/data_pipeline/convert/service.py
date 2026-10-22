"""convert 服务层:按格式查注册表 → 校验任务能力 → 调用转换函数。

能力门禁集中在这里(门口拦一次),各转换器只写自己会赢的分支。
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from od_platform.data_pipeline.convert.registry import ConvertOptions, get_converter


def convert_data_to_yolo(
    input_dir: Path,
    output_labels_dir: Path,
    annotation_format: str,
    options: ConvertOptions,
) -> List[str]:
    """把 input_dir 下的标注转换为 YOLO 标签,返回写出的类名列表。"""
    entry = get_converter(annotation_format)
    if not entry.supports(options.task):
        raise ValueError(
            f"格式 {annotation_format!r} 不支持 task={options.task!r}。"
            f"支持: {entry.supported_tasks}")
    return entry.func(input_dir, output_labels_dir, options)
