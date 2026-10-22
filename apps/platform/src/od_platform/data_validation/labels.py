"""YOLO 标签行解析:全框架唯一权威。
返回 (box, kind):box=(class_id, [coords]);kind=None 表示合法;box=None 时 kind 指明原因。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from od_platform.common.constants import Task

KIND_FIELD_COUNT = "field_count_mismatch"
KIND_PARSE = "parse_error"
KIND_POLY_FEW = "polygon_too_few_points"

ParsedBox = Tuple[int, List[float]]


def parse_label_line(line: str, task_type: str) -> Tuple[Optional[ParsedBox], Optional[str]]:
    """解析一行 YOLO 标签(调用方先 strip;空行返回 (None, None))。"""
    parts = line.split()
    if not parts:
        return None, None
    n = len(parts)
    if task_type == Task.SEGMENT:
        if n < 7:
            return None, KIND_POLY_FEW if n >= 3 else KIND_FIELD_COUNT
        if n % 2 == 0:
            return None, KIND_POLY_FEW
    else:
        if n != 5:
            return None, KIND_FIELD_COUNT
    try:
        class_id = int(parts[0])
        coords = [float(v) for v in parts[1:]]
    except ValueError:
        return None, KIND_PARSE
    return (class_id, coords), None
