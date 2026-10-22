"""check: label_format —— 每行标签的结构与取值合法吗。任何一行坏 → ERROR。"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from od_platform.common.constants import DETAILS_PREVIEW_LIMIT
from od_platform.data_validation.labels import ParsedBox, parse_label_line
from od_platform.data_validation.registry import (
    CheckContext, CheckResult, CheckSeverity, check,
)

_NAME = "label_format"
KIND_CLASS_ID_OUT_OF_RANGE = "class_id_out_of_range"
KIND_COORD_OUT_OF_RANGE = "coord_out_of_range"


def _validate_one_line(line: str, task_type: str,
                       nc: int) -> Tuple[Optional[ParsedBox], Optional[str]]:
    """纯函数:(box, kind);kind=None 即合法。"""
    box, kind = parse_label_line(line, task_type)
    if box is None:
        return None, kind
    class_id, coords = box
    if not (0 <= class_id < nc):
        return None, KIND_CLASS_ID_OUT_OF_RANGE
    if any(v < 0 or v > 1 for v in coords):
        return None, KIND_COORD_OUT_OF_RANGE
    return box, None


@check(_NAME)
def validate_label_format(ctx: CheckContext) -> CheckResult:
    snap = ctx.snapshot
    nc = snap.nc
    if not isinstance(nc, int):
        return CheckResult(_NAME, CheckSeverity.PASS,
                           "nc 不可用,跳过(见 yaml_schema)", {"reason": "nc_unavailable"})
    bad_lines: List[dict] = []
    error_kinds: Dict[str, int] = {}
    n_bad = n_total = 0
    for split in snap.splits:
        for lbl in snap.label_files_per_split[split]:
            try:
                text = lbl.read_text(encoding="utf-8")
            except OSError:
                continue
            for line_no, line in enumerate(text.splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                n_total += 1
                _box, kind = _validate_one_line(line, snap.task_type, nc)
                if kind is not None:
                    n_bad += 1
                    error_kinds[kind] = error_kinds.get(kind, 0) + 1
                    if len(bad_lines) < DETAILS_PREVIEW_LIMIT:
                        bad_lines.append({"split": split, "label": str(lbl),
                                          "line_no": line_no, "kind": kind,
                                          "line": line})
    if n_bad == 0:
        return CheckResult(_NAME, CheckSeverity.PASS,
                           f"全部 {n_total} 行标签结构与取值合法", {"total_lines": n_total})
    return CheckResult(_NAME, CheckSeverity.ERROR,
                       f"{n_bad}/{n_total} 行标签非法({error_kinds})",
                       {"total_lines": n_total, "n_bad": n_bad,
                        "error_kinds": error_kinds, "bad_lines_preview": bad_lines})
