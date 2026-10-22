"""check: box_geometry —— 退化框(零宽/零高/面积极小)+ 极端宽高比。
if-chain 优先级:一行只报一个问题。整体 WARNING:可训但建议清理。仅适用 detect。"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from od_platform.common.constants import (
    DEGENERATE_BOX_MIN_SIZE, DETAILS_PREVIEW_LIMIT, STATS_MAX_ASPECT_RATIO, Task,
)
from od_platform.data_validation.labels import parse_label_line
from od_platform.data_validation.registry import (
    CheckContext, CheckResult, CheckSeverity, check,
)

_NAME = "box_geometry"
KIND_ZERO_WIDTH = "zero_width"
KIND_ZERO_HEIGHT = "zero_height"
KIND_TINY_AREA = "tiny_area"
KIND_EXTREME_AR = "extreme_aspect_ratio"


def _geometry_issue(coords: List[float]) -> Optional[Tuple[str, str]]:
    """(kind, detail) 或 None。if-chain:退化优先于极端宽高比。"""
    _cx, _cy, w, h = coords
    if w <= 0:
        return KIND_ZERO_WIDTH, f"w={w}"
    if h <= 0:
        return KIND_ZERO_HEIGHT, f"h={h}"
    if w < DEGENERATE_BOX_MIN_SIZE or h < DEGENERATE_BOX_MIN_SIZE:
        return KIND_TINY_AREA, f"w={w}, h={h}"
    ratio = max(w / h, h / w)
    if ratio >= STATS_MAX_ASPECT_RATIO:
        return KIND_EXTREME_AR, f"aspect={ratio:.1f}"
    return None


@check(_NAME)
def validate_box_geometry(ctx: CheckContext) -> CheckResult:
    snap = ctx.snapshot
    if snap.task_type != Task.DETECT:
        return CheckResult(_NAME, CheckSeverity.PASS,
                           f"task={snap.task_type},几何检查仅适用 detect",
                           {"reason": "task_not_applicable"})
    issue_kinds: Dict[str, int] = {}
    bad_boxes: List[dict] = []
    total_boxes = n_skipped = 0
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
                box, kind = parse_label_line(line, snap.task_type)
                if box is None:
                    n_skipped += 1
                    continue
                total_boxes += 1
                issue = _geometry_issue(box[1])
                if issue is not None:
                    kind_s, detail = issue
                    issue_kinds[kind_s] = issue_kinds.get(kind_s, 0) + 1
                    if len(bad_boxes) < DETAILS_PREVIEW_LIMIT:
                        bad_boxes.append({"split": split, "label": str(lbl),
                                          "line_no": line_no, "kind": kind_s,
                                          "detail": detail})
    details = {"total_boxes": total_boxes, "issue_kinds": issue_kinds,
               "skipped_structure_errors": n_skipped,
               "thresholds": {"degenerate_min_size": DEGENERATE_BOX_MIN_SIZE,
                              "max_aspect_ratio": STATS_MAX_ASPECT_RATIO},
               "bad_boxes_preview": bad_boxes}
    if not issue_kinds:
        return CheckResult(_NAME, CheckSeverity.PASS,
                           f"全部 {total_boxes} 个框几何正常(无退化框/极端宽高比)", details)
    n_bad = sum(issue_kinds.values())
    return CheckResult(_NAME, CheckSeverity.WARNING,
                       f"{n_bad}/{total_boxes} 个框几何异常({issue_kinds})— 可训但建议清理",
                       details)
