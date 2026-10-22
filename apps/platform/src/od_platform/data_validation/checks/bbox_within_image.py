"""check: bbox_within_image —— 框相对图像出不出界(cx±w/2 合并判断)。
≥10% 出界 → ERROR;有但 <10% → WARNING。结构错误的行跳过(label_format 已报)。"""
from __future__ import annotations

from typing import List

from od_platform.common.constants import (
    BBOX_EDGE_TOLERANCE, BBOX_OUT_ERROR_RATIO, DETAILS_PREVIEW_LIMIT, Task,
)
from od_platform.data_validation.labels import parse_label_line
from od_platform.data_validation.registry import (
    CheckContext, CheckResult, CheckSeverity, check,
)

_NAME = "bbox_within_image"


def _overflow_amount(coords: List[float]) -> float:
    """cx,cy,w,h → 出界量(不出界为 0;容差 BBOX_EDGE_TOLERANCE)。"""
    cx, cy, w, h = coords
    overflow = max(0 - (cx - w / 2), (cx + w / 2) - 1,
                   0 - (cy - h / 2), (cy + h / 2) - 1)
    return max(0.0, overflow - BBOX_EDGE_TOLERANCE)


@check(_NAME)
def validate_bbox_within_image(ctx: CheckContext) -> CheckResult:
    snap = ctx.snapshot
    if snap.task_type == Task.SEGMENT:
        return CheckResult(_NAME, CheckSeverity.PASS,
                           "task=segment,出界检查仅适用 detect(多边形顶点允许在外)",
                           {"reason": "task_not_applicable"})
    out_boxes: List[dict] = []
    n_out = n_total = n_skipped = 0
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
                    n_skipped += 1                     # 结构错误:label_format 已报,跳过
                    continue
                n_total += 1
                if _overflow_amount(box[1]) > 0:
                    n_out += 1
                    if len(out_boxes) < DETAILS_PREVIEW_LIMIT:
                        out_boxes.append({"split": split, "label": str(lbl),
                                          "line_no": line_no, "coords": box[1]})
    details = {"total_boxes": n_total, "n_out": n_out,
               "out_ratio": round(n_out / n_total, 4) if n_total else 0.0,
               "skipped_structure_errors": n_skipped,
               "thresholds": {"error_ratio": BBOX_OUT_ERROR_RATIO,
                              "edge_tolerance": BBOX_EDGE_TOLERANCE},
               "out_boxes_preview": out_boxes}
    if n_total == 0:
        return CheckResult(_NAME, CheckSeverity.PASS, "没有可检查的框", details)
    ratio = n_out / n_total
    if ratio >= BBOX_OUT_ERROR_RATIO:
        return CheckResult(_NAME, CheckSeverity.ERROR,
                           f"{n_out}/{n_total} 个框出界({ratio:.1%})", details)
    if n_out:
        return CheckResult(_NAME, CheckSeverity.WARNING,
                           f"{n_out}/{n_total} 个框出界({ratio:.1%})— 少量,建议修源数据",
                           details)
    return CheckResult(_NAME, CheckSeverity.PASS, f"全部 {n_total} 个框均在图像内", details)
