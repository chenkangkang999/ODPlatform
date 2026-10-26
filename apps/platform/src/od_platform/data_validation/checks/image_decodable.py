"""check: image_decodable —— 图像文件本身能不能打开（结构完整性）。

★ 为什么需要这一项

    现有检查全都在看**标签**或**配对关系**: 标签结构(label_format)、框是否越界
    (bbox_within_image)、图有没有对应标签(pair_existence)、标注覆盖
    (annotation_coverage)…… **没有任何一项验证"图本身可不可读"**。

    于是一个末尾被截断的 JPEG 在它们眼里完全正常 —— 文件在、大小非零、有对应标签。
    直到训练时 DataLoader 才抛错, 而那时已经烧掉几小时。这一项就是把这个失败
    提前到质检阶段。

★ 实现选择: 用 PIL 的 verify() 做【结构级】校验, 不真解像素

    verify() 会读完文件头与所有数据块, 能抓到截断/损坏; 代价远低于整图
    imdecode（后者要按像素解码, 大分辨率图上是几十倍的开销）。
    代价是它抓不到"能解析但像素异常"的图 —— 那属于另一个层面的问题, 不在本项范围。
    注意 PIL 规定: verify() 之后该对象不可再用, 所以每张图都要新开一个。

★ 进度反馈: 这是逐图 I/O, 大数据集上是分钟级, 静默等待没有区分度。
"""
from __future__ import annotations

import logging
from typing import Dict, List

from od_platform.common.constants import DETAILS_PREVIEW_LIMIT, IMAGE_DECODE_ERROR_RATIO
from od_platform.common.progress import ProgressReporter
from od_platform.data_validation.registry import (
    CheckContext, CheckResult, CheckSeverity, check,
)

_NAME = "image_decodable"
logger = logging.getLogger(__name__)

try:                                    # pillow 在精简环境里可能缺席
    from PIL import Image as _PILImage
except ImportError:                     # pragma: no cover - 取决于运行环境
    _PILImage = None


@check(_NAME)
def validate_image_decodable(ctx: CheckContext) -> CheckResult:
    snap = ctx.snapshot
    if _PILImage is None:
        return CheckResult(_NAME, CheckSeverity.INFO,
                           "pillow 不可用, 跳过图片完整性校验",
                           {"reason": "pillow_missing"})

    candidates: List[tuple] = [(split, path) for split in snap.splits
                               for path in snap.images_per_split.get(split, ())]
    if not candidates:
        return CheckResult(_NAME, CheckSeverity.INFO, "没有可检查的图像", {})

    bad: List[Dict[str, str]] = []
    progress = ProgressReporter(f"D4 {_NAME}", len(candidates), logger=logger, unit="张图")
    for position, (split, path) in enumerate(candidates, 1):
        try:
            with _PILImage.open(path) as image:
                image.verify()
        except Exception as exc:  # noqa: BLE001 —— 各种损坏形式都算"打不开"
            bad.append({"split": split, "file": path.name, "error": type(exc).__name__})
        progress.update(position)
    progress.finish(done=len(candidates))

    total = len(candidates)
    ratio = len(bad) / total
    details = {
        "total_images": total,
        "n_bad": len(bad),
        "bad_ratio": round(ratio, 4),
        "thresholds": {"error_ratio": IMAGE_DECODE_ERROR_RATIO},
        "bad_preview": bad[:DETAILS_PREVIEW_LIMIT],
    }
    if ratio >= IMAGE_DECODE_ERROR_RATIO:
        return CheckResult(_NAME, CheckSeverity.ERROR,
                           f"{len(bad)}/{total} 张图打不开({ratio:.1%})— 疑似导出/传输损坏",
                           details)
    if bad:
        return CheckResult(_NAME, CheckSeverity.WARNING,
                           f"{len(bad)}/{total} 张图打不开({ratio:.1%})— 少量, 建议修源数据",
                           details)
    return CheckResult(_NAME, CheckSeverity.PASS,
                       f"全部 {total} 张图均可正常打开", details)