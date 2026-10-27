#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : runner.py
# @Project   : ODPlatform
# @Function  : 落地执行 —— 把计划变成 data/raw/<name>/{images,labels} + 体检报告

"""落地执行。

★★ 关于硬链接与"写前 unlink" —— 今天真实踩过的坑

    用硬链接拍平能**零字节复制**, 但代价是: 目标文件与源文件**是同一个 inode**。
    如果接着往目标路径写内容（例如把越界框裁回 [0,1] 后写回标签）, ``write_text``
    会**就地截断那个 inode** —— **源文件被一起改掉, 且不可恢复**。
    ``data/raw/`` 是本仓库明令"只读, 绝不就地改"的区域, 出了这种事就等于
    把原始数据毁了。

    所以本模块的铁律只有两条:
      · **内容需要改动** → 走 :func:`_write_text_fresh`（先 unlink 再写新文件）;
      · **内容不需要改动** → 才允许硬链接。
    代码里**不存在**"先硬链接、再往它写"的路径。

★ 另外两件必须做对的事

    · **越界框裁剪**: Roboflow 导出的 YOLO 归一化坐标带浮点舍入残渣, 会让框
      微微超出 [0,1]（实测 92/22878 个框越界 0.5px）。D4 的 ``bbox_within_image``
      会因此把总评拉成 WARNING, 而 WARNING 不允许开训 —— 所以在接入时裁干净。
    · **源文件指纹**: 落地前后各算一次源标签的内容哈希并比对, 把"源没被改"
      这件事变成**可验证的结论**写进报告, 而不是只靠"设计上应该没改"。

★ 重名冲突若选择 prefix 策略: 用来源目录的短摘要做后缀, 保证**改写后仍然唯一**,
  并且标签名跟着图片名一起改, 不让配对关系断掉。
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from od_platform.common.progress import ProgressReporter
from od_platform.data_pipeline.ingest.scanner import (
    IngestPlan,
    SourceSample,
    YOLO_FIELD_COUNT,
    parse_label_line,
)
from od_platform.data_pipeline.ingest.settings import (
    CONFLICT_PREFIX,
    IngestSettings,
)

logger = logging.getLogger(__name__)

LINK_MODE_HARDLINK = "hardlink"
LINK_MODE_COPY = "copy"
HASH_ALGORITHM = "sha1"
PREFIX_DIGEST_LENGTH = 8
LABEL_FILE_SUFFIX = "\n"


@dataclass
class IngestResult:
    """落地结果（写进报告, 也用来给人看）。"""

    target: Path
    linked: int = 0                  # 用硬链接落地的文件数
    copied: int = 0                  # 回退复制的文件数
    labels_written: int = 0          # 因裁剪而【新写】的标签（不是链接）
    labels_linked: int = 0
    clamped_boxes: int = 0           # 实际裁过的框
    dropped_boxes: int = 0           # 裁完退化的框（丢弃）
    source_guard: Dict[str, Any] = field(default_factory=dict)
    report_path: Optional[Path] = None

    @property
    def total_placed(self) -> int:
        return self.linked + self.copied

    def to_dict(self) -> Dict[str, Any]:
        return {
            "linked": self.linked,
            "copied": self.copied,
            "labels_written": self.labels_written,
            "labels_linked": self.labels_linked,
            "clamped_boxes": self.clamped_boxes,
            "dropped_boxes": self.dropped_boxes,
            "source_guard": self.source_guard,
        }

    def render_lines(self) -> List[str]:
        mode = []
        if self.linked:
            mode.append(f"硬链接 {self.linked}")
        if self.copied:
            mode.append(f"复制 {self.copied}")
        lines = [f"文件落地: {' + '.join(mode) if mode else '无'}"
                 f"（标签: 链接 {self.labels_linked} / 因裁剪新写 {self.labels_written}）"]
        if self.clamped_boxes or self.dropped_boxes:
            lines.append(f"越界框: 裁回 {self.clamped_boxes} 个, 丢弃退化 {self.dropped_boxes} 个")
        guard = self.source_guard
        if guard:
            changed = guard.get("changed", 0)
            mark = "✅" if changed == 0 else "❌"
            lines.append(f"{mark} 源文件未被改动: 校验 {guard.get('checked', 0)} 个标签, "
                         f"内容变化 {changed} 个")
        return lines


# ============================================================
# 底层文件操作
# ============================================================

def _write_text_fresh(path: Path, text: str) -> None:
    """写一个**全新**的文件: 先 unlink 再写。

    ★ 为什么必须先 unlink: 目标路径可能已经存在, 而且**可能是指向源文件的硬链接**
      （同一个 inode）。``write_text`` 会就地截断 —— 那等于改写源数据。
      本函数的语义是"这个路径要变成一个由我创建的新文件", 所以先摘掉旧目录项,
      再创建新 inode。这样源文件永远不受影响。
    """
    if path.exists() or path.is_symlink():
        path.unlink()
    path.write_text(text, encoding="utf-8")


def place_file(source_file: Path, target_file: Path, settings: IngestSettings) -> str:
    """把源文件"放到"目标路径: 硬链接优先, 失败自动回退复制。

    Returns:
        :data:`LINK_MODE_HARDLINK` 或 :data:`LINK_MODE_COPY`

    注意本函数**只搬运、不改内容**。任何需要改内容的场景都必须走
    :func:`_write_text_fresh`, 绝不能"先链接再写"。
    """
    target_file.parent.mkdir(parents=True, exist_ok=True)
    if target_file.exists() or target_file.is_symlink():
        # 已存在的目标一律先摘掉: 可能是上次运行留下的, 甚至是指向源文件的硬链接。
        # 摘掉目录项只是让"这个名字"不再指向旧 inode, 不会动旧 inode 里的数据。
        target_file.unlink()
    if settings.prefer_hardlink:
        try:
            os.link(source_file, target_file)
            return LINK_MODE_HARDLINK
        except OSError as exc:      # 跨卷 / 文件系统不支持 / 权限不足
            logger.debug("硬链接失败, 回退复制: %s -> %s (%s)",
                         source_file, target_file, exc)
    shutil.copy2(source_file, target_file)
    return LINK_MODE_COPY


# ============================================================
# 越界框裁剪
# ============================================================

def format_box(class_id: int, cx: float, cy: float, w: float, h: float,
               settings: IngestSettings) -> Optional[str]:
    """把一行 YOLO 框裁回 [0,1] 并格式化; 退化框返回 None。

    ★ 为什么先夹【四条边】再反推中心与宽高
      YOLO 存的是 ``(cx, cy, w, h)``; 越界时**只夹中心是错的** —— 一个中心在
      0.5、宽度 0.9 的框, 中心没越界, 而两条边早就飞到画面外了。

    ★ 为什么舍入不会再造出新的越界
      夹完的边都在 [0,1] 内, 而把 [0,1] 内的数舍入到 ``coordinate_precision``
      位小数**结果仍在 [0,1]**（0.9999999 → 1.000000 就是上限, 不可能舍成
      1.000001）。再由舍入后的边反推 cx/cy/w/h 并同样舍入一次, 累积误差不超过
      0.5×10^-precision, 小于 D4 的越界容差 ``box_edge_tolerance``。
    """
    precision = settings.coordinate_precision
    left = max(0.0, min(1.0, cx - w / 2.0))
    right = max(0.0, min(1.0, cx + w / 2.0))
    top = max(0.0, min(1.0, cy - h / 2.0))
    bottom = max(0.0, min(1.0, cy + h / 2.0))
    new_w, new_h = right - left, bottom - top
    if new_w <= 0.0 or new_h <= 0.0:
        return None                     # 退化框: 丢掉比留一个零面积框好
    new_cx, new_cy = (left + right) / 2.0, (top + bottom) / 2.0
    return (f"{class_id} {new_cx:.{precision}f} {new_cy:.{precision}f} "
            f"{new_w:.{precision}f} {new_h:.{precision}f}")


def clamp_label_text(text: str, settings: IngestSettings) -> Tuple[str, int, int, int]:
    """裁一份标签的全文。

    Returns:
        ``(裁剪后的文本, 实际裁过的框数, 丢弃的退化框数, 格式异常行数)``

    格式异常的行**原样保留** —— 本命令负责的是"裁回边界", 不是"修标签"。
    悄悄修掉坏行会让 D4 的 label_format 永远发现不了问题。
    """
    out: List[str] = []
    clamped = dropped = malformed = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parsed = parse_label_line(stripped)
        if parsed is None:
            malformed += 1
            out.append(stripped)          # 原样保留: 修标签不是本命令的职责
            continue
        class_id, cx, cy, w, h = parsed
        formatted = format_box(class_id, cx, cy, w, h, settings)
        if formatted is None:
            dropped += 1
            continue
        edges_before = (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
        edges_after = (max(0.0, min(1.0, edges_before[0])), max(0.0, min(1.0, edges_before[1])),
                       max(0.0, min(1.0, edges_before[2])), max(0.0, min(1.0, edges_before[3])))
        if edges_after != edges_before or len(stripped.split()) != YOLO_FIELD_COUNT:
            clamped += 1
        out.append(formatted)
    body = "\n".join(out)
    if body:
        body += LABEL_FILE_SUFFIX
    return body, clamped, dropped, malformed


# ============================================================
# 命名与源指纹
# ============================================================

def _prefix_digest(source_dir: Path) -> str:
    """由来源目录算一个稳定短摘要（用于重名加前缀）。"""
    return hashlib.new(HASH_ALGORITHM, str(source_dir).encode("utf-8")).hexdigest()[
        :PREFIX_DIGEST_LENGTH]


def resolve_name(name: str, source_dir: Path, conflicting: Sequence[str],
                 settings: IngestSettings) -> str:
    """拍平后的最终文件名; 若该名字冲突且策略是 prefix, 加来源摘要后缀。

    摘要由**来源目录**决定, 所以同一批文件重跑会得到同样的名字（可复现）,
    且不同来源目录不会撞车。
    """
    if settings.conflict_policy != CONFLICT_PREFIX or name not in set(conflicting):
        return name
    path = Path(name)
    return f"{path.stem}__{_prefix_digest(source_dir)}{path.suffix}"


def _label_name_for(image_name: str, settings: IngestSettings) -> str:
    """标签名跟着（可能已改名的）图片名走, 保证配对关系不断。"""
    return Path(image_name).with_suffix(settings.layouts.label_suffix).name


def snapshot_source_labels(samples: Sequence[SourceSample]) -> Dict[str, str]:
    """给源标签算内容哈希 —— 用来事后证明"源没被改动"。

    只算标签不算图片: 图片只可能被硬链接（从不写入）, 而标签是唯一可能被
    "裁剪后写回"的文件。对超大图片集全量哈希代价太高, 收益也不对应。
    """
    snapshot: Dict[str, str] = {}
    for sample in samples:
        if sample.label is None:
            continue
        try:
            digest = hashlib.new(HASH_ALGORITHM, sample.label.read_bytes()).hexdigest()
        except OSError:
            continue
        snapshot[str(sample.label)] = digest
    return snapshot


def verify_sources_untouched(snapshot: Dict[str, str]) -> Dict[str, Any]:
    """复查源标签内容有没有变; 返回可写进报告的结论。"""
    changed: List[str] = []
    missing: List[str] = []
    for path, digest in snapshot.items():
        try:
            now = hashlib.new(HASH_ALGORITHM, Path(path).read_bytes()).hexdigest()
        except OSError:
            missing.append(path)
            continue
        if now != digest:
            changed.append(path)
    return {
        "checked": len(snapshot),
        "changed": len(changed),
        "changed_preview": changed[:10],
        "missing": len(missing),
        "algorithm": HASH_ALGORITHM,
    }


# ============================================================
# 执行
# ============================================================

def apply_plan(plan: IngestPlan, settings: IngestSettings, *,
               force: bool = False) -> IngestResult:
    """执行计划: 建目标目录、落地图片与标签、裁框、写报告。

    Args:
        force: 目标目录已存在且非空时是否允许继续（默认拒绝）

    Raises:
        FileExistsError: 目标已存在且非空, 且没给 ``force``
    """
    target = Path(plan.target)
    if target.exists() and any(target.iterdir()) and not force:
        raise FileExistsError(
            f"目标目录已存在且非空: {target}\n"
            "  接入会往里写新数据, 为避免与既有数据集混在一起, 默认拒绝。\n"
            "  确认要覆盖就加 --force（注意: 不会删除既有文件, 只覆盖同名文件）")

    images_dir = target / settings.layouts.images_dirname
    labels_dir = target / settings.layouts.labels_dirname
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    conflicting = [c.name for c in plan.conflicts if c.kind == "image"]
    result = IngestResult(target=target)
    before = snapshot_source_labels(plan.samples)

    total_steps = max(1, len(plan.samples) * 2)
    progress = ProgressReporter("接入文件", total_steps, logger=logger, unit="个文件")
    done = 0
    for sample in plan.samples:
        image_name = resolve_name(sample.image.name, sample.image.parent,
                                  conflicting, settings)
        method = place_file(sample.image, images_dir / image_name, settings)
        if method == LINK_MODE_HARDLINK:
            result.linked += 1
        else:
            result.copied += 1
        done += 1
        progress.update(done)

        if sample.label is None:
            # 没有标签的图: 只放图。缺失会在 D4 的 pair_existence 里体现,
            # 这里不伪造一个空标签（伪造等于替 D4 撒谎）。
            done += 1
            progress.update(done)
            continue

        label_target = labels_dir / _label_name_for(image_name, settings)
        try:
            text = sample.label.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("标签读不出来, 跳过: %s (%s)", sample.label, exc)
            done += 1
            progress.update(done)
            continue

        if settings.clamp_boxes:
            clamped_text, clamped, dropped, _malformed = clamp_label_text(text, settings)
            if clamped or dropped:
                # ★ 内容变了 → 必须新写, 绝不能硬链接后改写（那会毁掉源文件）
                _write_text_fresh(label_target, clamped_text)
                result.labels_written += 1
                result.clamped_boxes += clamped
                result.dropped_boxes += dropped
                done += 1
                progress.update(done)
                continue
        # 内容没变 → 硬链接（零字节）
        method = place_file(sample.label, label_target, settings)
        result.labels_linked += 1
        if method == LINK_MODE_HARDLINK:
            result.linked += 1
        else:
            result.copied += 1
        done += 1
        progress.update(done)
    progress.finish(done=done)

    result.source_guard = verify_sources_untouched(before)
    result.report_path = write_report(plan, result, settings)
    return result


def write_report(plan: IngestPlan, result: IngestResult,
                 settings: IngestSettings) -> Path:
    """把体检报告写到目标目录里。"""
    import json

    payload: Dict[str, Any] = {
        "dataset_name": plan.name,
        "source": str(plan.source),
        "target": str(result.target),
        "layout": plan.layout,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "class_names": plan.class_names,
        "class_source": plan.class_source,
        "samples": len(plan.samples),
        "splits": plan.splits,
        "labels": plan.label_stats.to_dict(),
        "images": plan.image_health.to_dict(settings.small_image_side_px),
        "conflicts": [c.to_dict() for c in plan.conflicts],
        "result": result.to_dict(),
        "settings": plan.settings_audit,
    }
    path = Path(result.target) / settings.report_filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("体检报告已写入: %s", path)
    return path