#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : scanner.py
# @Project   : ODPlatform
# @Function  : 布局探测与体检 —— 把"各种原始布局"扫成一份可执行的接入计划

"""布局探测 + 体检。

★ 支持的布局（自动探测, 不写死某一个）

    flat         ``<src>/images/*`` + ``<src>/labels/*``
    presplit     ``<src>/<split>/images/*`` + ``<src>/<split>/labels/*``
                 （split ∈ train/valid/val/test —— Roboflow 导出就是这种）
    legacy       ``<src>/data/<split>/*.jpg`` + 同名 ``*.txt``
    images_only  只有图片没有标签 —— **明确拒绝**, 不静默产出空数据集

★ 为什么体检必须在"计划阶段"就做完

    ``odp-ingest`` 默认先预演。如果体检放到真落地时才做, 用户看到的预演
    就是"扫到 372 个文件"这种没有信息量的东西 —— 而真正该在动手前知道的是:
    **有多少框越界了、有多少标签是空的、有多少图根本解不开、分辨率够不够**。
    这些正是 D4 质检会报警告的根因, 提前暴露比事后排查便宜得多。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from od_platform.common.progress import ProgressReporter
from od_platform.data_pipeline.ingest.settings import IngestSettings, LayoutSettings

logger = logging.getLogger(__name__)

# 布局名（具名常量: 报告、日志、测试都引用它们, 不散落字符串）
LAYOUT_FLAT = "flat"
LAYOUT_PRESPLIT = "presplit"
LAYOUT_LEGACY = "legacy"
LAYOUT_IMAGES_ONLY = "images_only"
LAYOUT_UNKNOWN = "unknown"

LAYOUT_DESCRIPTIONS: Dict[str, str] = {
    LAYOUT_FLAT: "扁平: images/ + labels/",
    LAYOUT_PRESPLIT: "已分集合: <split>/images/ + <split>/labels/",
    LAYOUT_LEGACY: "旧布局: data/<split>/*.jpg + 同名 .txt",
    LAYOUT_IMAGES_ONLY: "只有图片、没有标签",
    LAYOUT_UNKNOWN: "无法识别",
}

# 一条合法 YOLO 标签行的字段数: class_id cx cy w h
YOLO_FIELD_COUNT = 5


class IngestError(Exception):
    """接入过程中"必须停下来让人处理"的错误（源不合法、冲突、覆盖风险…）。"""


@dataclass(frozen=True)
class SourceSample:
    """一个待接入的样本: 一张图 + 它的标签文件（可能没有）。"""

    image: Path
    label: Optional[Path]
    split: str            # 来源集合（未分集合时为空串）
    relative: str         # 相对源根的展示路径, 报错时能指回去

    @property
    def has_label(self) -> bool:
        return self.label is not None


@dataclass(frozen=True)
class Conflict:
    """拍平后出现的重名冲突（静默覆盖就是事故, 所以要显式拦下来）。"""

    kind: str                        # image | label
    name: str
    sources: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "sources": list(self.sources)}


@dataclass
class LabelStats:
    """标签体检结果。"""

    total_files: int = 0
    total_boxes: int = 0
    empty_label_files: int = 0          # 空标签 = 负样本, 合法但要如实报
    malformed_lines: int = 0            # 字段数不对/非数字
    out_of_range_boxes: int = 0         # 裁回 [0,1] 会改动多少个框
    dropped_boxes: int = 0              # 裁完边长 <= 0 的退化框（会被丢弃）
    images_without_label: int = 0
    boxes_per_class: Dict[int, int] = field(default_factory=dict)
    images_per_class: Dict[int, int] = field(default_factory=dict)
    malformed_preview: List[str] = field(default_factory=list)
    out_of_range_preview: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label_files": self.total_files,
            "total_boxes": self.total_boxes,
            "empty_label_files": self.empty_label_files,
            "malformed_lines": self.malformed_lines,
            "out_of_range_boxes": self.out_of_range_boxes,
            "dropped_boxes": self.dropped_boxes,
            "images_without_label": self.images_without_label,
            "boxes_per_class": {str(k): v for k, v in sorted(self.boxes_per_class.items())},
            "images_per_class": {str(k): v for k, v in sorted(self.images_per_class.items())},
            "malformed_preview": self.malformed_preview,
            "out_of_range_preview": self.out_of_range_preview,
        }


@dataclass
class ImageHealth:
    """图片体检结果。"""

    probed: int = 0
    undecodable: int = 0
    small: int = 0                      # 最小边 < small_image_side_px 的张数
    min_sides: List[int] = field(default_factory=list)
    undecodable_preview: List[str] = field(default_factory=list)

    @property
    def median_min_side(self) -> Optional[int]:
        if not self.min_sides:
            return None
        ordered = sorted(self.min_sides)
        return ordered[len(ordered) // 2]

    def to_dict(self, small_threshold: int) -> Dict[str, Any]:
        return {
            "probed": self.probed,
            "undecodable": self.undecodable,
            "undecodable_preview": self.undecodable_preview,
            "small_image_count": self.small,
            "small_image_side_px": small_threshold,
            "median_min_side_px": self.median_min_side,
            "min_min_side_px": min(self.min_sides) if self.min_sides else None,
            "max_min_side_px": max(self.min_sides) if self.min_sides else None,
        }


@dataclass
class IngestPlan:
    """接入计划（还没动盘）。"""

    name: str
    source: Path
    target: Path
    layout: str
    samples: List[SourceSample] = field(default_factory=list)
    class_names: Optional[List[str]] = None
    class_source: str = "未找到"
    conflicts: List[Conflict] = field(default_factory=list)
    label_stats: LabelStats = field(default_factory=LabelStats)
    image_health: ImageHealth = field(default_factory=ImageHealth)
    settings_audit: Dict[str, Any] = field(default_factory=dict)

    @property
    def splits(self) -> Dict[str, int]:
        """来源集合分布（未分集合的记为 ``(未分)``）。"""
        counts: Dict[str, int] = {}
        for sample in self.samples:
            counts[sample.split or "(未分)"] = counts.get(sample.split or "(未分)", 0) + 1
        return counts

    def class_name_of(self, class_id: int) -> str:
        if self.class_names and 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"<class_id={class_id}>"

    def render_lines(self) -> List[str]:
        """给人看的计划摘要（CLI 直接打这些）。"""
        lines = [
            f"源目录:   {self.source}",
            f"目标目录: {self.target}",
            f"探测布局: {self.layout}（{LAYOUT_DESCRIPTIONS.get(self.layout, '?')}）",
            f"类名:     {self.class_names if self.class_names else '未找到'}  ← {self.class_source}",
            f"样本:     {len(self.samples)} 个  " +
            "  ".join(f"{k}={v}" for k, v in sorted(self.splits.items())),
        ]
        stats = self.label_stats
        lines.append(
            f"标签:     {stats.total_files} 个文件 / {stats.total_boxes} 个框"
            f"（空标签 {stats.empty_label_files}, 格式异常行 {stats.malformed_lines}, "
            f"越界框 {stats.out_of_range_boxes}, 退化框 {stats.dropped_boxes}）")
        if stats.boxes_per_class:
            boxes = "  ".join(
                f"{self.class_name_of(cid)}={n}"
                for cid, n in sorted(stats.boxes_per_class.items()))
            lines.append(f"每类框数: {boxes}")
        if stats.images_per_class:
            images = "  ".join(
                f"{self.class_name_of(cid)}={n}"
                for cid, n in sorted(stats.images_per_class.items()))
            lines.append(f"每类图片: {images}")
        health = self.image_health
        if health.probed:
            lines.append(
                f"图片体检: 探针 {health.probed} 张, 解不开 {health.undecodable} 张, "
                f"最小边中位数 {health.median_min_side}px, "
                f"过小(<{self.settings_audit.get('small_image_side_px')}px) {health.small} 张")
        if stats.images_without_label:
            lines.append(f"⚠ 有 {stats.images_without_label} 张图没有对应标签"
                         f"（D4 的 pair_existence 会报警告）")
        if self.conflicts:
            lines.append(f"⚠ 重名冲突 {len(self.conflicts)} 处:")
            for conflict in self.conflicts[:10]:
                lines.append(f"    [{conflict.kind}] {conflict.name} ← {len(conflict.sources)} 处来源")
        return lines


# ============================================================
# 布局探测
# ============================================================

def _is_dir(path: Path) -> bool:
    return path.is_dir()


def detect_layout(source: Path, layouts: LayoutSettings) -> str:
    """探测源目录属于哪种布局。探测顺序: 最具体的先判。"""
    images_dir = source / layouts.images_dirname
    labels_dir = source / layouts.labels_dirname
    if _is_dir(images_dir) and _is_dir(labels_dir):
        return LAYOUT_FLAT
    for split in layouts.split_dirnames:
        split_dir = source / split
        if _is_dir(split_dir / layouts.images_dirname) and _is_dir(
                split_dir / layouts.labels_dirname):
            return LAYOUT_PRESPLIT
    nested = source / layouts.nested_dirname
    if nested.is_dir():
        for split in layouts.split_dirnames:
            if _is_dir(nested / split):
                return LAYOUT_LEGACY
    if _is_dir(images_dir):
        return LAYOUT_IMAGES_ONLY
    # 更宽松的兜底: 源里直接散着图片, 但一个标签都没有
    for suffix in layouts.normalized_suffixes():
        if any(source.rglob(f"*{suffix}")):
            return LAYOUT_IMAGES_ONLY
    return LAYOUT_UNKNOWN


# ============================================================
# 采样
# ============================================================

def _samples_from_split_dirs(images_dir: Path, labels_dir: Path, split: str,
                             layouts: LayoutSettings) -> List[SourceSample]:
    """``images/`` + ``labels/`` 这种成对布局; 标签与图片**相对路径同名**。"""
    samples: List[SourceSample] = []
    suffixes = layouts.normalized_suffixes()
    for image in sorted(images_dir.rglob("*")):
        if not image.is_file() or image.suffix.lower() not in suffixes:
            continue
        relative = image.relative_to(images_dir)
        label = labels_dir / relative.with_suffix(layouts.label_suffix)
        samples.append(SourceSample(
            image=image, label=label if label.is_file() else None,
            split=split, relative=str(image)))
    return samples


def _samples_from_flat_files(directory: Path, split: str,
                             layouts: LayoutSettings) -> List[SourceSample]:
    """旧布局: 图片与同名 ``.txt`` 同一个目录里。"""
    samples: List[SourceSample] = []
    suffixes = layouts.normalized_suffixes()
    for image in sorted(directory.rglob("*")):
        if not image.is_file() or image.suffix.lower() not in suffixes:
            continue
        label = image.with_suffix(layouts.label_suffix)
        samples.append(SourceSample(
            image=image, label=label if label.is_file() else None,
            split=split, relative=str(image)))
    return samples


def collect_samples(source: Path, layouts: LayoutSettings, layout: str) -> List[SourceSample]:
    """按探测到的布局收集样本。"""
    samples: List[SourceSample] = []
    if layout == LAYOUT_FLAT:
        samples = _samples_from_split_dirs(source / layouts.images_dirname,
                                          source / layouts.labels_dirname, "", layouts)
    elif layout == LAYOUT_PRESPLIT:
        for split in layouts.split_dirnames:
            split_dir = source / split
            if _is_dir(split_dir / layouts.images_dirname):
                samples += _samples_from_split_dirs(
                    split_dir / layouts.images_dirname,
                    split_dir / layouts.labels_dirname, split, layouts)
    elif layout == LAYOUT_LEGACY:
        nested = source / layouts.nested_dirname
        for split in layouts.split_dirnames:
            if _is_dir(nested / split):
                samples += _samples_from_flat_files(nested / split, split, layouts)
    elif layout == LAYOUT_IMAGES_ONLY:
        # 只有图没标签: 只收集图片, 后面在 scan() 里明确拒绝
        suffixes = layouts.normalized_suffixes()
        for image in sorted(source.rglob("*")):
            if image.is_file() and image.suffix.lower() in suffixes:
                samples.append(SourceSample(image=image, label=None, split="",
                                            relative=str(image)))
    return samples


def find_conflicts(samples: Sequence[SourceSample]) -> List[Conflict]:
    """检测**拍平后**的重名冲突。

    ★ 为什么必须拦: 拍平意味着 "train/a.jpg" 与 "test/a.jpg" 会落到同一个目标名。
      静默覆盖的结果是**数据集少了一批样本而不报错** —— 这种错误在训练完看指标时
      才发现, 排查成本极高。所以默认直接拒绝。
    """
    image_sources: Dict[str, List[str]] = {}
    label_sources: Dict[str, List[str]] = {}
    for sample in samples:
        image_sources.setdefault(sample.image.name, []).append(sample.relative)
        if sample.label is not None:
            # 用【标签自己的名字】当键 —— 用图片名的话, 标签冲突会顶着图片名报出来,
            # 排查时按那个名字去找标签会扑空。
            label_sources.setdefault(sample.label.name, []).append(str(sample.label))

    conflicts: List[Conflict] = []
    for name, sources in sorted(image_sources.items()):
        # 同一个文件被列两次不算冲突（去重后仍>1 才是真冲突）
        unique = sorted(set(sources))
        if len(unique) > 1:
            conflicts.append(Conflict("image", name, tuple(unique)))
    for name, sources in sorted(label_sources.items()):
        unique = sorted(set(sources))
        if len(unique) > 1:
            conflicts.append(Conflict("label", name, tuple(unique)))
    return conflicts


# ============================================================
# 类名
# ============================================================

def find_class_names(source: Path,
                     layouts: LayoutSettings) -> Tuple[Optional[List[str]], str]:
    """找类名 —— 顺序即 class_id, 必须原样保留。

    Returns:
        ``(names, 来源说明)``; 找不到时 names 为 None, 说明里写清找过哪些地方。
        **绝不编一个类名出来** —— 编出来的类名会让整份数据集静默错位。
    """
    import yaml

    candidates = [source / name for name in layouts.class_file_names]
    candidates += sorted(p for p in source.glob("*.yaml") if p.is_file())
    candidates += sorted(p for p in source.glob("*.yml") if p.is_file())
    seen: List[str] = []
    for path in candidates:
        if not path.is_file() or str(path) in seen:
            continue
        seen.append(str(path))
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, ValueError) as exc:
            logger.warning("类名文件读不出来, 跳过 %s: %s", path, exc)
            continue
        names = data.get("names") if isinstance(data, dict) else None
        if isinstance(names, dict):                    # {0: 'a', 1: 'b'} 这种也认
            names = [names[key] for key in sorted(names, key=lambda k: int(k))]
        if isinstance(names, list) and names:
            return [str(n) for n in names], f"{path.name} 的 names"
    return None, f"未找到类名（找过: {', '.join(Path(p).name for p in seen) or '无 yaml'}）"


# ============================================================
# 标签体检
# ============================================================

def parse_label_line(line: str) -> Optional[Tuple[int, float, float, float, float]]:
    """解析一行 YOLO 标签; 不合法返回 None（不抛异常 —— 坏行要能被统计出来）。"""
    parts = line.split()
    if len(parts) != YOLO_FIELD_COUNT:
        return None
    try:
        return (int(float(parts[0])), float(parts[1]), float(parts[2]),
                float(parts[3]), float(parts[4]))
    except (TypeError, ValueError):
        return None


def box_out_of_range(cx: float, cy: float, w: float, h: float, *,
                     tolerance: float) -> bool:
    """框是否超出 [0,1]（含容差）—— 与 D4 ``bbox_within_image`` 同一口径。

    注意判据用的是**四条边**而不是中心点加宽高: 一个中心在画面内、但宽度
    0.9 的框, 两边早就飞出去了。
    """
    left, right = cx - w / 2.0, cx + w / 2.0
    top, bottom = cy - h / 2.0, cy + h / 2.0
    return (left < -tolerance or top < -tolerance
            or right > 1.0 + tolerance or bottom > 1.0 + tolerance)


def scan_labels(samples: Sequence[SourceSample], settings: IngestSettings, *,
                classes: Optional[Sequence[str]] = None) -> LabelStats:
    """逐个标签文件做体检（字段/框数/每类/越界）。

    ★ 发现"标签里的 class_id 超出你给的类别列表"时**直接报错**:
      这是"标签和类别表对不上"的典型事故, 静默放过会让整份数据集的类别错位,
      而错位的类别在训练指标上**看不出来**（只是所有类都变差一点）。
    """
    stats = LabelStats()
    labelled = [s for s in samples if s.has_label]
    stats.images_without_label = len(samples) - len(labelled)
    stats.total_files = len(labelled)

    progress = ProgressReporter("读取标签", len(labelled), logger=logger, unit="个标签")
    for position, sample in enumerate(labelled, 1):
        assert sample.label is not None
        try:
            text = sample.label.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            stats.malformed_lines += 1
            if len(stats.malformed_preview) < settings.preview_limit:
                stats.malformed_preview.append(f"{sample.label.name}: 读不出来 ({exc})")
            progress.update(position)
            continue

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            stats.empty_label_files += 1
        classes_in_image: set = set()
        for line in lines:
            parsed = parse_label_line(line)
            if parsed is None:
                stats.malformed_lines += 1
                if len(stats.malformed_preview) < settings.preview_limit:
                    stats.malformed_preview.append(f"{sample.label.name}: {line[:60]}")
                continue
            class_id, cx, cy, w, h = parsed
            if classes is not None and not 0 <= class_id < len(classes):
                raise IngestError(
                    f"标签里的 class_id 超出类别表: {sample.label} 出现 class_id={class_id}, "
                    f"但类别表只有 {len(classes)} 类（{list(classes)}）—— "
                    "类别顺序即 class_id, 对不上会让整份数据集静默错位")
            stats.total_boxes += 1
            stats.boxes_per_class[class_id] = stats.boxes_per_class.get(class_id, 0) + 1
            classes_in_image.add(class_id)
            if box_out_of_range(cx, cy, w, h, tolerance=settings.box_edge_tolerance):
                stats.out_of_range_boxes += 1
                if len(stats.out_of_range_preview) < settings.preview_limit:
                    stats.out_of_range_preview.append(
                        f"{sample.label.name}: {class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            if w <= 0 or h <= 0:
                stats.dropped_boxes += 1
        for class_id in classes_in_image:
            stats.images_per_class[class_id] = stats.images_per_class.get(class_id, 0) + 1
        progress.update(position)
    progress.finish(done=len(labelled))
    return stats


# ============================================================
# 图片体检
# ============================================================

def probe_images(samples: Sequence[SourceSample], settings: IngestSettings) -> ImageHealth:
    """解码探针 + 分辨率统计。

    ★ 用 PIL 而不是 ``cv2.imread``: OpenCV 在 Windows 上读不了非 ASCII 路径,
      而且是**静默返回 None** —— 那会把"路径有中文"误报成"图片损坏"。
      PIL 没有这个问题（这一点在真实数据上实测过）。
    """
    health = ImageHealth()
    if not settings.probe_images:
        return health
    from PIL import Image

    progress = ProgressReporter("图片体检", len(samples), logger=logger, unit="张图")
    for position, sample in enumerate(samples, 1):
        health.probed += 1
        try:
            with Image.open(sample.image) as image:
                size = image.size
                # verify() 会读完文件头与所有数据块；代价远低于整图解码。
                # 注意 PIL 规定 verify() 之后该对象不可再用 —— 但 size 已经拿到了。
                image.verify()
        except Exception as exc:  # noqa: BLE001 —— 各种损坏形式都算"解不开"
            health.undecodable += 1
            if len(health.undecodable_preview) < settings.preview_limit:
                health.undecodable_preview.append(
                    f"{sample.relative}: {type(exc).__name__}")
            progress.update(position)
            continue
        min_side = int(min(size))
        health.min_sides.append(min_side)
        if min_side < settings.small_image_side_px:
            health.small += 1
        progress.update(position)
    progress.finish(done=len(samples))
    return health


# ============================================================
# 计划
# ============================================================

def scan(source: Path, target: Path, name: str, settings: IngestSettings, *,
         classes: Optional[Sequence[str]] = None) -> IngestPlan:
    """端到端扫描: 探测布局 → 采样 → 查冲突 → 体检 → 出计划。

    Raises:
        IngestError: 源不合法（不存在/布局不认/只有图没标签/有重名冲突/类别对不上）
    """
    source = Path(source)
    if not source.is_dir():
        raise IngestError(f"源目录不存在: {source}")

    layout = detect_layout(source, settings.layouts)
    if layout == LAYOUT_UNKNOWN:
        raise IngestError(
            f"认不出源目录的布局: {source}\n"
            f"  支持: {', '.join(f'{k} = {v}' for k, v in LAYOUT_DESCRIPTIONS.items() if k != LAYOUT_UNKNOWN)}\n"
            "  如果是一种新布局, 在 configs/runtime/ingest.yaml 的 layouts 节里加规则即可")

    samples = collect_samples(source, settings.layouts, layout)
    if not samples:
        raise IngestError(f"源目录里一张图片都没扫到: {source}（布局判定为 {layout}）")
    if layout == LAYOUT_IMAGES_ONLY:
        raise IngestError(
            f"源目录里只有图片、没有标签（{len(samples)} 张图）—— 检测训练需要框。\n"
            "  odp-transform 只吃带框的数据集; 若这是图像分类数据, 它走的是另一条通道,\n"
            "  不能用本命令产出检测数据集（静默产出一个空数据集比报错更糟）。")

    conflicts = find_conflicts(samples)
    if conflicts and settings.aborts_on_conflict:
        detail = "\n".join(
            f"    [{c.kind}] {c.name} ← {len(c.sources)} 处: {c.sources[:3]}"
            for c in conflicts[:10])
        raise IngestError(
            f"拍平后出现 {len(conflicts)} 处重名冲突, 已中止（默认不静默覆盖）:\n{detail}\n"
            "  处理办法: 加 --conflict-policy prefix 给重名文件加来源前缀, "
            "或先把源目录里的重名文件改掉")

    class_names, class_source = find_class_names(source, settings.layouts)
    if classes:
        class_names, class_source = [str(c) for c in classes], "命令行 --classes"

    stats = scan_labels(samples, settings, classes=class_names)
    health = probe_images(samples, settings)

    return IngestPlan(
        name=name, source=source, target=Path(target), layout=layout,
        samples=list(samples), class_names=class_names, class_source=class_source,
        conflicts=conflicts, label_stats=stats, image_health=health,
        settings_audit=settings.to_audit())