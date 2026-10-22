"""数据集快照:把 yaml + 磁盘状态装进一个只读数据结构,供所有 check 共享。
装数据的，不下结论 —— snapshot 只负责"如实呈现",判断是 check 的事。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from od_platform.common.constants import IMAGE_EXTENSIONS, Task
from od_platform.data_validation.labels import parse_label_line

logger = logging.getLogger(__name__)

_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SplitStats:
    """单个 split 的有界聚合统计(内存占用与样本数无关)。"""
    image_count: int = 0
    annotated_count: int = 0                        # 有 ≥1 个有效框的图数
    instance_count: int = 0                         # 有效框总数
    class_instances: Dict[int, int] = field(default_factory=dict)


@dataclass
class DatasetSnapshot:
    """一次质检运行共享的数据集快照。"""
    yaml_path: Path
    yaml_data: Dict = field(default_factory=dict)
    yaml_load_error: Optional[str] = None
    data_root: Optional[Path] = None
    nc: Optional[int] = None
    class_names: List[str] = field(default_factory=list)
    task_type: str = Task.DETECT
    images_per_split: Dict[str, Tuple[Path, ...]] = field(default_factory=dict)
    labels_per_split: Dict[str, Tuple[Path, ...]] = field(default_factory=dict)        # 期望存在
    label_files_per_split: Dict[str, Tuple[Path, ...]] = field(default_factory=dict)   # 磁盘实有
    stats_per_split: Dict[str, SplitStats] = field(default_factory=dict)
    scan_warnings: List[str] = field(default_factory=list)

    @property
    def splits(self) -> List[str]:
        return sorted(self.images_per_split)

    @property
    def total_images(self) -> int:
        return sum(len(v) for v in self.images_per_split.values())


# ---------------------------------------------------------------- 内部装配

def _load_yaml(yaml_path: Path) -> Tuple[Dict, Optional[str]]:
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {}, f"{type(e).__name__}: {e}"
    if not isinstance(data, dict):
        return {}, f"yaml 顶层不是 mapping: {type(data).__name__}"
    return data, None


def _resolve_data_root(yaml_path: Path, data: Dict) -> Optional[Path]:
    raw = data.get("path")
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_absolute() else (yaml_path.parent / p).resolve()


def _list_images(split_dir: Path) -> Tuple[Path, ...]:
    exts = {e.lower() for e in IMAGE_EXTENSIONS} | {e.upper() for e in IMAGE_EXTENSIONS}
    return tuple(sorted(p for p in split_dir.iterdir() if p.suffix in exts))


def _label_path_for_image(img: Path) -> Path:
    """.../images/train/x.jpg → .../labels/train/x.txt(替换最后一段 images)。"""
    parts = list(img.parts)
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def _list_label_files(data_root: Path, split: str) -> Tuple[Path, ...]:
    d = data_root / "labels" / split
    return tuple(sorted(d.glob("*.txt"))) if d.exists() else ()


def _normalize_names(names) -> List[str]:
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    if isinstance(names, list):
        return list(names)
    return []


def _build_stats(snap: DatasetSnapshot) -> None:
    """内容层:扫一遍期望标签路径,聚合 SplitStats。读取失败跳过(结构问题归 label_format)。"""
    nc = snap.nc
    for split, label_paths in snap.labels_per_split.items():
        n_annotated = n_inst = 0
        class_inst: Dict[int, int] = {}
        for lbl in label_paths:
            if not lbl.exists():
                continue
            try:
                text = lbl.read_text(encoding="utf-8")
            except OSError:
                continue
            n_valid = 0
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                box, kind = parse_label_line(line, snap.task_type)
                if box is None:
                    continue
                n_valid += 1
                cid = box[0]
                if isinstance(nc, int) and 0 <= cid < nc:
                    class_inst[cid] = class_inst.get(cid, 0) + 1
            if n_valid:
                n_annotated += 1
                n_inst += n_valid
        snap.stats_per_split[split] = SplitStats(
            image_count=len(snap.images_per_split.get(split, ())),
            annotated_count=n_annotated,
            instance_count=n_inst,
            class_instances=class_inst)


def build_snapshot(yaml_path: Path, task_type: Optional[str] = None) -> DatasetSnapshot:
    yaml_path = Path(yaml_path)
    snap = DatasetSnapshot(yaml_path=yaml_path)
    data, err = _load_yaml(yaml_path)
    snap.yaml_data, snap.yaml_load_error = data, err
    if err:
        return snap                                     # yaml 都读不了,后续 check 各自早返回

    snap.data_root = _resolve_data_root(yaml_path, data)
    nc = data.get("nc")
    snap.nc = nc if isinstance(nc, int) else None
    snap.class_names = _normalize_names(data.get("names"))

    odp_meta = data.get("odp_meta") or {}
    resolved = task_type or odp_meta.get("task") or Task.DETECT
    if task_type and odp_meta.get("task") and task_type != odp_meta["task"]:
        logger.warning("task_type=%s 与 yaml odp_meta.task=%s 不一致,以参数为准",
                       task_type, odp_meta["task"])
    snap.task_type = resolved

    if snap.data_root is None or not snap.data_root.exists():
        if snap.data_root is not None:
            snap.scan_warnings.append(f"split 根目录不存在: {snap.data_root}")
        return snap

    for split in _SPLITS:
        if split not in data:
            continue
        img_dir = snap.data_root / data[split]
        if not img_dir.exists():
            snap.scan_warnings.append(f"split '{split}' 的图像目录不存在: {img_dir}")
            snap.images_per_split[split] = ()
            snap.labels_per_split[split] = ()
            snap.label_files_per_split[split] = ()
            continue
        images = _list_images(img_dir)
        snap.images_per_split[split] = images
        snap.labels_per_split[split] = tuple(_label_path_for_image(p) for p in images)
        snap.label_files_per_split[split] = _list_label_files(snap.data_root, split)

    _build_stats(snap)
    return snap
