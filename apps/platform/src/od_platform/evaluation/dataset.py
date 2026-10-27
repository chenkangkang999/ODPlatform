#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : dataset.py
# @Project   : ODPlatform
# @Function  : 评估数据集视图（解析路径 + 硬链接搭视图 + 数真值）

"""评估数据集视图 —— 把数据集整理成 ultralytics 能吃的形态, 且**不污染原始数据**。

★ 为什么不能把 data.yaml 直接交给 ultralytics（三个真实问题）

    1. **原始数据是只读区**: ``data/raw`` 在本仓库里是只读的。而 ultralytics 在 val
       时会往数据集目录写一个 ``labels.cache`` —— 直接指过去, 就等于"跑一次评估
       改一次原始数据"。这跟仓库"raw 绝不就地改"的规矩直接冲突。
    2. **Roboflow 导出的相对路径常常是坏的**: 典型是 ``train: ../train/images``,
       而 data.yaml 就放在数据集根目录, 于是字面解析指向 ``data/raw/train/images``,
       那个目录根本不存在。这不是数据坏, 是它假设 yaml 被放在子目录里。
    3. **val-only 也必须给 ``train:``**: ultralytics 强制要求 train/val 两个键都在,
       缺一个直接 ``SyntaxError: 'train:' key missing``。

    所以本模块做三件事:
        解析出**真实的** split 目录 → 用**硬链接**在 ``runs/`` 下搭一个视图
        （零额外字节; 跨卷才退化为复制）→ 写一份只用相对路径的规范化 yaml。

★ 视图里"数真值"也是在这里做的

    类别不平衡警告需要"每类真值框数", 而 ``DetMetrics`` 并没有暴露它
    （``nt_per_class`` 在 8.4.x 里不存在）。真值就在磁盘上, 自己数一遍最直接,
    也不必去读框架的私有字段。
"""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from od_platform.common.progress import ProgressReporter

logger = logging.getLogger(__name__)

# 认得的图片后缀（与 web/demo 两端保持一致, 免得同一批图在两处被认得不一样）
IMAGE_SUFFIXES: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

VIEW_IMAGES_DIRNAME = "images"
VIEW_LABELS_DIRNAME = "labels"


@dataclass(frozen=True)
class DatasetView:
    """一次评估实际使用的数据集视图（可完整复现"这次评的是什么"）。"""

    root: Path
    yaml_path: Path
    images_dir: Path
    labels_dir: Path
    split: str
    class_names: Tuple[str, ...]
    n_images: int
    instances_per_class: Tuple[int, ...]
    source_yaml: Path
    source_split_dir: Path
    resolution_notes: Tuple[str, ...] = ()
    copied_files: int = 0
    limited_to: int = 0

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def total_instances(self) -> int:
        return int(sum(self.instances_per_class))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_yaml": str(self.source_yaml),
            "source_split_dir": str(self.source_split_dir),
            "view_root": str(self.root),
            "view_yaml": str(self.yaml_path),
            "split": self.split,
            "class_names": list(self.class_names),
            "n_images": self.n_images,
            "n_classes": self.n_classes,
            "total_instances": self.total_instances,
            "instances_per_class": {
                # 长度理应一致（都来自同一个 split 的统计）。这里 strict=False 是
                # 为了万一不一致时**报告仍能生成**, 而不是让一份指标报告因为配平
                # 问题直接崩掉 —— 数真值本来就是为了"如实呈现"。
                name: count for name, count in zip(self.class_names,
                                                   self.instances_per_class,
                                                   strict=False)
            },
            # 记录"相对路径是怎么找到的" —— 路径没解析对时, 这行注释能省掉半小时排查
            "resolution_notes": list(self.resolution_notes),
            # >0 说明没能用硬链接（跨卷）, 视图额外占了这么多文件的磁盘
            "copied_files": self.copied_files,
            # >0 说明只评了前 N 张, 指标不代表整个 split —— 报告里必须标注
            "limited_to": self.limited_to,
        }


# ---------------------------------------------------------------- yaml 读取

def load_dataset_yaml(path: Path) -> Dict[str, Any]:
    """读数据集 yaml; 顶层不是 mapping 时当场报错, 不静默用空配置往下跑。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"数据集 yaml 不存在: {path}")
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} 顶层不是 mapping: {type(data).__name__}")
    return data


def class_names_from(data: Dict[str, Any]) -> Tuple[str, ...]:
    """取出类名。

    支持两种写法, 因为两种都真实存在:
        · ``names: ['ball', 'player']``            —— Roboflow 导出用 list;
        · ``names: {0: ball, 1: player}``          —— 本仓库 configs/datasets/*.yaml 用 dict。
    只认一种的话, 换一份数据就得改代码。
    """
    names = data.get("names")
    if isinstance(names, dict):
        try:
            ordered = [names[key] for key in sorted(names, key=lambda k: int(k))]
        except (TypeError, ValueError):
            ordered = [value for _, value in sorted(names.items())]
        return tuple(str(n) for n in ordered)
    if isinstance(names, (list, tuple)):
        return tuple(str(n) for n in names)
    return ()


def reconcile_class_count(data: Dict[str, Any], names: Sequence[str]) -> int:
    """对齐 ``nc`` 与 ``names`` 的长度; 两者都有且不一致时当场报错。

    这类不一致最坑的地方是**不报错**: 后面每类指标会整体错位一格, 而数字看着很正常。
    """
    declared = data.get("nc")
    if declared is None:
        if not names:
            raise ValueError("数据集 yaml 里既没有 names 也没有 nc, 无法确定类别数")
        return len(names)
    declared = int(declared)
    if names and len(names) != declared:
        raise ValueError(f"数据集 yaml 自相矛盾: nc={declared} 但 names 有 {len(names)} 个 "
                         f"({list(names)})")
    return declared


# ---------------------------------------------------------------- 路径解析

def _candidate_forms(raw: str) -> List[str]:
    """把 ``../train/images`` 这类写法展开成若干候选相对路径。

    Roboflow 的习惯是"yaml 放根目录、路径却写成 ``../xxx``" —— 照字面解析会跑到
    数据集外面去。所以除了原样, 还要试"逐层剥掉前导 ``../``"的形式。
    """
    forms = [raw]
    stripped = raw
    while stripped.startswith("../") or stripped.startswith("..\\"):
        stripped = stripped[3:]
        if stripped and stripped not in forms:
            forms.append(stripped)
    return forms


def resolve_split_dir(raw: Any, *, bases: Sequence[Path]) -> Tuple[Optional[Path], List[str]]:
    """把 yaml 里的 split 值解析成真实目录。

    Args:
        raw: yaml 里 ``train`` / ``val`` / ``test`` 的值
        bases: 依次尝试的基准目录（数据集根、yaml 所在目录、其父目录…）

    Returns:
        (目录 或 None, 说明)。说明无论成败都会返回 —— 解析失败时它就是排查依据。
    """
    notes: List[str] = []
    if raw in (None, ""):
        notes.append("yaml 里没有这个 split 的路径")
        return None, notes

    text = str(raw).strip()
    as_path = Path(text)
    candidates: List[Tuple[Path, str]] = []
    if as_path.is_absolute():
        candidates.append((as_path, "绝对路径"))
    else:
        for form in _candidate_forms(text):
            for base in bases:
                candidates.append((Path(base) / form, f"{base}/ + {form}"))

    for candidate, how in candidates:
        try:
            if candidate.is_dir():
                resolved = candidate.resolve()
                notes.append(f"split 目录解析为 {resolved}（{how}）")
                return resolved, notes
        except OSError:                       # 路径非法/无权限: 继续试下一个候选
            continue

    notes.append(f"split 路径 {text!r} 在所有候选位置都不存在: "
                 + "; ".join(str(c) for c, _ in candidates[:6]))
    return None, notes


def candidate_bases(yaml_dir: Path, declared_root: Any) -> List[Path]:
    """组装解析 split 时依次尝试的基准目录。

    顺序有意义: yaml 里显式写的 ``path`` 优先级最高（那是 ultralytics 的语义）,
    其次是 yaml 所在目录, 最后是它的父目录。
    """
    bases: List[Path] = []
    if declared_root:
        root = Path(str(declared_root))
        if not root.is_absolute():
            root = yaml_dir / root
        bases.append(root)
    bases.append(yaml_dir)
    if yaml_dir.parent != yaml_dir:
        bases.append(yaml_dir.parent)
    # 去重但保持顺序
    unique: List[Path] = []
    for base in bases:
        if base not in unique:
            unique.append(base)
    return unique


# ---------------------------------------------------------------- 真值统计

def count_instances(labels_dir: Path, n_classes: int) -> Tuple[int, ...]:
    """数每一类的真值框数（从标签文件直接数, 不依赖框架内部字段）。

    读不了的文件跳过而不抛 —— 一个坏标签不该让整场评估失败; 真正的坏标签
    该由 D4 质检去抓（那是它的职责）。
    """
    counts = [0] * max(n_classes, 0)
    for label in sorted(Path(labels_dir).glob("*.txt")):
        try:
            text = label.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("标签读不了, 记为 0: %s (%s)", label, exc)
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                index = int(float(line.split()[0]))
            except (ValueError, IndexError):
                continue
            if 0 <= index < len(counts):
                counts[index] += 1
    return tuple(counts)


# ---------------------------------------------------------------- 搭视图

def _place(source: Path, target: Path, *, use_hardlinks: bool) -> bool:
    """把文件放进视图; 返回 True 表示发生了**真实复制**（而非硬链接）。

    硬链接零额外字节, 但要求同卷同文件系统; 跨卷时退化为复制, 并让上层知道
    多占了空间 —— 这是"静默降级"和"如实降级"的区别。
    """
    if target.exists():
        return False
    if use_hardlinks:
        try:
            os.link(source, target)
            return False
        except OSError as exc:
            logger.debug("硬链接失败, 退化为复制: %s (%s)", source, exc)
    shutil.copy2(source, target)
    return True


def list_split_images(images_dir: Path) -> List[Path]:
    """列出 split 里的图片（按名字排序, 保证 ``--limit`` 的结果可复现）。"""
    return sorted(p for p in Path(images_dir).iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def build_dataset_view(data_yaml: Path, *, split: str, view_root: Path,
                       limit: int = 0, use_hardlinks: bool = True) -> DatasetView:
    """搭一个只读安全的数据集视图, 返回它的全部事实。

    Args:
        data_yaml: 数据集 yaml（Roboflow 导出或本仓库生成的都行）
        split: val / test / train
        view_root: 视图落盘位置（通常放在本次评估的 run 目录下）
        limit: 只取前 N 张（0=全部）。**只用于链路冒烟**, 会让指标失去代表性。
        use_hardlinks: 优先用硬链接（零额外字节）

    Raises:
        FileNotFoundError: yaml 不在, 或 split 目录解析不出来（附全部候选路径）
    """
    data_yaml = Path(data_yaml)
    data = load_dataset_yaml(data_yaml)
    yaml_dir = data_yaml.parent

    names = class_names_from(data)
    n_classes = reconcile_class_count(data, names)
    if not names:
        names = tuple(f"class_{i}" for i in range(n_classes))

    bases = candidate_bases(yaml_dir, data.get("path"))
    images_dir, notes = resolve_split_dir(data.get(split), bases=bases)
    if images_dir is None:
        raise FileNotFoundError(
            f"数据集 yaml {data_yaml} 里的 {split!r} 路径解析不出来。\n"
            + "\n".join(notes)
            + "\n（Roboflow 导出的 ../xxx 与「yaml 在子目录」的假设有关, 请确认目录结构）")

    labels_dir = images_dir.parent / VIEW_LABELS_DIRNAME
    image_files = list_split_images(images_dir)
    if limit:
        image_files = image_files[:limit]
    if not image_files:
        raise FileNotFoundError(f"split 目录里一张图都没有: {images_dir}")

    view_root = Path(view_root)
    view_images = view_root / VIEW_IMAGES_DIRNAME
    view_labels = view_root / VIEW_LABELS_DIRNAME
    view_images.mkdir(parents=True, exist_ok=True)
    view_labels.mkdir(parents=True, exist_ok=True)

    copied = 0
    # 视图可能上千张图, 静默复制会让人以为卡住了 —— 接进度反馈（与 D3/D4 同一套）
    progress = ProgressReporter("搭建数据集视图", len(image_files),
                                logger=logger, unit="张图")
    for index, image in enumerate(image_files, 1):
        copied += int(_place(image, view_images / image.name, use_hardlinks=use_hardlinks))
        label = labels_dir / f"{image.stem}.txt"
        if label.is_file():
            copied += int(_place(label, view_labels / label.name, use_hardlinks=use_hardlinks))
        progress.update(index)
    progress.finish(done=len(image_files))

    instances = count_instances(view_labels, n_classes)

    # 规范化 yaml: 只用相对路径, 且 train/val 都给（ultralytics 强制要求）。
    # 请求的是 test 时再额外写一个 test 键 —— 三个键都指向同一个视图目录。
    view_yaml = view_root / "data.yaml"
    lines = [
        "# 由 odp-eval 自动生成的数据集视图（不要手改）。",
        f"# 来源: {data_yaml}",
        f"# {split} 目录实际解析为: {images_dir}",
        "# 以下路径都相对本文件所在目录 —— 这样视图可以整体搬走。",
        f"train: {VIEW_IMAGES_DIRNAME}",
        f"val: {VIEW_IMAGES_DIRNAME}",
    ]
    if split == "test":
        lines.append(f"test: {VIEW_IMAGES_DIRNAME}")
    lines += [f"nc: {n_classes}", "names:"]
    lines += [f"  {i}: {name}" for i, name in enumerate(names)]
    view_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")

    view = DatasetView(
        root=view_root, yaml_path=view_yaml, images_dir=view_images,
        labels_dir=view_labels, split=split, class_names=names,
        n_images=len(image_files), instances_per_class=instances,
        source_yaml=data_yaml, source_split_dir=images_dir,
        resolution_notes=tuple(notes), copied_files=copied,
        limited_to=len(image_files) if limit else 0)
    logger.info("数据集视图就绪: %s（%d 张图 / %d 个真值框%s）",
                view.yaml_path, view.n_images, view.total_instances,
                f", 复制了 {copied} 个文件（未能硬链接）" if copied else "，硬链接零额外占用")
    return view
