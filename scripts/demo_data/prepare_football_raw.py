#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : prepare_football_raw.py
# @Project   : ODPlatform
# @Function  : 为足球数据集生成 D3 能吃的扁平 raw 视图（硬链接，零字节复制）

"""把足球数据集接进 D3 流水线（odp-transform）——补上"没有出处"的缺口。

★ 要解决的缺口

    ``odp-train`` 开训前做数据身份核验: 数据集 yaml 必须带 D3(odp-transform)
    产出的 ``odp_meta.manifest_path``, 缺了直接拒绝开训, 而且 ``--skip-validate``
    也救不了（缺 manifest 是硬拦, 发生在核验判定之前, 见 data_validation/identity.py）。
    足球数据集是转换脚本直接产出的, 没有 manifest —— 于是走不了正式训练入口。

★ 为什么需要这一步

    D3 对原料目录有固定契约（见 data_pipeline/orchestrator.py）:

        data/raw/<name>/images/*.jpg     ← 全部图片, 扁平
        data/raw/<name>/labels/*.txt     ← 全部 YOLO 标签, 扁平, 与图片同名
        （COCO/VOC 则放 annotations/; YOLO 格式不看 annotations/）

    而足球数据是"一份数据三种视图"的单副本布局, 且按 train/valid/test 分了目录:

        data/raw/soccer/football-det/data/<split>/*.jpg + *.txt

    D3 会【自己重新划分】(它只管一个图片池 + 一个标签池), 所以三个 split 要先
    合并成一个扁平的池子。本脚本就干这一件事。

★ 单副本原则怎么守住

    用【硬链接】而不是复制: 同一卷上的硬链接与源文件共享同一份字节,
    因此 images/ 那一千多张图**不额外占用任何磁盘**, 却让 D3 看到它要的布局。
    （跨卷时会失败, 那时才需要 --copy。源与目标都在 data/raw/ 下, 正常不会跨卷。）

★ 之后怎么跑（脚本会把这三条打出来）

    odp-transform football --format yolo --classes player football \\
                  --split-strategy stratified_multilabel
    odp-validate apps/platform/configs/datasets/football.yaml
    odp-train --data football --device 0 --experiment-name football_v2

    注意: D3 会按自己的策略【重新划分】(默认 0.8/0.1/0.1, 随机种子固定),
    所以新划分与 Roboflow 原来的 859/243/130 不同 —— 这是有意的:
    划分可复现、有 manifest 记录, 而且可以按类别分层（足球类只占 5.4%,
    分层能保证每个 split 里都有球）。

用法::

    python scripts/demo_data/prepare_football_raw.py --dry-run   # 只看计划
    python scripts/demo_data/prepare_football_raw.py             # 生成硬链接视图
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from od_platform.common import paths

# 数据集名 —— 决定 data/raw/<name>、data/processed/<name>、configs/datasets/<name>.yaml
DATASET_NAME = "football"
SPLITS: Tuple[str, ...] = ("train", "valid", "test")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# D3 的 YOLO 转换器【必须】显式给类别顺序（yolo 文件里没有类名）。
# 顺序即 class_id, 必须与标签里写的编号一致: 我们的转换脚本产出 0=player 1=football。
CLASSES: Tuple[str, ...] = ("player", "football")


def _same_file(a: Path, b: Path) -> bool:
    """a 与 b 是否指向同一份物理数据（硬链接/同一个文件）。"""
    try:
        sa, sb = a.stat(), b.stat()
    except OSError:
        return False
    return (sa.st_dev, sa.st_ino) == (sb.st_dev, sb.st_ino)


def collect_sources(source_root: Path) -> Dict[str, Path]:
    """收集三个 split 下的图片与标签, 并检查文件名是否全局唯一。

    Returns:
        {文件名: 源路径}。图片与标签同名同 stem, 合成一个字典。

    Raises:
        FileNotFoundError: 源目录/文件缺失。
        ValueError: 出现重名 —— 扁平化会互相覆盖, 必须停下。
    """
    if not source_root.is_dir():
        raise FileNotFoundError(f"源数据集目录不存在: {source_root}")

    mapping: Dict[str, Path] = {}
    collisions: Dict[str, List[Path]] = {}
    for split in SPLITS:
        split_dir = source_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"缺少划分目录: {split_dir}")
        for path in sorted(split_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_SUFFIXES and path.suffix.lower() != ".txt":
                continue
            if path.name in mapping:
                collisions.setdefault(path.name, [mapping[path.name]]).append(path)
                continue
            mapping[path.name] = path

    if collisions:
        detail = "; ".join(f"{name}: {[str(p) for p in paths_]}"
                           for name, paths_ in list(collisions.items())[:5])
        raise ValueError(
            f"扁平化后有 {len(collisions)} 个重名文件会互相覆盖, 拒绝继续: {detail}")
    if not mapping:
        raise FileNotFoundError(f"{source_root} 下没有找到任何图片或标签")
    return mapping


def normalize_label(text: str, *, tolerance: float = 0.0) -> Tuple[str, int]:
    """把 YOLO 标签里的框裁进图像范围, 返回 (新内容, 修了几个框)。

    ★ 为什么需要这一步

      D4 质检抓到"92/22878 个框出界(0.4%)" —— 最大只超出 0.5 像素, 是 Roboflow
      导出 COCO 时的浮点舍入残渣。D4 因此把总评判为 WARNING, 而 odp-train 只接受
      PASS（除非显式 --skip-validate）。

      修法就是把框夹回 [0,1]: 越界 0.5px 的框会相应缩小 0.5px, 对训练没有影响,
      却让数据真正合规。源数据不动（raw 应当保持原样）, 标准化发生在这一层。

    Args:
        text: YOLO 标签文件内容（每行: cls cx cy w h）
        tolerance: 允许的越界量; 超过才修（默认 0 = 一点都不许出界）
    """
    lines, fixed = [], 0
    for raw in text.splitlines():
        parts = raw.split()
        if len(parts) != 5:
            lines.append(raw)
            continue
        cls, cx, cy, w, h = parts[0], *(float(v) for v in parts[1:])
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        cx1, cy1 = max(0.0, x1), max(0.0, y1)
        cx2, cy2 = min(1.0, x2), min(1.0, y2)
        # 只有真的超出容差才算修改, 避免把浮点格式差异当成修复
        if (abs(cx1 - x1) > tolerance or abs(cy1 - y1) > tolerance
                or abs(cx2 - x2) > tolerance or abs(cy2 - y2) > tolerance):
            fixed += 1
            w, h = cx2 - cx1, cy2 - cy1
            cx, cy = (cx1 + cx2) / 2, (cy1 + cy2) / 2
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        else:
            lines.append(raw)
    return ("\n".join(lines) + ("\n" if lines else "")), fixed


def link_file(source: Path, target: Path, *, dry_run: bool, copy: bool) -> str:
    """把 source 呈现为 target。返回实际采用的方式。"""
    if target.exists() or target.is_symlink():
        if not target.is_symlink() and _same_file(source, target):
            return "already-linked"
        if target.is_symlink():
            if not dry_run:
                target.unlink()
        else:
            raise FileExistsError(
                f"目标已存在且不是同一份数据, 拒绝覆盖: {target}\n"
                f"（如需重建, 请先手动删除 {target.parent}）")
    if dry_run:
        return "copy" if copy else "hardlink"
    target.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(source, target)
        return "copy"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError as exc:            # 跨卷 (EXDEV) 等
        raise OSError(
            f"无法为 {source.name} 建硬链接（{exc}）。若源与目标不在同一卷, "
            f"请加 --copy（会真正复制一份字节）。") from None


def prepare(source_root: Path, target_root: Path, *, dry_run: bool = False,
            copy: bool = False) -> Dict[str, int]:
    """生成 D3 需要的 data/raw/<name>/{images,labels} 视图。

    图片: 硬链接（零字节复制）。
    标签: 先做越界裁剪; 干净的硬链接, 需要修的写成规范化的新文件（体积极小）。
    """
    mapping = collect_sources(source_root)
    stats = {"images": 0, "labels": 0, "hardlink": 0, "copy": 0,
             "already": 0, "labels_rewritten": 0, "boxes_fixed": 0}

    for name, source in sorted(mapping.items()):
        is_image = source.suffix.lower() in IMAGE_SUFFIXES
        kind = "images" if is_image else "labels"
        stats[kind] += 1
        target = target_root / kind / name

        if not is_image:
            text, fixed = normalize_label(source.read_text(encoding="utf-8"))
            if fixed:
                stats["boxes_fixed"] += fixed
                stats["labels_rewritten"] += 1
                if not dry_run:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    # ★ 必须先摘掉旧条目: 它可能是指向源文件的硬链接, 直接 write
                    #   会【改写源数据】(同一 inode)。unlink 只去掉这个目录项。
                    if target.exists() or target.is_symlink():
                        target.unlink()
                    target.write_text(text, encoding="utf-8")
                continue

        how = link_file(source, target, dry_run=dry_run, copy=copy)
        if how == "hardlink":
            stats["hardlink"] += 1
        elif how == "copy":
            stats["copy"] += 1
        else:
            stats["already"] += 1
    return stats


def next_commands() -> List[str]:
    yaml_path = f"apps/platform/configs/datasets/{DATASET_NAME}.yaml"
    return [
        f"odp-transform --dataset {DATASET_NAME} --format yolo "
        f"--classes {' '.join(CLASSES)} --split-strategy stratified_multilabel",
        f"odp-validate {yaml_path}",
        f"odp-train --data {DATASET_NAME} --device 0 --experiment-name football_v2",
    ]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=paths.FOOTBALL_DATASET_DIR,
                        help="单副本数据集目录（含 train/valid/test）")
    parser.add_argument("--target", type=Path, default=paths.raw_dataset_root(DATASET_NAME),
                        help="D3 原料目录（默认 data/raw/football）")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划, 不改动磁盘")
    parser.add_argument("--copy", action="store_true",
                        help="真正复制字节（仅当源与目标跨卷、无法建硬链接时使用）")
    args = parser.parse_args(argv)

    print(f"源      : {args.source}")
    print(f"D3 原料 : {args.target}")
    print(f"类别顺序: {CLASSES}（必须与标签编号一致）")
    print(f"方式    : {'复制字节' if args.copy else '硬链接（零字节复制）'}"
          f"{'  [DRY-RUN]' if args.dry_run else ''}\n")

    try:
        stats = prepare(args.source, args.target, dry_run=args.dry_run, copy=args.copy)
    except (FileNotFoundError, ValueError, FileExistsError, OSError) as exc:
        print(f"失败: {exc}", file=sys.stderr)
        return 1

    print(f"图片 {stats['images']} 个 / 标签 {stats['labels']} 个")
    print(f"  硬链接 {stats['hardlink']} · 复制 {stats['copy']} · 已存在 {stats['already']}")
    if not args.copy and stats["hardlink"]:
        print("  ⇒ 这些文件与源文件共享同一份字节, 没有多占磁盘。")
    if stats["boxes_fixed"]:
        print(f"  标签越界裁剪: 修了 {stats['boxes_fixed']} 个框, "
              f"重写 {stats['labels_rewritten']} 个标签文件")
        print("  ⇒ 源数据未改动; 只有 D3 视图里的标签被规范化"
              "（D4 的 bbox_within_image 因此才会 PASS）。")

    print("\n接下来依次执行:")
    for command in next_commands():
        print(f"    {command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())