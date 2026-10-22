"""物化:按 manifest 把 (图,标签) 对从 raw 复制到 processed/<dataset>/{images,labels}/<split>/。

三原则:
  1. 一律 copy2,绝不 hardlink —— 链接一改源全毁
  2. 删旧只删自己造的叶子目录;删除前 _assert_within 拦围栏(沙盒外/沙盒根/符号链接一律拒绝)
  3. 以 manifest 为准:契约里有什么就落什么,不多不少
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from od_platform.common import paths
from od_platform.common.lineage import SplitManifest

logger = logging.getLogger(__name__)

_SPLITS = ("train", "val", "test")


def _assert_within(sandbox: Path, target: Path) -> None:
    """围栏:target 必须严格位于 sandbox 之内(且不等于 sandbox 本身)。"""
    sb = sandbox.resolve()
    tg = target.resolve()
    if tg == sb:
        raise ValueError(f"拒绝操作沙盒根目录本身: {tg}")
    try:
        tg.relative_to(sb)
    except ValueError:
        raise ValueError(f"拒绝操作沙盒外路径: {tg}(沙盒: {sb})") from None


def _safe_clear(leaf: Path) -> None:
    """清空叶子目录:不存在则建;是符号链接则拒绝;否则整体删除后重建。"""
    if leaf.is_symlink():
        raise ValueError(f"拒绝清空符号链接: {leaf}")
    if leaf.exists():
        shutil.rmtree(leaf)
    leaf.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SplitSourceDirs:
    """raw 侧的图源与标注源。"""
    images: Path
    labels: Path

    def resolve(self, stem: str) -> tuple[Path, Path]:
        """按 stem 找到 (图像文件, 标签文件)。同 stem 多扩展名给 warning;缺文件直接报错。"""
        imgs = sorted(self.images.glob(f"{stem}.*"))
        if not imgs:
            raise FileNotFoundError(f"图像缺失: {self.images}/{stem}.*")
        if len(imgs) > 1:
            logger.warning("同 stem 多扩展名,取第一个: %s(候选: %s)", imgs[0], imgs)
        lbl = self.labels / f"{stem}.txt"
        if not lbl.exists():
            raise FileNotFoundError(f"标签缺失: {lbl}")
        return imgs[0], lbl


@dataclass(frozen=True)
class SplitOutputDirs:
    """processed 侧的成品目录: root/<images|labels>/<split>。"""
    root: Path

    @property
    def leaves(self) -> List[Path]:
        return [self.root / kind / s for kind in ("images", "labels") for s in _SPLITS]

    @classmethod
    def for_dataset(cls, dataset: str) -> "SplitOutputDirs":
        if not dataset or "/" in dataset or "\\" in dataset or dataset.startswith("."):
            raise ValueError(f"非法数据集名: {dataset!r}")
        return cls(paths.dataset_processed_dir(dataset))

    @classmethod
    def under(cls, root: Path) -> "SplitOutputDirs":
        return cls(root)


def _copy_pair(img: Path, lbl: Path, out: SplitOutputDirs, split: str) -> None:
    shutil.copy2(img, out.root / "images" / split / img.name)
    shutil.copy2(lbl, out.root / "labels" / split / lbl.name)


def materialize(manifest: SplitManifest, source: SplitSourceDirs,
                out: SplitOutputDirs) -> Dict[str, int]:
    """按 manifest 物化。返回各 split 的样本数。"""
    for leaf in out.leaves:
        _assert_within(paths.PROCESSED_DATA_DIR, leaf)
        _safe_clear(leaf)
    counts: Dict[str, int] = {}
    for split in _SPLITS:
        stems = manifest.stems_of(split)
        for stem in stems:
            img, lbl = source.resolve(stem)
            _copy_pair(img, lbl, out, split)
        counts[split] = len(stems)
    return counts
