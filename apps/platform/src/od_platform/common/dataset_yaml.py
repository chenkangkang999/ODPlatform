"""数据集 yaml 的路径约定 —— 让签入 git 的 yaml 保持可移植, 不绑定某台机器的盘符。

约定(写与读两端都要遵守):

* **写**: ``path`` 一律写【相对仓库根】的 POSIX 路径(如 ``data/processed/demo``)。
  数据在仓库之外(用户自带数据集)时才退回绝对路径, 并记一条 warning 说明原因。
* **读**: 绝对路径原样使用; 相对路径按此顺序解析 ——
  1. 相对 yaml 所在目录(兼容外部/ultralytics 风格 yaml, 它们的相对路径本意如此);
  2. 相对仓库根(ODPlatform 自己的约定, 见 ``paths.ROOT_DIR``)。

为什么要独立一个模块: ultralytics 的规则与此不同 —— 它把相对 ``path`` 解析成
``DATASETS_DIR / path``(见 ``ultralytics/data/utils.py: check_det_dataset``), 那是它自己的
全局配置目录。所以喂给 ultralytics 之前必须先把 ``path`` 落成绝对路径,
用 :func:`write_resolved_copy` 生成一份"运行期副本", 源 yaml 保持可移植。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from od_platform.common.paths import ROOT_DIR

logger = logging.getLogger(__name__)


def to_repo_relative(path: Path) -> str:
    """把数据集根目录写成【相对仓库根】的 POSIX 路径; 在仓库外则原样返回绝对路径。

    仓库内的写法让 yaml 可以安全签入 git —— 换台机器/换个盘符 clone 下来依然可用。
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        logger.warning(
            "数据集根目录在仓库之外, yaml 只能写绝对路径(该文件不可移植): %s", resolved)
        return resolved.as_posix()


def resolve_dataset_root(yaml_path: Path, raw: str | None) -> Path | None:
    """解析 yaml 里的 ``path`` 字段。``raw`` 为空返回 None。"""
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute():
        return p
    by_yaml_dir = (Path(yaml_path).parent / p).resolve()
    if by_yaml_dir.exists():
        return by_yaml_dir
    # 仓库根相对是 ODPlatform 自己的约定; 目录不存在时也走这条(由调用方报"根目录不存在")
    return (ROOT_DIR / p).resolve()


def load_dataset_yaml(yaml_path: Path) -> dict[str, Any]:
    """读数据集 yaml, 顶层必须是 mapping。"""
    data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"数据集 yaml 顶层不是 mapping: {yaml_path}")
    return data


def write_resolved_copy(source_yaml: Path, dest_yaml: Path) -> Path:
    """把 ``source_yaml`` 的 ``path`` 解析成绝对路径后另存一份, 返回副本路径。

    给只认绝对路径(或按自己 DATASETS_DIR 解析相对路径)的消费方用, 目前是 ultralytics。
    源 yaml 保持可移植; 这份副本只活在运行现场目录(``runs/`` 下, 不进 git)。
    """
    source_yaml = Path(source_yaml)
    data = load_dataset_yaml(source_yaml)
    root = resolve_dataset_root(source_yaml, data.get("path"))
    if root is None:
        logger.warning("数据集 yaml 没有 path 字段, 无法解析副本: %s", source_yaml)
    else:
        data["path"] = root.as_posix()

    dest_yaml = Path(dest_yaml)
    dest_yaml.parent.mkdir(parents=True, exist_ok=True)
    dest_yaml.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    logger.info("数据集解析副本已生成: %s (path=%s)", dest_yaml, data.get("path"))
    return dest_yaml
