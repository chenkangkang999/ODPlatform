"""内部工具:从 yaml 的 odp_meta 找到并加载 D3 划分契约。下划线开头,不被自动发现。"""
from __future__ import annotations

from typing import Optional, Tuple

from od_platform.common import paths
from od_platform.common.lineage import SplitManifest


def load_manifest_from_yaml(
    yaml_data: dict,
) -> Tuple[Optional[SplitManifest], Optional[str], dict]:
    """返回 (manifest, error, odp_meta);error 非空即失败原因。"""
    odp_meta = yaml_data.get("odp_meta") or {}
    ref = odp_meta.get("manifest_path")
    if not ref:
        return None, "yaml 缺少 odp_meta.manifest_path(不是 odp-transform 产出?)", odp_meta
    mpath = paths.ROOT_DIR / ref
    if not mpath.exists():
        return None, f"manifest 文件不存在: {mpath}", odp_meta
    try:
        return SplitManifest.read(mpath), None, odp_meta
    except Exception as e:
        return None, f"manifest 读取失败: {type(e).__name__}: {e}", odp_meta
