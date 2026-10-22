"""模型路径解析桩."""
from __future__ import annotations

from pathlib import Path


def resolve_model_path(model: str, search_dirs=None) -> Path:
    """在 search_dirs 里找模型文件; 找不到原样返回(交给 ultralytics 下载)."""
    p = Path(model)
    if p.exists():
        return p
    for d in (search_dirs or []):
        candidate = Path(d) / p.name
        if candidate.exists():
            return candidate
    return p
