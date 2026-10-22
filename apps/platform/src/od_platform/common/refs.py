"""名字 ↔ 路径 的统一解析:CLI/脚本只记名字,落盘位置由 paths.py 定。
规则:ref 含路径分隔符或是绝对路径 → 当路径用;否则当名字,补默认目录和后缀。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from od_platform.common.paths import (
    MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR,
)


def resolve_ref(ref: str, *, base_dir: Path,
                default_suffix: Optional[str] = None) -> Path:
    """把用户给的引用解析成实际路径。"""
    p = Path(ref)
    if p.is_absolute() or len(p.parts) > 1:      # 明确路径:直接解析
        return p.resolve()
    if default_suffix and not p.suffix:          # 纯名字:补后缀
        p = p.with_suffix(default_suffix)
    return (base_dir / p).resolve()


def resolve_dataset(ref: str) -> Path:
    """'demo' → <repo>/data/raw/demo;也可传显式路径。"""
    return resolve_ref(ref, base_dir=RAW_DATA_DIR)


def resolve_yaml(ref: str) -> Path:
    """'demo' → <repo>/data/processed/demo.yaml;也可传显式路径。"""
    return resolve_ref(ref, base_dir=PROCESSED_DATA_DIR, default_suffix=".yaml")


def resolve_pretrained_model(ref: str) -> Path:
    """'yolo11n' → <repo>/models/pretrained/yolo11n.pt。"""
    return resolve_ref(ref, base_dir=MODELS_DIR / "pretrained", default_suffix=".pt")


def resolve_trained_model(ref: str) -> Path:
    """'demo_s40' → <repo>/models/trained/demo_s40.pt。"""
    return resolve_ref(ref, base_dir=MODELS_DIR / "trained", default_suffix=".pt")
