"""日志文件改名桩."""
from __future__ import annotations

import logging
from pathlib import Path


def rename_log_to_save_dir(output_dir: Path, model_stem: str) -> None:
    """把根 logger 的 FileHandler 路径改成 output_dir 下. 桩: 不做实际改名."""
    pass
