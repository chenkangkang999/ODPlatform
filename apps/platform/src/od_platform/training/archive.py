"""训练产物归档。runs/training/<run_id>/ 是过程现场, models/trained/ 是精选成品。"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

from od_platform.common.paths import TRAINED_MODELS_DIR

logger = logging.getLogger(__name__)


def archive_best_weight(save_dir: Path, stem: str) -> Optional[Path]:
    """把现场里的 best.pt 归档到 models/trained/<stem>-best.pt。复制而非移动。"""
    src = save_dir / "weights" / "best.pt"
    if not src.exists():
        logger.warning("现场里没有 best.pt, 跳过归档: %s", src)
        return None
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dst = TRAINED_MODELS_DIR / f"{stem}-best.pt"
    shutil.copy2(src, dst)
    logger.info("权重已归档: %s", dst)
    return dst
