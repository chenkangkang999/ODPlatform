#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : autobatch.py
# @Project   : ODPlatform
# @Function  : 按显存与输入尺寸估算批大小（替代 ultralytics AutoBatch 的 2× 探测）

"""批大小的动态估算。

★ 为什么不用 ultralytics 的 ``batch=-1`` (AutoBatch)

    本机实测（RTX 4060 Laptop / 8 GB, imgsz=1280）:

        AutoBatch: Computing optimal batch size for imgsz=2560 at 60.0% GPU memory utilization
        CUDA error: out of memory

    它会按 **2×imgsz** 去探测显存（1280 → 2560），8 GB 的卡在探测阶段就爆，
    而且爆在 ``empty_cache`` 里直接抛异常、连"退化成默认 batch"都做不到。
    大显存机器上它很好用; 消费级显卡上它不可用。

★ 本模块的规则（可解释、可复现、按机器自适应）

    以"实测能跑通的组合"为基准点, 按像素数与显存线性缩放:

        基准:  imgsz 640 + batch 16 在 8 GB 上训练正常（历史 3 轮训练即此配置）
        规则:  batch = base_batch × (base_imgsz / imgsz)² × (显存 / base_vram)

    激活占用近似随 batch × 像素数 增长, 所以这个缩放是对的量纲。
    例: 1280 / 8 GB → 8 × 0.25 × 1 = 2;  1280 / 16 GB → 4;  640 / 8 GB → 8。

★ 边界处理
    · 拿不到显存信息（无 CUDA / CPU 训练 / 查询失败）→ 按基准显存 8 GB 估,
      宁保守不冒险;
    · 显式给了 batch（正数或 (0,1] 的利用率）→ 原样返回, 不做任何干预。
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 基准点: 这三个数来自实测, 而且是【保守值】
#   实测事实(同一张 8GB 卡): imgsz 640 + batch 16 能正常训练,
#   但 imgsz 1280 + batch 4 就 OOM 了（free 4.69G 时还要再要 116MB）。
#   说明"按像素数线性缩放"在 1280 上偏乐观 —— 检测头/特征图开销不是纯线性。
#   所以基准取半, 让 1280/8GB 算出 2（实测能稳的值）。
BASE_IMGSZ: int = 640          # 历史训练用的输入尺寸
BASE_BATCH_AT_BASE_VRAM: int = 8     # 该尺寸下 8 GB 显存【保守】可训练的批大小
BASE_VRAM_GB: float = 8.0      # 基准显存
MIN_BATCH: int = 1             # 再小的卡也别低于 1

BYTES_PER_GB: float = 1024 ** 3


def available_vram_gb(device: Any = None) -> Optional[float]:
    """返回指定 GPU 的总显存(GB); 无 CUDA 或查询失败返回 None。

    Args:
        device: "0" / "0,1" / "cpu" / None。多卡时只看第一张。
    """
    if isinstance(device, str) and device.strip().lower() in ("", "cpu", "mps"):
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        index = 0
        if isinstance(device, str) and device.strip():
            first = device.split(",")[0].strip()
            if first.lstrip("-").isdigit():
                index = int(first)
        return float(torch.cuda.get_device_properties(index).total_memory) / BYTES_PER_GB
    except Exception as exc:  # noqa: BLE001 —— 估不出来就退回保守假设, 不该让训练挂掉
        logger.debug("读取显存失败, 按基准显存估计: %s", exc)
        return None


def suggest_batch(imgsz: int, vram_gb: Optional[float] = None,
                  *, base_imgsz: int = BASE_IMGSZ,
                  base_batch: int = BASE_BATCH_AT_BASE_VRAM,
                  base_vram_gb: float = BASE_VRAM_GB) -> int:
    """按输入尺寸与显存估算一个能跑通的批大小。"""
    if imgsz <= 0:
        raise ValueError(f"imgsz 必须为正数, 收到 {imgsz}")
    vram = base_vram_gb if vram_gb is None else max(vram_gb, 0.0)
    scale = (base_imgsz / imgsz) ** 2 * (vram / base_vram_gb)
    # 取最近整数而不是向下截断: 显卡报的 total_memory 常略小于标称值
    # （实测 8GB 的卡报 7.96GiB）, 向下截断会把基准点 16 算成 15 —— 白白保守。
    return max(MIN_BATCH, int(round(base_batch * scale)))


def resolve_batch(batch: Any, imgsz: int, device: Any = None) -> Any:
    """把"自动"(-1)解析成具体批大小; 显式值原样返回。

    Returns:
        显式 batch（正数 / (0,1] 的显存利用率）→ 原样；
        batch == -1 → 由 :func:`suggest_batch` 算出的整数。
    """
    if batch != -1:
        return batch
    vram = available_vram_gb(device)
    resolved = suggest_batch(int(imgsz), vram)
    logger.info("batch=-1(自动): imgsz=%s, 显存=%s → batch=%s", imgsz,
                f"{vram:.1f} GB" if vram else f"未知(按 {BASE_VRAM_GB:.0f} GB 估)", resolved)
    return resolved