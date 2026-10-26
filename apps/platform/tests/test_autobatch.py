#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""批大小动态估算的测试。

守两件事:
    · 估算规则必须与"实测能跑通的基准点"自洽, 且随 imgsz / 显存单调合理;
    · 显式给的值绝不能被自动规则改写（否则用户固定 batch 就没意义了）。
"""
from __future__ import annotations

import pytest

from od_platform.training.autobatch import (
    BASE_BATCH_AT_BASE_VRAM,
    BASE_IMGSZ,
    BASE_VRAM_GB,
    MIN_BATCH,
    available_vram_gb,
    resolve_batch,
    suggest_batch,
)


def test_baseline_point_reproduces_the_measured_configuration():
    """基准点必须原样复现: 640 / 8GB → 16（这正是实测跑通的那组）。"""
    assert suggest_batch(BASE_IMGSZ, BASE_VRAM_GB) == BASE_BATCH_AT_BASE_VRAM


def test_1280_on_8gb_gives_2_which_is_the_verified_safe_value():
    """本机真实场景: imgsz 1280 + 8GB → 2。

    这是【实测校准过】的值: batch 4 在同一张卡上 OOM 了（free 4.69G 还要再要
    116MB）, 所以基准取半让 1280 落到 2。改小这个数会让 8GB 卡重新 OOM。
    """
    assert suggest_batch(1280, 8.0) == 2


def test_batch_scales_with_imgsz_and_vram():
    assert suggest_batch(640, 16.0) == 16          # 显存翻倍
    assert suggest_batch(1280, 16.0) == 4          # 尺寸翻倍 + 显存翻倍
    assert suggest_batch(320, 8.0) == 32           # 尺寸减半
    # 像素数翻倍(边长), 批大小应约为 1/4
    assert suggest_batch(1280, 8.0) == suggest_batch(640, 8.0) // 4


def test_nominal_vram_slightly_under_the_label_still_hits_the_baseline():
    """显卡报的 total_memory 常略小于标称(实测 7.96GiB)。

    取最近整数才能让 8GB 卡算出基准点 8 / 1280 时算出 2;
    向下截断会得到 7 和 1 —— 白白保守一档。
    """
    assert suggest_batch(640, 7.96) == BASE_BATCH_AT_BASE_VRAM
    assert suggest_batch(1280, 7.96) == 2


def test_never_below_minimum_even_on_a_tiny_card():
    assert suggest_batch(1280, 1.0) == MIN_BATCH
    assert suggest_batch(1536, 2.0) >= MIN_BATCH


def test_unknown_vram_falls_back_to_the_baseline_conservatively():
    """拿不到显存时按 8GB 估 —— 宁可保守, 不能冒险。"""
    assert suggest_batch(1280, None) == suggest_batch(1280, BASE_VRAM_GB)


def test_rejects_nonsense_imgsz():
    with pytest.raises(ValueError):
        suggest_batch(0, 8.0)
    with pytest.raises(ValueError):
        suggest_batch(-640, 8.0)


@pytest.mark.parametrize("explicit", [1, 4, 8, 16, 32, 0.6, 0.9])
def test_explicit_batch_is_never_overridden(explicit):
    """显式值(含 (0,1] 的显存利用率)必须原样透传。"""
    assert resolve_batch(explicit, 1280, "0") == explicit


def test_auto_batch_is_resolved_to_a_concrete_int():
    resolved = resolve_batch(-1, 1280, "cpu")
    assert isinstance(resolved, int) and resolved >= MIN_BATCH


def test_available_vram_returns_none_for_cpu_or_unknown_device():
    assert available_vram_gb("cpu") is None
    assert available_vram_gb("") is None
    assert available_vram_gb("mps") is None
    # 无 CUDA 的环境里也不能抛异常
    assert available_vram_gb("0") is None or available_vram_gb("0") > 0