#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""进度反馈组件的测试。

守三件事:
    · 该报的时候报（跨档位 + 满足最小间隔）;
    · 不该报的时候闭嘴（总量太小、间隔太短）—— 否则小数据集会刷一屏;
    · finish() 永远有收尾行（哪怕中途一次没报过）。
"""
from __future__ import annotations

import logging

import pytest

from od_platform.common.progress import (
    MIN_TOTAL_FOR_INTERMEDIATE,
    ProgressReporter,
)


class FakeClock:
    """可控时钟（monotonic 在测试里不可控）。"""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture()
def clock():
    return FakeClock()


def make(name="任务", total=1000, clock=None, **kwargs):
    return ProgressReporter(name, total, clock=clock or FakeClock(), **kwargs)


def test_small_total_does_not_report_intermediately(caplog):
    """总量小于阈值时中间一次都不报（一屏就完事了, 报点是噪音）。"""
    clock = FakeClock()
    progress = make(total=MIN_TOTAL_FOR_INTERMEDIATE - 1, clock=clock)
    with caplog.at_level(logging.INFO):
        for index in range(1, MIN_TOTAL_FOR_INTERMEDIATE):
            clock.advance(0.5)
            progress.update(index)
    assert caplog.text == ""


def test_reports_once_per_step_when_time_allows(caplog):
    """总量够大 + 时间够 = 每跨一个档位报一次（默认 10 档）。"""
    clock = FakeClock()
    progress = make(total=1000, clock=clock)
    with caplog.at_level(logging.INFO):
        for index in range(1, 1001):
            clock.advance(0.05)          # 50 项 ≈ 2.5s, 满足最小间隔
            progress.update(index)
    lines = [r.message for r in caplog.records]
    assert len(lines) == 10, lines           # 10%/20%/.../100%
    assert "1000/1000" in lines[-1]
    assert "(100%)" in lines[-1]


def test_min_interval_suppresses_bursts(caplog):
    """时间没到就闭嘴 —— 否则 1000 项会报 10 次都挤在同一毫秒里。"""
    clock = FakeClock()
    progress = make(total=1000, clock=clock)
    with caplog.at_level(logging.INFO):
        for index in range(1, 1001):
            progress.update(index)       # 时钟不走
    assert caplog.text == ""


def test_skipped_thresholds_do_not_report_every_item(caplog):
    """一次性跳很多项时, 阈值要跟着推进, 不能之后每项都触发。"""
    clock = FakeClock()
    progress = make(total=1000, clock=clock)
    with caplog.at_level(logging.INFO):
        clock.advance(10)
        progress.update(500)             # 直接跳到 50%
        clock.advance(3)
        progress.update(501)             # 500 的档位已过, 不该再报
    assert len(caplog.records) == 1
    assert "500/1000" in caplog.records[0].message


def test_finish_always_reports_even_if_nothing_was_reported(caplog):
    clock = FakeClock()
    progress = make(total=10, clock=clock)
    clock.advance(2.5)
    with caplog.at_level(logging.INFO):
        message = progress.finish()
    assert "完成 10 项" in message
    assert "用时 2.50s" in message
    assert len(caplog.records) == 1


def test_unknown_total_stays_silent_until_finish(caplog):
    """总量未知（0）时不猜百分比, 只在收尾报实际数量。"""
    clock = FakeClock()
    progress = make(total=0, clock=clock)
    with caplog.at_level(logging.INFO):
        for index in range(1, 200):
            assert progress.update(index) is None
        message = progress.finish(done=199)
    assert "完成 199 项" in message
    assert len(caplog.records) == 1


def test_report_includes_rate_and_eta(caplog):
    clock = FakeClock()
    progress = make(total=100, clock=clock, unit="张图")
    with caplog.at_level(logging.INFO):
        for index in range(1, 51):
            clock.advance(0.5)           # 每项 0.5s → 50 项共 25s, 速率 2/s
            progress.update(index)
    message = caplog.records[-1].message
    assert "50/100 张图" in message
    assert "2.0/s" in message
    assert "剩余约 25s" in message


def test_eta_is_unknown_before_any_progress():
    """还没完成任何一项时不该编一个剩余时间出来。"""
    progress = make(total=100)
    assert progress._eta(0) == "--"
    assert "完成 0 项" in progress.finish(done=0)


def test_steps_must_be_positive():
    with pytest.raises(ValueError):
        make(steps=0)
    with pytest.raises(ValueError):
        make(steps=-1)