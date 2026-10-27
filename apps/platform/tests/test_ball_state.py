#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""球静止判定（geometric/ball_state.py）的测试。

这一项是"任意球/点球"的前置条件, 也是本次新增里最容易悄悄错掉的一环:
它必须区分【静止 / 在动 / 未知】三种状态 —— 把"样本不足"当成"静止",
球刚进画面那几帧就会凭空变成定位球证据。
"""
from __future__ import annotations

import pytest

from od_platform.geometric.ball_state import BallStationaryTracker
from od_platform.geometric.settings import RuleSettings

BALL = (52.5, 34.0)

# 测试用的显式策略: 窗口 10 帧、至少 4 个样本、位移上限 0.5 米
SETTINGS = RuleSettings(ball_stationary_window_frames=10,
                        ball_stationary_min_samples=4,
                        ball_stationary_max_displacement_m=0.5)


def feed(tracker, positions, *, start=0):
    for offset, position in enumerate(positions):
        tracker.update(position, start + offset)


# ---------------------------------------------------------------- 未知 vs 静止

def test_not_enough_samples_is_unknown_not_stationary():
    """★ 最重要的边界: 样本不足必须返回 None（未知）, 不能顺势当成静止。"""
    tracker = BallStationaryTracker(SETTINGS)
    feed(tracker, [BALL] * 3)                       # 少于 min_samples=4
    status = tracker.status()
    assert status["stationary"] is None
    assert status["samples"] == 3
    assert status["displacement_m"] is None
    assert "样本不足" in status["reason"]
    assert tracker.is_stationary is None


def test_a_still_ball_becomes_stationary_once_it_has_enough_samples():
    tracker = BallStationaryTracker(SETTINGS)
    feed(tracker, [BALL] * 4)
    status = tracker.status()
    assert status["stationary"] is True
    assert status["displacement_m"] == pytest.approx(0.0)


def test_a_jittering_ball_within_the_threshold_is_still_stationary():
    """标定抖动会让球心逐帧小幅抖动 —— 那是噪声, 不是球在跑。"""
    tracker = BallStationaryTracker(SETTINGS)
    jitter = [(BALL[0] + delta, BALL[1]) for delta in (0.0, 0.05, -0.05, 0.1, -0.1)]
    feed(tracker, jitter)
    assert tracker.status()["stationary"] is True


def test_a_rolling_ball_is_not_stationary():
    tracker = BallStationaryTracker(SETTINGS)
    feed(tracker, [(BALL[0] + 0.5 * index, BALL[1]) for index in range(5)])
    status = tracker.status()
    assert status["stationary"] is False
    assert status["displacement_m"] > SETTINGS.ball_stationary_max_displacement_m


def test_returns_in_the_middle_of_the_window_still_count_as_movement():
    """★ 两两最大位移而不是首尾距离: 球出去又回来, 首尾相同但确实动过。"""
    tracker = BallStationaryTracker(SETTINGS)
    feed(tracker, [BALL, (BALL[0] + 3.0, BALL[1]), (BALL[0] + 6.0, BALL[1]),
                   (BALL[0] + 3.0, BALL[1]), BALL])
    status = tracker.status()
    assert status["stationary"] is False
    assert status["displacement_m"] >= 6.0


# ---------------------------------------------------------------- 窗口边界

def test_window_forgets_old_positions():
    """久远的位置必须被窗口淘汰, 否则"很久以前动过"会永久污染判定。

    窗口是 10 帧, 所以要喂够 10 帧以上才会把第 0 帧挤出去。
    """
    tracker = BallStationaryTracker(SETTINGS)
    feed(tracker, [(BALL[0] + 50.0, BALL[1])])       # 第 0 帧跑得很远
    feed(tracker, [BALL] * 12, start=1)              # 之后一直静止, 直到挤掉第 0 帧
    status = tracker.status()
    assert status["stationary"] is True
    assert status["first_frame"] > 0


def test_stale_positions_expire_when_frames_are_skipped():
    """球连续多帧没检出来时, 旧观测不能还被当成"当前静止"。"""
    tracker = BallStationaryTracker(SETTINGS)
    feed(tracker, [BALL] * 4)
    tracker.update(BALL, 1000)                       # 中间跳了 996 帧
    status = tracker.status()
    assert status["samples"] == 1
    assert status["stationary"] is None              # 只剩一个有效观测 → 未知


def test_reset_clears_the_history():
    tracker = BallStationaryTracker(SETTINGS)
    feed(tracker, [BALL] * 6)
    assert tracker.status()["stationary"] is True
    tracker.reset()
    assert tracker.status()["samples"] == 0
    assert tracker.status()["stationary"] is None


# ---------------------------------------------------------------- 阈值来自配置

def test_thresholds_come_from_settings_not_from_code():
    """同一串位置, 换一套阈值结论就变 —— 证明没有硬编码。"""
    positions = [BALL] + [(BALL[0] + 0.4 * index, BALL[1]) for index in range(1, 5)]

    strict = BallStationaryTracker(RuleSettings(
        ball_stationary_window_frames=10, ball_stationary_min_samples=4,
        ball_stationary_max_displacement_m=0.1))
    relaxed = BallStationaryTracker(RuleSettings(
        ball_stationary_window_frames=10, ball_stationary_min_samples=4,
        ball_stationary_max_displacement_m=5.0))
    feed(strict, positions)
    feed(relaxed, positions)
    assert strict.status()["stationary"] is False
    assert relaxed.status()["stationary"] is True


def test_min_samples_greater_than_window_is_rejected_by_settings():
    with pytest.raises(ValueError, match="ball_stationary_min_samples"):
        RuleSettings(ball_stationary_window_frames=3, ball_stationary_min_samples=5)