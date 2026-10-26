#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""tracking 子系统回归测试。

守住的语义（每一条都对应历史上真实坏掉的行为）:
    · ID 跨帧稳定（历史版本用 hash(cls_name) 当 ID, 同队所有球员共用一个 ID）
    · 低分检测能靠第二阶段关联找回（ByteTrack 的核心思想）
    · 遮挡在 max_age 内能恢复同一 ID, 超过则换新 ID
    · 不同类别（球 / 球员）永不被关联到一起
    · 小目标（足球）靠中心距离兜底, 不会每帧换 ID
    · 非法输入被跳过而不是抛异常; 帧号必须单调不减
"""
import pytest

from od_platform.tracking import MultiObjectTracker, Track

FAST = {"backend": "botsort", "min_confidence": 0.1, "high_threshold": 0.5,
        "iou_threshold": 0.3, "max_age": 5, "min_hits": 2}


def _player(x, y=200, size=60):
    return {"cls_name": "player", "bbox": [x, y, x + size, y + size * 2], "conf": 0.9}


def _ball(x, y=400):
    return {"cls_name": "ball", "bbox": [x, y, x + 16, y + 16], "conf": 0.8}


def test_ids_are_stable_across_frames():
    """两个球员朝相反方向走 8 帧, ID 必须一直是 1 和 2。"""
    tracker = MultiObjectTracker(FAST)
    for frame in range(8):
        tracker.update([_player(100 + 10 * frame), _player(400 - 5 * frame)], frame)
        assert tracker.last_assignment == {0: 1, 1: 2}


def test_class_isolation_between_ball_and_player():
    """球和人不会被关联成同一条轨迹（即使位置重叠）。"""
    tracker = MultiObjectTracker(FAST)
    for frame in range(4):
        tracker.update([_player(300), _ball(305)], frame)
    assert tracker.last_assignment[0] != tracker.last_assignment[1]
    assert len(tracker.summary()["by_class"]) == 2


def test_low_confidence_detection_is_recovered():
    """某帧置信度掉到阈值以下时, 靠第二阶段关联仍保住原 ID（不新建轨迹）。"""
    tracker = MultiObjectTracker(FAST)
    for frame in range(3):
        tracker.update([_player(100 + 5 * frame)], frame)
    weak = _player(115)
    weak["conf"] = 0.3                        # 低于 high_threshold(0.5), 高于 min_confidence
    tracker.update([weak], 3)
    assert tracker.last_assignment == {0: 1}
    assert tracker.summary()["live_tracks"] == 1


def test_occlusion_within_max_age_keeps_id():
    """消失 3 帧（max_age=5 以内）后重新出现, 应当续上原来的 ID。"""
    tracker = MultiObjectTracker(FAST)
    for frame in range(3):
        tracker.update([_player(10 + 5 * frame)], frame)
    for frame in range(3, 6):
        tracker.update([], frame)             # 被遮挡
    tracker.update([_player(40)], 6)
    assert tracker.last_assignment == {0: 1}


def test_id_changes_after_max_age():
    """消失超过 max_age 后回来, 必须发【新 ID】（不能把两个人粘成一条轨迹）。"""
    tracker = MultiObjectTracker({**FAST, "max_age": 2})
    for frame in range(3):
        tracker.update([_player(10 + 5 * frame)], frame)
    assert tracker.last_assignment == {0: 1}
    for frame in range(3, 9):
        tracker.update([], frame)             # 长时间消失, 旧轨迹已被回收
    tracker.update([_player(10)], 9)
    assert tracker.last_assignment == {0: 2}, "重新出现的应当是全新身份"
    assert tracker.summary()["live_tracks"] == 1


def test_fast_small_ball_keeps_its_id():
    """★ 回归测试: 足球每帧位移与自身尺寸同量级, IoU 会跌破阈值。

    历史实现（以及只认 IoU 的实现）会让球每帧换一个 ID; 小目标中心距离
    兜底必须把它稳住。
    """
    tracker = MultiObjectTracker(FAST)
    ids = []
    for frame in range(8):
        tracker.update([_ball(300 + 12 * frame)], frame)
        ids.append(tracker.last_assignment[0])
    assert len(set(ids)) == 1, f"球每帧换了 ID: {ids}"


def test_large_objects_do_not_use_distance_fallback():
    """大目标（球员）之间不允许用中心距离兜底, 避免把两个挨着的人错配。"""
    tracker = MultiObjectTracker({**FAST, "min_hits": 1})
    tracker.update([_player(100)], 0)
    # 相距 40px 的另一个人: IoU 为 0, 且尺度远大于 small_object_extent_px
    tracker.update([_player(200)], 1)
    assert tracker.last_assignment == {0: 2}


def test_frame_index_must_not_decrease():
    tracker = MultiObjectTracker(FAST)
    tracker.update([_player(100)], 5)
    with pytest.raises(ValueError, match="单调不减"):
        tracker.update([_player(100)], 4)


def test_invalid_detections_are_skipped():
    """缺字段 / 退化框 / NaN 一律跳过, 不抛异常。"""
    tracker = MultiObjectTracker(FAST)
    tracks = tracker.update([
        {"cls_name": "player", "bbox": [0, 0, 0, 0], "conf": 0.9},      # 零面积
        {"cls_name": "player", "bbox": [1, 2, 3], "conf": 0.9},         # 少一个坐标
        {"bbox": [10, 10, 40, 90], "conf": 0.9},                        # 没有类别
        {"cls_name": "player", "bbox": [float("nan"), 0, 10, 10], "conf": 0.9},
        _player(100),
    ], 0)
    assert len(tracks) == 1
    assert tracker.last_assignment == {4: 1}


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="不支持的跟踪后端"):
        MultiObjectTracker({"backend": "magic"})


def test_bad_iou_threshold_is_rejected():
    with pytest.raises(ValueError, match="iou_threshold"):
        MultiObjectTracker({"iou_threshold": 1.5})


def test_reset_clears_state():
    tracker = MultiObjectTracker(FAST)
    tracker.update([_player(100)], 0)
    tracker.reset()
    assert tracker.tracks == []
    tracker.update([_player(100)], 0)
    assert tracker.last_assignment == {0: 1}


def test_track_exposes_foot_point_and_velocity():
    """脚点是投影到球场时该用的点（框底边中点）, 不是框中心。"""
    tracker = MultiObjectTracker(FAST)
    for frame in range(3):
        tracker.update([_player(100 + 6 * frame)], frame)
    track = tracker.tracks[0]
    assert isinstance(track, Track)
    assert track.foot[1] == pytest.approx(track.bbox[3])
    assert track.velocity[0] > 0
    assert track.confirmed


def test_confirmation_needs_min_hits():
    tracker = MultiObjectTracker({**FAST, "min_hits": 3})
    tracker.update([_player(100)], 0)
    assert tracker.tracks[0].state == "tentative"
    tracker.update([_player(102)], 1)
    tracker.update([_player(104)], 2)
    assert tracker.tracks[0].state == "confirmed"