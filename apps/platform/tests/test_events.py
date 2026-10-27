#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""判罚事件（红牌 / 黄牌 / 换人）的测试。

★ 这里全部用**造假检测结果**, 不下载也不跑真实模型 —— 因为当前权重根本
  检不出"牌", 真跑只会得到零事件, 什么也验证不了。判定逻辑是可以独立验证的。

守的核心是这几条（都是答辩会被追问的点）:
  · 颜色与模型类别矛盾时**必须降级为"证据不足"**, 不许直接信模型;
  · 牌离出示者太远 → **不构成事件**（不是"证据不足", 是压根不算）;
  · 没有可信标定的换人**只能是候选**, 且明写"未标定", 不假装确定;
  · 类别名走配置、大小写不敏感、认多个别名; 逻辑里没有类名字面量;
  · 冷却: 同一张牌连续多帧只报一次。
"""
from __future__ import annotations

import json
import logging

import cv2
import numpy as np
import pytest

from od_platform.events import (
    KIND_CARD,
    KIND_RED_CARD,
    KIND_SEVERITY,
    KIND_SUBSTITUTION,
    KIND_YELLOW_CARD,
    ROLE_GENERIC_CARD,
    ROLE_PLAYER,
    ROLE_RED_CARD,
    ROLE_REFEREE,
    ROLE_SUBSTITUTION_BOARD,
    ROLE_YELLOW_CARD,
    CardEventDetector,
    Detection,
    EventSettings,
    PenaltyEvent,
    SubstitutionDetector,
    cap_confidence,
    parse_detections,
)
from od_platform.events.settings import (
    CardSettings,
    ClassNameSettings,
    ColorCheckSettings,
    SubstitutionSettings,
)
from od_platform.geometric.calibration import PitchCalibration
from od_platform.geometric.homography import FieldHomography
from od_platform.geometric.pitch import DEFAULT_PITCH

# BGR
RED_BGR = (0, 0, 255)
YELLOW_BGR = (0, 255, 255)
WHITE_BGR = (255, 255, 255)

FRAME_SIZE = (400, 600)                  # (高, 宽)
CARD_BOX = (280, 120, 300, 160)          # 举在裁判头顶附近的牌
REFEREE_BOX = (250, 100, 350, 300)       # 裁判（高 200px, 中心 (300,200)）
REFEREE = {"cls": "referee", "conf": 0.9, "xyxy": list(REFEREE_BOX)}
PLAYER = {"cls": "player", "conf": 0.9, "xyxy": list(REFEREE_BOX)}


def make_frame(colour=RED_BGR, box=CARD_BOX):
    """造一张只有牌那块区域有颜色的图。"""
    frame = np.zeros((FRAME_SIZE[0], FRAME_SIZE[1], 3), dtype=np.uint8)
    x1, y1, x2, y2 = box
    frame[y1:y2, x1:x2] = colour
    return frame


def card(cls="red_card", conf=0.8, box=CARD_BOX):
    return {"cls": cls, "conf": conf, "xyxy": list(box)}


def event_of(events, index=0) -> PenaltyEvent:
    assert events, "期望产出事件, 实际为空"
    return events[index]


# ============================================================ 可信标定夹具

def _build_calibrated_homography() -> FieldHomography:
    """造一个【自洽】的可信标定。

    ★ 为什么不能手点几个坐标: ``PitchCalibration.from_landmarks`` 会做重投影
      校验, 手点的像素点往往互相矛盾（误差超过 1 m 直接被拒）。正确做法是先造
      一个真实的透视映射 H（米→像素）, 再由 H 反算各标记点的像素位置 ——
      这样点必然自洽（重投影误差 ~1e-14）。
    """
    metre_corners = [(0.0, 0.0), (DEFAULT_PITCH.length, 0.0),
                     (DEFAULT_PITCH.length, DEFAULT_PITCH.width),
                     (0.0, DEFAULT_PITCH.width)]
    pixel_corners = [(400.0, 300.0), (1500.0, 300.0), (1800.0, 1000.0), (100.0, 1000.0)]
    matrix = cv2.getPerspectiveTransform(np.float32(metre_corners), np.float32(pixel_corners))

    def to_pixel(metre_x: float, metre_y: float):
        vec = matrix @ np.array([metre_x, metre_y, 1.0])
        return [float(vec[0] / vec[2]), float(vec[1] / vec[2])]

    names = ("corner_top_left", "corner_top_right", "corner_bottom_left",
             "corner_bottom_right", "centre_spot", "halfway_top", "halfway_bottom")
    landmarks = {name: to_pixel(*DEFAULT_PITCH.landmark(name)) for name in names}
    return FieldHomography.from_calibration(
        PitchCalibration.from_landmarks(landmarks, image_size=(1920, 1080)))


@pytest.fixture(scope="module")
def homography() -> FieldHomography:
    return _build_calibrated_homography()


def board_at_metres(homography: FieldHomography, x_m: float, y_m: float,
                    half_width: float = 15.0, height: float = 30.0) -> dict:
    """把换人牌摆在指定的场地坐标上（**框的下边缘中点**落在那一点）。

    接地点用下边缘中点是刻意约定: 物体贴地时接地的是下边缘, 用框中心会在
    透视下系统性偏向场地内侧。
    """
    px, py = homography.meter_to_pixel(x_m, y_m)
    return {"cls": "换人牌", "conf": 0.85,
            "xyxy": [px - half_width, py - height, px + half_width, py]}


# ============================================================ 1. 正常出示

def test_red_card_shown_by_referee_is_confirmed():
    events = CardEventDetector().detect([REFEREE, card()], frame=make_frame(RED_BGR),
                                        frame_index=0)
    event = event_of(events)
    assert event.kind == KIND_RED_CARD
    assert event.confirmed is True
    assert event.severity == KIND_SEVERITY[KIND_RED_CARD]
    assert event.anchor_class == "referee"
    assert event.anchor_role == ROLE_REFEREE
    assert event.confidence == pytest.approx(0.8)
    assert event.evidence["red_pixel_ratio"] > 0.9
    assert "颜色二次确认通过" in " ".join(event.reasons)


def test_yellow_card_shown_by_referee_is_confirmed():
    events = CardEventDetector().detect([REFEREE, card("yellow_card")],
                                        frame=make_frame(YELLOW_BGR), frame_index=0)
    event = event_of(events)
    assert event.kind == KIND_YELLOW_CARD
    assert event.confirmed is True
    assert event.evidence["yellow_pixel_ratio"] > 0.9


def test_red_card_is_at_least_as_severe_as_yellow():
    assert KIND_SEVERITY[KIND_RED_CARD] > KIND_SEVERITY[KIND_YELLOW_CARD]


def test_event_reports_the_thresholds_it_used():
    """判据要能自解释: 阈值必须随事件带出来, 否则答辩时说不清"凭什么"。"""
    event = event_of(CardEventDetector().detect([REFEREE, card()],
                                                frame=make_frame(), frame_index=0))
    thresholds = event.evidence["thresholds"]
    assert thresholds["max_show_distance_body_heights"] > 0
    assert thresholds["color_min_pixel_ratio"] > 0
    assert event.evidence["distance_body_heights"] is not None


def test_player_is_used_as_anchor_when_no_referee_present():
    event = event_of(CardEventDetector().detect([PLAYER, card()], frame=make_frame(),
                                                frame_index=0))
    assert event.anchor_role == ROLE_PLAYER
    assert "player" in " ".join(event.reasons)


# ============================================================ 2. 颜色二次确认

def test_color_conflict_downgrades_to_insufficient_evidence():
    """★ 核心用例: 模型说红牌, 但牌那块区域是黄的 —— 不许直接信模型。"""
    events = CardEventDetector().detect([REFEREE, card("red_card")],
                                        frame=make_frame(YELLOW_BGR), frame_index=0)
    event = event_of(events)
    assert event.confirmed is False, "颜色与类别矛盾时必须以模型为准会误判, 应降级"
    assert event.confidence <= 0.4, "降级事件的置信度必须被压住"
    reasons = " ".join(event.reasons)
    assert "矛盾" in reasons and "降级" in reasons
    assert event.evidence["yellow_pixel_ratio"] > 0.9
    assert event.kind == KIND_RED_CARD, "牌型仍按模型给出, 但明确标为未确认"


def test_color_conflict_can_be_configured_to_trust_the_model():
    settings = EventSettings(cards=CardSettings(
        color=ColorCheckSettings(conflict_policy="trust_model")))
    event = event_of(CardEventDetector(settings).detect(
        [REFEREE, card("red_card")], frame=make_frame(YELLOW_BGR), frame_index=0))
    assert event.confirmed is True
    assert "trust_model" in " ".join(event.reasons)
    assert "矛盾" in " ".join(event.reasons), "即使采信也要把矛盾留痕"


def test_missing_frame_downgrades_instead_of_silently_trusting_the_model():
    """没给图像帧 → 颜色确认做不了 → 降级。("没量过"不等于"量过了")"""
    event = event_of(CardEventDetector().detect([REFEREE, card()], frame=None,
                                                frame_index=0))
    assert event.confirmed is False
    assert "无法做颜色二次确认" in " ".join(event.reasons)
    assert event.evidence["color_check"] == "no_frame"


def test_disabled_color_check_trusts_the_model_and_says_so():
    settings = EventSettings(cards=CardSettings(color=ColorCheckSettings(enabled=False)))
    event = event_of(CardEventDetector(settings).detect([REFEREE, card()], frame=None,
                                                       frame_index=0))
    assert event.confirmed is True
    assert "颜色二次确认已按配置关闭" in " ".join(event.reasons)
    assert event.evidence["color_check"] == "disabled"


def test_neither_red_nor_yellow_downgrades_a_specific_card_class():
    """裁到的是一块白（既不明显红也不明显黄）→ 颜色不成立 → 降级。"""
    event = event_of(CardEventDetector().detect([REFEREE, card("red_card")],
                                                frame=make_frame(WHITE_BGR), frame_index=0))
    assert event.confirmed is False
    assert "既没有明显的红也没有明显的黄" in " ".join(event.reasons)


def test_generic_card_class_lets_colour_decide_the_kind():
    """只训练了一个通用 'card' 类时, 红/黄由**颜色**决定。"""
    red_event = event_of(CardEventDetector().detect([REFEREE, card("card")],
                                                    frame=make_frame(RED_BGR), frame_index=0))
    assert red_event.kind == KIND_RED_CARD and red_event.confirmed is True
    assert "由颜色判定" in " ".join(red_event.reasons)

    yellow_event = event_of(CardEventDetector().detect([REFEREE, card("card")],
                                                       frame=make_frame(YELLOW_BGR),
                                                       frame_index=0))
    assert yellow_event.kind == KIND_YELLOW_CARD and yellow_event.confirmed is True


def test_generic_card_class_without_usable_colour_stays_unresolved():
    event = event_of(CardEventDetector().detect([REFEREE, card("card")],
                                                frame=make_frame(WHITE_BGR), frame_index=0))
    assert event.kind == KIND_CARD and event.confirmed is False
    assert "无法定牌型" in " ".join(event.reasons)


def test_colour_threshold_is_configurable():
    """牌区域只有一半是红的 → 阈值定在它两侧, 结论必须跟着翻。

    （用"区域整体都是红的"验证不了可配置性: 那种情况任何合理阈值都会通过。）
    """
    half_red = make_frame(RED_BGR)
    half_red[CARD_BOX[1] + 20:CARD_BOX[3], CARD_BOX[0]:CARD_BOX[2]] = 0   # 下半截涂黑
    lenient = EventSettings(cards=CardSettings(
        color=ColorCheckSettings(min_pixel_ratio=0.4)))
    strict = EventSettings(cards=CardSettings(
        color=ColorCheckSettings(min_pixel_ratio=0.6)))

    assert event_of(CardEventDetector(lenient).detect(
        [REFEREE, card()], frame=half_red, frame_index=0)).confirmed is True
    strict_event = event_of(CardEventDetector(strict).detect(
        [REFEREE, card()], frame=half_red, frame_index=0))
    assert strict_event.confirmed is False, "红色占比没到 0.6 就该降级"


# ============================================================ 3. 距离与形状

def test_card_too_far_from_anyone_is_not_an_event():
    """太远 → 不构成出示（连候选都不是）。"""
    far = card(box=(20, 20, 40, 60))
    assert CardEventDetector().detect([REFEREE, far], frame=make_frame(RED_BGR),
                                      frame_index=0) == []


def test_card_without_any_anchor_is_reported_unconfirmed():
    """没有人能当出示者 → 报候选并写明原因, 而不是丢掉或假装确定。"""
    event = event_of(CardEventDetector().detect([card()], frame=make_frame(RED_BGR),
                                                frame_index=0))
    assert event.confirmed is False
    assert "无法确认是谁出示" in " ".join(event.reasons)
    assert event.anchor_class == ""


def test_require_card_above_anchor_rejects_a_card_below_the_centre():
    settings = EventSettings(cards=CardSettings(require_card_above_anchor=True))
    below = card(box=(280, 240, 300, 290))          # 中心 y=265 > 裁判中心 y=200
    assert CardEventDetector(settings).detect([REFEREE, below], frame=make_frame(RED_BGR),
                                              frame_index=0) == []


def test_degenerate_shape_is_not_treated_as_a_card():
    """"压根不像牌"与"像牌但没确认"是两回事: 前者不产生任何事件。"""
    thin = card(box=(280, 100, 300, 400))           # 宽高比 0.067 < 0.35
    assert CardEventDetector().detect([REFEREE, thin], frame=make_frame(RED_BGR),
                                      frame_index=0) == []
    # 1x1 像素: 面积占比 4.2e-6, 低于默认下限 1e-5（默认值注释里写明"1080p 上约 21px"）
    tiny = card(box=(280, 120, 281, 121))
    assert CardEventDetector().detect([REFEREE, tiny], frame=make_frame(RED_BGR),
                                      frame_index=0) == []


def test_area_ratio_threshold_is_configurable():
    """把面积下限抬高, 10x10 的小框就被排掉 —— 证明它真的在读配置。"""
    settings = EventSettings(cards=CardSettings(min_area_ratio=0.001))
    small = card(box=(280, 120, 290, 130))          # 占比 4.2e-4 < 1e-3
    assert CardEventDetector(settings).detect([REFEREE, small], frame=make_frame(RED_BGR),
                                              frame_index=0) == []
    # 同样的框, 默认阈值下是能通过的（差别只来自配置）
    assert CardEventDetector().detect([REFEREE, small], frame=make_frame(RED_BGR),
                                      frame_index=0)


def test_low_confidence_detection_is_ignored():
    assert CardEventDetector().detect([REFEREE, card(conf=0.05)], frame=make_frame(),
                                      frame_index=0) == []


def test_unrecognised_class_produces_nothing():
    """球不是牌 —— 认不出的类名必须被忽略, 不能"反正有个框"就判。"""
    ball = {"cls": "football", "conf": 0.9, "xyxy": list(CARD_BOX)}
    assert CardEventDetector().detect([REFEREE, ball], frame=make_frame(RED_BGR),
                                      frame_index=0) == []


# ============================================================ 4. 冷却

def test_cooldown_suppresses_repeats_within_the_window():
    detector = CardEventDetector()                   # 默认冷却 30 帧
    frame = make_frame(RED_BGR)
    assert len(detector.detect([REFEREE, card()], frame=frame, frame_index=0)) == 1
    for index in range(1, 30):
        assert detector.detect([REFEREE, card()], frame=frame, frame_index=index) == [], \
            f"第 {index} 帧不该重复报同一个事件"
    assert len(detector.detect([REFEREE, card()], frame=frame, frame_index=30)) == 1


def test_cooldown_also_works_without_explicit_frame_index():
    detector = CardEventDetector()
    frame = make_frame(RED_BGR)
    assert len(detector.detect([REFEREE, card()], frame=frame)) == 1
    assert detector.detect([REFEREE, card()], frame=frame) == []


def test_reset_clears_the_cooldown():
    detector = CardEventDetector()
    frame = make_frame(RED_BGR)
    assert len(detector.detect([REFEREE, card()], frame=frame)) == 1
    detector.reset()
    assert len(detector.detect([REFEREE, card()], frame=frame)) == 1


def test_cooldown_window_is_configurable():
    settings = EventSettings(cards=CardSettings(cooldown_frames=2))
    detector = CardEventDetector(settings)
    frame = make_frame(RED_BGR)
    assert len(detector.detect([REFEREE, card()], frame=frame, frame_index=0)) == 1
    assert detector.detect([REFEREE, card()], frame=frame, frame_index=1) == []
    assert len(detector.detect([REFEREE, card()], frame=frame, frame_index=2)) == 1


# ============================================================ 5. 换人

def test_substitution_near_touchline_with_trustworthy_calibration(homography):
    detector = SubstitutionDetector(homography=homography)
    board = board_at_metres(homography, 20.0, 1.0)   # 距边线(y=0) 1 米
    event = event_of(detector.detect([board], frame_index=0))
    assert event.kind == KIND_SUBSTITUTION
    assert event.confirmed is True
    assert event.evidence["calibrated"] is True
    assert event.evidence["distance_to_touchline_m"] <= 3.0
    assert "位于场边" in " ".join(event.reasons)


def test_substitution_far_from_the_touchline_is_not_an_event(homography):
    """场地中央的牌子不是换人（有可信标定时必须排掉）。"""
    board = board_at_metres(homography, 52.5, 34.0)
    assert SubstitutionDetector(homography=homography).detect([board], frame_index=0) == []


@pytest.mark.parametrize("y_metre,expected", [(2.5, True), (4.0, False)])
def test_substitution_touchline_threshold_is_respected(homography, y_metre, expected):
    """阈值两侧的行为都要对（默认 3 m）。"""
    board = board_at_metres(homography, 20.0, y_metre)
    events = SubstitutionDetector(homography=homography).detect([board], frame_index=0)
    assert bool(events) is expected


def test_substitution_without_calibration_is_only_a_candidate():
    """★ 无标定 → 只能报候选, 且必须明写"未标定", 不能假装确定。"""
    board = board_at_metres(_build_calibrated_homography(), 20.0, 1.0)
    event = event_of(SubstitutionDetector(homography=None).detect([board], frame_index=0))
    assert event.confirmed is False
    assert event.confidence <= 0.4
    assert event.evidence["calibrated"] is False
    assert "未标定" in " ".join(event.reasons)
    assert "候选" in " ".join(event.reasons)


def test_ad_hoc_homography_counts_as_untrustworthy_calibration():
    """随手映射(calibrated=False)与"没有标定"同等对待 —— 它没有场上物理意义。"""
    ad_hoc = FieldHomography(src_points=[(0, 0), (100, 0), (100, 100), (0, 100)],
                             dst_points=[(0, 0), (105, 0), (105, 68), (0, 68)])
    board = {"cls": "换人牌", "conf": 0.85, "xyxy": [50, 20, 80, 50]}
    event = event_of(SubstitutionDetector(homography=ad_hoc).detect([board], frame_index=0))
    assert event.confirmed is False
    assert event.evidence["calibrated"] is False


def test_uncalibrated_candidates_can_be_turned_off():
    settings = EventSettings(substitution=SubstitutionSettings(
        allow_uncalibrated_candidates=False))
    board = {"cls": "换人牌", "conf": 0.85, "xyxy": [50, 20, 80, 50]}
    assert SubstitutionDetector(settings).detect([board], frame_index=0) == []


def test_require_calibration_false_accepts_a_board_without_position_check():
    """逃生口: 确实不在乎位置时, 可以只凭检出就采信。"""
    settings = EventSettings(substitution=SubstitutionSettings(require_calibration=False))
    board = {"cls": "换人牌", "conf": 0.85, "xyxy": [50, 20, 80, 50]}
    event = event_of(SubstitutionDetector(settings).detect([board], frame_index=0))
    assert event.confirmed is True
    assert event.evidence["calibrated"] is False


def test_substitution_area_ratio_uses_the_frame_when_available():
    """给了画面就能按面积比排掉"占满屏的牌子"; 不给则跳过该判据（不知道≠不合格）。"""
    frame = np.zeros((FRAME_SIZE[0], FRAME_SIZE[1], 3), dtype=np.uint8)
    big = {"cls": "换人牌", "conf": 0.9, "xyxy": [0, 0, FRAME_SIZE[1], FRAME_SIZE[0]]}
    assert SubstitutionDetector().detect([big], frame=frame, frame_index=0) == []
    # 同一个检测, 不给 frame → 面积比无从计算, 不会被判成不合格
    assert SubstitutionDetector().detect([big], frame_index=0)


def test_substitution_cooldown_suppresses_repeats():
    board = {"cls": "换人牌", "conf": 0.85, "xyxy": [50, 20, 80, 50]}
    detector = SubstitutionDetector()
    assert len(detector.detect([board], frame_index=0)) == 1
    assert detector.detect([board], frame_index=5) == []
    assert len(detector.detect([board], frame_index=30)) == 1


def test_unknown_class_is_not_a_substitution():
    assert SubstitutionDetector().detect([card("red_card")], frame_index=0) == []


# ============================================================ 6. 类别名与配置

@pytest.mark.parametrize("name,expected", [
    ("red_card", ROLE_RED_CARD),
    ("RED_CARD", ROLE_RED_CARD),
    ("  Red_Card  ", ROLE_RED_CARD),
    ("card_red", ROLE_RED_CARD),
    ("红牌", ROLE_RED_CARD),
    ("yellow_card", ROLE_YELLOW_CARD),
    ("YellowCard", ROLE_YELLOW_CARD),
    ("黄牌", ROLE_YELLOW_CARD),
    ("card", ROLE_GENERIC_CARD),
    ("referee", ROLE_REFEREE),
    ("Ref", ROLE_REFEREE),
    ("裁判", ROLE_REFEREE),
    ("player", ROLE_PLAYER),
    ("球员", ROLE_PLAYER),
    ("substitution_board", ROLE_SUBSTITUTION_BOARD),
    ("换人牌", ROLE_SUBSTITUTION_BOARD),
    ("football", ""),
    ("", ""),
    (None, ""),
])
def test_role_resolution_is_alias_based_and_case_insensitive(name, expected):
    assert ClassNameSettings().resolve_role(name) == expected


def test_specific_card_class_takes_priority_over_the_generic_one():
    """同时有 card 和 red_card 时, red_card 必须归到具体的红牌, 不被通用类吃掉。"""
    classes = ClassNameSettings()
    assert classes.resolve_role("red_card") == ROLE_RED_CARD


def test_config_override_replaces_the_alias_list():
    settings = EventSettings.from_config({"classes": {"red_card_aliases": ["my_red"]}})
    assert settings.classes.resolve_role("my_red") == ROLE_RED_CARD
    assert settings.classes.resolve_role("red_card") == "", "覆盖是替换而不是追加"


def test_config_override_reaches_the_detectors_selection():
    """配置改了类名, 检测器就得跟着改 —— 证明类名没有写死在逻辑里。"""
    settings = EventSettings.from_config(
        {"classes": {"red_card_aliases": ["my_own_red"]}})
    frame = make_frame(RED_BGR)
    assert CardEventDetector(settings).detect([REFEREE, card("my_own_red")],
                                              frame=frame, frame_index=0)
    assert CardEventDetector(settings).detect([REFEREE, card("red_card")],
                                              frame=frame, frame_index=0) == []


def test_a_single_string_alias_is_accepted():
    assert ClassNameSettings(player_aliases="Player").resolve_role("player") == ROLE_PLAYER


def test_repo_events_yaml_matches_the_code_defaults():
    """示例配置必须与代码默认值一致 —— 否则两边会随时间漂移而没人发现。"""
    assert EventSettings.load() == EventSettings()


def test_unknown_config_key_is_reported(caplog):
    with caplog.at_level(logging.WARNING):
        EventSettings.from_config({"cards": {"no_such_key": 1}})
    assert "no_such_key" in caplog.text


def test_nested_colour_section_is_readable_from_config():
    settings = EventSettings.from_config(
        {"cards": {"color": {"conflict_policy": "trust_model"}}})
    assert settings.cards.color.conflict_policy == "trust_model"


def test_colour_switch_can_be_put_at_the_cards_level():
    """便利键: 有些配置习惯把开关放在 cards 一级, 不能被静默丢掉。"""
    settings = EventSettings.from_config({"cards": {"color_check_enabled": False}})
    assert settings.cards.color.enabled is False


# ============================================================ 7. 设置层校验

@pytest.mark.parametrize("factory", [
    lambda: CardSettings(min_confidence=1.5),
    lambda: CardSettings(min_confidence=-0.1),
    lambda: CardSettings(cooldown_frames=-1),
    lambda: CardSettings(min_area_ratio=0.5, max_area_ratio=0.1),
    lambda: CardSettings(min_aspect_ratio=3.0, max_aspect_ratio=0.5),
    lambda: CardSettings(max_show_distance_body_heights=0),
    lambda: CardSettings(unconfirmed_confidence_ceiling=2.0),
    lambda: ColorCheckSettings(conflict_policy="nope"),
    lambda: ColorCheckSettings(min_pixel_ratio=1.5),
    lambda: ColorCheckSettings(min_crop_pixels=0),
    lambda: ColorCheckSettings(red_ranges=[[0, 1, 2]]),
    lambda: ColorCheckSettings(red_ranges=[[10, 0, 0, 5, 255, 255]]),
    lambda: ColorCheckSettings(red_ranges=[]),
    lambda: ClassNameSettings(red_card_aliases=[]),
    lambda: ClassNameSettings(red_card_aliases=[None, "  "]),
    lambda: ClassNameSettings(referee_aliases=123),
    lambda: SubstitutionSettings(touchline_margin_m=0),
    lambda: SubstitutionSettings(cooldown_frames=-2),
    lambda: SubstitutionSettings(min_area_ratio=-0.1),
])
def test_invalid_settings_are_rejected(factory):
    with pytest.raises(ValueError):
        factory()


def test_settings_audit_is_serialisable():
    json.dumps(EventSettings().to_audit())
    json.dumps(CardSettings().to_audit())
    json.dumps(SubstitutionSettings().to_audit())


# ============================================================ 8. 输入解析

def test_parse_detections_skips_malformed_rows():
    raw = [{"cls": "red_card", "xyxy": [1, 2, 3, 4]},
           {"cls": "bad", "xyxy": [1, 2]},          # 坐标不够
           "junk",                                   # 根本不是 dict
           None,
           {"cls": "bad2", "xyxy": "not-numbers"}]
    parsed = parse_detections(raw)
    assert len(parsed) == 1 and parsed[0].cls == "red_card"


def test_parse_detections_accepts_detection_objects():
    det = Detection(cls="referee", x1=0, y1=0, x2=10, y2=20)
    assert parse_detections([det]) == [det]


def test_missing_confidence_is_not_treated_as_zero():
    """★ "没有置信度"不等于"置信度为 0" —— 不能因此把它滤掉。"""
    parsed = parse_detections([{"cls": "red_card", "xyxy": list(CARD_BOX)}])
    assert parsed[0].conf is None
    event = event_of(CardEventDetector().detect(
        [REFEREE, {"cls": "red_card", "xyxy": list(CARD_BOX)}],
        frame=make_frame(RED_BGR), frame_index=0))
    assert event.confirmed is True
    assert event.evidence["detection_confidence"] is None
    assert event.confidence <= 0.4, "置信度未知时不该给出高置信度"


def test_empty_input_produces_nothing():
    assert CardEventDetector().detect([]) == []
    assert CardEventDetector().detect(None) == []
    assert SubstitutionDetector().detect([]) == []


# ============================================================ 9. 事件对象

def test_event_to_dict_is_json_serialisable():
    event = event_of(CardEventDetector().detect([REFEREE, card()], frame=make_frame(),
                                                frame_index=7))
    payload = event.to_dict()
    json.dumps(payload, ensure_ascii=False)
    assert payload["kind"] == KIND_RED_CARD
    assert payload["frame_index"] == 7
    assert payload["severity"] == 2
    assert payload["confirmed"] is True
    assert set(payload["evidence"]) >= {"card_class", "thresholds"}
    assert isinstance(payload["reasons"], list)


def test_cap_confidence_only_caps_unconfirmed_events():
    assert cap_confidence(0.9, True, 0.4) == pytest.approx(0.9)
    assert cap_confidence(0.9, False, 0.4) == pytest.approx(0.4)


def test_describe_mentions_state_and_reasons():
    text = event_of(CardEventDetector().detect([REFEREE, card()], frame=make_frame(),
                                               frame_index=0)).describe()
    assert "确认" in text and KIND_RED_CARD in text


def test_detection_geometry_helpers():
    det = Detection(cls="x", x1=0, y1=0, x2=10, y2=20)
    assert det.width == 10 and det.height == 20 and det.area == 200
    assert det.centre == (5.0, 10.0)
    assert det.bbox == (0, 0, 10, 20)
    assert det.bottom_centre == (5.0, 20.0)
    assert det.aspect_ratio() == pytest.approx(0.5)
    assert Detection(cls="x", x1=0, y1=0, x2=10, y2=0).aspect_ratio() == 0.0
    assert det.normalised_area((100, 100)) == pytest.approx(0.02)
    assert det.normalised_area(None) is None