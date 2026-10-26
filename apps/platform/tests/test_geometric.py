#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""geometric 子系统回归测试。

守住的语义（每条都对应历史上真实坏掉或从未被调用的行为）:
    · 没有真标定就不能产出"米" —— ad_hoc 映射必须自报 calibrated=False
    · 门线判定必须同时满足"越线"和"在门框内"（历史版本把 y 范围参数直接丢掉）
    · 出界必须看最后触球方, 缺它只能报 unknown, 不能猜
    · check_offside 必须真的被调用到, 并带 needs_evidence
    · 没有标定时 analyzer 一条几何事件都不产出（历史版本用画面四角伪造标定）
    · 场地尺寸/阈值全部来自 PitchSpec + Settings, 改配置全链路跟着变

用到的常量都从 PitchSpec 取, 测试里不重复写球场尺寸字面量。
"""
import numpy as np
import pytest

from od_platform.geometric import pitch as pitch_module
from od_platform.geometric.analyzer import PitchAnalyzer
from od_platform.geometric.calibration import (
    CALIBRATION_SUFFIX, CalibrationError, PitchCalibration, discover,
)
from od_platform.geometric.homography import FieldHomography
from od_platform.geometric.pitch import DEFAULT_PITCH, PitchSpec
from od_platform.geometric.rules import RuleEngine
from od_platform.geometric.settings import CalibrationSettings, GeometrySettings, RuleSettings

PITCH = DEFAULT_PITCH
IMAGE_SIZE = (1920, 1080)


# ============================================================
# 夹具: 一个"带透视"的合成机位（米 → 像素）
# ============================================================

def _projector():
    """返回 pitch(米) → pixel 的投影函数; 矩阵带透视项, 不是纯缩放。"""
    matrix = np.array([[12.0, 3.0, 400.0],
                       [1.5, 11.0, 300.0],
                       [0.0008, 0.0004, 1.0]])

    def to_pixel(gx: float, gy: float):
        vector = matrix @ np.array([gx, gy, 1.0])
        return (float(vector[0] / vector[2]), float(vector[1] / vector[2]))

    return to_pixel


LANDMARK_SAMPLE = [
    "corner_bottom_left", "corner_bottom_right",
    "penalty_area_left_top", "penalty_area_left_bottom",
    "penalty_spot_left", "centre_spot", "goal_area_right_bottom",
]


def _synthetic_calibration(image_size=IMAGE_SIZE, pitch=PITCH) -> PitchCalibration:
    to_pixel = _projector()
    pixels = {name: to_pixel(*pitch.landmark(name)) for name in LANDMARK_SAMPLE}
    return PitchCalibration.from_landmarks(pixels, image_size=image_size,
                                           source="synthetic-test", pitch=pitch)


def _engine(settings: RuleSettings | None = None, pitch=PITCH) -> RuleEngine:
    return RuleEngine(homography=FieldHomography.from_calibration(
        _synthetic_calibration(pitch=pitch)), settings=settings)


# ============================================================
# 场地模型
# ============================================================

def test_pitch_defaults_follow_ifab():
    assert (PITCH.length, PITCH.width) == (105.0, 68.0)
    assert PITCH.goal_width == 7.32
    assert PITCH.goal_y_min == pytest.approx(30.34)
    assert PITCH.goal_y_max == pytest.approx(37.66)
    assert PITCH.penalty_area_y_min == pytest.approx(13.84)
    assert PITCH.penalty_area_y_max == pytest.approx(54.16)
    assert PITCH.centre_x == 52.5
    # 越位线判定用到的"本方半场"边界就是中线
    assert PITCH.centre_x * 2 == PITCH.length


def test_pitch_spec_is_configurable():
    """★ 场地尺寸不是写死的: 换一份规格, 派生坐标全部跟着变。"""
    small = PitchSpec(length=40.0, width=20.0, goal_width=3.0,
                      penalty_area_depth=6.0, penalty_area_width=12.0,
                      goal_area_depth=3.0, goal_area_width=6.0,
                      penalty_spot_distance=5.0, centre_circle_radius=3.0)
    assert small.centre_x == 20.0
    assert small.goal_y_min == pytest.approx(small.centre_y - small.goal_width / 2)
    assert small.landmark("goal_left_post_bottom") == (
        0.0, pytest.approx(small.goal_y_max))
    assert small.landmark("penalty_spot_left")[0] == pytest.approx(5.0)
    assert small.landmark("penalty_spot_right")[0] == pytest.approx(35.0)
    assert small.within(39.9, 19.9) and not small.within(40.1, 10.0)
    # 换规格后, 用新规格构造的标定也必须自洽（尺寸与标定是配套的）
    assert FieldHomography.from_calibration(
        _synthetic_calibration(pitch=small)).pitch is small


def test_pitch_spec_reads_legacy_config_keys():
    spec = PitchSpec.from_config({"field_length": 100, "field_width": 64,
                                  "goal_width": 7.32})
    assert (spec.length, spec.width) == (100.0, 64.0)


def test_pitch_rejects_non_positive_dimension():
    with pytest.raises(ValueError, match="必须为正数"):
        PitchSpec(length=0.0)


def test_unknown_landmark_raises():
    with pytest.raises(KeyError, match="未知的场地关键点"):
        PITCH.landmark("not_a_landmark")


def test_landmark_table_is_symmetric():
    """左右两侧的对称点必须严格镜像（历史上 sign 写反过）。"""
    left = PITCH.landmark("penalty_area_left_top")
    right = PITCH.landmark("penalty_area_right_top")
    assert left[0] == pytest.approx(PITCH.penalty_area_depth)
    assert right[0] == pytest.approx(PITCH.length - PITCH.penalty_area_depth)
    assert left[1] == right[1]


# ============================================================
# 标定
# ============================================================

def test_calibration_needs_at_least_four_points():
    settings = CalibrationSettings()
    with pytest.raises(CalibrationError, match="至少需要"):
        PitchCalibration(image_points=[(0, 0), (1, 0), (1, 1)],
                         pitch_points=[(0, 0), (1, 0), (1, 1)],
                         pitch=PITCH, settings=settings)


def test_calibration_min_points_is_configurable():
    """把 min_points 调到 5, 4 个点的标定就该被拒 —— 证明门槛来自配置。"""
    settings = CalibrationSettings(min_points=5)
    with pytest.raises(CalibrationError, match="至少需要 5"):
        PitchCalibration(image_points=[(0, 0), (10, 0), (10, 10), (0, 10)],
                         pitch_points=[(0, 0), (PITCH.length, 0),
                                       (PITCH.length, PITCH.width), (0, PITCH.width)],
                         pitch=PITCH, settings=settings)
    # 默认配置下同样这 4 个点是合法的
    assert PitchCalibration(image_points=[(0, 0), (10, 0), (10, 10), (0, 10)],
                            pitch_points=[(0, 0), (PITCH.length, 0),
                                          (PITCH.length, PITCH.width), (0, PITCH.width)],
                            pitch=PITCH, image_size=(10, 10)).point_count == 4


def test_calibration_rejects_mismatched_counts():
    with pytest.raises(CalibrationError, match="数量不等"):
        PitchCalibration(image_points=[(0, 0), (1, 0), (1, 1), (0, 1)],
                         pitch_points=[(0, 0), (1, 0), (1, 1)], pitch=PITCH)


def test_calibration_rejects_duplicate_pixel_points():
    with pytest.raises(CalibrationError, match="重复点"):
        PitchCalibration(image_points=[(5, 5)] * 4,
                         pitch_points=[(0, 0), (105, 0), (105, 68), (0, 68)],
                         pitch=PITCH)


def test_calibration_rejects_pitch_point_outside_field():
    with pytest.raises(CalibrationError, match="落在球场"):
        PitchCalibration(image_points=[(0, 0), (100, 0), (100, 100), (0, 100)],
                         pitch_points=[(0, 0), (105, 0), (200, 68), (0, 68)],
                         pitch=PITCH)


def test_calibration_rejects_contradictory_points():
    """5 个点里有一个点错了, 重投影误差会超上限, 必须拒绝而不是照用。"""
    to_pixel = _projector()
    pixels = [to_pixel(*PITCH.landmark(name)) for name in LANDMARK_SAMPLE]
    ground = [PITCH.landmark(name) for name in LANDMARK_SAMPLE]
    pixels[3] = (pixels[3][0] + 400.0, pixels[3][1] - 250.0)   # 故意点错一个
    with pytest.raises(CalibrationError, match="重投影误差"):
        PitchCalibration(image_points=pixels, pitch_points=ground, pitch=PITCH)


def test_calibration_error_tolerance_is_configurable():
    """把误差上限放宽, 同一批矛盾点就能通过 —— 证明门槛来自配置而非写死。"""
    to_pixel = _projector()
    pixels = [to_pixel(*PITCH.landmark(name)) for name in LANDMARK_SAMPLE]
    ground = [PITCH.landmark(name) for name in LANDMARK_SAMPLE]
    pixels[3] = (pixels[3][0] + 400.0, pixels[3][1] - 250.0)
    relaxed = CalibrationSettings(max_reprojection_error_m=1e9, ransac_reprojection_px=5.0)
    calibration = PitchCalibration(image_points=pixels, pitch_points=ground,
                                   pitch=PITCH, settings=relaxed)
    assert calibration.reprojection_error_m > 0


def test_calibration_reprojection_error_is_small_for_consistent_points():
    calibration = _synthetic_calibration()
    # cv2 用 float32 求解, 微米级残差是正常的; 只要远小于 1 mm 即可
    assert calibration.reprojection_error_m < 1e-3
    assert calibration.point_count == len(LANDMARK_SAMPLE)


def test_calibration_roundtrip_through_json(tmp_path):
    calibration = _synthetic_calibration()
    path = calibration.save(tmp_path / "c.json")
    restored = PitchCalibration.load(path, pitch=PITCH)
    # to_dict 会把坐标四舍五入到可读精度（精度也来自配置）, 所以按近似比较
    for saved_point, original in zip(restored.image_points, calibration.image_points,
                                     strict=True):
        assert saved_point == pytest.approx(original, abs=1e-3)
    for saved_point, original in zip(restored.pitch_points, calibration.pitch_points,
                                     strict=True):
        assert saved_point == pytest.approx(original, abs=1e-4)
    assert np.allclose(restored.matrix, calibration.matrix, atol=1e-4)


def test_calibration_rejects_wrong_schema(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"schema": "football.pitch-calibration.v99", "points": []}',
                    encoding="utf-8")
    with pytest.raises(CalibrationError, match="schema"):
        PitchCalibration.load(path)


def test_calibration_reports_missing_file(tmp_path):
    with pytest.raises(CalibrationError, match="不存在"):
        PitchCalibration.load(tmp_path / "nope.json")


def test_full_frame_corners_are_flagged():
    """像素点就是整幅画面四角时, 必须能被识别出来（转播机位的典型凑数标定）。"""
    width, height = IMAGE_SIZE
    calibration = PitchCalibration(
        image_points=[(0, 0), (width, 0), (width, height), (0, height)],
        pitch_points=[(0, 0), (PITCH.length, 0), (PITCH.length, PITCH.width),
                      (0, PITCH.width)],
        image_size=IMAGE_SIZE, pitch=PITCH)
    assert calibration.full_frame_suspected is True


def test_discover_finds_sidecar_next_to_media(tmp_path):
    calibration = _synthetic_calibration()
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"")
    calibration.save(tmp_path / f"clip{CALIBRATION_SUFFIX}")
    found = discover(media, image_size=IMAGE_SIZE, pitch=PITCH)
    assert found is not None
    assert found.reprojection_error_m < 1e-3


def test_discover_returns_none_when_nothing_matches(tmp_path):
    """标定记录的分辨率与实际画面不一致时必须拒绝（换了机位就不能沿用旧标定）。"""
    calibration = _synthetic_calibration()
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"")
    calibration.save(tmp_path / f"clip{CALIBRATION_SUFFIX}")
    assert discover(media, image_size=(640, 640), pitch=PITCH) is None


# ============================================================
# 单应性: 区分"真标定"与"随手映射"
# ============================================================

def test_ad_hoc_homography_reports_not_calibrated():
    """★ 回归测试: 用两组点直接构造的映射绝不能自称"已标定"。

    历史版本正是用画面四角构造它, 并把 capabilities.calibrated_pitch 报成 True。
    """
    width, height = IMAGE_SIZE
    homography = FieldHomography(
        src_points=[(0, 0), (width, 0), (width, height), (0, height)],
        dst_points=[(0, 0), (PITCH.length, 0), (PITCH.length, PITCH.width),
                    (0, PITCH.width)])
    assert homography.calibrated is False
    assert homography.source == "ad_hoc"
    assert homography.quality["reprojection_error_m"] is None


def test_from_calibration_reports_calibrated():
    homography = FieldHomography.from_calibration(_synthetic_calibration())
    assert homography.calibrated is True
    assert homography.quality["reprojection_error_m"] < 1e-3
    assert homography.pitch is PITCH


def test_homography_requires_explicit_input():
    with pytest.raises(CalibrationError):
        FieldHomography()


def test_pixel_and_meter_round_trip():
    homography = FieldHomography.from_calibration(_synthetic_calibration())
    to_pixel = _projector()
    for gx, gy in [(0.0, 0.0), (PITCH.centre_x, PITCH.centre_y),
                   (PITCH.length, PITCH.width),
                   (PITCH.penalty_area_depth, PITCH.penalty_area_y_min)]:
        px, py = to_pixel(gx, gy)
        assert homography.pixel_to_meter(px, py) == pytest.approx((gx, gy), abs=1e-3)
        assert homography.meter_to_pixel(gx, gy) == pytest.approx((px, py), abs=1e-3)


def test_goal_line_requires_ball_inside_goal_mouth():
    """★ 回归测试: 门线判定必须看门框。

    历史实现 ball_crossed_goal_line(x, goal_y_min, goal_y_max) 把后两个参数
    直接丢掉, 只判断 x<0 或 x>105 —— 离球门 30 米远的边线球也会被判"过门线"。
    """
    homography = FieldHomography.from_calibration(_synthetic_calibration())
    centre_y = PITCH.centre_y
    assert homography.ball_crossed_goal_line(PITCH.length + 1.0, centre_y) is True
    assert homography.ball_crossed_goal_line(PITCH.length + 1.0, 10.0) is False
    assert homography.ball_crossed_goal_line(PITCH.length - 1.0, centre_y) is False
    assert homography.ball_crossed_goal_line(-1.0, centre_y, "left") is True
    assert homography.ball_crossed_goal_line(-1.0, centre_y, "right") is False


def test_penalty_area_and_goal_area_membership():
    homography = FieldHomography.from_calibration(_synthetic_calibration())
    centre_y = PITCH.centre_y
    assert homography.point_in_penalty_area(PITCH.penalty_area_depth - 1.0, centre_y,
                                            "left") is True
    assert homography.point_in_penalty_area(PITCH.penalty_area_depth + 1.0, centre_y,
                                            "left") is False
    assert homography.point_in_penalty_area(PITCH.length - 1.0, centre_y, "right") is True
    assert homography.point_in_goal_area(PITCH.goal_area_depth - 1.0, centre_y, "left") is True
    assert homography.point_in_goal_area(PITCH.penalty_area_depth, centre_y, "left") is False


# ============================================================
# 规则引擎
# ============================================================

def test_rules_refuse_to_run_without_calibration():
    """★ 未标定时规则引擎必须拒绝产出, 而不是把像素当米算。"""
    width, height = IMAGE_SIZE
    engine = RuleEngine(homography=FieldHomography(
        src_points=[(0, 0), (width, 0), (width, height), (0, height)],
        dst_points=[(0, 0), (PITCH.length, 0), (PITCH.length, PITCH.width),
                    (0, PITCH.width)]))
    assert engine.usable is False
    assert engine.skip_reason
    assert engine.check_collision([(10.0, 10.0), (10.5, 10.0)]) == []
    assert engine.check_ball_out((PITCH.length + 1.0, PITCH.centre_y)) is None
    assert engine.check_goal((PITCH.length + 5.0, PITCH.centre_y)) is None
    assert engine.check_offside([(100.0, 34.0)], [(95.0, 34.0), (97.0, 20.0)],
                                (90.0, 34.0)) == []


def test_offside_flags_player_beyond_second_last_defender():
    engine = _engine()
    defenders = [(99.0, 50.0), (97.0, 34.0), (95.0, 20.0)]     # 倒数第二 = 97
    attackers = [(98.5, 30.0), (90.0, 34.0)]
    candidates = engine.check_offside(attackers, defenders, (80.0, 34.0),
                                      attacking_direction="right", kick_frame=42)
    assert len(candidates) == 1
    assert candidates[0]["player_idx"] == 0
    assert candidates[0]["kick_frame"] == 42
    assert candidates[0]["offside_line_x"] == pytest.approx(97.0)
    assert candidates[0]["decision_status"] == "pending_human_review"
    assert candidates[0]["second_last_inferred"] is False


def test_offside_margin_comes_from_settings():
    """把容差放大后, 贴着越位线的球员就不再被判越位 —— 证明阈值来自配置。"""
    strict = _engine(RuleSettings(offside_margin_m=0.0))
    relaxed = _engine(RuleSettings(offside_margin_m=5.0))
    defenders = [(99.0, 50.0), (97.0, 34.0), (95.0, 20.0)]
    attackers = [(98.5, 30.0)]
    assert len(strict.check_offside(attackers, defenders, (80.0, 34.0),
                                    attacking_direction="right")) == 1
    assert relaxed.check_offside(attackers, defenders, (80.0, 34.0),
                                 attacking_direction="right") == []


def test_offside_ignores_players_in_own_half_and_behind_the_ball():
    engine = _engine()
    defenders = [(99.0, 50.0), (97.0, 34.0)]
    # 第一个在本方半场; 第二个在对方半场但没越过后卫; 第三个越过后卫但在球后面
    attackers = [(40.0, 34.0), (90.0, 34.0), (96.5, 34.0)]
    assert engine.check_offside(attackers, defenders, (99.5, 34.0),
                                attacking_direction="right") == []


def test_offside_marks_missing_kick_frame_as_needs_evidence():
    engine = _engine()
    candidates = engine.check_offside([(99.0, 34.0)], [(95.0, 20.0), (90.0, 50.0)],
                                      (85.0, 34.0), attacking_direction="right")
    assert candidates[0]["decision_status"] == "needs_evidence"
    assert "kick_moment" in candidates[0]["requires_evidence"]


def test_offside_works_for_leftward_attack():
    engine = _engine()
    defenders = [(1.0, 20.0), (3.0, 34.0), (5.0, 50.0)]        # 倒数第二 = 3
    candidates = engine.check_offside([(1.5, 34.0)], defenders, (20.0, 34.0),
                                      attacking_direction="left")
    assert len(candidates) == 1


def test_contact_candidate_reports_metre_distance():
    engine = _engine(RuleSettings(collision_distance_m=2.0))
    contacts = engine.check_collision([(30.0, 30.0), (31.0, 30.0), (60.0, 30.0)])
    assert len(contacts) == 1
    assert contacts[0]["dist_m"] == pytest.approx(1.0)
    assert contacts[0]["decision_status"] == "pending_human_review"
    assert "action_type" in contacts[0]["requires_evidence"]


def test_contact_threshold_comes_from_settings():
    near = _engine(RuleSettings(collision_distance_m=0.5))
    far = _engine(RuleSettings(collision_distance_m=5.0))
    pair = [(30.0, 30.0), (31.0, 30.0)]
    assert near.check_collision(pair) == []
    assert len(far.check_collision(pair)) == 1


def test_ball_out_needs_last_touch_and_reports_unknown():
    """★ 回归测试: 出界不看最后触球方就只能说 unknown（角球/球门球无法区分）。

    注意用门框之外的 y: 那才是"出界"; 门框之内是进球。
    """
    engine = _engine()
    verdict = engine.check_ball_out((PITCH.length + 1.0, 10.0))
    assert verdict is not None
    assert verdict["restart"] == "unknown"
    assert verdict["decision_status"] == "needs_evidence"


def test_ball_out_goal_kick_vs_corner():
    engine = _engine()
    endline = (PITCH.length + 1.0, 10.0)
    corner = engine.check_ball_out(endline, last_touch_team="B", goal_defending_team="B")
    assert corner["restart"] == "corner"
    assert corner["awarded_to"] == "A"

    goal_kick = engine.check_ball_out(endline, last_touch_team="A", goal_defending_team="B")
    assert goal_kick["restart"] == "goal_kick"
    assert goal_kick["awarded_to"] == "B"


def test_ball_out_inside_goal_mouth_is_not_a_restart():
    """门框内过线是进球, 不该同时报出界（避免重复计数）。"""
    engine = _engine()
    assert engine.check_ball_out((PITCH.length + 5.0, PITCH.centre_y),
                                 last_touch_team="A") is None


def test_ball_out_touchline_is_a_throw_in_to_the_other_team():
    engine = _engine()
    verdict = engine.check_ball_out((PITCH.centre_x, -1.0), last_touch_team="A")
    assert verdict["restart"] == "throw_in"
    assert verdict["awarded_to"] == "B"


def test_goal_candidate_requires_ball_inside_goal_mouth():
    engine = _engine()
    goal = engine.check_goal((PITCH.length + 5.0, PITCH.centre_y))
    assert goal is not None
    assert goal["side"] == "right"
    assert goal["decision_status"] == "needs_evidence"
    assert "no_offence_before_goal" in goal["requires_evidence"]
    assert engine.check_goal((PITCH.length + 5.0, 5.0)) is None


def test_goal_line_tolerance_falls_back_to_ball_radius():
    """goal_line_tolerance_m 不给时取球半径; 显式给了就用给的。"""
    assert RuleSettings().effective_goal_line_tolerance_m == RuleSettings().ball_radius_m
    explicit = RuleSettings(goal_line_tolerance_m=0.5)
    assert explicit.effective_goal_line_tolerance_m == 0.5


# ============================================================
# 配置模型
# ============================================================

def test_geometry_settings_read_from_config():
    config = {
        "pitch": {"length": 100, "width": 64},
        "calibration": {"max_reprojection_error_m": 2.5},
        "rules": {"collision_distance_m": 3.0, "attacking_direction": "left"},
        "analyzer": {"event_cooldown_frames": 7},
    }
    settings = GeometrySettings.from_config(config)
    assert settings.pitch.length == 100.0
    assert settings.calibration.max_reprojection_error_m == 2.5
    assert settings.rules.collision_distance_m == 3.0
    assert settings.rules.attacking_direction == "left"
    assert settings.analyzer.event_cooldown_frames == 7


def test_geometry_settings_accept_legacy_rule_keys():
    """历史配置里的旧键名仍然被认（避免旧配置静默失效）。"""
    settings = GeometrySettings.from_config({
        "rules": {"offside_margin": 1.5, "collision_dist": 3.0,
                  "ball_tol": 0.8, "goal_line_x_tolerance": 0.3},
        "event_cooldown_frames": 12,
    })
    assert settings.rules.offside_margin_m == 1.5
    assert settings.rules.collision_distance_m == 3.0
    assert settings.rules.ball_assignment_tolerance_m == 0.8
    assert settings.rules.goal_line_tolerance_m == 0.3
    assert settings.analyzer.event_cooldown_frames == 12


def test_geometry_settings_reject_bad_values():
    with pytest.raises(ValueError, match="collision_distance_m"):
        GeometrySettings.from_config({"rules": {"collision_distance_m": -1}})
    with pytest.raises(ValueError, match="attacking_direction"):
        GeometrySettings.from_config({"rules": {"attacking_direction": "up"}})


def test_settings_audit_snapshot_lists_every_effective_value():
    audit = GeometrySettings().to_audit()
    assert set(audit) == {"pitch", "calibration", "rules", "analyzer"}
    assert audit["pitch"]["length"] == PITCH.length
    assert audit["rules"]["goal_line_tolerance_m"] == RuleSettings().ball_radius_m


def test_pitch_module_carries_no_logic_constants():
    """场地尺寸只应存在于 PitchSpec, 模块级不应再散落尺寸常量。"""
    for name in ("PITCH_LENGTH", "PITCH_WIDTH", "GOAL_WIDTH", "PENALTY_AREA_DEPTH"):
        assert not hasattr(pitch_module, name), f"{name} 不应再是模块级常量"


# ============================================================
# 分析器: 端到端
# ============================================================

def _box(role, foot_pixel, *, team=None, conf=0.9, width=30.0, height=80.0):
    px, py = foot_pixel
    return {"cls": role, "conf": conf, "team": team,
            "cx": px, "cy": py - height / 2.0, "w": width, "h": height,
            "x1": px - width / 2.0, "y1": py - height, "x2": px + width / 2.0, "y2": py}


def _frame(index, boxes, timestamp=None):
    return {"frame_index": index, "timestamp": timestamp, "boxes": boxes}


def test_analyzer_without_calibration_emits_nothing():
    """★ 回归测试: 没有真标定时不产出任何几何事件, 且如实上报未标定。"""
    analyzer = PitchAnalyzer(1234, 567, config={})
    assert analyzer.ok is False
    assert analyzer.calibrated is False
    assert analyzer.skip_reason
    to_pixel = _projector()
    frame = _frame(0, [
        _box("player", to_pixel(30.0, 30.0)),
        _box("player", to_pixel(30.8, 30.0)),
        _box("ball", to_pixel(30.4, 30.5)),
    ])
    assert analyzer.analyze(frame) == []


def test_analyzer_emits_contact_candidate_with_calibration():
    analyzer = PitchAnalyzer(*IMAGE_SIZE, config={},
                             calibration=_synthetic_calibration())
    assert analyzer.calibrated is True
    to_pixel = _projector()
    events = analyzer.analyze(_frame(0, [
        _box("player", to_pixel(30.0, 30.0)),
        _box("player", to_pixel(30.8, 30.0)),          # 相距 0.8 m
        _box("ball", to_pixel(50.0, 40.0)),
    ]))
    contacts = [e for e in events if e["type"] == "collision"]
    assert len(contacts) == 1
    assert contacts[0]["distance_m"] == pytest.approx(0.8, abs=0.05)
    assert contacts[0]["decision_status"] == "pending_human_review"


def test_analyzer_calls_offside_when_teams_are_known():
    """★ 回归测试: check_offside 过去从未被调用; 现在必须真的产出越位候选。"""
    analyzer = PitchAnalyzer(*IMAGE_SIZE,
                             config={"rules": {"attacking_direction": "right",
                                               "attacking_team": "A"}},
                             calibration=_synthetic_calibration())
    to_pixel = _projector()
    events = analyzer.analyze(_frame(0, [
        _box("player", to_pixel(98.5, 30.0), team="A"),     # 越位
        _box("player", to_pixel(80.0, 34.0), team="A"),
        # 三名防守球员: 离本方球门线最近的是 99, 所以倒数第二 = 97 → 越位线 x=97
        _box("player", to_pixel(99.0, 50.0), team="B"),
        _box("player", to_pixel(97.0, 34.0), team="B"),
        _box("player", to_pixel(95.0, 20.0), team="B"),
        _box("ball", to_pixel(80.5, 34.0)),
    ]))
    offsides = [e for e in events if e["type"] == "offside"]
    assert len(offsides) == 1
    assert offsides[0]["team"] == "A"
    assert offsides[0]["offside_line_x"] == pytest.approx(97.0)
    assert offsides[0]["attacking_direction_inferred"] is False
    assert "kick_moment" in offsides[0]["requires_evidence"]


def test_analyzer_reports_why_offside_is_skipped():
    """没有队伍标签时越位不产出, 但必须给出可查的原因（而不是静默）。"""
    analyzer = PitchAnalyzer(*IMAGE_SIZE, config={},
                             calibration=_synthetic_calibration())
    to_pixel = _projector()
    events = analyzer.analyze(_frame(0, [
        _box("player", to_pixel(98.5, 30.0)),
        _box("ball", to_pixel(80.5, 34.0)),
    ]))
    assert [e for e in events if e["type"] == "offside"] == []
    assert "球队归属" in analyzer.summary()["offside_skipped_reason"]


def test_analyzer_cooldown_suppresses_repeats():
    """同一对球员连续多帧都贴近, 只应报一次。"""
    analyzer = PitchAnalyzer(*IMAGE_SIZE, config={},
                             calibration=_synthetic_calibration())
    to_pixel = _projector()
    total = 0
    for index in range(5):
        total += len([e for e in analyzer.analyze(_frame(index, [
            _box("player", to_pixel(30.0, 30.0)),
            _box("player", to_pixel(30.8, 30.0)),
            _box("ball", to_pixel(50.0, 40.0)),
        ])) if e["type"] == "collision"])
    assert total == 1


def test_analyzer_cooldown_comes_from_config():
    """把冷却帧数设为 1, 每帧都会重新报 —— 证明它来自配置。"""
    analyzer = PitchAnalyzer(*IMAGE_SIZE,
                             config={"analyzer": {"event_cooldown_frames": 1}},
                             calibration=_synthetic_calibration())
    to_pixel = _projector()
    total = 0
    for index in range(5):
        total += len([e for e in analyzer.analyze(_frame(index, [
            _box("player", to_pixel(30.0, 30.0)),
            _box("player", to_pixel(30.8, 30.0)),
            _box("ball", to_pixel(50.0, 40.0)),
        ])) if e["type"] == "collision"])
    assert total == 5


def test_analyzer_summary_is_auditable():
    analyzer = PitchAnalyzer(*IMAGE_SIZE, config={},
                             calibration=_synthetic_calibration())
    summary = analyzer.summary()
    assert summary["calibrated"] is True
    assert summary["calibration"]["points"] >= 4
    assert summary["tracker"]["backend"] == "botsort"
    assert summary["settings"]["analyzer"]["event_cooldown_frames"] == 30
    assert summary["settings"]["rules"]["collision_distance_m"] == 2.0