"""Catch import failures and elementary geometry errors without model downloads."""
from od_platform.action_recognition import ActionRecognizer
from od_platform.geometric import FieldHomography, PitchCalibration, RuleEngine, RuleSettings
from od_platform.tracking import MultiObjectTracker


def engine(settings=None):
    pixels = [(10.0, 10.0), (220.0, 10.0), (220.0, 146.0), (10.0, 146.0)]
    ground = [(0.0, 0.0), (105.0, 0.0), (105.0, 68.0), (0.0, 68.0)]
    calibration = PitchCalibration(image_points=pixels, pitch_points=ground,
                                   image_size=(320, 240), source="program-smoke-fixture")
    return RuleEngine(FieldHomography.from_calibration(calibration), settings=settings)


def test_geometry_module_imports_and_uses_configured_thresholds():
    value = engine(RuleSettings(penalty_spot_tolerance_m=0.5, collision_distance_m=1.5))
    assert value.settings.penalty_spot_tolerance_m == 0.5
    assert value.settings.collision_distance_m == 1.5


def test_set_piece_rules_emit_candidates_on_a_calibrated_pitch():
    """定位球判据（本次新增）在真标定下要能产出候选 —— 角球弧与罚球点各测一个。"""
    value = engine()
    corner = value.check_corner_kick((0.5, 0.5))              # 左上角球弧内
    assert corner is not None and corner["set_piece"] == "corner_kick"
    assert value.check_corner_kick((52.5, 34.0)) is None      # 中圈不是角球

    spot = value.check_penalty_kick((11.0, 34.0))              # 左侧罚球点
    assert spot is not None and spot["side"] == "left"


def test_unloaded_action_model_is_not_ready_and_returns_no_actions():
    recognizer = ActionRecognizer({"model_name": "custom", "checkpoint": ""})
    assert recognizer.ready is False
    assert recognizer.predict([]) == []
    assert callable(MultiObjectTracker)
