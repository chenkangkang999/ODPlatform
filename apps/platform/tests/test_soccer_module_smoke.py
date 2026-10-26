"""Catch import failures and elementary geometry errors without model downloads."""
import pytest

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
    value = engine(RuleSettings(offside_margin_m=0.5, collision_distance_m=1.5))
    assert value.settings.offside_margin_m == 0.5
    assert value.settings.collision_distance_m == 1.5


@pytest.mark.parametrize("direction,defenders,attackers,ball,expected", [
    ("right", [(10, 0), (30, 0), (70, 0), (90, 0)], [(50, 0), (80, 0)], (60, 0), [1]),
    ("left", [(10, 0), (30, 0), (70, 0), (90, 0)], [(20, 0), (50, 0)], (40, 0), [0]),
])
def test_offside_candidates_use_second_opponent_from_target_goal(direction, defenders, attackers, ball, expected):
    candidates = engine().check_offside(attackers, defenders, ball, direction)
    assert [candidate["player_idx"] for candidate in candidates] == expected


def test_inferred_second_opponent_is_marked_as_needing_evidence():
    candidates = engine().check_offside([(100, 0)], [(20, 0)], (0, 0))
    assert len(candidates) == 1
    assert candidates[0]["second_last_inferred"] is True
    assert candidates[0]["decision_status"] == "needs_evidence"
    assert "kick_moment" in candidates[0]["requires_evidence"]


def test_unloaded_action_model_is_not_ready_and_returns_no_actions():
    recognizer = ActionRecognizer({"model_name": "custom", "checkpoint": ""})
    assert recognizer.ready is False
    assert recognizer.predict([]) == []
    assert callable(MultiObjectTracker)
