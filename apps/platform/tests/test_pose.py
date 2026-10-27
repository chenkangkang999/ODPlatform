#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""姿态估计与姿态型动作判据的测试。

这一组不依赖真模型、不依赖权重、不依赖网络 —— 全部用**注入的假估计器**与
**手摆的关键点坐标**。理由: 判据是几何, 就该能用确定的坐标验证; 让测试去碰
5MB 的权重只会让 CI 变脆。

几何基准（STANDING, 图像 y 向下为正）:
    肩中点 (100, 70) / 髋中点 (100, 140) → 躯干长度 = 70
    外接框 (70, 20, 130, 220)            → 身高 = 200
    髋部高度比 = (220-140)/200 = 0.40    （站立）
    躯干倾角   = 0°                       （竖直）
    膝伸展角   ≈ 179°                     （直立腿）
    踝高于髋比 = (140-215)/70 ≈ -1.07     （踝在髋下方, 正常站立）
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest

from od_platform.pose import (
    ACTION_ELBOW_STRIKE,
    ACTION_HIGH_LEG,
    ACTION_JUDGES,
    ACTION_PULLING_PUSHING,
    ACTION_SLIDING_TACKLE,
    COCO_KEYPOINT_NAMES,
    EVIDENCE_CANDIDATE,
    EVIDENCE_INSUFFICIENT,
    EVIDENCE_JUDGEMENT,
    HIPS,
    KEYPOINT_INDEX,
    SHOULDERS,
    Keypoint,
    PersonPose,
    PoseEstimator,
    PoseModelSettings,
    PoseSettings,
    PoseThresholds,
    angle_at,
    default_config_path,
    judge_all,
    judge_elbow_strike,
    judge_high_leg,
    judge_pulling_pushing,
    judge_sliding_tackle,
)
from od_platform.pose.estimator import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)

# ---------------------------------------------------------------- 几何基准

STANDING: Dict[str, Tuple[float, float]] = {
    NOSE: (100, 40), "left_eye": (95, 36), "right_eye": (105, 36),
    "left_ear": (88, 38), "right_ear": (112, 38),
    LEFT_SHOULDER: (80, 70), RIGHT_SHOULDER: (120, 70),
    LEFT_ELBOW: (75, 110), RIGHT_ELBOW: (125, 110),
    LEFT_WRIST: (72, 145), RIGHT_WRIST: (128, 145),
    LEFT_HIP: (85, 140), RIGHT_HIP: (115, 140),
    LEFT_KNEE: (88, 180), RIGHT_KNEE: (112, 180),
    LEFT_ANKLE: (90, 215), RIGHT_ANKLE: (110, 215),
}
BBOX = (70.0, 20.0, 130.0, 220.0)
BBOX_TOP, BBOX_BOTTOM = BBOX[1], BBOX[3]
BODY_HEIGHT = BBOX_BOTTOM - BBOX_TOP          # 200
TORSO_LENGTH = 70.0


def make_pose(*, bbox=BBOX, conf=0.9, index=0, unavailable=(),
              **moves) -> PersonPose:
    """造一个人。

    Args:
        bbox: 外接框
        conf: 所有关键点的置信度
        index: 帧内序号
        unavailable: 要标成"不可用"的关键点名（模拟置信度不足或缺失）
        **moves: 关键点名=(x, y), 覆盖基准坐标
    """
    points = dict(STANDING)
    points.update(moves)
    keypoints = {
        name: Keypoint(x=float(x), y=float(y), conf=conf,
                       available=name not in unavailable)
        for name, (x, y) in points.items()
    }
    return PersonPose(keypoints=keypoints, confidence=conf, bbox=tuple(bbox), index=index)


def shift(pose: PersonPose, dx: float, dy: float) -> PersonPose:
    """整体平移一个人的所有关键点（用于摆对手的位置）。"""
    return PersonPose(
        keypoints={name: Keypoint(kp.x + dx, kp.y + dy, kp.conf, kp.available)
                   for name, kp in pose.keypoints.items()},
        confidence=pose.confidence,
        bbox=(pose.bbox[0] + dx, pose.bbox[1] + dy, pose.bbox[2] + dx, pose.bbox[3] + dy),
        index=pose.index)


def hips_at_ratio(ratio: float, *, bbox=BBOX, x_left: float = 85.0,
                  x_right: float = 115.0) -> Dict[str, Tuple[float, float]]:
    """生成一组髋坐标, 使其髋部高度比**正好等于**给定值（用于边界测试）。

    必须传入实际使用的那个 bbox —— 身高/最低点不同, 同一个 y 算出来的比值不同。
    """
    y = bbox[3] - ratio * (bbox[3] - bbox[1])
    return {LEFT_HIP: (x_left, y), RIGHT_HIP: (x_right, y)}


# ---------------------------------------------------------------- 假估计器

class FakeTensor:
    """模拟 torch.Tensor 的最小接口（cpu()/numpy()）。"""

    def __init__(self, data):
        self._data = np.asarray(data, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._data


class FakeBoxes:
    def __init__(self, xyxy, conf):
        self.xyxy = None if xyxy is None else FakeTensor(xyxy)
        self.conf = None if conf is None else FakeTensor(conf)


class FakeKeypoints:
    def __init__(self, xy, conf):
        self.xy = None if xy is None else FakeTensor(xy)
        self.conf = None if conf is None else FakeTensor(conf)


class FakeResult:
    def __init__(self, boxes=None, keypoints=None):
        self.boxes = boxes
        self.keypoints = keypoints


class FakeModel:
    """假 pose 模型: 记录调用参数, 返回预设结果。"""

    def __init__(self, results):
        self._results = results
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append(kwargs)
        return self._results


class ExplodingModel:
    def predict(self, image, **kwargs):
        raise RuntimeError("模拟推理失败")


def keypoint_arrays(points: Dict[str, Tuple[float, float]], conf: float = 0.9):
    """按 COCO 顺序把 {点名: (x,y)} 转成模型的 (1,17,2) 与 (1,17) 数组。"""
    xy = np.zeros((1, len(COCO_KEYPOINT_NAMES), 2), dtype=float)
    kconf = np.full((1, len(COCO_KEYPOINT_NAMES)), conf, dtype=float)
    for name, index in KEYPOINT_INDEX.items():
        if name in points:
            xy[0, index] = points[name]
    return xy, kconf


def fake_model_for(people, *, box_conf=0.9, kpt_conf=0.9, boxes=None) -> FakeModel:
    """用一组 {点名: (x,y)} 造一个假模型, 一次返回所有人。

    Args:
        boxes: 显式指定每个人的外接框; None 时按关键点范围取。
               需要精确控制"髋部高度比"这类依赖外接框的判据时必须显式给。
    """
    xyxy, confs, kxy, kconfs = [], [], [], []
    for index, person in enumerate(people):
        if boxes is not None:
            xyxy.append(list(boxes[index]))
        else:
            xs = [p[0] for p in person.values()]
            ys = [p[1] for p in person.values()]
            xyxy.append([min(xs), min(ys), max(xs), max(ys)])
        confs.append(box_conf)
        xy, kc = keypoint_arrays(person, kpt_conf)
        kxy.append(xy[0])
        kconfs.append(kc[0])
    result = FakeResult(
        boxes=FakeBoxes(np.array(xyxy, dtype=float), np.array(confs, dtype=float)),
        keypoints=FakeKeypoints(np.array(kxy, dtype=float), np.array(kconfs, dtype=float)))
    return FakeModel([result])


# ================================================================
# 一、配置层
# ================================================================

def test_defaults_are_valid_and_audit_is_complete():
    settings = PoseSettings()
    audit = settings.to_audit()
    assert set(audit) == {"model", "thresholds", "allow_uncalibrated_candidate"}
    assert audit["model"]["imgsz"] == 640
    assert audit["thresholds"]["high_leg_ankle_above_hip_ratio"] == 0.10
    # 审计快照必须能序列化（要写进结果文件）
    json.dumps(audit, ensure_ascii=False)


@pytest.mark.parametrize("kwargs, expected", [
    ({"imgsz": 0}, "imgsz"),
    ({"imgsz": -1}, "imgsz"),
    ({"weights": ""}, "weights"),
    ({"conf": 1.5}, "conf"),
    ({"conf": -0.1}, "conf"),
    ({"iou": 2.0}, "iou"),
    ({"keypoint_conf": -0.5}, "keypoint_conf"),
    ({"max_persons": 0}, "max_persons"),
])
def test_invalid_model_settings_raise(kwargs, expected):
    with pytest.raises(ValueError, match=expected):
        PoseModelSettings(**kwargs)


@pytest.mark.parametrize("kwargs, expected", [
    ({"high_leg_ankle_above_hip_ratio": -0.1}, "high_leg_ankle_above_hip_ratio"),
    ({"slide_hip_height_ratio": -1.0}, "slide_hip_height_ratio"),
    ({"slide_knee_extension_deg": 0}, "slide_knee_extension_deg"),
    ({"slide_knee_extension_deg": 200}, "slide_knee_extension_deg"),
    ({"slide_torso_incline_deg": -5}, "slide_torso_incline_deg"),
    ({"elbow_abduction_deg": 0}, "elbow_abduction_deg"),
    ({"elbow_abduction_deg": 181}, "elbow_abduction_deg"),
    ({"pull_contact_distance_m": 0}, "pull_contact_distance_m"),
    ({"min_hit_confidence": 1.5}, "min_hit_confidence"),
])
def test_invalid_thresholds_raise(kwargs, expected):
    with pytest.raises(ValueError, match=expected):
        PoseThresholds(**kwargs)


def test_yaml_overrides_apply_by_field_name():
    settings = PoseSettings.from_config({
        "model": {"imgsz": 1280, "device": "cpu", "keypoint_conf": 0.7},
        "thresholds": {"elbow_abduction_deg": 75.0, "pull_contact_distance_m": 1.5},
        "allow_uncalibrated_candidate": False,
    })
    assert settings.model.imgsz == 1280
    assert settings.model.device == "cpu"
    assert settings.model.keypoint_conf == 0.7
    assert settings.thresholds.elbow_abduction_deg == 75.0
    assert settings.thresholds.pull_contact_distance_m == 1.5
    assert settings.allow_uncalibrated_candidate is False
    # 没覆盖的项保持默认
    assert settings.model.weights == PoseModelSettings().weights


def test_embedded_pose_section_is_unwrapped():
    """嵌在 football.yaml 里的 pose: 节也要能读（与 action_recognition 同一约定）。"""
    settings = PoseSettings.from_config({"pose": {"model": {"imgsz": 960}}})
    assert settings.model.imgsz == 960


def test_unknown_keys_are_reported(caplog):
    with caplog.at_level(logging.DEBUG):
        PoseSettings.from_config({"model": {"not_a_field": 1},
                                  "thresholds": {"also_unknown": 2},
                                  "top_unknown": 3})
    text = caplog.text
    assert "not_a_field" in text
    assert "also_unknown" in text
    assert "top_unknown" in text


def test_empty_or_invalid_config_falls_back_to_defaults():
    assert PoseSettings.from_config(None) == PoseSettings()
    assert PoseSettings.from_config({}) == PoseSettings()
    assert PoseSettings.from_config("不是映射") == PoseSettings()


def test_repo_pose_yaml_matches_the_code_defaults():
    """仓库里那份 pose.yaml 必须与 dataclass 默认值一致。

    这是**契约测试**: 配置文件是给人看的文档, 一旦有人改了 dataclass 默认值却忘了
    同步 yaml（或反过来), 页面上写着 0.10 而实际跑的是别的值 —— 立刻失败。
    """
    path = default_config_path()
    assert path.is_file(), f"缺少配置文件: {path}"
    loaded = PoseSettings.load(path)
    assert loaded == PoseSettings(), "pose.yaml 与代码默认值不一致"


def test_missing_config_file_uses_defaults(tmp_path):
    assert PoseSettings.load(tmp_path / "nope.yaml") == PoseSettings()


def test_invalid_yaml_top_level_raises(tmp_path):
    path = tmp_path / "pose.yaml"
    path.write_text("- 这是一个列表\n- 不是映射\n", encoding="utf-8")
    with pytest.raises(ValueError):
        PoseSettings.load(path)


def test_resolved_weights_prefers_a_local_file(tmp_path, monkeypatch):
    """本地有权重就给路径（离线环境唯一可靠的路）, 没有才把名字交给 ultralytics。"""
    import od_platform.pose.settings as settings_module
    monkeypatch.setattr(settings_module, "PRETRAINED_MODELS_DIR", tmp_path)
    missing = PoseModelSettings(weights="yolo11n-pose.pt")
    assert missing.resolved_weights() == "yolo11n-pose.pt"      # 本地没有 → 原样返回

    (tmp_path / "yolo11n-pose.pt").write_bytes(b"stub")
    present = PoseModelSettings(weights="yolo11n-pose.pt")
    assert Path(present.resolved_weights()) == tmp_path / "yolo11n-pose.pt"


def test_resolved_weights_keeps_an_explicit_absolute_path(tmp_path):
    weights = tmp_path / "custom.pt"
    weights.write_bytes(b"stub")
    assert PoseModelSettings(weights=str(weights)).resolved_weights() == str(weights)


# ================================================================
# 二、PersonPose 的几何访问器
# ================================================================

def test_centers_and_torso_length():
    pose = make_pose()
    assert pose.shoulder_center() == pytest.approx((100.0, 70.0))
    assert pose.hip_center() == pytest.approx((100.0, 140.0))
    assert pose.torso_center() == pytest.approx((100.0, 105.0))
    assert pose.torso_length() == pytest.approx(TORSO_LENGTH)
    assert pose.body_height == pytest.approx(BODY_HEIGHT)
    assert pose.hip_height_ratio() == pytest.approx(0.40)
    assert pose.torso_incline_deg() == pytest.approx(0.0, abs=1e-6)


def test_unavailable_keypoints_are_invisible_to_geometry():
    pose = make_pose(unavailable=(LEFT_HIP,))
    assert pose.get(LEFT_HIP) is None
    assert pose.point(LEFT_HIP) is None
    # 髋中心退化成"只剩右髋", 而不是把缺失点当 0（那会把中心拉到图像左上角）
    assert pose.hip_center() == pytest.approx((115.0, 140.0))
    assert LEFT_HIP in pose.missing(HIPS)
    assert LEFT_HIP not in pose.available_names()


def test_center_of_requires_enough_points():
    pose = make_pose(unavailable=(LEFT_SHOULDER, RIGHT_SHOULDER))
    assert pose.shoulder_center() is None
    assert pose.torso_length() is None          # 基准不可用 → 明确返回 None
    assert pose.center_of(SHOULDERS, min_available=2) is None


def test_missing_lists_are_deduplicated_and_sorted_in_verdict():
    verdict = judge_high_leg(make_pose(unavailable=(LEFT_ANKLE, RIGHT_ANKLE,
                                                   LEFT_HIP, RIGHT_HIP)))
    assert verdict.evidence_level == EVIDENCE_INSUFFICIENT
    assert verdict.hit is False
    assert verdict.missing == sorted(set(verdict.missing))


def test_hip_height_ratio_none_when_bbox_is_degenerate():
    pose = make_pose(bbox=(10.0, 10.0, 20.0, 10.0))     # 高度 0
    assert pose.hip_height_ratio() is None


# ================================================================
# 三、角度helper
# ================================================================

def test_angle_at_right_angle_and_straight_line():
    assert angle_at((0, 0), (1, 0), (0, 1)) == pytest.approx(90.0)
    assert angle_at((0, 0), (-1, 0), (1, 0)) == pytest.approx(180.0)


def test_angle_at_degenerate_returns_none_not_zero():
    """零长向量必须返回 None —— 返回 0 会被下游当成"完全折叠", 是假的结论。"""
    assert angle_at((5, 5), (5, 5), (10, 10)) is None
    assert angle_at((5, 5), (10, 10), (5, 5)) is None


# ================================================================
# 四、高抬腿
# ================================================================

def test_high_leg_not_hit_when_standing():
    verdict = judge_high_leg(make_pose())
    assert verdict.hit is False
    assert verdict.evidence_level == EVIDENCE_JUDGEMENT
    assert verdict.measurements["踝高于髋的比例"] == pytest.approx(-1.0714, abs=1e-3)


def test_high_leg_boundary_exactly_at_threshold_is_hit():
    """刚好等于门槛算命中（判据是 >=）。"""
    ankle_y = 140.0 - 0.10 * TORSO_LENGTH          # 比值正好 0.10
    verdict = judge_high_leg(make_pose(**{LEFT_ANKLE: (90.0, ankle_y)}))
    assert verdict.measurements["踝高于髋的比例"] == pytest.approx(0.10)
    assert verdict.hit is True


def test_high_leg_boundary_just_below_threshold_is_not_hit():
    ankle_y = 140.0 - 0.09 * TORSO_LENGTH          # 比值 0.09
    verdict = judge_high_leg(make_pose(**{LEFT_ANKLE: (90.0, ankle_y)}))
    assert verdict.measurements["踝高于髋的比例"] == pytest.approx(0.09)
    assert verdict.hit is False


def test_high_leg_takes_the_higher_of_two_legs():
    """左腿没抬、右腿抬了 → 命中, 且理由里点明是右腿。"""
    verdict = judge_high_leg(make_pose(**{RIGHT_ANKLE: (110.0, 70.0)}))
    assert verdict.hit is True
    assert "right" in " ".join(verdict.reasons)


def test_high_leg_needs_only_one_usable_ankle():
    """一条腿的关键点不可用不该让整条判据变成"证据不足"（另一条腿还能判）。"""
    verdict = judge_high_leg(make_pose(unavailable=(LEFT_ANKLE,),
                                       **{RIGHT_ANKLE: (110.0, 60.0)}))
    assert verdict.evidence_level == EVIDENCE_JUDGEMENT
    assert verdict.hit is True


def test_high_leg_without_any_ankle_is_insufficient():
    verdict = judge_high_leg(make_pose(unavailable=(LEFT_ANKLE, RIGHT_ANKLE)))
    assert verdict.evidence_level == EVIDENCE_INSUFFICIENT
    assert set(verdict.missing) >= {LEFT_ANKLE, RIGHT_ANKLE}


def test_high_leg_insufficient_when_torso_scale_unusable():
    """肩/髋不可用 → 归一化基准没有 → 不许猜。"""
    verdict = judge_high_leg(make_pose(unavailable=(LEFT_SHOULDER, RIGHT_SHOULDER)))
    assert verdict.evidence_level == EVIDENCE_INSUFFICIENT


# ================================================================
# 五、滑铲
# ================================================================

SLIDING_MOVES = {
    LEFT_SHOULDER: (60.0, 150.0), RIGHT_SHOULDER: (75.0, 160.0),
    LEFT_HIP: (100.0, 180.0), RIGHT_HIP: (110.0, 185.0),
    LEFT_KNEE: (120.0, 195.0), LEFT_ANKLE: (105.0, 215.0),      # 屈腿
    RIGHT_KNEE: (150.0, 190.0), RIGHT_ANKLE: (190.0, 195.0),    # 伸腿
    NOSE: (55.0, 140.0),
}
SLIDING_BBOX = (50.0, 130.0, 200.0, 200.0)


def sliding_pose(**overrides) -> PersonPose:
    moves = dict(SLIDING_MOVES)
    moves.update(overrides)
    return make_pose(bbox=SLIDING_BBOX, **moves)


def test_sliding_tackle_hit():
    verdict = judge_sliding_tackle(sliding_pose())
    assert verdict.hit is True
    assert verdict.evidence_level == EVIDENCE_JUDGEMENT
    assert verdict.measurements["膝关节伸展角"] == pytest.approx(180.0, abs=0.5)
    assert verdict.measurements["躯干倾角"] > 45.0
    assert verdict.measurements["髋部高度比"] < 0.35


def test_sliding_tackle_not_hit_when_standing():
    verdict = judge_sliding_tackle(make_pose())
    assert verdict.hit is False
    assert verdict.measurements["髋部高度比"] == pytest.approx(0.40)


def test_sliding_tackle_hip_height_boundary():
    """髋部高度比刚好等于门槛算命中; 略高一点就不算。"""
    at = sliding_pose(**hips_at_ratio(0.35, bbox=SLIDING_BBOX, x_left=100.0, x_right=110.0))
    assert judge_sliding_tackle(at).measurements["髋部高度比"] == pytest.approx(0.35)
    assert judge_sliding_tackle(at).hit is True

    above = sliding_pose(**hips_at_ratio(0.36, bbox=SLIDING_BBOX, x_left=100.0, x_right=110.0))
    assert judge_sliding_tackle(above).hit is False


def test_sliding_tackle_requires_a_straight_knee():
    """髋沉下去 + 躯干倒了, 但两条腿都是屈的 → 不命中（三条判据缺一不可）。"""
    verdict = judge_sliding_tackle(sliding_pose(**{RIGHT_KNEE: (140.0, 210.0),
                                                   RIGHT_ANKLE: (110.0, 190.0)}))
    assert verdict.measurements["膝关节伸展角"] < 150.0
    assert verdict.hit is False


def test_sliding_tackle_requires_torso_incline():
    """髋沉了、腿直了, 但躯干还立着 → 不命中。"""
    upright = sliding_pose(**{LEFT_SHOULDER: (95.0, 130.0), RIGHT_SHOULDER: (105.0, 130.0)})
    verdict = judge_sliding_tackle(upright)
    assert verdict.measurements["躯干倾角"] < 45.0
    assert verdict.hit is False


def test_sliding_tackle_insufficient_without_legs():
    verdict = judge_sliding_tackle(make_pose(unavailable=(LEFT_KNEE, RIGHT_KNEE,
                                                         LEFT_ANKLE, RIGHT_ANKLE)))
    assert verdict.evidence_level == EVIDENCE_INSUFFICIENT


# ================================================================
# 六、肘击
# ================================================================

ELBOW_RAISED = {LEFT_ELBOW: (160.0, 70.0)}          # 上臂水平抬起 → 外展约 86°


def opponent_with_head_at(x: float, y: float, index: int = 1) -> PersonPose:
    """造一个对手, 使其头部中心落在 (x, y)。"""
    base = make_pose(index=index)
    head = base.head_center()
    return shift(base, x - head[0], y - head[1])


def test_elbow_strike_hit():
    striker = make_pose(**ELBOW_RAISED)
    verdict = judge_elbow_strike(striker, [opponent_with_head_at(170.0, 75.0)])
    assert verdict.hit is True
    assert verdict.evidence_level == EVIDENCE_JUDGEMENT
    assert verdict.measurements["肩关节外展角"] == pytest.approx(85.9, abs=1.5)
    assert verdict.measurements["肘到最近对手头部距离"] < 1.0


def test_elbow_strike_not_hit_when_arm_is_down():
    """手臂自然下垂: 外展角约 15°, 不达门槛。"""
    verdict = judge_elbow_strike(make_pose(), [opponent_with_head_at(105.0, 110.0)])
    assert verdict.measurements["肩关节外展角"] < 60.0
    assert verdict.hit is False


def test_elbow_strike_abduction_boundary():
    """外展角门槛的两侧: 略低于门槛不命中, 略高于门槛命中。

    这里刻意留 0.5° 的余量而不是卡在门槛上 —— 门槛处是浮点刀锋, 旋转构造出来的
    角度必然带 1e-13 级误差, 卡死会变成"测试在赌浮点"。真正的 >= 语义由
    test_high_leg_boundary_exactly_at_threshold_is_hit 用干净的 0.10 比值验证。
    """
    thresholds = PoseThresholds()
    shoulder, hip = STANDING[LEFT_SHOULDER], STANDING[LEFT_HIP]

    def elbow_for(angle_deg: float) -> Tuple[float, float]:
        angle = math.radians(angle_deg)
        vx, vy = hip[0] - shoulder[0], hip[1] - shoulder[1]
        length = math.hypot(vx, vy)
        ux, uy = vx / length, vy / length
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        return (shoulder[0] + 50.0 * (ux * cos_a - uy * sin_a),
                shoulder[1] + 50.0 * (ux * sin_a + uy * cos_a))

    below = elbow_for(thresholds.elbow_abduction_deg - 0.5)
    verdict = judge_elbow_strike(make_pose(**{LEFT_ELBOW: below}),
                                 [opponent_with_head_at(below[0], below[1] - 20.0)])
    assert verdict.measurements["肩关节外展角"] < thresholds.elbow_abduction_deg
    assert verdict.hit is False

    above = elbow_for(thresholds.elbow_abduction_deg + 0.5)
    verdict = judge_elbow_strike(make_pose(**{LEFT_ELBOW: above}),
                                 [opponent_with_head_at(above[0], above[1] - 20.0)])
    assert verdict.measurements["肩关节外展角"] > thresholds.elbow_abduction_deg
    assert verdict.hit is True


def test_elbow_strike_not_hit_when_head_far():
    striker = make_pose(**ELBOW_RAISED)
    verdict = judge_elbow_strike(striker, [opponent_with_head_at(400.0, 400.0)])
    assert verdict.measurements["肘到最近对手头部距离"] > 1.0
    assert verdict.hit is False


def test_elbow_strike_without_opponent_is_insufficient():
    """肘击必须有个被肘的人 —— 场上没人时不能"猜"命中。"""
    verdict = judge_elbow_strike(make_pose(**ELBOW_RAISED), [])
    assert verdict.hit is False
    assert verdict.evidence_level == EVIDENCE_INSUFFICIENT
    assert any("对手" in reason for reason in verdict.reasons)


def test_elbow_strike_ignores_opponent_without_a_head():
    verdict = judge_elbow_strike(make_pose(**ELBOW_RAISED),
                                 [make_pose(unavailable=("nose", "left_eye", "right_eye",
                                                         "left_ear", "right_ear"))])
    assert verdict.evidence_level == EVIDENCE_INSUFFICIENT


def test_elbow_strike_picks_the_nearest_opponent():
    striker = make_pose(**ELBOW_RAISED)
    far = opponent_with_head_at(500.0, 500.0, index=1)
    near = opponent_with_head_at(168.0, 74.0, index=2)
    verdict = judge_elbow_strike(striker, [far, near])
    assert verdict.hit is True
    assert "#2" in " ".join(verdict.reasons)


# ================================================================
# 七、拉人推人
# ================================================================

REACHING_WRIST = (10.0, 85.0)        # 左腕伸向左前方（明显抬离身体）
TARGET_TORSO_AT = (40.0, 85.0)       # 对手躯干中心, 距左腕 30 像素
BASE_TORSO_CENTER = (100.0, 105.0)   # STANDING 的躯干中心
TARGET_SHIFT = (TARGET_TORSO_AT[0] - BASE_TORSO_CENTER[0],
                TARGET_TORSO_AT[1] - BASE_TORSO_CENTER[1])


def pusher_and_target() -> Tuple[PersonPose, PersonPose]:
    """造"伸臂的人"与"在 30 像素外的对手"。

    伸臂: 左肩(80,70) → 左腕(10,85)
        手臂张开角 = (肩-腕) 与 (肩-左髋) 的夹角 ≈ 82°   ≥ 门槛 45°
        手臂伸展度 = |肩-腕| / 70 ≈ 1.02                 ≥ 门槛 0.90
    对手: 平移 STANDING, 使躯干中心落在 (40,85) → 距左腕 30 像素
        · 无标定: 30/70 ≈ 0.43 躯干长 ≤ 1.20 → 候选命中
        · 有标定: 30 像素 × (1/20) 米/像素 = 1.5 m ≤ 2.0 → 判据命中
    """
    pusher = make_pose(**{LEFT_WRIST: REACHING_WRIST})
    target = shift(make_pose(index=1), *TARGET_SHIFT)
    return pusher, target


def test_pulling_pushing_geometry_baseline():
    """先把基准几何钉住, 让下面几条"差一点点"的测试有可信参照。

    否则阈值一改, 没人分得清失败的是被测逻辑还是测试里手算的坐标。
    """
    pusher, target = pusher_and_target()
    shoulder = pusher.point(LEFT_SHOULDER)
    wrist = pusher.point(LEFT_WRIST)
    assert angle_at(shoulder, wrist, pusher.point(LEFT_HIP)) == pytest.approx(82.0, abs=1.5)
    assert (math.dist(shoulder, wrist) / pusher.torso_length()
            == pytest.approx(1.02, abs=0.03))
    assert math.dist(wrist, target.torso_center()) == pytest.approx(30.0)


def test_pulling_pushing_hit_with_calibration_is_a_judgement():
    pusher, target = pusher_and_target()
    meters_per_px = 1.0 / 20.0                     # 20 像素 = 1 米

    def distance_fn(p, q):
        return math.dist(p, q) * meters_per_px

    verdict = judge_pulling_pushing(pusher, [target], distance_fn=distance_fn)
    assert verdict.hit is True
    assert verdict.evidence_level == EVIDENCE_JUDGEMENT
    assert verdict.measurements["手到对手躯干距离"] == pytest.approx(1.5)
    assert "m" in " ".join(verdict.reasons)


def test_pulling_pushing_without_calibration_is_only_a_candidate():
    """无标定 → 按身体比例近似 → 只能是候选, 绝不当判据。"""
    pusher, target = pusher_and_target()
    verdict = judge_pulling_pushing(pusher, [target])
    assert verdict.hit is True
    assert verdict.evidence_level == EVIDENCE_CANDIDATE
    assert "未标定" in " ".join(verdict.reasons)


def test_pulling_pushing_contact_distance_boundary_in_meters():
    """门槛两侧: 算出来正好 2.0 m 算命中, 2.01 m 不算。"""
    pusher, target = pusher_and_target()
    distance_px = math.dist(pusher.point(LEFT_WRIST), target.torso_center())

    def at(p, q):
        return math.dist(p, q) * (2.0 / distance_px)

    def beyond(p, q):
        return math.dist(p, q) * (2.01 / distance_px)

    verdict = judge_pulling_pushing(pusher, [target], distance_fn=at)
    assert verdict.measurements["手到对手躯干距离"] == pytest.approx(2.0)
    assert verdict.hit is True
    assert judge_pulling_pushing(pusher, [target], distance_fn=beyond).hit is False


def test_pulling_pushing_requires_the_arm_to_leave_the_body():
    """手臂自然下垂（张开角约 10°）且手正好贴着对手 → **不命中**。

    这是那个设计缺陷的回归测试: 若用"肩-腕距离 / 躯干长"当关键判据, 自然下垂的
    手臂比值是 1.08, 会把每个站着的人都判成在拉人。
    """
    pusher = make_pose()
    wrist = pusher.point(LEFT_WRIST)
    target = shift(make_pose(index=1), wrist[0] - BASE_TORSO_CENTER[0],
                   wrist[1] - BASE_TORSO_CENTER[1])
    verdict = judge_pulling_pushing(pusher, [target])
    assert verdict.measurements["手臂张开角"] < 45.0
    assert verdict.measurements["手臂伸展度"] > 0.90      # 垂着也是伸直的
    assert verdict.hit is False


def test_pulling_pushing_requires_the_arm_not_to_be_folded():
    """张开角够大, 但手离肩很近（折叠）→ 被"伸展度"守卫挡住。"""
    pusher = make_pose(**{LEFT_WRIST: (110.0, 60.0)})
    target = shift(make_pose(index=1), 110.0 - BASE_TORSO_CENTER[0],
                   60.0 - BASE_TORSO_CENTER[1])
    verdict = judge_pulling_pushing(pusher, [target])
    assert verdict.measurements["手臂张开角"] >= 45.0
    assert verdict.measurements["手臂伸展度"] < 0.90
    assert verdict.hit is False


def test_pulling_pushing_disallowed_uncalibrated_is_insufficient():
    """配置不允许无标定近似时, 没有 distance_fn 就明确不下结论。"""
    settings = PoseSettings(allow_uncalibrated_candidate=False)
    pusher, target = pusher_and_target()
    verdict = judge_pulling_pushing(pusher, [target], settings=settings)
    assert verdict.hit is False
    assert verdict.evidence_level == EVIDENCE_INSUFFICIENT
    assert any("标定" in reason for reason in verdict.reasons)


def test_pulling_pushing_tolerates_a_broken_distance_fn():
    """标定实现抛异常时降级, 不炸整条判据。"""
    def broken(p, q):
        raise RuntimeError("单应矩阵损坏")

    pusher, target = pusher_and_target()
    verdict = judge_pulling_pushing(pusher, [target], distance_fn=broken)
    assert verdict.evidence_level in (EVIDENCE_CANDIDATE, EVIDENCE_INSUFFICIENT)
    assert verdict.hit in (True, False)          # 关键是没抛异常


def test_pulling_pushing_without_opponent_is_insufficient():
    verdict = judge_pulling_pushing(make_pose(**{LEFT_WRIST: REACHING_WRIST}), [])
    assert verdict.evidence_level == EVIDENCE_INSUFFICIENT


# ================================================================
# 八、通用行为: 降级、可解释、注册表
# ================================================================

@pytest.mark.parametrize("judge, need_pair", [
    (judge_high_leg, False), (judge_sliding_tackle, False),
    (judge_elbow_strike, True), (judge_pulling_pushing, True),
])
def test_every_judge_is_insufficient_without_keypoints(judge, need_pair):
    """没有任何关键点时必须给"证据不足", 而不是猜一个结论。"""
    empty = PersonPose(keypoints={}, confidence=0.0, bbox=(0.0, 0.0, 0.0, 0.0))
    verdict = judge(empty, []) if need_pair else judge(empty)
    assert verdict.hit is False
    assert verdict.evidence_level == EVIDENCE_INSUFFICIENT
    assert verdict.missing
    assert verdict.confidence == 0.0


def test_low_confidence_hit_is_downgraded_to_candidate():
    """几何上确实命中, 但关键点置信度太低 → 降级为候选。"""
    verdict = judge_high_leg(make_pose(conf=0.3, **{LEFT_ANKLE: (90.0, 70.0)}))
    assert verdict.hit is True
    assert verdict.evidence_level == EVIDENCE_CANDIDATE
    assert any("降级为候选" in reason for reason in verdict.reasons)


def test_every_reason_contains_measured_value_and_threshold():
    """每个结论都要能说清"算出来多少、门槛多少" —— 答辩会被追问。"""
    verdict = judge_high_leg(make_pose(**{LEFT_ANKLE: (90.0, 60.0)}))
    text = " ".join(verdict.reasons)
    assert "门槛" in text
    assert "0.100" in text
    assert verdict.measurements


def test_verdict_to_dict_is_json_serialisable():
    verdict = judge_high_leg(make_pose(**{LEFT_ANKLE: (90.0, 60.0)}))
    payload = verdict.to_dict()
    json.dumps(payload, ensure_ascii=False)
    assert set(payload) >= {"action", "hit", "confidence", "evidence_level",
                            "reasons", "missing_keypoints", "measurements"}


def test_judge_all_covers_every_registered_action():
    verdicts = judge_all(make_pose(), [])
    assert [v.action for v in verdicts] == list(ACTION_JUDGES)
    assert len(verdicts) == len(ACTION_JUDGES) == 4
    # 清单里那四项, 一个都不能漏
    assert set(ACTION_JUDGES) == {ACTION_HIGH_LEG, ACTION_SLIDING_TACKLE,
                                  ACTION_ELBOW_STRIKE, ACTION_PULLING_PUSHING}


def test_judge_all_flags_a_high_leg_among_the_four():
    verdicts = {v.action: v for v in judge_all(make_pose(**{LEFT_ANKLE: (90.0, 60.0)}))}
    assert verdicts[ACTION_HIGH_LEG].hit is True
    assert verdicts[ACTION_SLIDING_TACKLE].hit is False
    assert verdicts[ACTION_ELBOW_STRIKE].evidence_level == EVIDENCE_INSUFFICIENT


# ================================================================
# 九、估计器
# ================================================================

def test_estimator_degrades_gracefully_when_weights_are_missing():
    """拿不到权重: 不抛异常、available=False、estimate() 返回空、原因可读。"""
    estimator = PoseEstimator(weights="definitely-not-a-real-weight.pt")
    assert estimator.available is False
    assert estimator.estimate(np.zeros((64, 64, 3), dtype=np.uint8)) == []
    assert estimator.load_error
    assert estimator.weights == "definitely-not-a-real-weight.pt"


def test_estimator_with_injected_model_is_available_without_torch():
    estimator = PoseEstimator(model=FakeModel([]))
    assert estimator.available is True
    assert estimator.load_error == ""


def test_estimator_parses_keypoints_and_bbox():
    settings = PoseSettings()
    estimator = PoseEstimator(settings, model=fake_model_for([STANDING]))
    poses = estimator.estimate(np.zeros((720, 1280, 3), dtype=np.uint8))

    assert len(poses) == 1
    pose = poses[0]
    assert pose.confidence == pytest.approx(0.9)
    assert pose.point(LEFT_SHOULDER) == pytest.approx((80.0, 70.0))
    assert pose.point(RIGHT_ANKLE) == pytest.approx((110.0, 215.0))
    assert pose.torso_length() == pytest.approx(TORSO_LENGTH)
    assert pose.bbox[3] > pose.bbox[1]


def test_estimator_marks_low_confidence_keypoints_unavailable():
    settings = PoseSettings(model=PoseModelSettings(keypoint_conf=0.95))
    estimator = PoseEstimator(settings, model=fake_model_for([STANDING], kpt_conf=0.9))
    pose = estimator.estimate(np.zeros((64, 64, 3), dtype=np.uint8))[0]
    assert pose.get(NOSE) is None
    assert pose.available_names() == []


def test_estimator_passes_configured_predict_kwargs():
    settings = PoseSettings(model=PoseModelSettings(imgsz=960, conf=0.4, iou=0.6,
                                                    device="cpu"))
    model = fake_model_for([STANDING])
    PoseEstimator(settings, model=model).estimate(np.zeros((64, 64, 3), dtype=np.uint8))
    assert model.calls[0]["imgsz"] == 960
    assert model.calls[0]["conf"] == 0.4
    assert model.calls[0]["iou"] == 0.6
    assert model.calls[0]["device"] == "cpu"
    assert model.calls[0]["verbose"] is False


def test_estimator_omits_device_when_not_configured():
    model = fake_model_for([STANDING])
    PoseEstimator(PoseSettings(), model=model).estimate(np.zeros((64, 64, 3), dtype=np.uint8))
    assert "device" not in model.calls[0]


def test_estimator_respects_max_persons():
    settings = PoseSettings(model=PoseModelSettings(max_persons=2))
    model = fake_model_for([STANDING, STANDING, STANDING])
    estimator = PoseEstimator(settings, model=model)
    assert len(estimator.estimate(np.zeros((64, 64, 3), dtype=np.uint8))) == 2


def test_estimator_survives_a_failing_model():
    estimator = PoseEstimator(PoseSettings(), model=ExplodingModel())
    assert estimator.estimate(np.zeros((64, 64, 3), dtype=np.uint8)) == []


@pytest.mark.parametrize("result", [
    FakeResult(boxes=None, keypoints=None),
    FakeResult(boxes=FakeBoxes(np.zeros((0, 4)), np.zeros((0,))),
               keypoints=FakeKeypoints(np.zeros((0, 17, 2)), np.zeros((0, 17)))),
])
def test_estimator_handles_empty_model_output(result):
    estimator = PoseEstimator(PoseSettings(), model=FakeModel([result]))
    assert estimator.estimate(np.zeros((64, 64, 3), dtype=np.uint8)) == []


def test_estimator_handles_no_results_and_none_image():
    estimator = PoseEstimator(PoseSettings(), model=FakeModel([]))
    assert estimator.estimate(np.zeros((64, 64, 3), dtype=np.uint8)) == []
    assert estimator.estimate(None) == []


def test_estimator_survives_a_missing_keypoint_confidence_channel():
    """有的后端不给关键点置信度通道。

    此时不能把置信度当 1.0（那是凭空相信）, 而应按 0.0 处理 → 该人的所有关键点
    都标成不可用; 但**这个人本身仍然要返回**（外接框还有用, 下游判据会如实报
    "证据不足", 比直接把人丢掉更有信息量）。
    """
    result = FakeResult(
        boxes=FakeBoxes(np.array([[70.0, 20.0, 130.0, 220.0]]), np.array([0.9])),
        keypoints=FakeKeypoints(np.array([[[80.0, 70.0]] * 17]), None))
    estimator = PoseEstimator(PoseSettings(), model=FakeModel([result]))
    poses = estimator.estimate(np.zeros((64, 64, 3), dtype=np.uint8))

    assert len(poses) == 1
    assert poses[0].available_names() == []
    assert poses[0].bbox == pytest.approx((70.0, 20.0, 130.0, 220.0))
    assert judge_high_leg(poses[0]).evidence_level == EVIDENCE_INSUFFICIENT


def test_estimator_rejects_malformed_keypoint_shape():
    """后端给出畸形的关键点数组（0 维 / 形状不对）时按"本帧无结果"处理, 不许炸。"""
    for bad in (np.array(None, dtype=object), np.zeros((2, 3))):
        result = FakeResult(
            boxes=FakeBoxes(np.array([[70.0, 20.0, 130.0, 220.0]]), np.array([0.9])),
            keypoints=FakeKeypoints(bad, None))
        estimator = PoseEstimator(PoseSettings(), model=FakeModel([result]))
        assert estimator.estimate(np.zeros((64, 64, 3), dtype=np.uint8)) == []


# ================================================================
# 十、端到端: 图像 → 姿态 → 动作判定（假模型, 不碰真权重）
# ================================================================

def test_image_to_action_chain_with_a_fake_estimator():
    """整条链路: 假模型出关键点 → PersonPose → 几何判据 → 可解释的结论。"""
    raised = dict(STANDING)
    raised[LEFT_ANKLE] = (95.0, 60.0)              # 把左踝抬到髋以上
    estimator = PoseEstimator(PoseSettings(), model=fake_model_for([raised]))

    poses = estimator.estimate(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert len(poses) == 1

    verdicts = {v.action: v for v in judge_all(poses[0])}
    high_leg = verdicts[ACTION_HIGH_LEG]
    assert high_leg.hit is True
    assert high_leg.evidence_level == EVIDENCE_JUDGEMENT
    assert high_leg.measurements["踝高于髋的比例"] > 0.10
    assert any("门槛" in reason for reason in high_leg.reasons)


def test_end_to_end_sliding_tackle_from_the_estimator():
    """外接框要显式给: 髋部高度比依赖它, 按关键点范围现取会得到另一个比值。"""
    estimator = PoseEstimator(PoseSettings(),
                              model=fake_model_for([SLIDING_MOVES], boxes=[SLIDING_BBOX]))
    pose = estimator.estimate(np.zeros((720, 1280, 3), dtype=np.uint8))[0]
    assert pose.hip_height_ratio() == pytest.approx(0.25)
    assert judge_sliding_tackle(pose).hit is True


def test_end_to_end_elbow_strike_between_two_people():
    striker = dict(STANDING)
    striker[LEFT_ELBOW] = (160.0, 70.0)
    opponent = {name: (x + 70.0, y + 37.4) for name, (x, y) in STANDING.items()}
    estimator = PoseEstimator(PoseSettings(), model=fake_model_for([striker, opponent]))

    poses = estimator.estimate(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert len(poses) == 2
    verdict = judge_elbow_strike(poses[0], poses[1:])
    assert verdict.hit is True


def test_estimator_keeps_person_order_stable_by_index():
    """帧内序号要稳定（后续与轨迹 id 对齐时靠它）。"""
    estimator = PoseEstimator(PoseSettings(), model=fake_model_for([STANDING, STANDING]))
    poses = estimator.estimate(np.zeros((64, 64, 3), dtype=np.uint8))
    assert [pose.index for pose in poses] == [0, 1]


# ================================================================
# 十一、真权重（有就用, 没有就跳过 —— 不让 CI 依赖 5MB 文件）
# ================================================================

def test_real_pose_model_if_available():
    """本机有 models/pretrained/yolo11n-pose.pt 时, 真跑一次空图确认不炸。

    没有权重就 skip: 离线环境下这条不该让整个测试套件变红。
    """
    settings = PoseSettings()
    if not Path(settings.model.resolved_weights()).is_file():
        pytest.skip("本地没有 pose 权重, 跳过真模型测试")
    estimator = PoseEstimator(settings)
    if not estimator.available:
        pytest.skip(f"pose 模型未能加载: {estimator.load_error}")
    poses = estimator.estimate(np.zeros((320, 320, 3), dtype=np.uint8))
    assert isinstance(poses, list)          # 空图 -> 空列表是正常结果
