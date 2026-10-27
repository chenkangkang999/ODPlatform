#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""visualization 绘制测试。

这一组守的是"真的画了" —— 此前 ``draw()`` 是桩, 直接 ``return image``,
而 ``odp-infer`` 在美化开启时走的正是这条路径, 于是推断结果一个框都没有。
所以第一条测试就是: 画完之后像素必须变了。
"""
from __future__ import annotations

import numpy as np
import pytest

from od_platform.visualization import (
    SKELETON_EDGES,
    BeautifyVisualizer,
    DrawStyle,
    keypoint_pixel,
)

BOX_COLOR = (0, 0, 255)          # BGR 红
DEFAULT_COLOR = (0, 255, 0)      # BGR 绿


def blank(height: int = 120, width: int = 160) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def detection(cls="player", conf=0.9, xyxy=(10, 10, 60, 80), **extra) -> dict:
    return {"cls": cls, "conf": conf, "xyxy": list(xyxy), **extra}


def color_present(region: np.ndarray, color) -> bool:
    """区域内是否出现过该颜色（避免卡死在"边线到底画在第几行"上）。"""
    return bool((region == np.array(color, dtype=np.uint8)).all(axis=-1).any())


# ---------------------------------------------------------------- 核心: 画出来了

def test_boxes_are_actually_drawn_and_original_array_is_reused():
    image = blank()
    before = image.copy()
    result = BeautifyVisualizer(color_mapping={"player": BOX_COLOR}).draw(
        image, [detection()], style=DrawStyle(box_thickness=4))
    assert result is image, "绘制约定是原地修改并返回同一个数组（与 HUD 一致）"
    assert not np.array_equal(before, image), "桩会原样返回; 真实现必须改变像素"


def test_box_uses_the_mapped_class_color():
    image = blank()
    BeautifyVisualizer(color_mapping={"player": BOX_COLOR}).draw(
        image, [detection()], style=DrawStyle(box_thickness=4))
    assert color_present(image[6:20, 5:70], BOX_COLOR)


def test_unknown_class_falls_back_to_default_color():
    image = blank()
    BeautifyVisualizer(default_color=DEFAULT_COLOR).draw(
        image, [detection(cls="goalkeeper")], style=DrawStyle(box_thickness=4))
    assert color_present(image[6:20, 5:70], DEFAULT_COLOR)


def test_each_class_gets_its_own_color():
    image = blank()
    visualizer = BeautifyVisualizer(
        color_mapping={"player": BOX_COLOR, "ball": DEFAULT_COLOR})
    visualizer.draw(image, [detection(xyxy=(10, 10, 40, 40)),
                            detection(cls="ball", xyxy=(100, 10, 130, 40))],
                    style=DrawStyle(box_thickness=4))
    assert color_present(image[6:20, 5:50], BOX_COLOR)
    assert color_present(image[6:20, 95:140], DEFAULT_COLOR)


def test_color_lookup_is_case_insensitive():
    visualizer = BeautifyVisualizer(color_mapping={"Player": BOX_COLOR})
    assert visualizer.color_for("player") == BOX_COLOR
    assert visualizer.color_for("PLAYER") == BOX_COLOR
    assert visualizer.color_for("nobody") == visualizer.default_color


# ---------------------------------------------------------------- 说明文字

def test_caption_contains_class_track_id_and_confidence():
    visualizer = BeautifyVisualizer()
    style = DrawStyle()
    text = visualizer.caption_for(detection(conf=0.912, track_id=7), style)
    assert text == "player #7 0.91"


def test_label_mapping_renames_the_caption():
    visualizer = BeautifyVisualizer(label_mapping={"player": "球员", "ball": "足球"})
    style = DrawStyle()
    assert visualizer.caption_for(detection(), style) == "球员 0.90"
    assert visualizer.caption_for(detection(), style, use_label_mapping=False) == "player 0.90"
    assert visualizer.caption_for(detection(cls="Player"), style) == "球员 0.90"


def test_show_confidence_can_be_turned_off():
    visualizer = BeautifyVisualizer()
    text = visualizer.caption_for(detection(), DrawStyle(show_confidence=False))
    assert text == "player"


def test_caption_without_class_name_still_has_id():
    visualizer = BeautifyVisualizer()
    text = visualizer.caption_for({"cls": "", "track_id": 3}, DrawStyle())
    assert text == "#3"


def test_conf_decimals_are_configurable():
    visualizer = BeautifyVisualizer()
    assert visualizer.caption_for(detection(conf=0.91234),
                                 DrawStyle(conf_decimals=4)).endswith("0.9123")


# ---------------------------------------------------------------- 边界

def test_empty_detections_leave_image_untouched():
    image = blank()
    before = image.copy()
    assert BeautifyVisualizer().draw(image, [], style=DrawStyle()) is image
    assert np.array_equal(before, image)


def test_none_image_is_returned_as_is():
    assert BeautifyVisualizer().draw(None, [detection()], style=DrawStyle()) is None


def test_tiny_boxes_are_skipped_as_noise():
    image = blank()
    before = image.copy()
    BeautifyVisualizer(color_mapping={"player": BOX_COLOR}).draw(
        image, [detection(xyxy=(10, 10, 11, 11))], style=DrawStyle(min_box_pixels=5))
    assert np.array_equal(before, image), "短边小于 min_box_pixels 的框不该画"


def test_detections_with_bad_coordinates_are_ignored_not_fatal():
    image = blank()
    BeautifyVisualizer().draw(image, [{"cls": "player", "xyxy": [1, 2, 3]}], style=DrawStyle())
    BeautifyVisualizer().draw(image, [{"cls": "player"}], style=DrawStyle())


def test_image_shape_is_preserved():
    image = blank(64, 96)
    result = BeautifyVisualizer().draw(image, [detection(xyxy=(5, 5, 50, 50))], style=DrawStyle())
    assert result.shape == (64, 96, 3)


# ---------------------------------------------------------------- 输入适配

def test_from_yolo_results_builds_detection_dicts():
    dets = BeautifyVisualizer.from_yolo_results(
        boxes=[[10, 10, 60, 80], [1, 2, 3, 4]],
        confidences=[0.9, 0.5],
        labels=["player", "ball"])
    assert dets == [{"cls": "player", "conf": 0.9, "xyxy": [10.0, 10.0, 60.0, 80.0]},
                    {"cls": "ball", "conf": 0.5, "xyxy": [1.0, 2.0, 3.0, 4.0]}]


def test_from_yolo_results_tolerates_missing_confidence_or_labels():
    dets = BeautifyVisualizer.from_yolo_results(boxes=[[1, 2, 3, 4]], labels=None)
    assert dets == [{"cls": "", "xyxy": [1.0, 2.0, 3.0, 4.0]}]


def test_from_yolo_results_skips_malformed_rows():
    dets = BeautifyVisualizer.from_yolo_results(boxes=[[1, 2, 3], [1, 2, 3, 4]],
                                                confidences=[0.1, 0.2], labels=["a", "b"])
    assert len(dets) == 1 and dets[0]["cls"] == "b"


def test_from_yolo_results_handles_ultralytics_empty_arrays():
    """service.py 在没有检测时会传 (0,4) 与 (0,) 的空数组。"""
    dets = BeautifyVisualizer.from_yolo_results(
        boxes=np.zeros((0, 4), dtype=float), confidences=np.zeros((0,), dtype=float), labels=[])
    assert dets == []


# ---------------------------------------------------------------- 样式自适应

def test_style_scales_up_on_large_frames():
    small = DrawStyle.from_image_size(720, 1280)
    big = DrawStyle.from_image_size(2160, 3840)
    assert big.text_scale > small.text_scale
    assert big.box_thickness >= small.box_thickness
    assert small.box_thickness >= 1


def test_style_overrides_win_over_auto_scaling():
    style = DrawStyle.from_image_size(2160, 3840, box_thickness=1, text_scale=0.4)
    assert style.box_thickness == 1
    assert style.text_scale == 0.4


def test_style_ignores_unknown_override_keys():
    style = DrawStyle.from_image_size(720, 1280, not_a_field=123)
    assert not hasattr(style, "not_a_field")


@pytest.mark.parametrize("size", [(0, 0), (-1, 10)])
def test_style_handles_degenerate_sizes(size):
    style = DrawStyle.from_image_size(*size)
    assert style.box_thickness >= 1


# ---------------------------------------------------------------- 姿态骨骼绘制
# 姿态子系统（od_platform.pose）的结果要靠这里才能被看见。

class _Keypoint:
    """姿态模块 Keypoint 的轻量替身（本模块只依赖鸭子类型）。"""

    def __init__(self, x, y, conf=0.9, available=True):
        self.x, self.y, self.conf, self.available = x, y, conf, available


class _Pose:
    def __init__(self, keypoints, bbox=(10.0, 10.0, 100.0, 130.0)):
        self.keypoints = keypoints
        self.bbox = bbox


BODY_COORDS = {
    "nose": (60, 20), "left_eye": (57, 17), "right_eye": (63, 17),
    "left_ear": (54, 18), "right_ear": (66, 18),
    "left_shoulder": (50, 35), "right_shoulder": (70, 35),
    "left_elbow": (45, 55), "right_elbow": (75, 55),
    "left_wrist": (42, 75), "right_wrist": (78, 75),
    "left_hip": (52, 70), "right_hip": (68, 70),
    "left_knee": (50, 95), "right_knee": (70, 95),
    "left_ankle": (48, 118), "right_ankle": (72, 118),
}


def _body(**unavailable):
    """造一个完整人体; 传入 ``name=False`` 可把某个关键点标成不可用。"""
    keypoints = {}
    for name, (x, y) in BODY_COORDS.items():
        usable = unavailable.get(name, True)
        keypoints[name] = _Keypoint(x, y, available=usable)
    return _Pose(keypoints)


def test_skeleton_edges_only_use_coco_keypoint_names():
    """★ 跨模块一致性: 骨骼连线的名字必须都在姿态模块的 COCO-17 里, 否则永远画不出来。

    两个模块刻意不互相 import（visualization 要保持轻量）, 所以这条由测试来守。
    """
    from od_platform.pose import COCO_KEYPOINT_NAMES

    used = {name for edge in SKELETON_EDGES for name in edge}
    missing = used - set(COCO_KEYPOINT_NAMES)
    assert not missing, f"这些名字不在 COCO-17 里: {missing}"


def test_draw_skeleton_actually_draws_and_reuses_the_array():
    image = blank(150, 120)
    before = image.copy()
    result = BeautifyVisualizer().draw_skeleton(image, [_body()], style=DrawStyle())
    assert result is image
    assert not np.array_equal(before, image)


def test_draw_skeleton_uses_the_configured_colors():
    image = blank(150, 120)
    style = DrawStyle(skeleton_thickness=3, keypoint_radius=2,
                      keypoint_color=(0, 255, 255))
    BeautifyVisualizer(default_color=(0, 0, 255)).draw_skeleton(image, [_body()],
                                                                style=style)
    assert color_present(image[30:45, 45:75], (0, 0, 255))     # 骨骼用 default_color
    assert color_present(image[15:25, 55:65], (0, 255, 255))   # 关节点用 keypoint_color


def test_unavailable_keypoints_are_never_drawn():
    """★ 硬规矩: 不可用的关键点既不画点, 也不画连到它的线。

    拿 0 坐标去画会得到一条横穿画面的假肢体 —— 那比不画更糟。
    """
    kept = ("left_hip", "right_hip", "left_shoulder", "right_shoulder")
    pose = _Pose({name: _Keypoint(*BODY_COORDS[name], available=(name in kept))
                  for name in BODY_COORDS})
    image = blank(150, 120)
    before = image.copy()
    BeautifyVisualizer(default_color=BOX_COLOR).draw_skeleton(image, [pose],
                                                              style=DrawStyle())
    assert not np.array_equal(before, image)            # 躯干那几条线还在
    # 左踝 (48, 118) 不可用 → 它所在的下肢区域一条线都不该出现
    assert not color_present(image[100:125, 40:58], BOX_COLOR)


def test_keypoint_pixel_rejects_missing_unavailable_and_invalid():
    assert keypoint_pixel({}, "nose") is None
    assert keypoint_pixel({"nose": _Keypoint(1, 2, available=False)}, "nose") is None
    assert keypoint_pixel({"nose": _Keypoint("x", 2)}, "nose") is None
    assert keypoint_pixel({"nose": _Keypoint(3.4, 5.6)}, "nose") == (3, 6)


def test_draw_skeleton_box_is_opt_in():
    without = blank(150, 120)
    BeautifyVisualizer().draw_skeleton(without, [_body()], style=DrawStyle())

    with_box = blank(150, 120)
    BeautifyVisualizer().draw_skeleton(
        with_box, [_body()],
        style=DrawStyle(draw_skeleton_box=True, box_thickness=4))
    # 开了人体框之后, 框的角落附近会多出颜色
    assert not np.array_equal(without, with_box)
    assert color_present(with_box[8:16, 8:16], DEFAULT_COLOR)


def test_draw_skeleton_on_empty_input_is_a_no_op():
    image = blank()
    before = image.copy()
    assert BeautifyVisualizer().draw_skeleton(image, [], style=DrawStyle()) is image
    assert BeautifyVisualizer().draw_skeleton(None, [_body()], style=DrawStyle()) is None
    assert np.array_equal(before, image)


def test_skeleton_style_scales_with_image_size():
    big = DrawStyle.from_image_size(2160, 3840).skeleton_thickness
    small = DrawStyle.from_image_size(720, 1280).skeleton_thickness
    assert big >= small >= 1