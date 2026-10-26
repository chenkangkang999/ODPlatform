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

from od_platform.visualization import BeautifyVisualizer, DrawStyle

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