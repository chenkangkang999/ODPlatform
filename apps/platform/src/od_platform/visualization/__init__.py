#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : __init__.py
# @Project   : ODPlatform
# @Function  : 检测结果美化绘制（D7：把框画出来）

"""visualization —— 检测结果美化绘制。

★ 此前这里是**桩**: ``draw()`` 直接 ``return image``（原图直出）。
  影响不是"少了点美化" —— ``odp-infer`` 的推断结果**一个框都没有**:
  ``inference/service.py`` 在美化开启时走的就是这条路径
  （``visualizer.draw(image, dets, ...)``）, 而它拿到的永远是原图。
  YOLO 自带的 ``result.plot()`` 只在美化关闭时才用。

★ 绘制约定（与 ``inference/overlay.py`` 的 HUD 一致: 原地修改并返回同一个数组）

  · 每个检测画一个矩形, 颜色按类别名从 ``color_mapping`` 取, 取不到用 ``default_color``;
  · 框上方画一行说明: ``[类名] [#轨迹号] [置信度]``, 类名可经 ``label_mapping`` 改名;
  · 说明文字先铺半透明底衬再写字 —— 否则浅色背景上白字看不见;
  · 所有几何/字号/颜色都来自 :class:`DrawStyle`, 本模块不写魔法数字。

★ 样式可调: ``configs/runtime/infer_pipeline.yaml`` 的 ``visualization.style``
  会经 ``DrawStyle.from_image_size(h, w, **overrides)`` 覆盖到这里的默认值。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# cv2.FONT_HERSHEY_SIMPLEX 的字面值就是 0。
# 这里写字面值、不在模块顶层 import cv2: 本模块要被 odp-infer 之外的地方
# 轻量引用（以及单元测试）, 顶层 import cv2 会把整条 opencv 依赖带进来。
FONT_HERSHEY_SIMPLEX = 0

# 分辨率自适应的基准: 以 720p 为 1.0 倍, 4K 上字号/线宽自动放大
STYLE_BASE_HEIGHT = 720


@dataclass
class DrawStyle:
    """绘制样式。默认值即"720p 下看着合适"的一套。"""

    box_thickness: int = 2
    text_scale: float = 0.6
    text_color: Tuple[int, int, int] = (255, 255, 255)
    text_bg_alpha: float = 0.6
    # ↓ 桩时期没有、真实绘制需要的项（都给默认值, 不破坏既有构造方式）
    text_bg_color: Tuple[int, int, int] = (0, 0, 0)
    text_thickness: int = 1
    text_pad_x: int = 4
    text_pad_y: int = 3
    font: int = FONT_HERSHEY_SIMPLEX
    min_box_pixels: int = 2          # 短边小于此值的框不画（噪声框）
    show_confidence: bool = True
    conf_decimals: int = 2

    @classmethod
    def from_image_size(cls, h: int, w: int, **overrides) -> "DrawStyle":
        """按画面尺寸自适应字号/线宽, 再用 overrides 覆盖。

        桩时期这个函数收下 h/w 却完全没用。现在它按短边相对 720p 的比例缩放:
        1080p 上字号和线宽不变, 4K 上自动放大一倍, 小图则收敛到最小 1px。
        """
        style = cls()
        if h > 0 and w > 0:
            factor = max(min(h, w) / STYLE_BASE_HEIGHT, 0.5)
            style.box_thickness = max(1, round(style.box_thickness * factor))
            style.text_thickness = max(1, round(style.text_thickness * factor))
            style.text_scale = round(style.text_scale * factor, 3)
            style.text_pad_x = max(1, round(style.text_pad_x * factor))
            style.text_pad_y = max(1, round(style.text_pad_y * factor))
        for key, value in overrides.items():
            if hasattr(style, key):
                setattr(style, key, value)
        return style


class BeautifyVisualizer:
    """检测框绘制器。

    Args:
        labels: 类别名列表（模型的 names, 仅作记录/兜底）
        label_mapping: 类别名 → 显示名（如 player→球员）
        color_mapping: 类别名 → BGR 颜色
        default_color: 没在 color_mapping 里出现时的颜色
        font_path: 预留（cv2 的 Hershey 字体不需要外部字体文件）
    """

    def __init__(self, labels: Optional[Sequence[str]] = None,
                 label_mapping: Optional[Dict[str, str]] = None,
                 color_mapping: Optional[Dict[str, Any]] = None,
                 default_color: Any = (0, 255, 0),
                 font_path: Optional[str] = None):
        self.labels = list(labels) if labels else []
        self.label_mapping = dict(label_mapping or {})
        self.color_mapping = dict(color_mapping or {})
        self.default_color = tuple(default_color)
        self.font_path = font_path
        # 大小写不敏感查找: 配置里写 "Player"、模型给 "player" 时也要命中
        self._lower_label_mapping = {k.lower(): v for k, v in self.label_mapping.items()}
        self._lower_color_mapping = {k.lower(): v for k, v in self.color_mapping.items()}

    # ---------------------------------------------------------------- 输入适配
    @staticmethod
    def from_yolo_results(boxes=None, confidences=None, labels=None) -> List[Dict[str, Any]]:
        """把 ultralytics 的三段数组合成绘制用的检测列表。

        桩时期的实现是 ``list(zip(labels or []))`` —— 返回的是 ``[(name,), ...]``,
        既丢了坐标也丢了置信度, 画不出任何东西。现在返回每个检测一个字典::

            [{"cls": "player", "conf": 0.91, "xyxy": [x1, y1, x2, y2]}, ...]

        三段长度不一致时按最短的可用信息拼装（不炸、不猜）。
        """
        coords_seq = [] if boxes is None else list(boxes)
        label_seq = [] if labels is None else list(labels)
        conf_seq = [] if confidences is None else list(confidences)

        detections: List[Dict[str, Any]] = []
        for index, raw in enumerate(coords_seq):
            coords = [float(value) for value in list(raw)[:4]]
            if len(coords) != 4:
                continue
            detection: Dict[str, Any] = {
                "cls": str(label_seq[index]) if index < len(label_seq) else "",
                "xyxy": coords,
            }
            if index < len(conf_seq):
                detection["conf"] = float(conf_seq[index])
            detections.append(detection)
        return detections

    # ---------------------------------------------------------------- 取色/取字
    def color_for(self, cls_name: Any) -> Tuple[int, int, int]:
        """按类别名取 BGR 颜色; 没配就用 default_color。"""
        if cls_name is None:
            return self.default_color
        key = str(cls_name).strip()
        if key in self.color_mapping:
            return tuple(self.color_mapping[key])
        return tuple(self._lower_color_mapping.get(key.lower(), self.default_color))

    def display_name(self, cls_name: Any, use_label_mapping: bool = True) -> str:
        """类名的显示名（label_mapping 可把它改成中文等）。"""
        if cls_name is None:
            return ""
        key = str(cls_name).strip()
        if not use_label_mapping:
            return key
        if key in self.label_mapping:
            return str(self.label_mapping[key])
        return str(self._lower_label_mapping.get(key.lower(), key))

    def caption_for(self, detection: Dict[str, Any], style: Optional[DrawStyle] = None,
                    use_label_mapping: bool = True) -> str:
        """框上那行字: ``类名 #轨迹号 置信度``（有哪段写哪段）。"""
        style = style or DrawStyle()
        parts: List[str] = []
        name = self.display_name(detection.get("cls"), use_label_mapping)
        if name:
            parts.append(name)
        track_id = detection.get("track_id")
        if track_id is not None:
            parts.append(f"#{track_id}")
        confidence = detection.get("conf")
        if style.show_confidence and confidence is not None:
            parts.append(f"{float(confidence):.{style.conf_decimals}f}")
        return " ".join(parts)

    # ---------------------------------------------------------------- 绘制
    def draw(self, image, detections, style: Optional[DrawStyle] = None,
             use_label_mapping: bool = True):
        """在 image 上画框与说明, **原地修改**并返回同一个数组。

        Args:
            image: BGR 图像（numpy 数组）
            detections: ``from_yolo_results`` 的输出, 或同形状的
                ``[{"cls", "conf", "xyxy", "track_id"?}, ...]``
            style: 绘制样式; None 时用默认而不是按图尺寸自适应
            use_label_mapping: 是否套用 label_mapping 改显示名

        Returns:
            画好的图像（与传入的是同一个对象）
        """
        import cv2

        style = style or DrawStyle()
        if image is None or not detections:
            return image

        captions: List[Tuple[str, int, int]] = []       # (文字, x, y)
        for detection in detections:
            coords = list(detection.get("xyxy") or [])
            if len(coords) != 4:
                continue
            x1, y1, x2, y2 = (int(round(float(value))) for value in coords)
            if (x2 - x1) < style.min_box_pixels or (y2 - y1) < style.min_box_pixels:
                continue
            cv2.rectangle(image, (x1, y1), (x2, y2),
                          self.color_for(detection.get("cls")), style.box_thickness)
            text = self.caption_for(detection, style, use_label_mapping)
            if text:
                captions.append((text, x1, y1))

        if captions:
            sizes = [cv2.getTextSize(text, style.font, style.text_scale, style.text_thickness)[0]
                     for text, _, _ in captions]
            # 先把所有底衬画在副本上, 一次性按 alpha 混合 —— 逐块混合会让
            # 重叠区域的透明度叠加, 颜色越压越黑。
            backdrop = image.copy()
            for (_, x, y), (text_w, text_h) in zip(captions, sizes):
                cv2.rectangle(backdrop, (x, max(0, y - text_h - 2 * style.text_pad_y)),
                              (x + text_w + 2 * style.text_pad_x, y), style.text_bg_color, -1)
            cv2.addWeighted(backdrop, style.text_bg_alpha,
                            image, 1.0 - style.text_bg_alpha, 0, image)
            for (text, x, y), _ in zip(captions, sizes):
                cv2.putText(image, text, (x + style.text_pad_x, max(style.text_pad_y, y - style.text_pad_y)),
                            style.font, style.text_scale, style.text_color,
                            style.text_thickness, cv2.LINE_AA)
        return image