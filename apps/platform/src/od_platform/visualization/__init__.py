"""visualization 包桩 — 美化绘制."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DrawStyle:
    box_thickness: int = 2
    text_scale: float = 0.6
    text_color: tuple = (255, 255, 255)
    text_bg_alpha: float = 0.6

    @classmethod
    def from_image_size(cls, h: int, w: int, **overrides) -> "DrawStyle":
        d = cls()
        for k, v in overrides.items():
            if hasattr(d, k):
                setattr(d, k, v)
        return d


class BeautifyVisualizer:
    def __init__(self, labels=None, label_mapping=None, color_mapping=None,
                 default_color=(0, 255, 0), font_path=None):
        self.labels = labels
        self.label_mapping = label_mapping or {}
        self.color_mapping = color_mapping or {}
        self.default_color = default_color
        self.font_path = font_path

    @staticmethod
    def from_yolo_results(boxes=None, confidences=None, labels=None):
        return list(zip(labels or []))

    def draw(self, image, detections, style=None, use_label_mapping=True):
        """桩: 直接返回原图(真实实现用 cv2/PIL 画框)."""
        return image
