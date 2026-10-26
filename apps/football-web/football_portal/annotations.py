"""Compact role-colored overlays; render only, never change detector outputs."""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

STYLE_VERSION = "football.roles.v2"
ROLE_COLORS = {"ball": "#FDE047", "goalkeeper": "#C084FC", "player": "#38BDF8", "referee": "#FB923C"}


def _intersects(first, second):
    return first[0] < second[2] and first[2] > second[0] and first[1] < second[3] and first[3] > second[1]


def _label_position(box, label_width, label_height, size, occupied):
    x1, y1, x2, y2 = box
    width, height = size
    candidates = ((x1, y1 - label_height - 2), (x2 + 3, y1),
                  (x1 - label_width - 3, y1), (x1, y2 + 3), (x1, y1 + 3))
    best, fewest = None, float("inf")
    for x, y in candidates:
        x, y = int(max(0, min(x, width - label_width))), int(max(0, min(y, height - label_height)))
        rectangle = (x, y, x + label_width, y + label_height)
        overlaps = sum(_intersects(rectangle, previous) for previous in occupied)
        if overlaps < fewest:
            best, fewest = rectangle, overlaps
        if overlaps == 0:
            break
    occupied.append(best)
    return best


def render_annotations(image: Image.Image, boxes: list[dict]) -> Image.Image:
    """Use original pixel coordinates. Only confidence text is drawn on the photo.

    Ball rings/crosshairs are rendered last so nearby person boxes cannot hide them.
    The ring's radius is a display aid, not an enlarged predicted bounding box.
    """
    canvas = image.convert("RGB").copy()
    width, height = canvas.size
    if width < 1 or height < 1:
        raise ValueError("图片尺寸无效")
    draw = ImageDraw.Draw(canvas)
    scale = max(.7, min(1.6, min(width, height) / 1000))
    line = max(1, round(1.5 * scale))
    font = ImageFont.load_default(size=max(10, round(12 * scale)))
    ordered = sorted(boxes, key=lambda item: item.get("cls") == "ball")
    labels = []
    for box in ordered:
        color = ROLE_COLORS[box["cls"]]
        x1, y1, x2, y2 = (max(0, min(float(box[key]), limit)) for key, limit in
                          (("x1", width - 1), ("y1", height - 1), ("x2", width - 1), ("y2", height - 1)))
        if x2 <= x1 or y2 <= y1:
            continue
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line)
        if box["cls"] == "ball":
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            radius = max(10 * scale, max(x2 - x1, y2 - y1) / 2 + 5 * scale)
            outer = (cx - radius, cy - radius, cx + radius, cy + radius)
            draw.ellipse(outer, outline="#142117", width=max(3, round(5 * scale)))
            draw.ellipse(outer, outline=color, width=max(2, round(2 * scale)))
            gap, tip = radius + 3 * scale, radius + 9 * scale
            for start, end in (((cx - tip, cy), (cx - gap, cy)), ((cx + gap, cy), (cx + tip, cy)),
                               ((cx, cy - tip), (cx, cy - gap)), ((cx, cy + gap), (cx, cy + tip))):
                draw.line((start, end), fill=color, width=max(1, line))
            label_box = (cx + radius + 5, cy - radius, cx + radius + 5, cy + radius)
        else:
            label_box = (x1, y1, x2, y2)
        labels.append((label_box, f"{box['conf']:.2f}", color, box["cls"] == "ball"))

    # Reserve the ball confidence first; person labels then avoid occupied labels.
    occupied = []
    placements = []
    for box, text, color, is_ball in sorted(labels, key=lambda item: not item[3]):
        bounds = draw.textbbox((0, 0), text, font=font)
        tag_width, tag_height = bounds[2] - bounds[0] + 8, bounds[3] - bounds[1] + 5
        rect = _label_position(box, tag_width, tag_height, (width, height), occupied)
        placements.append((rect, text, color, is_ball, bounds[1]))
    for rect, text, color, is_ball, text_top in sorted(placements, key=lambda item: item[3]):
        draw.rounded_rectangle(rect, radius=2, fill=color if is_ball else "#10231E")
        draw.text((rect[0] + 4, rect[1] + 2 - text_top), text, font=font, fill="#152117" if is_ball else color)
    return canvas


def render_bgr(image, boxes):
    """Small adapter for ODPlatform/OpenCV frames; Pillow stays usable in Web."""
    import numpy as np

    result = render_annotations(Image.fromarray(image[:, :, ::-1]), boxes)
    return np.asarray(result)[:, :, ::-1].copy()
