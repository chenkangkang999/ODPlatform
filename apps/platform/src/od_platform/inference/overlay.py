"""推理画面信息叠加 + 帧率指标."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


class FPSCounter:
    def __init__(self, window_size: int = 30) -> None:
        self._samples: deque[float] = deque(maxlen=window_size)
        self._last_ms: float = 0.0

    def update(self, ms: float) -> None:
        if ms > 0:
            self._samples.append(ms)
            self._last_ms = ms

    @property
    def fps(self) -> float:
        if not self._samples:
            return 0.0
        avg = sum(self._samples) / len(self._samples)
        return 1000.0 / avg if avg > 0 else 0.0

    @property
    def inst(self) -> float:
        return 1000.0 / self._last_ms if self._last_ms > 0 else 0.0


@dataclass
class Metrics:
    capture: FPSCounter = field(default_factory=FPSCounter)
    infer:   FPSCounter = field(default_factory=FPSCounter)
    render:  FPSCounter = field(default_factory=FPSCounter)
    loop:    FPSCounter = field(default_factory=FPSCounter)
    _spp: float = 0.0
    _sinf: float = 0.0
    _spost: float = 0.0
    _sn: int = 0

    def add_speed(self, speed: dict | None) -> None:
        if not speed:
            return
        self._spp += float(speed.get("preprocess", 0.0))
        self._sinf += float(speed.get("inference", 0.0))
        self._spost += float(speed.get("postprocess", 0.0))
        self._sn += 1
        if speed.get("inference"):
            self.infer.update(float(speed["inference"]))

    def speed_avg_ms(self) -> dict[str, float]:
        if not self._sn:
            return {}
        return {
            "preprocess":  round(self._spp / self._sn, 2),
            "inference":   round(self._sinf / self._sn, 2),
            "postprocess": round(self._spost / self._sn, 2),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "capture_fps": round(self.capture.fps, 1),
            "infer_fps":   round(self.infer.fps, 1),
            "render_fps":  round(self.render.fps, 1),
            "loop_fps":    round(self.loop.fps, 1),
            "current_fps": round(self.loop.inst, 1),
            "speed_ms":    self.speed_avg_ms(),
        }


_C_LABEL = (200, 200, 200)
_C_CAPTURE = (0, 200, 255)
_C_INFER = (0, 230, 0)
_C_RENDER = (255, 120, 220)
_C_LOOP = (255, 200, 0)
_C_CURRENT = (255, 255, 255)
_C_OBJ = (120, 220, 255)

# HUD 面板布局 / REC 标记(具名, 免得数字散落在绘制逻辑里)
HUD_PANEL_X = 12
HUD_PANEL_Y = 12
HUD_PANEL_WIDTH = 230
HUD_TEXT_SCALE = 0.5
HUD_TEXT_THICKNESS = 1
HUD_ROW_HEIGHT = 22
HUD_TEXT_BASELINE = 14
HUD_TEXT_OFFSET_X = 6
HUD_PADDING = 10
HUD_LABEL_WIDTH = 84
HUD_BG_COLOR = (20, 20, 20)
HUD_BG_ALPHA = 0.55
HUD_ACCENT_COLOR = (0, 200, 255)
HUD_HINT_COLOR = (210, 210, 210)
HUD_REC_COLOR = (0, 0, 255)
HUD_REC_DOT_OFFSET_X = 70
HUD_REC_DOT_Y = 24
HUD_REC_DOT_RADIUS = 7
HUD_REC_TEXT_OFFSET_X = 56
HUD_REC_TEXT_Y = 30
HUD_REC_TEXT_SCALE = 0.6
HUD_REC_TEXT_THICKNESS = 2
PAUSE_TEXT = "PAUSED"
PAUSE_TEXT_SCALE = 1.8
PAUSE_TEXT_THICKNESS = 3
PAUSE_DIM_ALPHA = 0.45
PAUSE_HINT = "Press SPACE to resume  |  Q / Esc to quit"
PAUSE_HINT_SCALE = 0.6
PAUSE_HINT_OFFSET_Y = 44


def draw_hud(frame, metrics: Metrics, *, n_dets: int = 0,
             recording: bool = False, show_info: bool = True) -> None:
    import cv2
    if not show_info:
        return
    rows = [
        ("Capture", f"{metrics.capture.fps:5.1f} FPS", _C_CAPTURE),
        ("Infer",   f"{metrics.infer.fps:5.1f} FPS", _C_INFER),
        ("Render",  f"{metrics.render.fps:5.1f} FPS", _C_RENDER),
        ("Loop",    f"{metrics.loop.fps:5.1f} FPS", _C_LOOP),
        ("Current", f"{metrics.loop.inst:5.1f} FPS", _C_CURRENT),
        ("Objects", f"{n_dets}", _C_OBJ),
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th, lh = HUD_TEXT_SCALE, HUD_TEXT_THICKNESS, HUD_ROW_HEIGHT
    pad = HUD_PADDING
    label_w = HUD_LABEL_WIDTH
    panel_w = HUD_PANEL_WIDTH
    panel_h = pad * 2 + lh * len(rows)
    x0, y0 = HUD_PANEL_X, HUD_PANEL_Y
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), HUD_BG_COLOR, -1)
    cv2.addWeighted(overlay, HUD_BG_ALPHA, frame, 1 - HUD_BG_ALPHA, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + 4, y0 + panel_h), HUD_ACCENT_COLOR, -1)
    y = y0 + pad + HUD_TEXT_BASELINE
    for label, value, color in rows:
        cv2.putText(frame, label, (x0 + pad + HUD_TEXT_OFFSET_X, y), font, fs, _C_LABEL, th, cv2.LINE_AA)
        cv2.putText(frame, value, (x0 + pad + label_w, y), font, fs, color, th, cv2.LINE_AA)
        y += lh
    if recording:
        h, w = frame.shape[:2]
        cv2.circle(frame, (w - HUD_REC_DOT_OFFSET_X, HUD_REC_DOT_Y), HUD_REC_DOT_RADIUS, HUD_REC_COLOR, -1)
        cv2.putText(frame, "REC", (w - HUD_REC_TEXT_OFFSET_X, HUD_REC_TEXT_Y), font,
                   HUD_REC_TEXT_SCALE, HUD_REC_COLOR, HUD_REC_TEXT_THICKNESS, cv2.LINE_AA)


def draw_pause(frame) -> None:
    import cv2
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, PAUSE_DIM_ALPHA, frame, 1 - PAUSE_DIM_ALPHA, 0, frame)
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text, fscale, fth = PAUSE_TEXT, PAUSE_TEXT_SCALE, PAUSE_TEXT_THICKNESS
    (tw, tht), _ = cv2.getTextSize(text, font, fscale, fth)
    cx, cy = (w - tw) // 2, (h + tht) // 2
    cv2.putText(frame, text, (cx, cy), font, fscale, (255, 255, 255), fth, cv2.LINE_AA)
    hint = PAUSE_HINT
    (hw, _), _ = cv2.getTextSize(hint, font, 0.6, 1)
    cv2.putText(frame, hint, ((w - hw) // 2, cy + PAUSE_HINT_OFFSET_Y), font, PAUSE_HINT_SCALE, HUD_HINT_COLOR, 1, cv2.LINE_AA)
