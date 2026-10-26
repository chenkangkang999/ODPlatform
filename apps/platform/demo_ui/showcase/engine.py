#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : engine.py
# @Project   : ODPlatform / demo_ui
# @Function  : 上传解码 → 检测 → 绘制（检测器可注入, 便于测试）

"""推理演示的执行层。

分三步, 每步都单独可测:

  1. :func:`decode_image` —— 上传字节 → BGR 图像（上传即内存处理, 不落盘）
  2. 检测 —— :func:`build_ultralytics_detector` 用 ultralytics 跑足球权重,
     也可以用 ``detector=`` 注入一个假检测器, 于是测试不需要真的加载模型
  3. :func:`render` —— 用 ``od_platform.visualization.BeautifyVisualizer`` 画框

★ 这里刻意不 import Django: 视图只做 HTTP 那点事, 算法与绘制放这儿, 测试可以直接调。

★ 绘制用的颜色/线宽复用 ``configs/runtime/infer_pipeline.yaml``（odp-infer 的同一份
  美化配置）, 只有"类别名→中文"的展示映射是本演示自己的 —— 见 demo_config 的说明。
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from od_platform.common.football_config import load_config
from od_platform.detection.settings import DetectionSettings
from od_platform.inference.pipeline_config import load_pipeline_config
from od_platform.visualization import BeautifyVisualizer, DrawStyle

from showcase import demo_config

logger = logging.getLogger(__name__)

# 检测器签名: 吃一张 BGR 图, 吐 [{"cls": str, "conf": float, "xyxy": [x1,y1,x2,y2]}]
Detector = Callable[[np.ndarray], List[Dict[str, Any]]]

# 已加载的 YOLO 权重缓存: 键 = (权重路径, 修改时间)。
# 为什么缓存: 每次请求都 YOLO(...) 会把页面拖到十几秒; 而只按路径缓存又会在
# 重新训练后继续用旧权重, 所以把 mtime 一起做键。
_MODEL_CACHE: Dict[tuple, Any] = {}


class UploadError(ValueError):
    """上传内容不可用（空 / 太大 / 后缀不对 / 解不开）。"""


# ============================================================
# 1. 解码
# ============================================================

def decode_image(payload: bytes) -> np.ndarray:
    """把上传的字节解成 BGR 图像。

    Raises:
        UploadError: 空内容或不是能解码的图片。
    """
    if not payload:
        raise UploadError("上传内容为空, 请重新选择图片")
    buffer = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise UploadError("这张图解不开 —— 可能不是图片, 或者文件已损坏")
    return image


def suffix_allowed(filename: str) -> bool:
    """后缀白名单: 先挡一道明显不对的文件, 真正的判据仍然是能不能解码。"""
    return Path(filename or "").suffix.lower() in demo_config.ALLOWED_IMAGE_SUFFIXES


# ============================================================
# 2. 检测
# ============================================================

def _load_model(weights: Path):
    """加载（并缓存）YOLO 权重。"""
    key = (str(weights), weights.stat().st_mtime)
    model = _MODEL_CACHE.get(key)
    if model is None:
        from ultralytics import YOLO          # 延迟导入: torch 很重, 别拖慢启动

        model = YOLO(str(weights))
        _MODEL_CACHE.clear()                  # 只留当前这一份, 别把显存/内存堆满
        _MODEL_CACHE[key] = model
        logger.info("已加载权重: %s", weights)
    return model


def build_ultralytics_detector(weights: Optional[Path] = None, *,
                               device: Optional[str] = None,
                               settings: Optional[DetectionSettings] = None) -> Detector:
    """构造真实检测器: ``(image) -> detections``。

    参数一律走 DetectionSettings（即 football.yaml 的 detection 节）,
    不在本文件里写阈值。
    """
    weights = Path(weights) if weights else demo_config.trained_weights_path()
    if not weights.is_file():
        raise UploadError(f"找不到权重文件: {weights}")
    settings = settings or DetectionSettings.from_app_config(load_config())
    device = device or demo_config.DEFAULT_DEVICE

    def detect(image: np.ndarray) -> List[Dict[str, Any]]:
        model = _load_model(weights)
        result = model.predict(image, imgsz=settings.imgsz, conf=settings.conf,
                               iou=settings.iou, max_det=settings.max_det,
                               device=device, verbose=False)[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        return BeautifyVisualizer.from_yolo_results(
            boxes=boxes.xyxy.cpu().numpy(),
            confidences=boxes.conf.cpu().numpy(),
            labels=[result.names[int(index)] for index in boxes.cls.int().cpu().tolist()],
        )

    return detect


# ============================================================
# 3. 绘制与编码
# ============================================================

def render(image: np.ndarray, detections: List[Dict[str, Any]], *,
           label_mapping: Optional[Dict[str, str]] = None) -> np.ndarray:
    """把框画到图像副本上（原图保持不动, 页面要同时显示"原图 / 带框图"）。

    颜色、默认色、线宽覆盖都来自 ``infer_pipeline.yaml``; 只有类别显示名是本演示
    自己的映射 —— 那份共享配置映射的是通用 COCO 名字, 不覆盖球员/球。
    """
    pipeline = load_pipeline_config(None)
    height, width = image.shape[:2]
    style = DrawStyle.from_image_size(height, width, **pipeline.style_overrides)
    mapping = dict(demo_config.LABEL_MAPPING)
    if label_mapping:
        mapping.update(label_mapping)
    visualizer = BeautifyVisualizer(
        label_mapping=mapping,
        color_mapping=pipeline.color_mapping,
        default_color=pipeline.default_color,
        font_path=pipeline.font_path,
    )
    return visualizer.draw(image.copy(), detections, style=style,
                           use_label_mapping=pipeline.use_label_mapping)


def to_data_uri(image: np.ndarray) -> str:
    """图像 → 可内联进 HTML 的 data URI（JPEG, 质量见 demo_config）。"""
    ok, buffer = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), demo_config.IMAGE_JPEG_QUALITY])
    if not ok:
        raise UploadError("图像编码失败, 无法回显")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


# ============================================================
# 编排
# ============================================================

def display_name(cls_name: Optional[str]) -> str:
    """类别名 → 页面显示名。

    与画在框上的那行字用同一份映射（demo_config.LABEL_MAPPING）,
    这样页面上的计数表和图上的标签不会一个中文一个英文。
    """
    if not cls_name:
        return "未命名"
    key = str(cls_name).strip()
    return demo_config.LABEL_MAPPING.get(key.lower(), key)


def analyse(image: np.ndarray, *, detector: Optional[Detector] = None,
            weights: Optional[Path] = None, device: Optional[str] = None,
            settings: Optional[DetectionSettings] = None) -> Dict[str, Any]:
    """跑一次完整演示: 检测 + 绘制 + 统计。

    Args:
        image: BGR 图像
        detector: 注入的检测器（测试用）; None 时构造真实的 ultralytics 检测器
        weights / device / settings: 传给真实检测器

    Returns:
        ``{"detections", "counts", "counts_raw", "total", "original", "annotated",
        "parameters", "device"}``

        · ``counts``     —— 按**显示名**计数（页面表格用, 中文）
        · ``counts_raw`` —— 按模型原始类别名计数（给程序看, 不随展示映射变）
    """
    settings = settings or DetectionSettings.from_app_config(load_config())
    if detector is None:
        detector = build_ultralytics_detector(weights, device=device, settings=settings)

    detections = detector(image) or []
    counts_raw: Dict[str, int] = {}
    counts: Dict[str, int] = {}
    for detection in detections:
        raw = str(detection.get("cls") or "未命名")
        counts_raw[raw] = counts_raw.get(raw, 0) + 1
        shown = display_name(raw)
        counts[shown] = counts.get(shown, 0) + 1

    def _ranked(table: Dict[str, int]) -> Dict[str, int]:
        return dict(sorted(table.items(), key=lambda item: (-item[1], item[0])))

    return {
        "detections": detections,
        "counts": _ranked(counts),
        "counts_raw": _ranked(counts_raw),
        "total": len(detections),
        "original": to_data_uri(image),
        "annotated": to_data_uri(render(image, detections)),
        "parameters": settings.to_audit(),
        # 实际跑在哪个设备上（football.yaml 的 detection.device 只是默认值,
        # 演示站点为了不抢队友门户/训练的卡, 一律覆盖成 CPU）
        "device": device or demo_config.DEFAULT_DEVICE,
    }