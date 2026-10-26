#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""多目标跟踪 —— ByteTrack 风格的两阶段关联 + 卡尔曼运动模型。

★ 为什么自己实现而不是直接调库:

    ultralytics / supervision 的跟踪器都要求"检测结果对象"（Results/Detections），
    和它们的推理入口绑死。而本项目需要的是: 拿【已经落盘的、纯字典的】逐帧检测
    就能跟踪 —— 例如离线处理 frames.jsonl、或给几何分析提供跨帧轨迹。
    所以这里实现一个不依赖任何推理框架、只吃 numpy 的跟踪器:
    输入 list[dict]、输出 list[Track]，可以脱离 torch 单独测试。

★ 相比历史版本的差别:

    历史版本 update() 根本【没有调用】底层跟踪器, 而是给每个框分配
    track_id = hash(cls_name) % 9999 —— 同一类的所有球员会拿到同一个 ID,
    而且 `from supervision.tracker import ...` 这个模块路径并不存在,
    所以它实际上从未工作过。本实现是真正可跑的。

★ 所有参数（含卡尔曼噪声权重、代价矩阵结构常量、历史长度）都来自
  TrackingSettings（tracking/settings.py）, 本文件里不出现可调数字字面量。

算法要点（ByteTrack, Zhang et al. 2022 的简化落地）:
    1. 卡尔曼滤波: 8 维状态 (x, y, a, h, vx, vy, va, vh), 匀速模型
    2. 两阶段关联: 先用高分检测匹配所有轨迹; 再用【低分检测】去救
       那些没匹配上的轨迹（遮挡/模糊导致的低置信度框正是找回目标的线索）
    3. 按类别分开关联: 球和人不可能互相匹配
    4. 生命周期: tentative → confirmed（连续命中 min_hits 帧）→ lost → removed
    5. 匈牙利指派（scipy）; scipy 不可用时退化为贪心, 行为仍然确定
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from od_platform.tracking.settings import KalmanSettings, TrackingSettings

logger = logging.getLogger(__name__)

STATE_TENTATIVE = "tentative"
STATE_CONFIRMED = "confirmed"
STATE_LOST = "lost"
STATE_REMOVED = "removed"


@dataclass
class Track:
    """单条跟踪结果。

    兼容历史字段（track_id / cls_name / bbox / conf / frame_idx），
    新增字段都有默认值，不影响老调用方。
    """

    track_id: int
    cls_name: str
    bbox: Tuple[float, float, float, float]      # x1, y1, x2, y2（像素）
    conf: float
    frame_idx: int
    state: str = STATE_TENTATIVE
    hits: int = 1
    age: int = 1                                  # 出生以来经过的帧数
    time_since_update: int = 0                    # 连续多少帧没被匹配
    velocity: Tuple[float, float] = (0.0, 0.0)    # 中心点速度（像素/帧）
    history: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def confirmed(self) -> bool:
        return self.state in (STATE_CONFIRMED, STATE_LOST)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def foot(self) -> Tuple[float, float]:
        """脚点（框底边中点）—— 投影到球场时应该用这个点, 而不是框中心。"""
        x1, _, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, float(y2)

    def to_dict(self) -> Dict[str, Any]:
        return {"track_id": self.track_id, "cls_name": self.cls_name,
                "bbox": [round(v, 2) for v in self.bbox], "conf": round(self.conf, 4),
                "frame_idx": self.frame_idx, "state": self.state,
                "hits": self.hits, "time_since_update": self.time_since_update}


class _KalmanFilterXYAH:
    """8 维匀速模型的卡尔曼滤波（x, y, a, h 及其速度）。

    a = 宽高比 (w/h), h = 高度。用 a 而不是 w 是为了让宽高比的变化相对独立。
    噪声参数全部来自 KalmanSettings。
    """

    def __init__(self, settings: Optional[KalmanSettings] = None) -> None:
        self.settings = settings or KalmanSettings()
        self._motion = np.eye(8)
        for i in range(4):
            self._motion[i, i + 4] = 1.0
        self._update = np.eye(4, 8)

    def initiate(self, measurement: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.settings
        mean = np.r_[np.asarray(measurement, dtype=np.float64), np.zeros(4)]
        height = max(float(measurement[3]), 1.0)
        position = cfg.initiate_position_factor * cfg.std_weight_position * height
        velocity = cfg.initiate_velocity_factor * cfg.std_weight_velocity * height
        std = [position, position, cfg.aspect_std, position,
               velocity, velocity, cfg.aspect_velocity_std, velocity]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.settings
        height = max(float(mean[3]), 1.0)
        position = cfg.std_weight_position * height
        velocity = cfg.std_weight_velocity * height
        std_pos = [position, position, cfg.aspect_std, position]
        std_vel = [velocity, velocity, cfg.aspect_velocity_std, velocity]
        noise = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = self._motion @ mean
        covariance = self._motion @ covariance @ self._motion.T + noise
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.settings
        height = max(float(mean[3]), 1.0)
        position = cfg.std_weight_position * height
        std = [position, position, cfg.measurement_aspect_std, position]
        noise = np.diag(np.square(std))
        return self._update @ mean, self._update @ covariance @ self._update.T + noise

    def update(self, mean: np.ndarray, covariance: np.ndarray,
               measurement: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_cov = self.project(mean, covariance)
        try:
            kalman_gain = covariance @ self._update.T @ np.linalg.inv(projected_cov)
        except np.linalg.LinAlgError:
            return mean, covariance
        innovation = np.asarray(measurement, dtype=np.float64) - projected_mean
        mean = mean + kalman_gain @ innovation
        covariance = covariance - kalman_gain @ self._update @ covariance
        return mean, covariance


class _Tracklet:
    """内部轨迹状态（卡尔曼均值/协方差 + 生命周期计数）。"""

    def __init__(self, track_id: int, cls_name: str, measurement: Sequence[float],
                 conf: float, frame_idx: int, kalman: _KalmanFilterXYAH,
                 settings: TrackingSettings) -> None:
        self.track_id = track_id
        self.cls_name = cls_name
        self.conf = conf
        self.frame_idx = frame_idx
        self.settings = settings
        self.mean, self.covariance = kalman.initiate(measurement)
        self.state = STATE_TENTATIVE
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self._last_center = _center_of_xyah(measurement)
        self._velocity = (0.0, 0.0)
        self.history: List[Tuple[float, float]] = [self._last_center]

    # ---- 一步 ----

    def predict(self, kalman: _KalmanFilterXYAH) -> None:
        self.mean, self.covariance = kalman.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, measurement: Sequence[float], conf: float, frame_idx: int,
               kalman: _KalmanFilterXYAH) -> None:
        self.mean, self.covariance = kalman.update(self.mean, self.covariance, measurement)
        self.conf = conf
        self.frame_idx = frame_idx
        self.hits += 1
        self.time_since_update = 0
        center = _center_of_xyah(measurement)
        self._velocity = (center[0] - self._last_center[0], center[1] - self._last_center[1])
        self._last_center = center
        self.history.append(center)
        limit = self.settings.history_length
        if len(self.history) > limit:                   # 有界历史, 防止长视频吃内存
            del self.history[:-limit]
        if self.hits >= self.settings.min_hits:
            self.state = STATE_CONFIRMED

    def mark_lost(self) -> None:
        self.state = (STATE_REMOVED if self.time_since_update > self.settings.max_age
                      else STATE_LOST)

    # ---- 查询 ----

    @property
    def confirmed(self) -> bool:
        return self.state in (STATE_CONFIRMED, STATE_LOST)

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        x, y, a, h = self.mean[:4]
        width = max(a * h, 1.0)
        height = max(h, 1.0)
        return (float(x - width / 2), float(y - height / 2),
                float(x + width / 2), float(y + height / 2))

    @property
    def velocity(self) -> Tuple[float, float]:
        return self._velocity

    def to_track(self) -> Track:
        return Track(track_id=self.track_id, cls_name=self.cls_name, bbox=self.bbox,
                     conf=self.conf, frame_idx=self.frame_idx, state=self.state,
                     hits=self.hits, age=self.age,
                     time_since_update=self.time_since_update, velocity=self.velocity,
                     history=list(self.history))


class MultiObjectTracker:
    """多目标跟踪器（纯 numpy, 不依赖推理框架）。

    Args:
        settings: TrackingSettings 实例, 或直接给 tracking 节的 dict。
                  不给则全部采用出厂默认值。

    用法::

        tracker = MultiObjectTracker(TrackingSettings.from_app_config(config))
        for frame_idx, detections in enumerate(stream):
            tracks = tracker.update(detections, frame_idx)
        for box_index, track_id in tracker.last_assignment.items():
            boxes[box_index]["track_id"] = track_id
    """

    def __init__(self, settings: Optional[TrackingSettings | Mapping] = None):
        if isinstance(settings, Mapping):
            settings = TrackingSettings.from_config(settings)
        self.settings = settings or TrackingSettings()
        self.backend = self.settings.backend.lower()
        self._kalman = _KalmanFilterXYAH(self.settings.kalman)
        self._tracklets: List[_Tracklet] = []
        self._next_id = 1
        self._last_frame_idx = -1
        # 最近一次 update 的"检测下标 → track_id"; 调用方据此把 ID 贴回自己的框
        self.last_assignment: Dict[int, int] = {}
        logger.info("跟踪器就绪: backend=%s min_conf=%.2f high=%.2f iou=%.2f "
                    "max_age=%d min_hits=%d max_tracks=%d",
                    self.backend, self.settings.min_confidence,
                    self.settings.high_threshold, self.settings.iou_threshold,
                    self.settings.max_age, self.settings.min_hits,
                    self.settings.max_tracks)

    # ------------------------------------------------------------------ 主入口

    def update(self, detections: Sequence[Dict[str, Any]], frame_idx: int,
               image: Any = None) -> List[Track]:
        """喂入一帧检测结果, 返回本帧被成功关联的轨迹（含 tentative）。

        Args:
            detections: [{"cls_name": "player", "bbox": [x1,y1,x2,y2], "conf": .9}, ...]
                        也接受桥接层用的键名 {"cls": ..., "xyxy": ...}
            frame_idx: 当前帧号（必须单调不减）
            image: 保留参数（外观特征用）; 本实现不使用, 传 None 即可

        Returns:
            本帧被关联到的轨迹, track_id 跨帧稳定
        """
        if frame_idx < self._last_frame_idx:
            raise ValueError(f"frame_idx 必须单调不减: {frame_idx} < {self._last_frame_idx}")
        self._last_frame_idx = frame_idx
        settings = self.settings

        high: List[Dict[str, Any]] = []
        low: List[Dict[str, Any]] = []
        for source_index, detection in enumerate(detections):
            normalized = _normalize_detection(detection, frame_idx, source_index)
            if normalized is None:
                continue
            if normalized["conf"] >= settings.high_threshold:
                high.append(normalized)
            elif normalized["conf"] >= settings.min_confidence:
                low.append(normalized)

        self.last_assignment = {}
        for tracklet in self._tracklets:
            tracklet.predict(self._kalman)

        # 阶段 1: 高分检测 ↔ 所有轨迹
        matches, unmatched_tracks, unmatched_high = self._associate(self._tracklets, high)
        for track_index, det_index in matches:
            tracklet = self._tracklets[track_index]
            self._apply(tracklet, high[det_index], frame_idx)
            self.last_assignment[high[det_index]["source_index"]] = tracklet.track_id

        # 阶段 2: 低分检测 ↔ 上一阶段没匹配上的【仍在跟踪中】的轨迹
        candidates = [t for i, t in enumerate(self._tracklets)
                      if i in unmatched_tracks and t.time_since_update <= 1]
        low_matches, _, _ = self._associate(candidates, low)
        for track_index, det_index in low_matches:
            tracklet = candidates[track_index]
            self._apply(tracklet, low[det_index], frame_idx)
            self.last_assignment[low[det_index]["source_index"]] = tracklet.track_id

        # 阶段 3: 高分未匹配 → 新建轨迹
        for det_index in unmatched_high:
            tracklet = self._spawn(high[det_index], frame_idx)
            if tracklet is not None:
                self.last_assignment[high[det_index]["source_index"]] = tracklet.track_id

        # 阶段 4: 生命周期收尾
        for tracklet in self._tracklets:
            if tracklet.time_since_update > 0:
                tracklet.mark_lost()
        live = [t for t in self._tracklets if t.state != STATE_REMOVED]
        if len(live) > settings.max_tracks:          # 极端场景保护: 保新弃旧
            live.sort(key=lambda t: (t.time_since_update, -t.hits))
            for tracklet in live[settings.max_tracks:]:
                tracklet.state = STATE_REMOVED
            live = live[:settings.max_tracks]
        self._tracklets = live

        return [t.to_track() for t in self._tracklets if t.time_since_update == 0]

    def reset(self) -> None:
        """清空状态（复用一个实例处理下一个视频时必须调）。"""
        self._tracklets.clear()
        self._next_id = 1
        self._last_frame_idx = -1
        self.last_assignment = {}

    # ------------------------------------------------------------------ 查询

    @property
    def tracks(self) -> List[Track]:
        return [t.to_track() for t in self._tracklets]

    @property
    def confirmed_count(self) -> int:
        return sum(1 for t in self._tracklets if t.confirmed)

    def summary(self) -> Dict[str, Any]:
        """给结果文件/审计用的跟踪摘要。"""
        by_class: Dict[str, int] = {}
        for tracklet in self._tracklets:
            by_class[tracklet.cls_name] = by_class.get(tracklet.cls_name, 0) + 1
        return {"backend": self.backend, "live_tracks": len(self._tracklets),
                "confirmed": self.confirmed_count, "by_class": by_class,
                "settings": {name: getattr(self.settings, name)
                             for name in ("max_age", "min_hits", "iou_threshold",
                                          "min_confidence", "high_threshold",
                                          "max_tracks", "small_object_extent_px",
                                          "center_distance_gate")}}

    # ------------------------------------------------------------------ 内部

    def _spawn(self, detection: Dict[str, Any], frame_idx: int) -> Optional[_Tracklet]:
        if len(self._tracklets) >= self.settings.max_tracks:
            return None
        tracklet = _Tracklet(self._next_id, detection["cls_name"], detection["measurement"],
                             detection["conf"], frame_idx, self._kalman, self.settings)
        self._next_id += 1
        self._tracklets.append(tracklet)
        return tracklet

    def _apply(self, tracklet: _Tracklet, detection: Dict[str, Any], frame_idx: int) -> None:
        tracklet.update(detection["measurement"], detection["conf"], frame_idx, self._kalman)

    def _associate(self, tracklets: Sequence[_Tracklet],
                   detections: Sequence[Dict[str, Any]]
                   ) -> Tuple[List[Tuple[int, int]], set, set]:
        """按类别 + IoU 做关联, 返回 (匹配对, 未匹配轨迹下标, 未匹配检测下标)。"""
        if not tracklets or not detections:
            return [], set(range(len(tracklets))), set(range(len(detections)))

        settings = self.settings
        invalid = settings.invalid_cost
        cost = np.full((len(tracklets), len(detections)), invalid, dtype=np.float64)
        for i, tracklet in enumerate(tracklets):
            box_a = tracklet.bbox
            for j, detection in enumerate(detections):
                if tracklet.cls_name != detection["cls_name"]:
                    continue                              # 球和人互不匹配
                box_b = detection["bbox"]
                iou = _iou(box_a, box_b)
                if iou >= settings.iou_threshold:
                    cost[i, j] = 1.0 - iou
                    continue
                # 小目标兜底: 足球只有十几像素, 每帧挪动的距离和自身尺寸同量级,
                # IoU 会在两帧之间直接跌破阈值 —— 光靠 IoU 会让球每帧换一个 ID。
                # 只对"小目标"启用中心距离兜底; 球员（大目标）仍然只认 IoU,
                # 避免把两个挨着的球员错配成同一条轨迹。
                extent_a, extent_b = _extent(box_a), _extent(box_b)
                if max(extent_a, extent_b) <= settings.small_object_extent_px:
                    ratio = (_center_distance(box_a, box_b)
                             / max((extent_a + extent_b) / 2.0, 1.0))
                    if ratio <= settings.center_distance_gate:
                        cost[i, j] = settings.distance_cost_base + min(ratio, 1.0)

        rows, cols = _solve_assignment(cost, gate=invalid)
        matches = [(int(r), int(c)) for r, c in zip(rows, cols, strict=False)
                   if cost[r, c] < invalid]
        matched_tracks = {r for r, _ in matches}
        matched_dets = {c for _, c in matches}
        return (matches,
                set(range(len(tracklets))) - matched_tracks,
                set(range(len(detections))) - matched_dets)


# ============================================================
# 工具函数
# ============================================================

def _normalize_detection(detection: Dict[str, Any], frame_idx: int,
                         source_index: int = -1) -> Optional[Dict[str, Any]]:
    """把各种键名的检测字典统一成内部格式; 非法检测返回 None（跳过, 不炸）。"""
    cls_name = detection.get("cls_name", detection.get("cls"))
    box = detection.get("bbox", detection.get("xyxy"))
    conf = detection.get("conf", detection.get("confidence", 0.0))
    if cls_name is None or box is None or len(box) != 4:
        return None
    x1, y1, x2, y2 = (float(v) for v in box)
    width, height = x2 - x1, y2 - y1
    if not np.all(np.isfinite([x1, y1, x2, y2, float(conf)])):
        return None
    if width <= 0 or height <= 0:
        return None
    measurement = [(x1 + x2) / 2.0, (y1 + y2) / 2.0, width / height, height]
    return {"cls_name": str(cls_name), "bbox": (x1, y1, x2, y2),
            "conf": float(conf), "measurement": measurement, "frame_idx": frame_idx,
            "source_index": source_index}


def _extent(box: Sequence[float]) -> float:
    """目标"尺度"的保守估计: 面积开方。用于判断是否属于小目标。"""
    x1, y1, x2, y2 = box
    return float(np.sqrt(max(0.0, x2 - x1) * max(0.0, y2 - y1)))


def _center_distance(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax, ay = (box_a[0] + box_a[2]) / 2.0, (box_a[1] + box_a[3]) / 2.0
    bx, by = (box_b[0] + box_b[2]) / 2.0, (box_b[1] + box_b[3]) / 2.0
    return float(np.hypot(ax - bx, ay - by))


def _iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _solve_assignment(cost: np.ndarray, gate: float) -> Tuple[Sequence[int], Sequence[int]]:
    """最小代价指派; scipy 不可用时退化为贪心（结果仍然确定）。"""
    try:
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment(cost)
    except ImportError:
        logger.warning("scipy 不可用, 关联退化为贪心匹配")
        return _greedy_assignment(cost, gate)


def _greedy_assignment(cost: np.ndarray, gate: float) -> Tuple[List[int], List[int]]:
    rows, cols = [], []
    used_rows: set = set()
    used_cols: set = set()
    order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
    for row, col in order:
        row, col = int(row), int(col)
        if cost[row, col] >= gate or row in used_rows or col in used_cols:
            continue
        rows.append(row)
        cols.append(col)
        used_rows.add(row)
        used_cols.add(col)
    return rows, cols


def _center_of_xyah(measurement: Sequence[float]) -> Tuple[float, float]:
    return float(measurement[0]), float(measurement[1])