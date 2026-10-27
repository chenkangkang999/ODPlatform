#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""球场标定 —— 把"画面上的像素"与"球场上的真实米"对应起来。

这是几何层的地基: 没有标定, 任何以"米"为单位的结论（球员间距、出界、
定位球是否摆在角球弧/罚球点上）都是假的。所以本模块的设计原则是:

    1. 标定必须【显式提供】—— 不猜、不默认、不用画面四角凑数;
    2. 标定必须【可验证】—— 记录重投影误差, 误差过大直接拒绝;
    3. 标定必须【可追溯】—— 记下来源、图像尺寸、备注, 落成 JSON 进版本库;
    4. 没有标定就【返回 None】—— 让上层老实说"未标定", 而不是假装能算。

★ 所有阈值（重投影误差上限、RANSAC 内点阈值、四角判定容差…）都来自
  CalibrationSettings（见 geometric/settings.py）, 本文件里不出现魔法数字。

标定从哪来（按优先级）:
    · 代码里直接构造 PitchCalibration
    · 素材旁边的同名 sidecar: <stem>.calibration.json
    · 素材同目录的 calibration.json（整个目录共用一份）
    · configs/runtime/calibrations/<stem>.json
    · configs/runtime/calibrations.json 里的分节

标定文件格式（schema = football.pitch-calibration.v1）::

    {
      "schema": "football.pitch-calibration.v1",
      "image_size": [1920, 1080],
      "source": "manual",
      "notes": "主场机位 A",
      "points": [
        {"landmark": "penalty_area_left_top",    "pixel": [512.0, 402.0]},
        {"landmark": "penalty_area_left_bottom", "pixel": [498.0, 655.0]},
        {"landmark": "halfway_bottom",           "pixel": [960.0, 604.0]},
        {"landmark": "goal_left_post_bottom",    "pixel": [371.0, 470.0]}
      ]
    }

也支持直接给成对坐标（不写 landmark）::

    {"pixel": [512.0, 402.0], "pitch": [16.5, 13.84]}

用 `odp-calibrate --list-landmarks` 查看所有可用的关键点名。
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from od_platform.geometric.pitch import DEFAULT_PITCH, PitchSpec
from od_platform.geometric.settings import CalibrationSettings

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "football.pitch-calibration.v1"
CALIBRATION_SUFFIX = ".calibration.json"

Point = Tuple[float, float]


class CalibrationError(ValueError):
    """标定缺失、格式不对或质量不合格。"""


# ============================================================
# 纯数学: 单应性求解与应用
# ============================================================

def solve_homography(image_points: Sequence[Point], pitch_points: Sequence[Point],
                     settings: Optional[CalibrationSettings] = None) -> np.ndarray:
    """由 >=min_points 组对应点求 3x3 单应性矩阵（像素 → 米）。"""
    settings = settings or CalibrationSettings()
    if len(image_points) < settings.min_points:
        raise CalibrationError(
            f"标定至少需要 {settings.min_points} 组对应点, 只给了 {len(image_points)} 组")
    src = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    dst = np.asarray(pitch_points, dtype=np.float64).reshape(-1, 1, 2)
    matrix = _find_homography(src, dst, len(image_points), settings)
    if matrix is None or not np.all(np.isfinite(matrix)):
        raise CalibrationError("单应性矩阵求解失败: 标定点退化（共线或重复），请重新标定")
    return matrix


def _find_homography(src: np.ndarray, dst: np.ndarray, point_count: int,
                     settings: CalibrationSettings):
    """恰好 min_points 个点时用直接解, 多于它时用 RANSAC 抗离群点。"""
    import cv2
    if point_count == settings.min_points:
        return cv2.findHomography(src, dst, 0)[0]
    return cv2.findHomography(src, dst, cv2.RANSAC, settings.ransac_reprojection_px)[0]


def apply_homography(matrix: np.ndarray, x: float, y: float) -> Point:
    """单点投影（不做异常吞没: 标定错了就该炸出来）。"""
    import cv2
    point = np.array([[[float(x), float(y)]]], dtype=np.float64)
    out = cv2.perspectiveTransform(point, matrix)[0][0]
    return float(out[0]), float(out[1])


# ============================================================
# 标定对象
# ============================================================

@dataclass(frozen=True)
class PitchCalibration:
    """一组"画面像素 ↔ 球场米"对应点 + 由它算出的单应性矩阵。

    推荐用 from_landmarks / load / discover 构造, 而不是手写 image_points。
    """

    image_points: Tuple[Point, ...]
    pitch_points: Tuple[Point, ...]
    image_size: Optional[Tuple[int, int]] = None
    source: str = "explicit"
    notes: str = ""
    pitch: PitchSpec = DEFAULT_PITCH
    settings: CalibrationSettings = field(default_factory=CalibrationSettings)
    _matrix: np.ndarray = field(default=None, repr=False, compare=False)
    _reprojection_error: float = field(default=float("nan"), repr=False, compare=False)

    # ---- 构造 ----

    def __post_init__(self) -> None:
        image_points = tuple((float(p[0]), float(p[1])) for p in self.image_points)
        pitch_points = tuple((float(p[0]), float(p[1])) for p in self.pitch_points)
        object.__setattr__(self, "image_points", image_points)
        object.__setattr__(self, "pitch_points", pitch_points)

        self._validate_inputs()
        matrix = solve_homography(image_points, pitch_points, self.settings)
        error = _reprojection_error(matrix, image_points, pitch_points)
        if error > self.settings.max_reprojection_error_m:
            raise CalibrationError(
                f"标定重投影误差 {error:.3f} m 超过上限 "
                f"{self.settings.max_reprojection_error_m} m; "
                f"标定点互相矛盾, 请检查像素点是否点错")
        object.__setattr__(self, "_matrix", matrix)
        object.__setattr__(self, "_reprojection_error", error)

    def _validate_inputs(self) -> None:
        minimum = self.settings.min_points
        if len(self.image_points) != len(self.pitch_points):
            raise CalibrationError(
                f"像素点 {len(self.image_points)} 个与场地对应点 "
                f"{len(self.pitch_points)} 个数量不等")
        if len(self.image_points) < minimum:
            raise CalibrationError(f"标定至少需要 {minimum} 组对应点, 只给了 {len(self.image_points)} 组")

        for name, points in (("像素点", self.image_points), ("场地坐标", self.pitch_points)):
            for x, y in points:
                if not (math.isfinite(x) and math.isfinite(y)):
                    raise CalibrationError(f"{name}里有非有限数值: ({x}, {y})")

        if len(set(self.image_points)) < minimum:
            raise CalibrationError("像素点里有重复点, 无法求解单应性")

        margin = self.settings.finite_margin
        for x, y in self.pitch_points:
            if not self.pitch.within(x, y, margin=margin):
                raise CalibrationError(
                    f"场地坐标 ({x}, {y}) 落在球场 "
                    f"{self.pitch.length}x{self.pitch.width} m 之外, "
                    f"请检查 landmark 名与像素点的对应")

        if self.image_size is not None:
            width, height = self.image_size
            if width <= 0 or height <= 0:
                raise CalibrationError(f"图像尺寸非法: {self.image_size}")
            factor = self.settings.point_range_factor
            outside = [p for p in self.image_points
                       if not (-width <= p[0] <= (1 + factor) * width
                               and -height <= p[1] <= (1 + factor) * height)]
            if outside:
                raise CalibrationError(
                    f"有 {len(outside)} 个像素点远离图像范围 {self.image_size}, 疑似点错坐标")

    @classmethod
    def from_landmarks(
        cls,
        image_points: Dict[str, Sequence[float]],
        *,
        image_size: Optional[Tuple[int, int]] = None,
        source: str = "manual",
        notes: str = "",
        pitch: PitchSpec = DEFAULT_PITCH,
        settings: Optional[CalibrationSettings] = None,
    ) -> "PitchCalibration":
        """用具名关键点构造: {"penalty_area_left_top": [x, y], ...}。"""
        settings = settings or CalibrationSettings()
        if len(image_points) < settings.min_points:
            raise CalibrationError(
                f"至少需要 {settings.min_points} 个具名关键点, 只给了 {len(image_points)} 个")
        pixels: List[Point] = []
        ground: List[Point] = []
        for name, pixel in image_points.items():
            ground.append(pitch.landmark(name))
            pixels.append((float(pixel[0]), float(pixel[1])))
        return cls(tuple(pixels), tuple(ground), image_size=image_size, source=source,
                   notes=notes, pitch=pitch, settings=settings)

    @classmethod
    def from_config(cls, config: Dict[str, Any], *,
                    pitch: PitchSpec = DEFAULT_PITCH,
                    settings: Optional[CalibrationSettings] = None) -> "PitchCalibration":
        """从配置的一节构造。

        支持两种写法:
            points: [{landmark: ..., pixel: [...]}, ...]   ← 推荐
            image_points: [[x, y], ...] + pitch_points: [[x, y], ...]
        """
        settings = settings or CalibrationSettings()
        image_size = _as_image_size(config.get("image_size"))
        points = config.get("points")
        if points:
            named = {entry["landmark"]: entry["pixel"] for entry in points if "landmark" in entry}
            pairs = [(entry["pixel"], entry["pitch"]) for entry in points
                     if "landmark" not in entry and "pixel" in entry and "pitch" in entry]
            if named and pairs:
                raise CalibrationError("标定配置里 landmark 写法与 pixel/pitch 写法不能混用")
            if pairs:
                return cls(tuple((float(p[0][0]), float(p[0][1])) for p in pairs),
                           tuple((float(p[1][0]), float(p[1][1])) for p in pairs),
                           image_size=image_size, source=str(config.get("source", "config")),
                           notes=str(config.get("notes", "")), pitch=pitch, settings=settings)
            return cls.from_landmarks(named, image_size=image_size,
                                      source=str(config.get("source", "config")),
                                      notes=str(config.get("notes", "")),
                                      pitch=pitch, settings=settings)
        if config.get("image_points") and config.get("pitch_points"):
            return cls(tuple(tuple(p) for p in config["image_points"]),
                       tuple(tuple(p) for p in config["pitch_points"]),
                       image_size=image_size, source=str(config.get("source", "config")),
                       notes=str(config.get("notes", "")), pitch=pitch, settings=settings)
        raise CalibrationError("标定配置里没有 points / image_points+pitch_points, 无法构造标定")

    # ---- 查询 ----

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @property
    def reprojection_error_m(self) -> float:
        return self._reprojection_error

    @property
    def point_count(self) -> int:
        return len(self.image_points)

    @property
    def full_frame_suspected(self) -> bool:
        """像素点是否就是整幅画面的四角（转播机位上这是典型的凑数标定）。"""
        return _looks_like_full_frame(self.image_points, self.image_size,
                                      self.settings.full_frame_tolerance_px)

    def quality(self) -> Dict[str, Any]:
        """给报告/审计用的标定质量摘要。"""
        return {
            "source": self.source,
            "points": self.point_count,
            "reprojection_error_m": round(self._reprojection_error,
                                          self.settings.coordinate_precision_m),
            "image_size": list(self.image_size) if self.image_size else None,
            "full_frame_suspected": self.full_frame_suspected,
            "notes": self.notes,
        }

    # ---- 序列化 ----

    def to_dict(self) -> Dict[str, Any]:
        px_digits = self.settings.coordinate_precision_px
        m_digits = self.settings.coordinate_precision_m
        return {
            "schema": SCHEMA_VERSION,
            "image_size": list(self.image_size) if self.image_size else None,
            "source": self.source,
            "notes": self.notes,
            "points": [
                {"pixel": [round(px, px_digits), round(py, px_digits)],
                 "pitch": [round(gx, m_digits), round(gy, m_digits)]}
                for (px, py), (gx, gy) in zip(self.image_points, self.pitch_points,
                                              strict=True)
            ],
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *,
                  pitch: PitchSpec = DEFAULT_PITCH,
                  settings: Optional[CalibrationSettings] = None) -> "PitchCalibration":
        if not isinstance(data, dict):
            raise CalibrationError(f"标定文件顶层应为对象, 收到 {type(data).__name__}")
        schema = data.get("schema")
        if schema and schema != SCHEMA_VERSION:
            raise CalibrationError(
                f"标定文件 schema 为 {schema!r}, 本版本只认 {SCHEMA_VERSION!r}")
        if not data.get("points") and not data.get("image_points"):
            raise CalibrationError("标定文件缺少 points / image_points")
        return cls.from_config(data, pitch=pitch, settings=settings)

    @classmethod
    def load(cls, path: Path, *, pitch: PitchSpec = DEFAULT_PITCH,
             settings: Optional[CalibrationSettings] = None) -> "PitchCalibration":
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise CalibrationError(f"标定文件不存在: {path}") from None
        except json.JSONDecodeError as exc:
            raise CalibrationError(f"标定文件不是合法 JSON: {path} ({exc})") from None
        return cls.from_dict(data, pitch=pitch, settings=settings)


# ============================================================
# 发现与校验: 自动找标定, 找不到就老实返回 None
# ============================================================

def candidate_paths(media_path: Path, *, extra_dirs: Iterable[Path] = ()) -> List[Path]:
    """按优先级列出可能的标定文件路径（只列, 不读）。"""
    media_path = Path(media_path)
    stem = media_path.stem
    found: List[Path] = [
        media_path.with_name(stem + CALIBRATION_SUFFIX),          # 素材同名 sidecar
        media_path.with_name("calibration.json"),                  # 同目录共用
    ]
    for directory in list(extra_dirs) + _default_calibration_dirs():
        directory = Path(directory)
        found.append(directory / f"{stem}.json")
        found.append(directory / CALIBRATION_SUFFIX)
    return found


def _default_calibration_dirs() -> List[Path]:
    from od_platform.geometric.settings import default_config_path
    base = default_config_path().parent
    return [base / "calibrations", base]


def default_calibration_dir() -> Path:
    """标定文件的默认存放目录（odp-calibrate --save <名字> 落在这里）。"""
    from od_platform.geometric.settings import default_config_path
    return default_config_path().parent / "calibrations"


def discover(
    media_path: Path,
    *,
    image_size: Optional[Tuple[int, int]] = None,
    extra_dirs: Iterable[Path] = (),
    pitch: PitchSpec = DEFAULT_PITCH,
    settings: Optional[CalibrationSettings] = None,
) -> Optional[PitchCalibration]:
    """自动寻找并加载标定。

    找不到返回 None —— 调用方必须据此声明"未标定", 不要自己编一个。
    找到但非法/尺寸不匹配, 记 warning 后返回 None（不让坏标定污染结果）。
    """
    settings = settings or CalibrationSettings()
    for candidate in candidate_paths(media_path, extra_dirs=extra_dirs):
        if not candidate.is_file():
            continue
        try:
            calibration = PitchCalibration.load(candidate, pitch=pitch, settings=settings)
        except CalibrationError as exc:
            logger.warning("标定文件 %s 不可用: %s", candidate, exc)
            continue

        if image_size and calibration.image_size and tuple(calibration.image_size) != tuple(image_size):
            logger.warning("标定文件 %s 记录的分辨率 %s 与实际画面 %s 不一致, 拒绝使用",
                           candidate, calibration.image_size, image_size)
            continue
        if image_size and calibration.image_size is None:
            logger.info("标定文件 %s 未记录分辨率, 无法核对机位变更", candidate)
        logger.info("已加载球场标定: %s (来源 %s, 重投影误差 %.3f m)",
                    candidate, calibration.source, calibration.reprojection_error_m)
        return calibration
    return None


# ============================================================
# 内部工具
# ============================================================

def _reprojection_error(matrix: np.ndarray, image_points: Sequence[Point],
                        pitch_points: Sequence[Point]) -> float:
    """把像素点投影到米, 与给出的场地坐标比, 取最大误差（米）。"""
    import cv2
    src = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(src, matrix).reshape(-1, 2)
    expected = np.asarray(pitch_points, dtype=np.float64).reshape(-1, 2)
    return float(np.max(np.linalg.norm(projected - expected, axis=1)))


def _looks_like_full_frame(image_points: Sequence[Point],
                           image_size: Optional[Tuple[int, int]],
                           tolerance: float) -> bool:
    if image_size is None or not image_points:
        return False
    width, height = float(image_size[0]), float(image_size[1])
    corners = {(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)}
    for px, py in image_points:
        if not any(abs(px - cx) <= tolerance and abs(py - cy) <= tolerance
                   for cx, cy in corners):
            return False
    return True


def _as_image_size(value: Any) -> Optional[Tuple[int, int]]:
    if not value or len(value) != 2:
        return None
    return int(value[0]), int(value[1])