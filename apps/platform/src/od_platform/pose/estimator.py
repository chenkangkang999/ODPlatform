#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : estimator.py
# @Project   : ODPlatform
# @Function  : 姿态估计器与人体姿态数据结构（包装 ultralytics pose 模型）

"""姿态估计: 把一帧图像变成一组结构化的"人体姿态"。

★ 依赖说明: 不需要新装任何东西

    ``ultralytics`` 自带姿态估计（``YOLO("yolo11n-pose.pt")``, COCO 17 关键点）,
    实测 ``task=pose`` / ``kpt_shape=[17, 3]``。所以本模块只用已有依赖。

★ 权重从哪来（这是本机真实踩到的坑, 不是假设）

    实测: 用 ultralytics 自动下载 ``yolo11n-pose.pt`` 会失败 ——
    ``SSLCertVerificationError: unable to get local issuer certificate``。
    这是 Python 的证书链拿不到本地 CA 导致的（同一时刻 PowerShell 能正常访问
    github.com, 因为它走 Windows 证书库）。
    所以 :meth:`PoseModelSettings.resolved_weights` 优先把
    ``models/pretrained/`` 下的**本地文件路径**交给 ultralytics; 拿不到权重时
    **优雅降级**: 构造时记一条说明清楚的日志, ``estimate()`` 返回空列表,
    绝不抛异常打断调用方。

★ 可注入

    构造函数接受 ``model=``（鸭子类型: 只要有 ``predict``）。测试用它注入假模型,
    既不依赖真权重, 也不需要在测试里跑 torch。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from od_platform.pose.settings import PoseSettings

logger = logging.getLogger(__name__)

# ============================================================
# COCO 17 关键点: 具名常量 + 顺序表
# 顺序必须与模型输出严格一致（kpt_shape=[17,3]）, 所以集中声明一次,
# 逻辑代码里只出现名字, 不出现索引数字。
# ============================================================
NOSE = "nose"
LEFT_EYE = "left_eye"
RIGHT_EYE = "right_eye"
LEFT_EAR = "left_ear"
RIGHT_EAR = "right_ear"
LEFT_SHOULDER = "left_shoulder"
RIGHT_SHOULDER = "right_shoulder"
LEFT_ELBOW = "left_elbow"
RIGHT_ELBOW = "right_elbow"
LEFT_WRIST = "left_wrist"
RIGHT_WRIST = "right_wrist"
LEFT_HIP = "left_hip"
RIGHT_HIP = "right_hip"
LEFT_KNEE = "left_knee"
RIGHT_KNEE = "right_knee"
LEFT_ANKLE = "left_ankle"
RIGHT_ANKLE = "right_ankle"

# 索引 = 模型输出的关键点下标。顺序是 COCO 标准, 不可改。
COCO_KEYPOINT_NAMES: Tuple[str, ...] = (
    NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
)
KEYPOINT_INDEX: Dict[str, int] = {name: index for index, name in enumerate(COCO_KEYPOINT_NAMES)}
KEYPOINT_COUNT: int = len(COCO_KEYPOINT_NAMES)

# 判据用到的关键点分组（避免在各处重复写左右两个名字）
SHOULDERS: Tuple[str, ...] = (LEFT_SHOULDER, RIGHT_SHOULDER)
HIPS: Tuple[str, ...] = (LEFT_HIP, RIGHT_HIP)
ELBOWS: Tuple[str, ...] = (LEFT_ELBOW, RIGHT_ELBOW)
WRISTS: Tuple[str, ...] = (LEFT_WRIST, RIGHT_WRIST)
KNEES: Tuple[str, ...] = (LEFT_KNEE, RIGHT_KNEE)
ANKLES: Tuple[str, ...] = (LEFT_ANKLE, RIGHT_ANKLE)
# 头部: 用五点里能用的那些求中心, 比只用鼻子稳（侧面/低头时鼻子可能不可用）
HEAD_NAMES: Tuple[str, ...] = (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR)

# 按"侧"组织的手臂/腿关键点: (近端关节, 中间关节, 远端, 同侧髋)
# 写死成表而不是在判据里用 f-string 拼名字 —— 名字拼错不会报错, 只会静默判不出结果。
ARMS: Dict[str, Tuple[str, str, str, str]] = {
    "left": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, LEFT_HIP),
    "right": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, RIGHT_HIP),
}
# (髋, 膝, 踝) —— 膝关节伸展角要用这三个点
LEGS: Dict[str, Tuple[str, str, str]] = {
    "left": (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
    "right": (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
}
SIDES: Tuple[str, ...] = ("left", "right")


def _to_numpy(value: Any):
    """把 tensor / 数组 / 列表统一成 numpy 数组; None 原样返回。"""
    if value is None:
        return None
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        value = numpy()
    import numpy as np
    return np.asarray(value)


@dataclass(frozen=True)
class Keypoint:
    """一个关键点。

    ``available`` 是**已经按阈值判过**的结论: 置信度不够的点在这里就被标成不可用,
    后面的判据必须跳过依赖它的动作, 而不是拿着一个噪声坐标（或者 0）去算 ——
    那会得出"看起来很像"的假结论。
    """

    x: float
    y: float
    conf: float
    available: bool = True

    @property
    def point(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class PersonPose:
    """一个人的姿态: 一组具名关键点 + 整体置信度 + 外接框。"""

    keypoints: Dict[str, Keypoint] = field(default_factory=dict)
    confidence: float = 0.0
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    index: int = 0                       # 本帧内的序号（便于与轨迹 id 对齐）

    # ---------------------------------------------------------- 取点
    def get(self, name: str) -> Optional[Keypoint]:
        """取一个可用的关键点; 缺失或置信度不足都返回 None。"""
        keypoint = self.keypoints.get(name)
        if keypoint is None or not keypoint.available:
            return None
        return keypoint

    def point(self, name: str) -> Optional[Tuple[float, float]]:
        keypoint = self.get(name)
        return keypoint.point if keypoint else None

    def available_names(self) -> List[str]:
        return [name for name in COCO_KEYPOINT_NAMES if self.get(name) is not None]

    def missing(self, names: Sequence[str]) -> List[str]:
        """给定一组关键点, 返回其中不可用的那些（用来解释"证据不足"）。"""
        return [name for name in names if self.get(name) is None]

    # ---------------------------------------------------------- 组合点
    def center_of(self, names: Sequence[str], *, min_available: int = 1
                  ) -> Optional[Tuple[float, float]]:
        """一组关键点的中心; 可用点少于 ``min_available`` 时返回 None。

        注意这里用的是**可用点的平均**而不是"把缺失点当 0" —— 后者会把中心
        拉到图像左上角, 得出完全错误的几何关系。
        """
        points = [self.point(name) for name in names]
        usable = [p for p in points if p is not None]
        if len(usable) < min_available:
            return None
        return (sum(p[0] for p in usable) / len(usable),
                sum(p[1] for p in usable) / len(usable))

    def shoulder_center(self) -> Optional[Tuple[float, float]]:
        return self.center_of(SHOULDERS)

    def hip_center(self) -> Optional[Tuple[float, float]]:
        return self.center_of(HIPS)

    def torso_center(self) -> Optional[Tuple[float, float]]:
        """躯干中心 = 肩中点与髋中点的中点（两对点都要可用）。"""
        shoulder, hip = self.shoulder_center(), self.hip_center()
        if shoulder is None or hip is None:
            return None
        return ((shoulder[0] + hip[0]) / 2.0, (shoulder[1] + hip[1]) / 2.0)

    def head_center(self) -> Optional[Tuple[float, float]]:
        return self.center_of(HEAD_NAMES)

    def torso_length(self) -> Optional[float]:
        """归一化基准: 肩中点 → 髋中点的距离。

        为什么是它而不是外接框高度: 外接框会随四肢伸展剧烈变化（铲球躺下时又扁又宽）,
        拿它当尺度等于用"被测量的动作"定义尺度本身。躯干在四肢运动时基本不变。
        """
        shoulder, hip = self.shoulder_center(), self.hip_center()
        if shoulder is None or hip is None:
            return None
        return math.dist(shoulder, hip)

    # ---------------------------------------------------------- 尺度
    @property
    def body_height(self) -> float:
        """外接框高度（像素）。"""
        return max(0.0, float(self.bbox[3]) - float(self.bbox[1]))

    def hip_height_ratio(self) -> Optional[float]:
        """髋部离自身最低点的高度 / 自身高度。站立约 0.45~0.50, 滑铲降到 0.1~0.2。"""
        hip = self.hip_center()
        height = self.body_height
        if hip is None or height <= 0:
            return None
        lowest = float(self.bbox[3])
        return (lowest - hip[1]) / height

    def torso_incline_deg(self) -> Optional[float]:
        """躯干相对竖直方向的倾角（度）。正常跑动 < 20, 滑铲/摔倒明显倾倒。"""
        shoulder, hip = self.shoulder_center(), self.hip_center()
        if shoulder is None or hip is None:
            return None
        dx = abs(hip[0] - shoulder[0])
        dy = abs(hip[1] - shoulder[1])
        if dx == 0.0 and dy == 0.0:
            return None
        # 与竖直方向(0,1)的夹角: 竖直时 dx=0 → 0 度
        return math.degrees(math.atan2(dx, dy))


class PoseEstimator:
    """姿态估计器。

    Args:
        settings: 姿态配置; None 时用默认值
        model: 注入的模型（鸭子类型: 有 ``predict`` 即可）; 给了就不再自行加载
        weights: 覆盖权重路径; None 时用 ``settings.model.resolved_weights()``
    """

    def __init__(self, settings: Optional[PoseSettings] = None, *,
                 model: Any = None, weights: Optional[str] = None):
        self.settings = settings or PoseSettings()
        self._model = model
        self._weights = weights if weights is not None else self.settings.model.resolved_weights()
        self._load_error = ""
        if model is not None:
            self._available = True
        else:
            self._available = self._load_model()

    # ---------------------------------------------------------- 状态
    @property
    def available(self) -> bool:
        """模型是否真的加载成功。False 时 :meth:`estimate` 恒返回空列表。"""
        return self._available

    @property
    def load_error(self) -> str:
        """加载失败的原因; 成功时为空串。"""
        return self._load_error

    @property
    def weights(self) -> str:
        return self._weights

    def _load_model(self) -> bool:
        """加载 ultralytics pose 模型; 失败不抛异常, 只记日志并降级。"""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            self._load_error = f"ultralytics 不可用: {exc}"
            logger.warning("姿态估计已降级（%s）: 请安装 ultralytics", self._load_error)
            return False
        try:
            self._model = YOLO(self._weights)
        except Exception as exc:  # noqa: BLE001 —— 权重缺失/损坏/下载失败都走降级
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "姿态模型加载失败, 本模块已降级为空结果: %s\n"
                "  权重参数: %s\n"
                "  提示: 本机 Python 的证书链拿不到本地 CA, 让 ultralytics 自动下载会报 "
                "SSLCertVerificationError; 请把 yolo11n-pose.pt 放到 models/pretrained/ 下 "
                "（或用 PowerShell 等走系统证书库的工具下载）",
                self._load_error, self._weights)
            return False
        logger.info("姿态模型已加载: %s（task=%s）", self._weights,
                    getattr(self._model, "task", "?"))
        return True

    # ---------------------------------------------------------- 推理
    def _predict_kwargs(self) -> Dict[str, Any]:
        model = self.settings.model
        kwargs: Dict[str, Any] = {"imgsz": model.imgsz, "conf": model.conf,
                                  "iou": model.iou, "verbose": False}
        if model.device:
            kwargs["device"] = model.device
        return kwargs

    def estimate(self, image: Any) -> List[PersonPose]:
        """估计一帧里所有人的姿态。

        Args:
            image: BGR 图像（numpy 数组）或图像路径

        Returns:
            姿态列表; 模型不可用、图像为空、或没有检测到人时返回**空列表**（不抛异常）。
        """
        if not self._available:
            return []
        if image is None:
            return []
        try:
            results = self._model.predict(image, **self._predict_kwargs())
        except Exception as exc:  # noqa: BLE001 —— 单帧失败不该打断视频流
            logger.warning("姿态推理失败, 本帧按空结果处理: %s: %s", type(exc).__name__, exc)
            return []
        if not results:
            return []
        return self._parse(results[0])

    def _parse(self, result: Any) -> List[PersonPose]:
        """把 ultralytics 的单个 Result 解析成 PersonPose 列表。"""
        boxes = getattr(result, "boxes", None)
        keypoints = getattr(result, "keypoints", None)
        if boxes is None or keypoints is None:
            return []

        xy = _to_numpy(getattr(keypoints, "xy", None))
        # 形状必须是 (N, 17, 2)。后端异常/缺通道时可能给出 None 或 0 维数组,
        # 直接往下走会在 len() 上炸 —— 这里明确挡掉, 当作"本帧没结果"。
        if xy is None or getattr(xy, "ndim", 0) < 3 or xy.size == 0:
            return []
        kconf = _to_numpy(getattr(keypoints, "conf", None))
        bboxes = _to_numpy(getattr(boxes, "xyxy", None))
        bconf = _to_numpy(getattr(boxes, "conf", None))

        threshold = self.settings.model.keypoint_conf
        count = min(len(xy), self.settings.model.max_persons,
                    len(bboxes) if bboxes is not None else len(xy))

        # 按整体置信度从高到低取前 max_persons 个
        order = list(range(len(xy)))
        if bconf is not None and len(bconf) >= len(xy):
            order.sort(key=lambda i: -float(bconf[i]))
        order = order[:count]

        poses: List[PersonPose] = []
        for slot, person_index in enumerate(order):
            poses.append(self._build_pose(
                person_index, slot, xy[person_index], kconf, bboxes, bconf, threshold))
        return poses

    def _build_pose(self, person_index: int, slot: int, xy_row: Any, kconf: Any,
                    bboxes: Any, bconf: Any, threshold: float) -> PersonPose:
        keypoints: Dict[str, Keypoint] = {}
        for name, kpt_index in KEYPOINT_INDEX.items():
            if kpt_index >= len(xy_row):
                continue
            x, y = float(xy_row[kpt_index][0]), float(xy_row[kpt_index][1])
            conf = 0.0
            if kconf is not None and person_index < len(kconf) and kpt_index < len(kconf[person_index]):
                conf = float(kconf[person_index][kpt_index])
            usable = (conf >= threshold
                      and math.isfinite(x) and math.isfinite(y))
            keypoints[name] = Keypoint(x=x, y=y, conf=conf, available=usable)

        bbox = (0.0, 0.0, 0.0, 0.0)
        if bboxes is not None and person_index < len(bboxes):
            bbox = tuple(float(v) for v in bboxes[person_index][:4])
        confidence = 0.0
        if bconf is not None and person_index < len(bconf):
            confidence = float(bconf[person_index])
        return PersonPose(keypoints=keypoints, confidence=confidence,
                          bbox=bbox, index=slot)
