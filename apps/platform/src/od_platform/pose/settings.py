#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : settings.py
# @Project   : ODPlatform
# @Function  : 姿态估计与姿态型动作判据的全部可调参数

"""姿态估计子系统的配置模型 —— 所有可调数字的唯一声明处。

★ 为什么要有这个子系统

    清单里的"铲球 / 高抬腿 / 肘击 / 拉人推人"属于**身体姿态型**动作。用 VideoMAE
    那类语义分类模型去做它们并不合适: 那是在 400 个通用动作类里挑一个最像的,
    而"假摔"在 Kinetics-400 里根本没有对应类别。

    这些动作其实都有**明确的几何定义**:
        高抬腿 = 踝关节高于髋关节
        滑铲   = 髋部贴近地面 + 单腿伸展 + 躯干倾倒
        肘击   = 上臂外展抬起 + 肘部贴近对手头部
        拉人推人 = 手臂伸展 + 手部贴近对手躯干
    所以这里的核心是**几何判据**, 不是分类模型 —— 判据可解释、可复算、答辩时能当场画出来。

★ 设计原则（对应"全局不要硬编码"）

    1. 默认值只写在 dataclass 字段上, 逻辑代码里【不出现任何字面量】;
    2. 每一项都能被 ``configs/runtime/pose.yaml`` 覆盖, 覆盖键名与字段同名;
    3. 非法值在 ``__post_init__`` 里抛 ``ValueError`` —— 不静默接受半个配置。

★ 阈值为什么全部用"相对自身躯干长度"的比值

    用像素绝对量做阈值, 换分辨率、换机位、球员离镜头远近一变化, 阈值就全废了。
    用"相对自身身体尺度"的比值, 则与远近、与分辨率无关。

    归一化基准选的是**肩-髋躯干长度**(左右肩中点 → 左右髋中点), 不是外接框:
    外接框会随着四肢伸展剧烈变化 —— 铲球时人躺下来, 外接框又扁又宽, 拿它当尺度
    等于用"被测量的动作"去定义尺度本身。躯干长度在四肢运动时基本不变, 才是稳的基准。

★ pose.yaml 里对应的节

    model:       权重/imgsz/置信度/设备
    thresholds:  四个动作的几何判据阈值（全部相对躯干长度或角度）
    allow_uncalibrated_candidate: 无标定时是否按身体比例近似（只能当候选）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from od_platform.common.config_utils import (
    as_bool,
    build_settings,
    read_section,
    report_unknown_keys,
)
from od_platform.common.paths import PRETRAINED_MODELS_DIR

logger = logging.getLogger(__name__)

SECTION = "pose"

# 取值范围（用于 __post_init__ 校验, 不参与判据计算）
CONF_MIN: float = 0.0
CONF_MAX: float = 1.0
ANGLE_MIN_DEG: float = 0.0
ANGLE_MAX_DEG: float = 180.0
# 躯干长度的退化保护: 低于此像素值的尺度不可信, 判据直接报"证据不足"。
# 这是数值退化保护（除零/噪声）, 不是判据阈值, 所以允许是绝对量。
MIN_TORSO_LENGTH_PX: float = 1.0

# 顶层标量字段名（用于报告未知键）
_TOP_LEVEL_SCALARS = ("allow_uncalibrated_candidate",)


@dataclass(frozen=True)
class PoseModelSettings:
    """姿态模型相关参数。"""

    weights: str = "yolo11n-pose.pt"   # 文件名（相对 models/pretrained 解析）或路径
    imgsz: int = 640                   # 推理分辨率
    conf: float = 0.25                 # 人体检测置信度下限
    iou: float = 0.7                   # NMS 的 IoU 阈值
    device: str = ""                   # 空 = 交给 ultralytics 自动选
    max_persons: int = 30              # 单帧最多处理的人数（画面里人很多时保护性能）
    keypoint_conf: float = 0.50        # 单个关键点置信度下限, 低于它视为"不可用"

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("weights 不能为空")
        if self.imgsz <= 0:
            raise ValueError(f"imgsz 必须为正数, 收到 {self.imgsz}")
        if self.max_persons < 1:
            raise ValueError(f"max_persons 必须 >= 1, 收到 {self.max_persons}")
        for name, value in (("conf", self.conf), ("iou", self.iou),
                            ("keypoint_conf", self.keypoint_conf)):
            if not CONF_MIN <= float(value) <= CONF_MAX:
                raise ValueError(f"{name} 必须在 [{CONF_MIN}, {CONF_MAX}] 内, 收到 {value}")

    def resolved_weights(self) -> str:
        """返回可直接交给 ultralytics 的权重参数。

        能给本地文件就【给文件路径】而不是名字 —— 名字会让 ultralytics 去 GitHub
        下载, 而实测本机 Python 的证书链拿不到本地 CA, 会直接
        ``SSLCertVerificationError`` 失败（PowerShell 能下是因为它走 Windows 证书库）。
        所以离线/内网环境下, 本地权重文件是唯一可靠的路。
        只在 ``models/pretrained/`` 下按文件名找, 不写任何绝对路径。
        """
        candidate = Path(self.weights)
        if candidate.is_absolute():
            # 配了绝对路径: 存在就用, 不存在原样返回（让 ultralytics 自己报错, 不隐藏）
            return str(candidate) if candidate.is_file() else self.weights
        local = PRETRAINED_MODELS_DIR / candidate
        return str(local) if local.is_file() else self.weights


@dataclass(frozen=True)
class PoseThresholds:
    """四个姿态型动作的几何判据阈值。

    每个阈值后面都注明了"正常站立时大概是多少", 这样调参时知道该往哪边挪,
    也知道离正常值有多远（否则 0.1 和 0.35 看起来毫无差别）。
    """

    # ---- 高抬腿: 踝高于髋多少（单位: 躯干长度）----
    # 判据量 = (髋_y - 踝_y) / 躯干长度, 图像 y 向下为正。
    # 正常站立时踝在髋下方很远, 该量约 -1.5; 抬到与髋同高时约 0; 高于髋为正。
    high_leg_ankle_above_hip_ratio: float = 0.10

    # ---- 滑铲 ----
    # 髋部离自身最低点的高度 / 自身高度; 站立时约 0.45~0.50, 滑铲时降到 0.1~0.2
    slide_hip_height_ratio: float = 0.35
    # 膝关节伸展角（大腿-小腿夹角）; 伸直约 180, 滑铲的伸腿那侧接近伸直
    slide_knee_extension_deg: float = 150.0
    # 躯干相对竖直方向的倾角; 正常跑动 < 20, 滑铲/摔倒时明显倾倒
    slide_torso_incline_deg: float = 45.0

    # ---- 肘击 ----
    # 上臂(肩→肘)与躯干轴(肩→髋)的夹角, 即肩关节外展角。
    # 注意: "肘关节外展"在解剖上不成立（肘只能屈伸）, 真正抬肘来自肩关节外展,
    # 所以这里量的是上臂相对躯干抬起的角度; 手臂自然下垂时约 10~20。
    elbow_abduction_deg: float = 60.0
    # 肘部到"最近对手头部"的距离 / 自身躯干长度
    elbow_head_distance_ratio: float = 1.00

    # ---- 拉人推人 ----
    # ★ 手臂【张开角】才是关键判据: 在肩处量 (肩-腕) 与躯干轴 (肩-髋) 的夹角。
    #   实测基准: 手臂自然下垂约 10°, 伸手抓人/推人约 60~90°。
    #   为什么不用"肩-腕距离/躯干长度"当关键判据: 自然下垂的手臂也是完全伸直的,
    #   那个比值照样是 1.08 —— 它区分不了"垂着"和"伸出去", 只能守住"手臂没折叠"。
    pull_reach_abduction_deg: float = 45.0
    # 手臂伸展度 = 肩-腕距离 / 躯干长度。屈臂约 0.35, 完全伸直约 1.3。
    # 它只当"手臂没有折叠"的下限守卫（自然下垂 1.08 也过, 那是预期的）。
    pull_reach_extension_ratio: float = 0.90
    # 手部到对手躯干中心的距离 / 自身躯干长度（**无标定时的近似判据, 只能当候选**）。
    pull_contact_distance_ratio: float = 1.20
    # 手部到对手躯干中心的距离上限（米）; 有标定时优先用这个（这才是物理判据）。
    pull_contact_distance_m: float = 2.00

    # ---- 通用 ----
    # 命中结果的置信度低于此值时, 降级为"候选"（有迹象但不足以作判据）
    min_hit_confidence: float = 0.50

    def __post_init__(self) -> None:
        for name in ("high_leg_ankle_above_hip_ratio", "slide_hip_height_ratio",
                     "elbow_head_distance_ratio", "pull_reach_extension_ratio",
                     "pull_contact_distance_ratio"):
            value = float(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} 不能为负, 收到 {value}")
        for name in ("slide_knee_extension_deg", "slide_torso_incline_deg",
                     "elbow_abduction_deg", "pull_reach_abduction_deg"):
            value = float(getattr(self, name))
            if not ANGLE_MIN_DEG < value <= ANGLE_MAX_DEG:
                raise ValueError(f"{name} 必须在 ({ANGLE_MIN_DEG}, {ANGLE_MAX_DEG}] 内, 收到 {value}")
        if self.pull_contact_distance_m <= 0:
            raise ValueError(f"pull_contact_distance_m 必须为正数, 收到 {self.pull_contact_distance_m}")
        if not CONF_MIN <= self.min_hit_confidence <= CONF_MAX:
            raise ValueError(f"min_hit_confidence 必须在 [{CONF_MIN}, {CONF_MAX}] 内, "
                             f"收到 {self.min_hit_confidence}")


@dataclass(frozen=True)
class PoseSettings:
    """姿态子系统的完整配置。"""

    model: PoseModelSettings = field(default_factory=PoseModelSettings)
    thresholds: PoseThresholds = field(default_factory=PoseThresholds)
    # 没有场地标定时, 是否允许退化成"按身体比例/像素近似"。
    # ★ 这样得到的结论只能标注为【候选】, 不能当判据 —— 因为它不知道镜头远近,
    #   两个像素挨着的人可能在真实场地里差着两米。
    allow_uncalibrated_candidate: bool = True

    @classmethod
    def from_config(cls, config: Any, *, where: str = f"{SECTION}.yaml") -> "PoseSettings":
        """从配置 mapping 构造。

        允许两种形态:
            · 独立文件 —— 顶层就是 ``model:`` / ``thresholds:``;
            · 嵌在 football.yaml 里 —— 顶层有 ``pose:`` 节, 自动往下钻一层。
        """
        if not isinstance(config, Mapping):
            return cls()
        nested = config.get(SECTION)
        if isinstance(nested, Mapping):
            config = nested

        unknown: List[str] = []
        settings = cls(
            model=_build(PoseModelSettings, config, "model", where, unknown),
            thresholds=_build(PoseThresholds, config, "thresholds", where, unknown),
            allow_uncalibrated_candidate=_read_allow_uncalibrated(config, where),
        )
        report_unknown_keys(where, unknown)
        return settings

    @classmethod
    def load(cls, path: Optional[Any] = None) -> "PoseSettings":
        """从 yaml 文件加载; 默认 configs/runtime/pose.yaml。"""
        path = Path(path) if path else default_config_path()
        if not path.is_file():
            logger.info("未找到姿态配置 %s, 全部采用默认值", path)
            return cls()
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, Mapping):
            raise ValueError(f"{path} 顶层不是 mapping: {type(data).__name__}")
        logger.info("已加载姿态配置: %s", path)
        return cls.from_config(data, where=str(path))

    def to_audit(self) -> Dict[str, Any]:
        """给结果文件/审计用的参数快照（说明"这次是按什么参数判的"）。"""
        return {
            "model": {
                "weights": self.model.weights,
                "resolved_weights": self.model.resolved_weights(),
                "imgsz": self.model.imgsz,
                "conf": self.model.conf,
                "iou": self.model.iou,
                "device": self.model.device or "(auto)",
                "max_persons": self.model.max_persons,
                "keypoint_conf": self.model.keypoint_conf,
            },
            "thresholds": {
                "high_leg_ankle_above_hip_ratio": self.thresholds.high_leg_ankle_above_hip_ratio,
                "slide_hip_height_ratio": self.thresholds.slide_hip_height_ratio,
                "slide_knee_extension_deg": self.thresholds.slide_knee_extension_deg,
                "slide_torso_incline_deg": self.thresholds.slide_torso_incline_deg,
                "elbow_abduction_deg": self.thresholds.elbow_abduction_deg,
                "elbow_head_distance_ratio": self.thresholds.elbow_head_distance_ratio,
                "pull_reach_abduction_deg": self.thresholds.pull_reach_abduction_deg,
                "pull_reach_extension_ratio": self.thresholds.pull_reach_extension_ratio,
                "pull_contact_distance_ratio": self.thresholds.pull_contact_distance_ratio,
                "pull_contact_distance_m": self.thresholds.pull_contact_distance_m,
                "min_hit_confidence": self.thresholds.min_hit_confidence,
            },
            "allow_uncalibrated_candidate": self.allow_uncalibrated_candidate,
        }


def default_config_path() -> Path:
    """pose.yaml 的默认位置（相对仓库根, 不写绝对路径）。"""
    from od_platform.common.paths import RUNTIME_CONFIGS_DIR
    return RUNTIME_CONFIGS_DIR / f"{SECTION}.yaml"


def _read_allow_uncalibrated(config: Mapping[str, Any], where: str) -> bool:
    """读顶层开关, 同时把顶层出现的其它未知键报出来。"""
    unknown = [key for key in config
               if key not in ("model", "thresholds", *_TOP_LEVEL_SCALARS)]
    if unknown:
        report_unknown_keys(where, unknown)
    return as_bool(config.get("allow_uncalibrated_candidate"), default=True)


def _build(cls, config: Mapping[str, Any], section: str, where: str, unknown: List[str]):
    """构造一个子配置节, 并把未知键收集进 unknown。"""
    settings = build_settings(cls, read_section(config, section), unknown_sink=unknown)
    return settings
