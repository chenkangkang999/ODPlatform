"""几何/规则引擎 —— 场地标定、坐标投影、接触/出界/定位球（角球/点球/任意球）【候选】。

分层（自下而上）:
    pitch        球场尺寸规格（PitchSpec）与具名关键点 —— 尺寸的唯一真相源
    settings     本子系统的配置模型（标定质量门槛 / 规则阈值 / 分析器参数）
    calibration  标定对象 + 质量校验 + 自动发现（没有标定就返回 None）
    homography   像素 ↔ 米 的投影，并区分"真标定"与"随手映射"
    ball_state   球静止判定 —— 任意球/点球的前置条件（看窗口内总位移, 不看逐帧速度）
    rules        接触 / 出界 / 定位球（角球 / 点球 / 任意球）的候选判据
    analyzer     把以上接起来，产出带证据的事件候选（算法侧入口）
"""
from . import pitch
from .analyzer import PitchAnalyzer
from .calibration import (
    CALIBRATION_SUFFIX, CalibrationError, PitchCalibration, default_calibration_dir, discover,
)
from .homography import FieldHomography
from .pitch import DEFAULT_PITCH, PitchSpec
from .rules import RuleEngine
from .settings import (
    AnalyzerSettings, CalibrationSettings, GeometrySettings, RuleSettings,
)

__all__ = [
    "pitch",
    "PitchAnalyzer",
    "PitchCalibration",
    "CalibrationError",
    "CALIBRATION_SUFFIX",
    "default_calibration_dir",
    "discover",
    "FieldHomography",
    "RuleEngine",
    "PitchSpec",
    "DEFAULT_PITCH",
    "GeometrySettings",
    "CalibrationSettings",
    "RuleSettings",
    "AnalyzerSettings",
]