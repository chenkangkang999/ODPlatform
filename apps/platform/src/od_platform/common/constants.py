"""跨模块共享的纯定义:不加新值就永远是干净的依赖底层(可独立成包)。
新增规则:值属于"两个以上子系统都要引用的业务决策"才进这里;子系统私有默认值留在子系统。
"""
from typing import Tuple


class AnnotationFormat:
    """标注格式标识(注册表键、CLI choices、转换路由共用)。"""
    PASCAL_VOC = "pascal_voc"
    COCO = "coco"
    YOLO = "yolo"

    @classmethod
    def all(cls) -> Tuple[str, ...]:
        return (cls.PASCAL_VOC, cls.COCO, cls.YOLO)


class Task:
    """任务类型:检测/分割。CLI choices + convert 能力路由 + 转换器几何分支共用。"""
    DETECT = "detect"
    SEGMENT = "segment"

    @classmethod
    def all(cls) -> Tuple[str, ...]:
        return (cls.DETECT, cls.SEGMENT)


DEFAULT_RANDOM_STATE: int = 42         # 全局默认种子
DEFAULT_TRAIN_RATE: float = 0.8        # 划分比例默认 8:1:1
DEFAULT_VAL_RATE: float = 0.1
RATE_EPSILON: float = 1e-6             # 浮点比例和的容差


class SplitStrategy:
    """划分策略名(注册表键、CLI choices 共用)。当前两种:随机 / 多标签分层。"""
    RANDOM = "random"
    STRATIFIED_MULTILABEL = "stratified_multilabel"

    @classmethod
    def all(cls) -> Tuple[str, ...]:
        return (cls.RANDOM, cls.STRATIFIED_MULTILABEL)


DEFAULT_SPLIT_STRATEGY: str = SplitStrategy.RANDOM

RARE_WARN_MIN_EXPECTED: float = 3.0    # 类别在最小 split 的期望样本量低于此值 → 稀有类预警
IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# 覆盖率前置闸门:转换前检查"标注数/图像数"。
COVERAGE_HARD_THRESHOLD: float = 0.5   # 低于此值 → 直接拒跑(疑似导出中断/路径配错)
COVERAGE_SOFT_THRESHOLD: float = 0.9   # 低于此值 → 警告但继续

# 统计类阈值:D3 只读报告与 D4 质检共用(本课立、D4 消费)
STATS_MAX_ASPECT_RATIO: float = 20.0   # bbox 宽高比超过此值 → 极端形状(质检 WARNING)
STATS_MAX_IMBALANCE_RATIO: float = 50.0  # 类别实例数失衡比超过此值 → 质检 WARNING
