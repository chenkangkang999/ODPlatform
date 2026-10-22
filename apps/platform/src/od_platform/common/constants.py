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

# ---- D4 质检阈值(阈值是业务决策,进 constants;check 里不写魔法数) ----
PAIR_MISSING_ERROR_RATIO: float = 0.5    # 缺标签图占比 ≥ 此值 → ERROR(疑似导出中断)
PAIR_MISSING_WARN_RATIO: float = 0.05    # ≥ 此值 → WARNING
PAIR_MAX_DETAIL: int = 20                # details 里缺失清单的展示上限
LINEAGE_MAX_DETAIL: int = 20             # manifest_lineage 的问题清单展示上限
DETAILS_PREVIEW_LIMIT: int = 20          # 各 check details 预览列表的统一上限
BBOX_EDGE_TOLERANCE: float = 1e-6        # 出界判定的浮点容差
BBOX_OUT_ERROR_RATIO: float = 0.10       # 出界框占比 ≥ 10% → ERROR,否则 WARNING
DEGENERATE_BOX_MIN_SIZE: float = 1e-3    # 框宽/高/面积低于此值 → 退化框
UNLABELED_WARN_RATIO: float = 0.30       # 无标注图占比 ≥ 此值 → WARNING
UNLABELED_INFO_RATIO: float = 0.05       # ≥ 此值 → INFO
SPLIT_MIN_IMAGES: int = 30               # split 图数低于此值 → INFO(评估方差偏大)
CLASS_IMBALANCE_INFO_RATIO: float = 10.0 # 类别失衡比 ≥ 此值 → INFO(≥50 由 STATS_MAX 接管)
ORPHAN_LABELS_WARN_RATIO: float = 0.10   # 孤儿标签占比 ≥ 此值 → WARNING
