#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : settings.py
# @Project   : ODPlatform
# @Function  : 模型评估的全部可调项（默认值只声明一次）

"""``odp-eval`` 的可调项 —— 对应 ``configs/runtime/eval.yaml``。

★ 为什么要有独立的评估子系统

    ``odp-train`` 只在训练过程中顺带打印指标, 没法"拿一个已经训好的权重单独跑评估、
    出结构化报告"。而"训完看 mAP50 / 每类 AP / IoU / 置信度"恰恰是最需要**可复现**的
    一步 —— 论文和答辩里的数字必须能随时重跑出来, 而不是只存在于某次训练的日志里。

★ 两处默认值的取向（不是随手填的）

    · ``device`` 默认 ``cpu``: 本机显存只有 8GB, 评估不该去抢训练的卡。
    · ``conf`` 默认 ``0.001``: **val 阶段必须用极低阈值**。用 0.25 之类的部署阈值去 val,
      会凭空砍掉 P-R 曲线的左半段, 算出来的 mAP 会系统性偏低。
      注意它和 ``conf_thresholds`` 是两件事: 后者是"部署时选哪个阈值"的扫描点。

★ 报告的两个"如实"开关（答辩会被追问）

    · ``min_images_for_reliable_map``: 验证集样本太少时, mAP 的抖动会大过结论本身;
    · ``min_instances_for_reliable_ap`` / ``imbalance_ratio_warn``: 某类框太少会被
      按类平均的 mAP 拖累（例如 ball 与 player 差一个数量级）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from od_platform.common.config_utils import build_settings, read_section, report_unknown_keys

logger = logging.getLogger(__name__)

SECTION = "evaluation"

# 合法 split —— ultralytics 的三个名字, 不要自创
SPLITS: Tuple[str, ...] = ("val", "test", "train")

# imgsz 必须能被 32 整除: YOLO 的下采样最大 stride 是 32, 不整除会在配置层报错
IMGSZ_STRIDE: int = 32
IMGSZ_MIN: int = 32
IMGSZ_MAX: int = 1536

# batch 与本仓库其它子系统保持一致: -1 = 自动
BATCH_AUTO: int = -1


@dataclass(frozen=True)
class EvalSettings:
    """评估参数。

    `split` / `imgsz` / `batch` / `device` 等字段名与 ultralytics 的 `val()` 同名,
    这样"配置项 → 框架参数"的映射不需要额外的翻译表。
    """

    # ---- 模型与数据 ----
    split: str = "val"                   # val | test | train
    imgsz: int = 640                     # 评估输入边长（须为 32 的倍数）
    batch: int = 8                       # 每批图片数; -1=自动
    device: str = "cpu"                  # cpu | 0 | 0,1
    workers: int = 2                     # 数据加载进程数
    max_det: int = 300                   # 每图最多保留多少框
    conf: float = 0.001                  # ★ val 用极低阈值, P-R 曲线才完整
    iou: float = 0.7                     # NMS 的 IoU 阈值（不是 mAP 的 IoU 门限）
    half: bool = False                   # 半精度（CPU 上无效, 默认关）
    plots: bool = False                  # 是否让 ultralytics 出图（默认关: 报告自己出）
    verbose: bool = True

    # ---- 置信度扫描（"部署时选哪个阈值"用）----
    conf_sweep: bool = True
    conf_thresholds: Tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.25,
                                          0.3, 0.4, 0.5, 0.6, 0.7)
    target_precision: float = 0.90       # "推荐阈值"要满足的精确率下限

    # ---- 如实警告的触发线 ----
    min_images_for_reliable_map: int = 100    # 验证集少于这么多张 → 警告 mAP 不可靠
    min_instances_for_reliable_ap: int = 100  # 某类真值框少于这么多 → 警告 AP 不可靠
    imbalance_ratio_warn: float = 10.0        # 最多类/最少类 的框数比超过它 → 警告

    # ---- 报告 ----
    decimals: int = 4                    # 指标保留几位小数
    include_pr_curves: bool = True       # 是否把每类 P-R 曲线点写进报告
    pr_curve_points: int = 200           # 曲线采样点数（原始 1000 点太占地方）
    limit: int = 0                       # 只评估前 N 张（0=全部）; 仅供链路冒烟, 报告会标注
    task_label: str = "detect"           # 报告目录名用的任务标签: runs/<task_label>_eval/

    def __post_init__(self) -> None:
        if str(self.split) not in SPLITS:
            raise ValueError(f"split 只能是 {SPLITS}, 收到 {self.split!r}")
        if self.imgsz < IMGSZ_MIN or self.imgsz > IMGSZ_MAX:
            raise ValueError(f"imgsz 必须在 [{IMGSZ_MIN}, {IMGSZ_MAX}], 收到 {self.imgsz}")
        if self.imgsz % IMGSZ_STRIDE:
            raise ValueError(f"imgsz 必须是 {IMGSZ_STRIDE} 的倍数, 收到 {self.imgsz}")
        if self.batch == 0:
            raise ValueError("batch 不能为 0（-1 表示自动）")
        if self.workers < 0:
            raise ValueError("workers 不能为负")
        if self.max_det < 1:
            raise ValueError("max_det 必须 >= 1")
        for name in ("conf", "iou"):
            value = getattr(self, name)
            if not 0 < value <= 1:
                raise ValueError(f"{name} 必须在 (0, 1], 收到 {value}")
        if not 0 < self.target_precision <= 1:
            raise ValueError(f"target_precision 必须在 (0, 1], 收到 {self.target_precision}")
        if not self.conf_thresholds:
            raise ValueError("conf_thresholds 不能为空（否则扫描没有意义）")
        for value in self.conf_thresholds:
            if not 0 < value <= 1:
                raise ValueError(f"conf_thresholds 里的值必须在 (0, 1], 收到 {value}")
        if list(self.conf_thresholds) != sorted(set(self.conf_thresholds)):
            raise ValueError("conf_thresholds 必须严格递增且不重复, "
                             f"收到 {self.conf_thresholds}")
        for name in ("min_images_for_reliable_map", "min_instances_for_reliable_ap"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} 必须 >= 1")
        if self.imbalance_ratio_warn <= 1:
            raise ValueError("imbalance_ratio_warn 必须 > 1（比值小于 1 没有意义）")
        if self.decimals < 0:
            raise ValueError("decimals 不能为负")
        if self.pr_curve_points < 2:
            raise ValueError("pr_curve_points 至少为 2")
        if self.limit < 0:
            raise ValueError("limit 不能为负（0 表示不限制）")
        if not str(self.task_label).strip():
            raise ValueError("task_label 不能为空（它是报告目录名的一部分）")

    # ---------------------------------------------------------------- 构造

    @classmethod
    def from_config(cls, config: Any, *, where: str = "eval.yaml") -> "EvalSettings":
        """从整份配置构造（取 ``evaluation`` 节）。"""
        unknown: List[str] = []
        settings = build_settings(cls, read_section(config, SECTION),
                                  aliases=_ALIASES, unknown_sink=unknown)
        report_unknown_keys(f"{where} 的 {SECTION} 节", unknown)
        return settings

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "EvalSettings":
        """从 yaml 加载; 默认用 configs/runtime/eval.yaml; 文件不在就用默认值。"""
        path = Path(path) if path else default_config_path()
        if not path.is_file():
            logger.info("未找到评估配置 %s, 全部采用默认值", path)
            return cls()
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} 顶层不是 mapping: {type(data).__name__}")
        logger.info("已加载评估配置: %s", path)
        return cls.from_config(data, where=str(path))

    # ---------------------------------------------------------------- 审计

    def to_audit(self) -> Dict[str, Any]:
        """给报告用的配置快照 —— 说明"这次评估是按什么参数跑的"。"""
        return {
            "split": self.split,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "device": self.device,
            "workers": self.workers,
            "max_det": self.max_det,
            "conf": self.conf,
            "iou": self.iou,
            "half": self.half,
            "conf_sweep": self.conf_sweep,
            "conf_thresholds": list(self.conf_thresholds),
            "target_precision": self.target_precision,
            "min_images_for_reliable_map": self.min_images_for_reliable_map,
            "min_instances_for_reliable_ap": self.min_instances_for_reliable_ap,
            "imbalance_ratio_warn": self.imbalance_ratio_warn,
            "decimals": self.decimals,
            "limit": self.limit,
            "task_label": self.task_label,
        }

    def round(self, value: Any) -> Any:
        """按配置的小数位取整; 非数字原样返回（None 不该变成 0）。"""
        if isinstance(value, bool) or value is None:
            return value
        try:
            return round(float(value), self.decimals)
        except (TypeError, ValueError):
            return value


# 历史/简写键名 → 字段名。目前只为"把话说短"服务, 不含已废弃的语义。
_ALIASES: Dict[str, str] = {
    "size": "imgsz",
    "thresholds": "conf_thresholds",
    "min_images": "min_images_for_reliable_map",
    "min_instances": "min_instances_for_reliable_ap",
}


def default_config_path() -> Path:
    """eval.yaml 的默认位置。

    路径只在 :mod:`od_platform.common.paths` 里写一次 —— 本函数是个薄封装,
    免得每个子系统各写一份"configs/runtime"的字面量。

    注意 ``runtime_config_path`` 会自己补 ``.yaml`` 后缀, 所以这里传的是文件名主体。
    """
    from od_platform.common import paths
    return paths.runtime_config_path("eval")
