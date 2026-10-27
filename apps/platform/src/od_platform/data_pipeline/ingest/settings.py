#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : settings.py
# @Project   : ODPlatform
# @Function  : 数据接入的可调参数（默认值只声明一次）

"""``odp-ingest`` 的配置模型 —— 所有可调项的唯一声明处。

设计原则（对应"全局不要硬编码"）:
    1. 默认值只写在 dataclass 字段上, 逻辑代码里【不出现任何字面量】;
    2. 每一项都能被 ``configs/runtime/ingest.yaml`` 覆盖, 覆盖键名与字段同名;
    3. 配置缺项/类型不对时给出明确报错, 不静默用半个配置。

ingest.yaml 里对应的节:
    layouts:  布局探测规则（目录名/后缀名/类名来源文件名）
    ingest:   接入策略（硬链接优先、是否裁框、重名冲突怎么办、报告文件名…）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from od_platform.common.config_utils import build_settings, read_section, report_unknown_keys
from od_platform.common.football_config import default_config_path

logger = logging.getLogger(__name__)

SECTION = "ingest"
LAYOUTS_SECTION = "layouts"

# 重名冲突的处理策略（具名常量, 免得代码里散落字符串）
CONFLICT_ABORT = "abort"       # 直接拒绝, 中止接入（默认: 不静默覆盖）
CONFLICT_PREFIX = "prefix"     # 给重名文件加来源前缀
CONFLICT_POLICIES: Tuple[str, ...] = (CONFLICT_ABORT, CONFLICT_PREFIX)


@dataclass(frozen=True)
class LayoutSettings:
    """布局探测规则。

    ★ 为什么把这些名字做成配置而不是写死在代码里:

      真实下载来的数据布局五花八门, 而且**换一个导出工具/换一个数据集就变**。
      写死的结果是"上个月能接的布局, 这个月不认了", 而排查成本很高。
      把它们提成配置, 遇到新布局改 yaml 就行, 不用改代码、不用重发版。
    """

    images_dirname: str = "images"          # flat/presplit 布局的图片子目录名
    labels_dirname: str = "labels"          # flat/presplit 布局的标签子目录名
    # 已分好集合的目录名。Roboflow 用 train/valid/test, 有的用 val。
    split_dirnames: Tuple[str, ...] = ("train", "valid", "val", "test")
    # legacy 布局: <src>/<nested_dirname>/<split>/*.jpg + 同名 *.txt
    nested_dirname: str = "data"
    image_suffixes: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    label_suffix: str = ".txt"
    # 类名从哪个文件读（Roboflow 导出会带 data.yaml）
    class_file_names: Tuple[str, ...] = ("data.yaml", "dataset.yaml", "data.yml")

    def __post_init__(self) -> None:
        if not self.split_dirnames:
            raise ValueError("split_dirnames 不能为空 —— 否则已分集合的数据会被当成未分")
        if not self.image_suffixes:
            raise ValueError("image_suffixes 不能为空")
        for name in ("images_dirname", "labels_dirname", "nested_dirname"):
            if not getattr(self, name):
                raise ValueError(f"{name} 不能为空")
        if not self.label_suffix.startswith("."):
            raise ValueError(f"label_suffix 必须以点开头, 收到 {self.label_suffix!r}")

    def normalized_suffixes(self) -> Tuple[str, ...]:
        """统一小写后缀, 免得 ``.JPG`` 漏掉。"""
        return tuple(s.lower() for s in self.image_suffixes)


@dataclass(frozen=True)
class IngestSettings:
    """接入策略。默认值偏保守: 宁可拒绝, 也不要静默产出错的数据集。"""

    layouts: LayoutSettings = field(default_factory=LayoutSettings)

    # ---- 文件落地 ----
    prefer_hardlink: bool = True       # 硬链接优先（零字节复制）; 失败自动回退复制
    # 重名冲突: 拍平后不同来源出现同名文件怎么办
    conflict_policy: str = CONFLICT_ABORT
    conflict_prefix: str = "dup"        # prefix 策略用的前缀
    report_filename: str = "ingest_report.json"

    # ---- 标签处理 ----
    clamp_boxes: bool = True            # 把 YOLO 归一化坐标夹回 [0,1]
    # 裁剪后保留的小数位。D4 的越界容差是 1e-6, 所以 6 位足够且不会因为舍入
    # 又造出新的越界（详见 runner._format_box 的注释）。
    coordinate_precision: int = 6
    box_edge_tolerance: float = 1e-6    # 判定"越界"的容差（与 D4 bbox_within_image 同源）

    # ---- 体检 ----
    probe_images: bool = True           # 是否真去解码探针（能发现截断/损坏的图）
    small_image_side_px: int = 320      # 最小边小于此值 → 报告里提示（小目标风险）
    preview_limit: int = 20             # 报告/日志里每类最多列几个名字

    def __post_init__(self) -> None:
        if self.conflict_policy not in CONFLICT_POLICIES:
            raise ValueError(
                f"conflict_policy 只能是 {CONFLICT_POLICIES}, 收到 {self.conflict_policy!r}")
        if self.coordinate_precision < 1:
            raise ValueError("coordinate_precision 必须 >= 1")
        if self.box_edge_tolerance < 0:
            raise ValueError("box_edge_tolerance 不能为负")
        if self.small_image_side_px < 1:
            raise ValueError("small_image_side_px 必须 >= 1")
        if self.preview_limit < 0:
            raise ValueError("preview_limit 不能为负")
        if not self.report_filename:
            raise ValueError("report_filename 不能为空")

    @property
    def aborts_on_conflict(self) -> bool:
        return self.conflict_policy == CONFLICT_ABORT

    # ---------------------------------------------------------------- 构造

    @classmethod
    def from_config(cls, config: Any, *, where: str = "ingest.yaml") -> "IngestSettings":
        """从整份配置构造（``configs/runtime/ingest.yaml`` 的内容）。"""
        if not isinstance(config, dict):
            return cls()
        unknown: List[str] = []
        layouts = build_settings(LayoutSettings, read_section(config, LAYOUTS_SECTION),
                                 unknown_sink=unknown)
        settings = build_settings(cls, read_section(config, SECTION), unknown_sink=unknown)
        # layouts 在【另一个 yaml 节】下, 所以用它覆盖掉整类构造时拿到的那份 ——
        # 保证 yaml 的两节都真正生效, 而不是只有一节被读到。
        settings = replace(settings, layouts=layouts)
        report_unknown_keys(f"{where} 的 {LAYOUTS_SECTION} 节", unknown)
        return settings

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "IngestSettings":
        """从 yaml 加载; 默认用 configs/runtime/ingest.yaml。

        文件不存在时用默认值并打一条 INFO —— 接入是个"随时能用"的工具,
        不该因为少一个配置文件就跑不起来。
        """
        path = Path(path) if path else default_config_path().parent / "ingest.yaml"
        if not path.is_file():
            logger.info("未找到接入配置 %s, 全部采用默认值", path)
            return cls()
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} 顶层不是 mapping: {type(data).__name__}")
        logger.info("已加载接入配置: %s", path)
        return cls.from_config(data, where=str(path))

    def to_audit(self) -> Dict[str, Any]:
        """给报告/审计用的配置快照（说明"这次是按什么参数接入的"）。"""
        return {
            "prefer_hardlink": self.prefer_hardlink,
            "conflict_policy": self.conflict_policy,
            "clamp_boxes": self.clamp_boxes,
            "coordinate_precision": self.coordinate_precision,
            "box_edge_tolerance": self.box_edge_tolerance,
            "probe_images": self.probe_images,
            "small_image_side_px": self.small_image_side_px,
            "layouts": {
                "images_dirname": self.layouts.images_dirname,
                "labels_dirname": self.layouts.labels_dirname,
                "split_dirnames": list(self.layouts.split_dirnames),
                "nested_dirname": self.layouts.nested_dirname,
                "image_suffixes": list(self.layouts.image_suffixes),
                "label_suffix": self.layouts.label_suffix,
            },
        }