#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : __init__.py
# @Project   : ODPlatform
# @Function  : 数据接入 —— 把各种原始布局整理成 D3 能吃的标准原始目录

"""数据接入（``odp-ingest`` 的实现）。

★ 解决什么问题

    ``odp-transform --dataset <name>`` 只认**扁平**布局::

        data/raw/<name>/images/*.jpg
        data/raw/<name>/labels/*.txt

    但真实下载来的数据布局五花八门 —— 尤其是 **Roboflow 导出的已经分好
    train/valid/test**, D3 直接吃不了, 用户每次都得手动搬。本包把这一步自动化,
    并顺手把几个**真实踩过的坑**一次处理掉:

    · **越界框**: Roboflow 的 YOLO 归一化坐标带浮点残渣, 会让框微微超出 [0,1]
      （实测 92/22878 个框越界 0.5px）→ 在这里裁回边界, 免得 D4 报 WARNING 卡住开训。
    · **硬链接改写源数据**: 硬链接拍平后目标与源是同一个 inode, 往目标写内容会
      就地截断**源文件**。本包只允许"内容需要改 → 新写文件", 绝无"先链接再写"。
    · **重名静默覆盖**: 不同 split 里的同名文件拍平后会互相覆盖, 数据集悄悄少一批
      样本而不报错 → 默认直接拒绝。

★ 分层

    :mod:`~od_platform.data_pipeline.ingest.settings`  可调项（只在这里声明默认值）
    :mod:`~od_platform.data_pipeline.ingest.scanner`   布局探测 + 体检 + 出计划
    :mod:`~od_platform.data_pipeline.ingest.runner`    落地执行 + 报告

★ 默认先预演: 不加 ``--apply`` 只扫描并打印计划, **一个字节都不写**。
"""
from od_platform.data_pipeline.ingest.runner import (
    IngestResult,
    apply_plan,
    clamp_label_text,
    format_box,
    place_file,
    resolve_name,
    snapshot_source_labels,
    verify_sources_untouched,
    write_report,
)
from od_platform.data_pipeline.ingest.scanner import (
    LAYOUT_DESCRIPTIONS,
    LAYOUT_FLAT,
    LAYOUT_IMAGES_ONLY,
    LAYOUT_LEGACY,
    LAYOUT_PRESPLIT,
    LAYOUT_UNKNOWN,
    Conflict,
    ImageHealth,
    IngestError,
    IngestPlan,
    LabelStats,
    SourceSample,
    box_out_of_range,
    collect_samples,
    detect_layout,
    find_class_names,
    find_conflicts,
    parse_label_line,
    probe_images,
    scan,
    scan_labels,
)
from od_platform.data_pipeline.ingest.settings import (
    CONFLICT_ABORT,
    CONFLICT_POLICIES,
    CONFLICT_PREFIX,
    IngestSettings,
    LayoutSettings,
)

__all__ = [
    # 设置
    "IngestSettings",
    "LayoutSettings",
    "CONFLICT_ABORT",
    "CONFLICT_PREFIX",
    "CONFLICT_POLICIES",
    # 布局与计划
    "LAYOUT_FLAT",
    "LAYOUT_PRESPLIT",
    "LAYOUT_LEGACY",
    "LAYOUT_IMAGES_ONLY",
    "LAYOUT_UNKNOWN",
    "LAYOUT_DESCRIPTIONS",
    "SourceSample",
    "Conflict",
    "LabelStats",
    "ImageHealth",
    "IngestPlan",
    "IngestError",
    "detect_layout",
    "collect_samples",
    "find_conflicts",
    "find_class_names",
    "scan",
    "parse_label_line",
    "box_out_of_range",
    "scan_labels",
    "probe_images",
    # 落地
    "IngestResult",
    "apply_plan",
    "place_file",
    "format_box",
    "clamp_label_text",
    "resolve_name",
    "snapshot_source_labels",
    "verify_sources_untouched",
    "write_report",
]