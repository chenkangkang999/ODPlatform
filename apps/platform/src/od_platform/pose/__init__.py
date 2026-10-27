#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : __init__.py
# @Project   : ODPlatform
# @Function  : pose —— 姿态估计与姿态型动作判据子系统

"""pose —— 姿态估计与姿态型动作判据。

★ 这个子系统解决什么

    清单里的"铲球 / 高抬腿 / 肘击 / 拉人推人"是**身体姿态型**动作。拿 VideoMAE 那类
    语义分类模型去做并不合适 —— 那是在 400 个通用动作类里挑一个最像的, 而"假摔"
    在 Kinetics-400 里根本没有对应类别。

    这些动作都有明确的几何定义:
        高抬腿   = 踝关节高于髋关节
        滑铲     = 髋部贴近地面 + 单腿伸展 + 躯干倾倒
        肘击     = 上臂外展抬起 + 肘部贴近对手头部
        拉人推人 = 手臂伸展 + 手部贴近对手躯干
    所以本子系统的核心是**几何判据**, 不是分类模型: 结论可解释、可复算。

★ 三层结构

    :mod:`od_platform.pose.settings`
        全部可调参数（默认值只声明一次）; 对应 ``configs/runtime/pose.yaml``。
    :mod:`od_platform.pose.estimator`
        包装 ultralytics pose 模型（COCO 17 关键点）, 产出 :class:`PersonPose`;
        权重拿不到时优雅降级（``estimate()`` 返回空列表, 不抛异常）。
    :mod:`od_platform.pose.actions`
        **纯几何判据**, 只吃 :class:`PersonPose`, 不 import ultralytics ——
        于是可以脱离模型单测: 摆一组关键点坐标就能验证判据。

★ 两个必须知道的口径

    · **归一化用"躯干长度"**（肩中点→髋中点）而不是外接框: 外接框会随四肢伸展
      剧烈变化（铲球躺下时又扁又宽）, 拿它当尺度等于用被测量的动作定义尺度本身。
    · **没有场地标定时, 接触类结论只能算"候选"**: 身体比例近似不知道镜头远近。
      要米制判据就把几何子系统的单应矩阵 ``distance_m`` 作为 ``distance_fn`` 注入。

★ 依赖

    不需要新装任何东西 —— ``ultralytics`` 自带姿态估计（``yolo11n-pose.pt``）。
    权重放 ``models/pretrained/`` 下即可（本机 Python 直接下载会因证书链失败,
    见 :meth:`~od_platform.pose.settings.PoseModelSettings.resolved_weights` 的说明）。
"""
from __future__ import annotations

from od_platform.pose.actions import (
    ACTION_ELBOW_STRIKE,
    ACTION_HIGH_LEG,
    ACTION_JUDGES,
    ACTION_PULLING_PUSHING,
    ACTION_SLIDING_TACKLE,
    EVIDENCE_CANDIDATE,
    EVIDENCE_INSUFFICIENT,
    EVIDENCE_JUDGEMENT,
    ActionVerdict,
    angle_at,
    judge_all,
    judge_elbow_strike,
    judge_high_leg,
    judge_pulling_pushing,
    judge_sliding_tackle,
)
from od_platform.pose.estimator import (
    ANKLES,
    ARMS,
    COCO_KEYPOINT_NAMES,
    HEAD_NAMES,
    HIPS,
    KEYPOINT_INDEX,
    KNEES,
    LEGS,
    SHOULDERS,
    WRISTS,
    Keypoint,
    PersonPose,
    PoseEstimator,
)
from od_platform.pose.settings import (
    PoseModelSettings,
    PoseSettings,
    PoseThresholds,
    default_config_path,
)

__all__ = [
    # 配置
    "PoseSettings",
    "PoseModelSettings",
    "PoseThresholds",
    "default_config_path",
    # 估计
    "PoseEstimator",
    "PersonPose",
    "Keypoint",
    "COCO_KEYPOINT_NAMES",
    "KEYPOINT_INDEX",
    "SHOULDERS",
    "HIPS",
    "ARMS",
    "LEGS",
    "KNEES",
    "ANKLES",
    "WRISTS",
    "HEAD_NAMES",
    # 判据
    "ActionVerdict",
    "judge_high_leg",
    "judge_sliding_tackle",
    "judge_elbow_strike",
    "judge_pulling_pushing",
    "judge_all",
    "angle_at",
    # 常量
    "ACTION_HIGH_LEG",
    "ACTION_SLIDING_TACKLE",
    "ACTION_ELBOW_STRIKE",
    "ACTION_PULLING_PUSHING",
    "ACTION_JUDGES",
    "EVIDENCE_JUDGEMENT",
    "EVIDENCE_CANDIDATE",
    "EVIDENCE_INSUFFICIENT",
]
