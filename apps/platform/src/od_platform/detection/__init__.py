#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""检测层（YOLO 推理参数与校验护栏）。

所有可调参数来自 :class:`~od_platform.detection.settings.DetectionSettings`
（对应 ``football.yaml`` 的 ``detection:`` 节）, 逻辑代码里不出现字面量。
"""
from od_platform.detection.settings import SECTION, DetectionSettings

__all__ = ["DetectionSettings", "SECTION"]