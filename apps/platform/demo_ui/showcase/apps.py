#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : apps.py
# @Project   : ODPlatform / demo_ui
"""演示应用的 AppConfig。"""

from django.apps import AppConfig


class ShowcaseConfig(AppConfig):
    """算法能力展示应用（总览 + 推理演示）。"""

    name = "showcase"
    verbose_name = "算法能力演示"
    default_auto_field = "django.db.models.BigAutoField"