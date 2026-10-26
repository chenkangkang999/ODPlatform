#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : urls.py
# @Project   : ODPlatform / demo_ui
"""演示站点的顶层 URL 表。

只有两页: 总览（能力与参数）与推理演示。没有后台、没有账号 —— 见 settings 的取舍说明。
"""

from django.urls import include, path

urlpatterns = [
    path("", include("showcase.urls")),
]