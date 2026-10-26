#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : urls.py
# @Project   : ODPlatform / demo_ui
"""演示应用的 URL 表。"""

from django.urls import path

from showcase import views

app_name = "showcase"

urlpatterns = [
    path("", views.overview, name="overview"),
    path("infer/", views.infer, name="infer"),
]