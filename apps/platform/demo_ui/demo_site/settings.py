#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : settings.py
# @Project   : ODPlatform / demo_ui
# @Function  : 演示站点的 Django 设置

"""答辩演示站点的设置。

★ 定位: 这是**演示器**, 不是生产配置。三个刻意的取舍:

  1. **不接数据库** —— 上传的图片在内存里算完就以 base64 内联回页面, 不落盘、不
     需要历史记录。所以 DATABASES 留空, 也不装 auth/admin/session/contenttypes,
     于是没有迁移、没有账号, `runserver` 起来即成。
  2. **不装任何新依赖** —— Django 6.1.2 / pillow / opencv / ultralytics 都已经在
     仓库的 .venv 里。这个站点只是把它们和算法包接起来。
  3. **路径全部经 od_platform.common.paths 拼** —— 不出现盘符绝对路径
     （CI 的硬编码守卫会扫所有被跟踪的配置文件）。

★ 边界: 本目录与队友的 ``apps/football-web/`` 完全独立, 互不引用、互不修改。
"""
from __future__ import annotations

import os
from pathlib import Path

from showcase.demo_config import (
    DEFAULT_HOST,
    DEMO_PORT,
    DEVICE_ENV_VAR,
    DEFAULT_DEVICE,
    UPLOAD_MAX_BYTES,
)

# 演示站点根目录（= apps/platform/demo_ui）
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- 安全
# 仅供本机演示: 密钥允许用环境变量覆盖; 没给就用这个开发用常量。
# 站点只监听 127.0.0.1 且 DEBUG 只用于演示, 不具备对外服务的条件。
DEV_SECRET_KEY: str = "demo-ui-development-only-not-for-production"
SECRET_KEY: str = os.environ.get("DEMO_UI_SECRET_KEY", DEV_SECRET_KEY)

DEBUG: bool = True
# testserver 是 Django 测试客户端的默认 Host, 少了它测试会 400。
ALLOWED_HOSTS: list = [DEFAULT_HOST, "localhost", "testserver"]

# ---------------------------------------------------------------- 应用
INSTALLED_APPS: list = [
    "django.contrib.staticfiles",
    "showcase",
]

MIDDLEWARE: list = [
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF: str = "demo_site.urls"
WSGI_APPLICATION: str = "demo_site.wsgi.application"

TEMPLATES: list = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,          # 模板放在 showcase/templates/showcase/ 下
        "OPTIONS": {"context_processors": []},
    },
]

# 空字典 = 本次演示不使用数据库（见模块文档字符串的取舍说明）
DATABASES: dict = {}

DEFAULT_AUTO_FIELD: str = "django.db.models.BigAutoField"
STATIC_URL: str = "static/"

# ---------------------------------------------------------------- 本地化
LANGUAGE_CODE: str = "zh-hans"
TIME_ZONE: str = "Asia/Shanghai"
USE_I18N: bool = False
USE_TZ: bool = True

# ---------------------------------------------------------------- 上传限制
# 两道: Django 自己的内存阈值, 以及视图里的显式检查（见 views._read_upload）
DATA_UPLOAD_MAX_MEMORY_SIZE: int = UPLOAD_MAX_BYTES
FILE_UPLOAD_MAX_MEMORY_SIZE: int = UPLOAD_MAX_BYTES
DATA_UPLOAD_MAX_NUMBER_FILES: int = 1

# ---------------------------------------------------------------- 演示站点自己的旋钮
DEMO_DEVICE: str = os.environ.get(DEVICE_ENV_VAR, DEFAULT_DEVICE)
DEMO_UPLOAD_MAX_BYTES: int = UPLOAD_MAX_BYTES
DEMO_PORT: int = DEMO_PORT