#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : wsgi.py
# @Project   : ODPlatform / demo_ui
"""WSGI 入口（部署演示用; 答辩现场用 manage.py runserver 即可）。"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "demo_site.settings")

application = get_wsgi_application()