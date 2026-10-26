#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : manage.py
# @Project   : ODPlatform / demo_ui
# @Function  : 演示站点管理入口（runserver 默认走 8100）

"""演示站点的管理命令入口。

与 ``django-admin`` 的唯一差别: ``runserver`` 不带地址时**自动补上演示端口**。
不补的话 Django 会用 8000, 而 8001 是队友的足球门户 —— 演示现场手忙脚乱地
改端口是最容易出岔子的一步, 干脆在这里定死。

用法::

    python apps/platform/demo_ui/manage.py runserver
    python apps/platform/demo_ui/manage.py check
"""
import os
import sys

from showcase.demo_config import DEFAULT_HOST, DEMO_PORT


def _with_default_addr(argv):
    """runserver 没给地址时补 ``host:port``; 用户显式给了就一个字不动。"""
    if len(argv) < 2 or argv[1] != "runserver":
        return argv
    positional = [item for item in argv[2:] if not item.startswith("-")]
    if any((":" in item) or item.isdigit() for item in positional):
        return argv                      # 用户自己指定了端口/地址
    patched = list(argv)
    patched.insert(2, f"{DEFAULT_HOST}:{DEMO_PORT}")
    return patched


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "demo_site.settings")
    from django.core.management import execute_from_command_line

    execute_from_command_line(_with_default_addr(sys.argv))


if __name__ == "__main__":
    main()