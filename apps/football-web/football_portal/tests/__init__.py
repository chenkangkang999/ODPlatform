import os
import unittest

if os.environ.get("DJANGO_SETTINGS_MODULE") != "football_portal.project_settings":
    raise unittest.SkipTest("足球扩展测试需显式使用football_portal.project_settings，不影响原项目CI。")
