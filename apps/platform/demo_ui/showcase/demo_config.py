#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : demo_config.py
# @Project   : ODPlatform / demo_ui
# @Function  : 演示站点自己的具名常量（不读外部配置, 也不写魔法数字）

"""演示站点的本地常量。

为什么单独一个模块而不是散在 settings/views 里:
    "这个演示器自己的设定"（端口、上传上限、权重位置、展示映射）与
    "被演示的算法参数"（来自 configs/runtime/*.yaml）是两回事。
    前者是演示器的主观选择, 后者必须从配置层如实读出。分开写,
    才不会过两天有人把端口写进模板、把阈值写进视图。

★ 本文件不得出现盘符绝对路径（形如 盘符冒号反斜杠 或 盘符冒号斜杠）:
  所有路径经 od_platform.common.paths 拼。
  CI 的硬编码守卫会扫所有被 git 跟踪的 *.py/*.yaml/*.yml/*.json/*.toml, 命中即失败。
"""
from __future__ import annotations

from od_platform.common import paths

# ---------------------------------------------------------------- 服务
# 端口选择: 8000 是 Django 默认（容易被顺手占用）, 8001 是队友的足球门户,
# 所以演示站点固定用 8100, 并在 manage.py 里作为 runserver 的默认地址。
DEMO_PORT: int = 8100
DEFAULT_HOST: str = "127.0.0.1"

# ---------------------------------------------------------------- 被演示的权重
# 训练产物的位置: <TRAINED_MODELS_DIR>/<run 名>/weights/<文件名>
# 这两段是"名字"不是"魔法数字", 但仍然具名, 以免在多个文件里各写一遍字符串。
TRAINED_RUN_NAME: str = "football_yolo"
WEIGHTS_FILENAME: str = "best.pt"

# 训练过程记录（用于如实展示"只训了多少轮"）—— 与权重同目录
TRAIN_RESULTS_FILENAME: str = "results.csv"
TRAIN_ARGS_FILENAME: str = "args.yaml"

# "冒烟级"的判定线: 轮数不超过此值就如实标注为冒烟级, 而不是当成可用模型。
# 3 轮显然远未收敛（yolo11n 一般要 50~300 轮）, 这里给一点余量。
SMOKE_EPOCH_THRESHOLD: int = 10

# ---------------------------------------------------------------- 推理
# 设备兜底: 本机显存只有 8GB, 演示站点一律走 CPU, 不去抢队友门户和训练的卡。
# 想改可以用环境变量 DEMO_UI_DEVICE。
DEVICE_ENV_VAR: str = "DEMO_UI_DEVICE"
DEFAULT_DEVICE: str = "cpu"

# 上传限制: 演示只处理单张图, 8MB 足够覆盖 1280x720 的照片与截图。
UPLOAD_MAX_BYTES: int = 8 * 1024 * 1024
ALLOWED_IMAGE_SUFFIXES: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# 回显用 JPEG 质量: 页面内联 base64, 用 PNG 会把 1280x720 撑到 MB 级。
IMAGE_JPEG_QUALITY: int = 85

# ---------------------------------------------------------------- 展示映射
# 类别名 → 中文显示名。**只属于本演示的展示层**, 故意不去改
# configs/runtime/infer_pipeline.yaml: 那份配置是 odp-infer 共用的, 它映射的是
# 通用 COCO 名字（person/car/bicycle）, 不覆盖球员/球; 为演示去改它等于改了
# 另一个模块的行为。颜色与线宽仍然复用 infer_pipeline.yaml（见 engine.py）。
LABEL_MAPPING: dict = {
    "player": "球员",
    "football": "足球",
    "ball": "足球",
    "referee": "裁判",
    "goalkeeper": "守门员",
}


def trained_weights_path():
    """本演示默认使用的足球检测权重。"""
    return paths.TRAINED_MODELS_DIR / TRAINED_RUN_NAME / "weights" / WEIGHTS_FILENAME


def trained_run_dir():
    """该次训练的产物目录（results.csv / args.yaml 所在处）。"""
    return paths.TRAINED_MODELS_DIR / TRAINED_RUN_NAME