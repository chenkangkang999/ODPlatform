#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""base_smoke —— D5 配置层的端到端冒烟测试。

★ 这一项的来历

    答辩清单 2.1 第二条原本写的是"``base_smoke.py`` 是 checkpoint 脚本而非 pytest
    测试, 要改成正式测试框架"。但全仓已经搜不到 ``base_smoke.py`` 了, 所以按用户
    要求**新建一个正式 pytest 版本**。

★ 它补的是什么空白

    生成器、加载器、合并器、BaseConfig 各自都有单测（``test_generator.py`` /
    ``test_loaders.py`` / ``test_merger.py`` / ``test_base.py``）, 但**"模板生成出来的
    yaml 能不能真的被 build_*_config 吃下去, 并转成 ultralytics 参数"这条跨子系统
    链路此前没有测试覆盖**。单测都绿而链路断掉, 是这类分层代码最典型的失效方式。

    所以这里做的是串联冒烟, 不是重复单测:
        生成模板 → 落盘 → build_*_config 加载+合并 → to_ultralytics_kwargs
"""
from __future__ import annotations

import pytest

from od_platform.common import paths
from od_platform.runtime_config import (
    CONFIG_REGISTRY,
    ConfigGenerator,
    ConfigSource,
    YOLOTrainConfig,
    build_infer_config,
    build_train_config,
    build_val_config,
)

# 注册表名 → 对应的 build_* 入口
_BUILDERS = {
    "train": build_train_config,
    "val": build_val_config,
    "infer": build_infer_config,
}

# ultralytics 参数里不该出现的"框架私有字段"（BaseConfig.FRAMEWORK_ONLY_FIELDS）
_FRAMEWORK_ONLY = {"verbose"}


def test_generated_template_round_trips_into_ultralytics_kwargs(tmp_path):
    """模板 → 落盘 → 加载合并 → ultralytics 参数, 整条链路走得通。"""
    assert ConfigGenerator().generate(YOLOTrainConfig, tmp_path / "train.yaml") is True

    config, merger = build_train_config(yaml_path="train.yaml", yaml_dir=tmp_path)
    assert isinstance(config, YOLOTrainConfig)
    assert merger is not None

    kwargs = config.to_ultralytics_kwargs()
    assert isinstance(kwargs, dict) and kwargs
    # 训练必需的核心项必须在
    for key in ("epochs", "imgsz", "batch", "optimizer"):
        assert key in kwargs, f"生成的模板里 {key} 丢了"
    # 框架私有字段不该喂给 ultralytics
    assert _FRAMEWORK_ONLY.isdisjoint(kwargs)


def test_cli_still_wins_over_a_generated_template(tmp_path):
    """生成出来的模板是 DEFAULT+YAML 两层, CLI 必须仍然最高优先级。"""
    ConfigGenerator().generate(YOLOTrainConfig, tmp_path / "train.yaml")
    config, merger = build_train_config(yaml_path="train.yaml", yaml_dir=tmp_path,
                                        cli_args={"epochs": 7})
    assert config.epochs == 7
    assert merger.get_metadata("epochs").source == ConfigSource.CLI


@pytest.mark.parametrize("name", sorted(CONFIG_REGISTRY))
def test_every_registered_config_generates_and_loads(name, tmp_path):
    """注册表里的每一种配置都要能"生成 → 加载" —— 新增配置类型时自动被这条兜住。"""
    config_cls = CONFIG_REGISTRY[name][0]
    yaml_name = f"{name}.yaml"
    assert ConfigGenerator().generate(config_cls, tmp_path / yaml_name) is True

    config, _ = _BUILDERS[name](yaml_path=yaml_name, yaml_dir=tmp_path)
    assert config is not None, f"{name}: 生成的模板没能加载出配置"
    assert isinstance(config.to_ultralytics_kwargs(), dict)


def test_repo_train_yaml_keeps_the_football_tuning():
    """仓库里那份 train.yaml 的调参值必须还在。

    这条是**契约测试**: 那些值有实测依据(见 train.yaml 头部注释),
    被无意改掉(或随模板重新生成冲掉)时必须立刻失败, 而不是悄悄用回默认值。
    """
    config, _ = build_train_config(
        yaml_path=str(paths.RUNTIME_CONFIGS_DIR / "train.yaml"))
    assert config.imgsz == 1280, "imgsz 是足球小目标最关键的参数, 不该退回 640"
    assert config.optimizer == "AdamW", "auto 会忽略 lr0, 必须显式指定优化器"
    assert config.lr0 == 0.001
    assert config.cache == "disk", "ram 缓存叠加多 worker 会把内存吃光(实测教训)"
    assert config.multi_scale is False, "1280 + multi_scale 在 8GB 卡上会 OOM(实测教训)"
    assert config.patience == 30
    assert config.close_mosaic == 20
    assert config.cos_lr is True


def test_repo_configs_all_survive_the_pipeline():
    """仓库里现有的 train/val/infer 三份 yaml 都要能走完整条链路。"""
    for name in sorted(CONFIG_REGISTRY):
        yaml_path = paths.RUNTIME_CONFIGS_DIR / f"{name}.yaml"
        if not yaml_path.is_file():
            continue
        config, _ = _BUILDERS[name](yaml_path=str(yaml_path))
        assert config is not None, f"{yaml_path.name} 加载不出配置"
        kwargs = config.to_ultralytics_kwargs()
        assert isinstance(kwargs, dict)
        assert _FRAMEWORK_ONLY.isdisjoint(kwargs)