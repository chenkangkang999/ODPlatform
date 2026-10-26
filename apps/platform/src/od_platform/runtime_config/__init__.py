"""runtime_config — YOLO 训练 / 验证 / 推理配置子系统.

公共 API:
  配置类: BaseConfig, YOLOTrainConfig, YOLOValConfig, YOLOInferConfig
  加载器: YAMLLoader, CLILoader, load_all_sources
  合并器: ConfigMerger, ConfigSource, ConfigMetadata
  生成器: ConfigGenerator
  注册表: CONFIG_REGISTRY
  一键 build: build_train_config, build_val_config, build_infer_config
"""
from __future__ import annotations

from argparse import Namespace
from pathlib  import Path
from typing   import Any, Dict, List, Mapping, Optional, Tuple, Union

from od_platform.runtime_config.base   import BaseConfig
from od_platform.runtime_config.train  import YOLOTrainConfig
from od_platform.runtime_config.val    import YOLOValConfig
from od_platform.runtime_config.infer  import YOLOInferConfig

from od_platform.runtime_config.loaders import (
    YAMLLoader,
    CLILoader,
    load_all_sources,
)

from od_platform.runtime_config.merger import (
    ConfigMerger,
    ConfigSource,
    ConfigMetadata,
)

from od_platform.runtime_config.generator import ConfigGenerator
from od_platform.runtime_config.registry  import CONFIG_REGISTRY


def _build_config(
    config_cls: type,
    yaml_path: Optional[Union[str, Path]],
    cli_args: Optional[Union[Namespace, Dict[str, Any]]],
    *,
    yaml_dir: Optional[Union[str, Path]] = None,
    cli_exclude: Optional[List[str]] = None,
    cli_mapping: Optional[Dict[str, str]] = None,
    extra_sources: Optional[List[Tuple[Union[ConfigSource, str], Mapping[str, Any]]]] = None,
    track_sources: bool = True,
    dry_run: bool = False,
) -> Tuple[Optional[Any], ConfigMerger]:
    """三个 build_*_config 的公共实现。

    它们的差别只有"目标配置类"一件事; 此前是三段逐字重复约 20 行的代码。
    重复的代价不是多打几个字, 而是**改一处漏两处** —— 源装配顺序（YAML → 自定义源 →
    CLI, 优先级由低到高）、cli 映射、审计开关、dry-run 分支, 这四样里任何一样漏改
    一个入口, 三个 build_* 的行为就会悄悄不一致。

    Args:
        config_cls: 目标配置类（YOLOTrainConfig / YOLOValConfig / YOLOInferConfig）
        yaml_path: yaml 文件名或路径; None 表示不读 yaml
        cli_args: CLI 覆盖项（Namespace 或 dict）
        yaml_dir: yaml 所在目录; None 时用 paths.RUNTIME_CONFIGS_DIR
        cli_exclude / cli_mapping: CLI 键的排除与改名规则
        extra_sources: 额外的自定义配置源（优先级低于 CLI、高于 YAML）
        track_sources: 是否记录每个字段的来源链（审计用）
        dry_run: True 时只预览合并计划, 不产出配置实例

    Returns:
        (配置实例或 None, 合并器)
    """
    from od_platform.common.paths import RUNTIME_CONFIGS_DIR

    sources = load_all_sources(
        yaml_path=yaml_path, yaml_dir=yaml_dir or RUNTIME_CONFIGS_DIR,
        cli_args=cli_args, cli_exclude=cli_exclude, cli_mapping=cli_mapping,
    )
    # 顺序即优先级（后者覆盖前者）: YAML → 自定义源 → CLI
    ordered: List[Tuple[Union[ConfigSource, str], Mapping[str, Any]]] = []
    if sources["yaml"]:
        ordered.append((ConfigSource.YAML, sources["yaml"]))
    if extra_sources:
        ordered.extend(extra_sources)
    if sources["cli"]:
        ordered.append((ConfigSource.CLI, sources["cli"]))

    merger = ConfigMerger(track_sources=track_sources)
    if dry_run:
        merger.preview(config_cls, sources=ordered)
        return None, merger
    return merger.merge(config_cls, sources=ordered), merger


def build_train_config(
    yaml_path: Optional[Union[str, Path]] = "train.yaml",
    cli_args:  Optional[Union[Namespace, Dict[str, Any]]] = None,
    *,
    yaml_dir:      Optional[Union[str, Path]] = None,
    cli_exclude:   Optional[List[str]]        = None,
    cli_mapping:   Optional[Dict[str, str]]   = None,
    extra_sources: Optional[List[Tuple[Union[ConfigSource, str], Mapping[str, Any]]]] = None,
    track_sources: bool = True,
    dry_run:       bool = False,
) -> Tuple[Optional[YOLOTrainConfig], ConfigMerger]:
    """一键完成 train 配置流水线: YAML + CLI (+ 自定义源) → 合并 → 验证。"""
    return _build_config(
        YOLOTrainConfig, yaml_path, cli_args, yaml_dir=yaml_dir,
        cli_exclude=cli_exclude, cli_mapping=cli_mapping,
        extra_sources=extra_sources, track_sources=track_sources, dry_run=dry_run,
    )


def build_val_config(
    yaml_path: Optional[Union[str, Path]] = "val.yaml",
    cli_args:  Optional[Union[Namespace, Dict[str, Any]]] = None,
    *,
    yaml_dir:      Optional[Union[str, Path]] = None,
    cli_exclude:   Optional[List[str]]        = None,
    cli_mapping:   Optional[Dict[str, str]]   = None,
    extra_sources: Optional[List[Tuple[Union[ConfigSource, str], Mapping[str, Any]]]] = None,
    track_sources: bool = True,
    dry_run:       bool = False,
) -> Tuple[Optional[YOLOValConfig], ConfigMerger]:
    """一键完成 val 配置流水线: YAML + CLI (+ 自定义源) → 合并 → 验证。"""
    return _build_config(
        YOLOValConfig, yaml_path, cli_args, yaml_dir=yaml_dir,
        cli_exclude=cli_exclude, cli_mapping=cli_mapping,
        extra_sources=extra_sources, track_sources=track_sources, dry_run=dry_run,
    )


def build_infer_config(
    yaml_path: Optional[Union[str, Path]] = "infer.yaml",
    cli_args:  Optional[Union[Namespace, Dict[str, Any]]] = None,
    *,
    yaml_dir:      Optional[Union[str, Path]] = None,
    cli_exclude:   Optional[List[str]]        = None,
    cli_mapping:   Optional[Dict[str, str]]   = None,
    extra_sources: Optional[List[Tuple[Union[ConfigSource, str], Mapping[str, Any]]]] = None,
    track_sources: bool = True,
    dry_run:       bool = False,
) -> Tuple[Optional[YOLOInferConfig], ConfigMerger]:
    """一键完成 infer 配置流水线: YAML + CLI (+ 自定义源) → 合并 → 验证。"""
    return _build_config(
        YOLOInferConfig, yaml_path, cli_args, yaml_dir=yaml_dir,
        cli_exclude=cli_exclude, cli_mapping=cli_mapping,
        extra_sources=extra_sources, track_sources=track_sources, dry_run=dry_run,
    )


__all__ = [
    "BaseConfig", "YOLOTrainConfig", "YOLOValConfig", "YOLOInferConfig",
    "YAMLLoader", "CLILoader", "load_all_sources",
    "ConfigMerger", "ConfigSource", "ConfigMetadata",
    "ConfigGenerator", "CONFIG_REGISTRY",
    "build_train_config", "build_val_config", "build_infer_config",
]