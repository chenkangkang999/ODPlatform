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
    """一键完成 train 配置流水线: YAML + CLI (+ 自定义源) → 合并 → 验证."""
    from od_platform.common.paths import RUNTIME_CONFIGS_DIR

    sources = load_all_sources(
        yaml_path=yaml_path, yaml_dir=yaml_dir or RUNTIME_CONFIGS_DIR,
        cli_args=cli_args, cli_exclude=cli_exclude, cli_mapping=cli_mapping,
    )
    sources_list: List[Tuple[Union[ConfigSource, str], Mapping[str, Any]]] = []
    if sources["yaml"]:
        sources_list.append((ConfigSource.YAML, sources["yaml"]))
    if extra_sources:
        sources_list.extend(extra_sources)
    if sources["cli"]:
        sources_list.append((ConfigSource.CLI, sources["cli"]))

    merger = ConfigMerger(track_sources=track_sources)
    if dry_run:
        merger.preview(YOLOTrainConfig, sources=sources_list)
        return None, merger
    config = merger.merge(YOLOTrainConfig, sources=sources_list)
    return config, merger


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
    """一键完成 val 配置流水线."""
    from od_platform.common.paths import RUNTIME_CONFIGS_DIR

    sources = load_all_sources(
        yaml_path=yaml_path, yaml_dir=yaml_dir or RUNTIME_CONFIGS_DIR,
        cli_args=cli_args, cli_exclude=cli_exclude, cli_mapping=cli_mapping,
    )
    sources_list: List[Tuple[Union[ConfigSource, str], Mapping[str, Any]]] = []
    if sources["yaml"]:
        sources_list.append((ConfigSource.YAML, sources["yaml"]))
    if extra_sources:
        sources_list.extend(extra_sources)
    if sources["cli"]:
        sources_list.append((ConfigSource.CLI, sources["cli"]))

    merger = ConfigMerger(track_sources=track_sources)
    if dry_run:
        merger.preview(YOLOValConfig, sources=sources_list)
        return None, merger
    config = merger.merge(YOLOValConfig, sources=sources_list)
    return config, merger


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
    """一键完成 infer 配置流水线."""
    from od_platform.common.paths import RUNTIME_CONFIGS_DIR

    sources = load_all_sources(
        yaml_path=yaml_path, yaml_dir=yaml_dir or RUNTIME_CONFIGS_DIR,
        cli_args=cli_args, cli_exclude=cli_exclude, cli_mapping=cli_mapping,
    )
    sources_list: List[Tuple[Union[ConfigSource, str], Mapping[str, Any]]] = []
    if sources["yaml"]:
        sources_list.append((ConfigSource.YAML, sources["yaml"]))
    if extra_sources:
        sources_list.extend(extra_sources)
    if sources["cli"]:
        sources_list.append((ConfigSource.CLI, sources["cli"]))

    merger = ConfigMerger(track_sources=track_sources)
    if dry_run:
        merger.preview(YOLOInferConfig, sources=sources_list)
        return None, merger
    config = merger.merge(YOLOInferConfig, sources=sources_list)
    return config, merger


__all__ = [
    "BaseConfig", "YOLOTrainConfig", "YOLOValConfig", "YOLOInferConfig",
    "YAMLLoader", "CLILoader", "load_all_sources",
    "ConfigMerger", "ConfigSource", "ConfigMetadata",
    "ConfigGenerator", "CONFIG_REGISTRY",
    "build_train_config", "build_val_config", "build_infer_config",
]
