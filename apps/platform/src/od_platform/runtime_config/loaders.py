"""配置加载器: 从不同来源加载配置(不负责验证和合并)."""
from __future__ import annotations

import logging
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import yaml

logger = logging.getLogger(__name__)


def _drop_none(d: Mapping[str, Any]) -> Dict[str, Any]:
    """过滤 None 值, 保留 False / 0 / '' 等显式值."""
    return {k: v for k, v in d.items() if v is not None}


class YAMLLoader:
    """加载 YAML 配置文件 → dict."""

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        self.config_dir = Path(config_dir) if config_dir else None

    def load(self, filename: Union[str, Path]) -> Dict[str, Any]:
        filepath = self._resolve_path(filename)
        if not filepath.exists():
            raise FileNotFoundError(
                f"YAML 配置文件不存在: {filepath}\n\n"
                f"请先生成默认配置模板:\n"
                f"  odp-gen-config {filepath.stem}\n\n"
                f"生成后编辑该文件再重新运行."
            )
        try:
            content = filepath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 解码失败, 尝试系统默认编码: {filepath}")
            content = filepath.read_text()
        if not content.strip():
            logger.debug(f"YAML 文件为空: {filepath}")
            return {}
        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise ValueError(
                f"YAML 格式错误: {filepath}\n"
                f"原始错误: {e}\n"
                f"提示: 检查缩进、引号匹配、冒号后是否有空格."
            ) from e
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError(
                f"YAML 顶层必须是字典, 当前是 {type(data).__name__}: {filepath}\n"
                f"内容预览: {str(data)[:100]}"
            )
        return _drop_none(data)

    def _resolve_path(self, filename: Union[str, Path]) -> Path:
        path = Path(filename)
        if path.is_absolute():
            return path
        if self.config_dir:
            return (self.config_dir / path).resolve()
        return path.resolve()


class CLILoader:
    """加载命令行参数 → dict."""

    DEFAULT_EXCLUDE: set[str] = {
        "help", "config", "cfg", "yaml_path", "debug", "version",
    }

    def __init__(
        self,
        exclude: Optional[List[str]] = None,
        mapping: Optional[Dict[str, str]] = None,
    ):
        self.exclude = self.DEFAULT_EXCLUDE | set(exclude or [])
        self.mapping = mapping or {}

    def load(
        self,
        args: Optional[Union[Namespace, Dict[str, Any]]] = None,
        filter_none: bool = True,
    ) -> Dict[str, Any]:
        if args is None:
            return {}
        if isinstance(args, Namespace):
            raw = vars(args)
        elif isinstance(args, dict):
            raw = args
        else:
            raise TypeError(
                f"args 必须是 argparse.Namespace 或 dict, "
                f"当前是 {type(args).__name__}"
            )
        result: Dict[str, Any] = {}
        for key, value in raw.items():
            if key in self.exclude or key.startswith("_"):
                continue
            if filter_none and value is None:
                continue
            mapped_key = self.mapping.get(key, key)
            result[mapped_key] = value
        return result


def load_all_sources(
    yaml_path: Optional[Union[str, Path]] = None,
    yaml_dir: Optional[Union[str, Path]] = None,
    cli_args: Optional[Union[Namespace, Dict[str, Any]]] = None,
    cli_exclude: Optional[List[str]] = None,
    cli_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """一次性加载所有配置源 → {'yaml': ..., 'cli': ...}"""
    yaml_config: Dict[str, Any] = {}
    if yaml_path:
        loader = YAMLLoader(config_dir=yaml_dir)
        yaml_config = loader.load(yaml_path)
    cli_loader = CLILoader(exclude=cli_exclude, mapping=cli_mapping)
    cli_config = cli_loader.load(cli_args)
    return {"yaml": yaml_config, "cli": cli_config}
