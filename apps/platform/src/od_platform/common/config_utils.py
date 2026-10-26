#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : config_utils.py
# @Project   : ODPlatform
# @Function  : 配置 → dataclass 的统一构造工具（消灭散落的数字与键名）

"""把 dict 配置灌进 dataclass 的公共件。

为什么要有这个模块:
    geometric 与 tracking 两个子系统都需要"从 football.yaml 的某一节读配置,
    没给的字段走 dataclass 默认值"。这种需求出现第二次就该抽公共件（D3 的规矩）。
    更重要的目的: 【默认值只在 dataclass 里声明一次】, 业务逻辑里不出现字面量。
    于是"哪里可以调、默认是多少"永远只有一个答案。

约定:
    · 配置里出现的键, 必须与 dataclass 字段同名（或用 aliases 声明别名）;
    · 不认识的键会被【收集并报告】, 而不是静默忽略 —— 拼错键名是常见事故;
    · 值的类型按字段默认值的类型转换（float/int/bool/str/tuple）;
    · 配置缺失（None / 非 dict）时原样返回默认实例, 不让几何/跟踪因缺配置而挂掉。
"""
from __future__ import annotations

import logging
from dataclasses import fields, is_dataclass, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 会被当成长度/数量一类的"数组"配置项按 tuple 处理时的元素类型
_ARRAY_TYPES = (list, tuple)


def read_section(config: Any, name: str) -> Dict[str, Any]:
    """取配置里的某一节; 缺了或类型不对就返回空字典。"""
    if not isinstance(config, Mapping):
        return {}
    section = config.get(name)
    return dict(section) if isinstance(section, Mapping) else {}


def build_settings(
    cls: Type[T],
    config: Any,
    *,
    aliases: Optional[Mapping[str, str]] = None,
    unknown_sink: Optional[List[str]] = None,
) -> T:
    """用 config 覆盖 cls 的默认值构造实例。

    Args:
        cls: 目标 dataclass
        config: 配置字典（通常来自 yaml 的某一节）
        aliases: 旧键名 → 字段名 的别名表（兼容历史配置）
        unknown_sink: 收集"配置里有、dataclass 里没有"的键名

    Returns:
        cls 的实例; config 不是映射时返回 cls 的默认实例
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} 不是 dataclass, 无法从配置构造")
    if not isinstance(config, Mapping):
        return cls()

    aliases = dict(aliases or {})
    known = {field.name: field for field in fields(cls)}
    overrides: Dict[str, Any] = {}
    for raw_key, value in config.items():
        key = aliases.get(raw_key, raw_key)
        field = known.get(key)
        if field is None:
            if unknown_sink is not None:
                unknown_sink.append(raw_key)
            continue
        overrides[key] = _coerce(field.default, value, key)
    return cls(**overrides)


def override_settings(instance: T, overrides: Optional[Mapping[str, Any]]) -> T:
    """用 overrides 覆盖 dataclass 实例的同名字段, 返回新实例。

    用途: 运行期覆盖（Web 任务 request.json / 环境变量）。
    · 只认与字段同名的键, 其余【静默跳过】—— 因为调用方常直接把整个请求 dict 递进来,
      里面本来就有 input_path / output_dir 这类不相干的键;
    · 值为 None 的键跳过（"没传"不等于"要设成空"）;
    · 类型按字段当前值转换, 转换结果会重新走一遍 __post_init__ 校验。
    """
    if not isinstance(overrides, Mapping) or not overrides:
        return instance
    known = {field.name: field for field in fields(instance)}
    changes: Dict[str, Any] = {}
    for key, value in overrides.items():
        if key not in known or value is None:
            continue
        changes[key] = _coerce(getattr(instance, key), value, key)
    return replace(instance, **changes) if changes else instance


def report_unknown_keys(where: str, keys: Sequence[str]) -> None:
    """把不认识的配置键报出来（拼错键名不该静默生效一半）。"""
    if keys:
        logger.warning("%s 里有无法识别的配置键（已忽略）: %s", where, sorted(set(keys)))


def _coerce(default: Any, value: Any, key: str) -> Any:
    """按默认值的类型转换; 转不过去就报错, 不静默塞进去。"""
    try:
        if isinstance(default, bool):
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if isinstance(default, float):
            return float(value)
        if isinstance(default, int):
            return int(value)
        if isinstance(default, str):
            return str(value)
        if isinstance(default, _ARRAY_TYPES):
            return tuple(value) if isinstance(default, tuple) else list(value)
        if default is None:
            return value
    except (TypeError, ValueError) as exc:
        raise ValueError(f"配置项 {key!r} 的值 {value!r} 无法转换为 "
                         f"{type(default).__name__}: {exc}") from None
    return value


def as_bool(value: Any, *, default: bool = False) -> bool:
    """把"可能来自 yaml / 环境变量"的值转成布尔。"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def first_present(config: Any, names: Iterable[str]) -> Any:
    """按顺序返回第一个存在的键值（兼容多套历史键名）。"""
    if not isinstance(config, Mapping):
        return None
    for name in names:
        if name in config:
            return config[name]
    return None