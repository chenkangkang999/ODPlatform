"""转换器注册表:格式名 → 转换函数 + 能力声明(supported_tasks)。

设计:模块级 dict + 装饰器自注册 + 懒加载自动发现。
新增格式 = 在 converters/ 下加一个文件,装饰器自动注册;注册表代码不动。
为什么重复注册是 ERROR:两个文件争同一个格式名,静默覆盖等于埋雷;宁可当场炸。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

ConverterFunc = Callable[[Path, Path, "ConvertOptions"], List[str]]


@dataclass
class ConvertOptions:
    """传给所有转换器的横切选项(参数包)。加新选项时这里加字段,签名不变。"""
    task: str = "detect"                        # detect / segment
    classes: Optional[List[str]] = field(default=None)  # None=不过滤;[]=全排除;[...]=白名单


@dataclass(frozen=True)
class ConverterEntry:
    func: ConverterFunc
    supported_tasks: Tuple[str, ...]            # 注册时声明的能力

    def supports(self, task: str) -> bool:
        return task in self.supported_tasks


_CONVERTERS: Dict[str, ConverterEntry] = {}
_LAZY_INITIALIZED = False


def register_converter(format_name: str, *, supported_tasks: Tuple[str, ...]):
    """装饰器:把转换函数注册为某格式的入口,并声明它支持的任务。"""
    def deco(func: ConverterFunc) -> ConverterFunc:
        if format_name in _CONVERTERS:
            raise ValueError(
                f"转换器重复注册: {format_name!r} 已被 "
                f"{_CONVERTERS[format_name].func.__module__} 占用")
        _CONVERTERS[format_name] = ConverterEntry(func=func, supported_tasks=supported_tasks)
        return func
    return deco


def _lazy_init() -> None:
    """首次查询时,导入 converters 包内所有实现模块(触发装饰器自注册)。"""
    global _LAZY_INITIALIZED
    if _LAZY_INITIALIZED:
        return
    from od_platform.common.registry_utils import import_submodules
    from od_platform.data_pipeline.convert import converters
    import_submodules(converters)
    _LAZY_INITIALIZED = True


def get_converter(format_name: str) -> ConverterEntry:
    _lazy_init()
    try:
        return _CONVERTERS[format_name]
    except KeyError:
        raise ValueError(
            f"不支持的标注格式: {format_name!r}。可用: {sorted(_CONVERTERS)}") from None


def list_converters() -> Dict[str, ConverterEntry]:
    _lazy_init()
    return dict(_CONVERTERS)
