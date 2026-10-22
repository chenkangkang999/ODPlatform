"""质检注册表 + 严重度 + 结果类型 + 执行上下文。

严重度四级:ERROR(必须修)> WARNING(可训但建议修)> INFO(提示)> PASS(通过);
rank() 供聚合取"全场最严"。重复注册直接 ValueError —— 静默覆盖等于埋雷。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

from od_platform.data_validation.snapshot import DatasetSnapshot


class CheckSeverity:
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    PASS = "PASS"

    _ORDER = {ERROR: 3, WARNING: 2, INFO: 1, PASS: 0}

    @classmethod
    def rank(cls, severity: str) -> int:
        return cls._ORDER[severity]


@dataclass
class CheckResult:
    name: str
    severity: str
    summary: str                                     # 一句话结论(给人看)
    details: Dict[str, Any] = field(default_factory=dict)  # 结构化证据(给程序看)

    @property
    def passed(self) -> bool:
        return self.severity in (CheckSeverity.PASS, CheckSeverity.INFO)


@dataclass
class CheckContext:
    """一次质检运行共享的上下文。"""
    yaml_path: Path
    snapshot: DatasetSnapshot


CheckFunc = Callable[[CheckContext], CheckResult]


@dataclass(frozen=True)
class CheckEntry:
    name: str
    func: CheckFunc


_REGISTRY: Dict[str, CheckEntry] = {}
_LAZY_INITIALIZED = False


def check(name: str):
    """装饰器:把函数注册为质检项。"""
    def deco(func: CheckFunc) -> CheckFunc:
        if name in _REGISTRY:
            raise ValueError(f"质检项重复注册: {name!r}")
        _REGISTRY[name] = CheckEntry(name=name, func=func)
        return func
    return deco


def _lazy_init() -> None:
    global _LAZY_INITIALIZED
    if _LAZY_INITIALIZED:
        return
    from od_platform.common.registry_utils import import_submodules
    from od_platform.data_validation import checks
    import_submodules(checks)
    _LAZY_INITIALIZED = True


def get_all_checks() -> List[CheckEntry]:
    _lazy_init()
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def get_check(name: str) -> CheckEntry:
    _lazy_init()
    return _REGISTRY[name]          # 名字写错就该炸(KeyError)


def list_check_names() -> List[str]:
    _lazy_init()
    return sorted(_REGISTRY)
