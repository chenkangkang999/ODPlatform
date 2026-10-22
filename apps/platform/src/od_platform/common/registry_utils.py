"""注册表公共件:子模块自动发现与导入。

convert/split/... 的注册表都遵循同一模式:
模块级 dict + 装饰器自注册 + 首次使用时导入所有实现模块(触发注册)。
导入副作用即注册,因此"发现哪些模块要导入"是公共逻辑,放这里。
"""
import importlib
import pkgutil
from types import ModuleType


def import_submodules(package: ModuleType) -> None:
    """导入 package 下所有非下划线开头的子模块(触发其模块级副作用)。

    下划线开头(_geometry、_common、_shared 等)视为纯工具模块,不被自动发现,
    由实现模块按需自行 import。
    """
    for info in pkgutil.iter_modules(package.__path__):
        if not info.name.startswith("_"):
            importlib.import_module(f"{package.__name__}.{info.name}")
