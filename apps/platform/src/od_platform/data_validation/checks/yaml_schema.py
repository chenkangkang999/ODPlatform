"""check: yaml_schema —— 配置文件本身合法吗(最前置的一道)。
yaml 无法加载是前置错,早返回;结构问题收集齐了一次报全。"""
from __future__ import annotations

from typing import List

from od_platform.data_validation.registry import (
    CheckContext, CheckResult, CheckSeverity, check,
)

_NAME = "yaml_schema"


@check(_NAME)
def validate_yaml_schema(ctx: CheckContext) -> CheckResult:
    if ctx.snapshot.yaml_load_error:
        return CheckResult(_NAME, CheckSeverity.ERROR, "yaml 无法加载",
                           {"error": ctx.snapshot.yaml_load_error})
    data = ctx.snapshot.yaml_data
    problems: List[str] = []
    nc, names = data.get("nc"), data.get("names")
    if not isinstance(nc, int) or nc <= 0:
        problems.append(f"nc 缺失或不是正整数: {nc!r}")
    if not isinstance(names, (list, dict)):
        problems.append(f"names 缺失或类型不对(应为 list/dict): {type(names).__name__}")
    if isinstance(nc, int) and isinstance(names, (list, dict)) and len(names) != nc:
        problems.append(f"nc={nc} 与 names 数量 {len(names)} 不一致")
    if not data.get("path"):
        problems.append("缺少 path 字段(数据集根目录)")
    for s in ("train", "val", "test"):
        if s in data and not isinstance(data[s], str):
            problems.append(f"{s} 字段应为字符串路径: {data[s]!r}")
    if problems:
        return CheckResult(_NAME, CheckSeverity.ERROR,
                           f"yaml 结构有 {len(problems)} 处问题", {"problems": problems})
    return CheckResult(_NAME, CheckSeverity.PASS, "yaml 结构合法", {"nc": nc})
