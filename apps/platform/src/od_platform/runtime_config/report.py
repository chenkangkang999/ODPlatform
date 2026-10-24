#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : report.py
# @Project   : ODPlatform
# @Function  : 配置溯源报告 —— 把 ConfigMerger 的合并痕迹渲染成人能读的两张表
"""配置溯源报告。住 runtime_config/(渲染的是 D5 的数据, D7 也用)。

两张表, 两个场景, 同一份数据:
    render_effective() 反查: 每个字段现在什么值、谁给的 —— 完整, 一个不少
    render_overrides() 追因: 哪些字段在合并中【被改写过】 —— 只列真正变过的

跟 merger.get_source_report() 分工: 那个按【来源】分组(概览); 这两张按【字段】索引(反查/追因)。
渲染 vs 输出分离: render_*() 返回 list[str] 可直接断言; log_config_report() 才碰 logger。
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

from od_platform.common.string_utils import pad_to_width

_KEY_WIDTH = 20
_WIDTH = 60


def _chain_oldest_first(meta: Any) -> List[Any]:
    """ConfigMerger.chain() 是 newest-first, 翻成 oldest-first(人读时间线从左到右)。
    ★ 符号必须跟着翻: oldest→newest 配 '→'。翻了顺序不翻符号, 箭头就指反方向。
    """
    return list(reversed(meta.chain()))


def _was_overridden(meta: Any) -> bool:
    """链上是否真的发生过【值变化】(本模块题眼)。
    16(DEFAULT)→16(YAML) 不算(复述); 100→300 才算。把'链长>1'当覆盖会淹没真信息。
    """
    chain = _chain_oldest_first(meta)
    return any(chain[i].value != chain[i - 1].value for i in range(1, len(chain)))


def render_effective(config: Any, merger: Any, *, width: int = _WIDTH) -> List[str]:
    """反查: 每字段现在什么值、谁给的。完整快照, 不过滤。"""
    lines = ["=" * width, "配置参数信息".center(width), "-" * width]
    for name in type(config).model_fields:
        value = getattr(config, name)
        meta = merger.get_metadata(name)
        # 没 metadata = 没有任何源覆盖过它(纯默认值), 对当前 config 很常见, 标 DEFAULT。
        source = meta.source_label if meta is not None else "DEFAULT"
        lines.append(f"{pad_to_width(name, _KEY_WIDTH)}: {value}  (来源: {source})")
    return lines


def render_overrides(config: Any, merger: Any, *, width: int = _WIDTH) -> List[str]:
    """追因: 只列合并中【被改写过】的字段及完整链路。"""
    changed: List[Tuple[str, str]] = []
    for name in type(config).model_fields:
        meta = merger.get_metadata(name)
        if meta is not None and _was_overridden(meta):
            chain = _chain_oldest_first(meta)
            changed.append((name, " → ".join(f"{m.value}({m.source_label})" for m in chain)))

    total = len(type(config).model_fields)
    lines = ["-" * width, "配置覆盖情况".center(width), "-" * width]
    if not changed:
        lines.append(f"没有字段被改写: {total} 个字段各源取值一致。")
        return lines
    # ★ 这句统计不能省(归纳≠省略, 省略让人不安)
    lines.append(f"{len(changed)} 个字段在合并中被改写; "
                 f"其余 {total - len(changed)} 个字段各源取值一致, 未列出。")
    lines += [f"{pad_to_width(n, _KEY_WIDTH)}: {c}" for n, c in changed]
    return lines


def log_config_report(config: Any, merger: Any, *,
                      logger: Optional[logging.Logger] = None) -> None:
    """把两张表打进 logger。业务模块只发声, handler 由 cli/ 挂。"""
    log = logger or logging.getLogger(__name__)
    for line in render_effective(config, merger):
        log.info(line)
    for line in render_overrides(config, merger):
        log.info(line)
