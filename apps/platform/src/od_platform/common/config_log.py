"""配置日志桩: 复用 runtime_config.report 的渲染."""
from __future__ import annotations

import logging

from od_platform.runtime_config.report import render_effective, render_overrides


def log_effective_config(config, merger, *, logger=None) -> None:
    log = logger or logging.getLogger(__name__)
    for line in render_effective(config, merger):
        log.info(line)


def log_override_chains(config, merger, *, logger=None) -> None:
    log = logger or logging.getLogger(__name__)
    for line in render_overrides(config, merger):
        log.info(line)
