"""运行上下文:每次 odp-* 命令运行,runs/<subsystem>/<run_id>/ 下放审计材料。
规则:谁产生不可复现的运行,谁开 RunContext;产物路径由 context 发,不散落拼接。
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from od_platform.common import paths

_SAFE = re.compile(r"^[a-z0-9_]+$")


class RunContext:
    """with RunContext('data_pipeline') as run: run.artifact_path('manifest.json')"""

    def __init__(self, subsystem: str):
        if not _SAFE.match(subsystem):
            raise ValueError(f"非法子系统名: {subsystem!r}(只允许小写字母/数字/下划线)")
        self.subsystem = subsystem
        now = datetime.now()                          # 单次获取,run_id 与 created_at 同源
        self.run_id: str = now.strftime("%Y%m%d-%H%M%S")
        self._created_at: str = now.isoformat(timespec="seconds")
        self.run_dir: Path = paths.RUNS_DIR / subsystem / self.run_id

    @property
    def created_at(self) -> str:
        return self._created_at

    def artifact_path(self, *parts: str) -> Path:
        p = self.run_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def __enter__(self) -> "RunContext":
        base, suffix = self.run_dir, 1
        while self.run_dir.exists():                  # 同秒冲突:run_id 加 -2/-3 后缀
            suffix += 1
            self.run_id = f"{base.name}-{suffix}"
            self.run_dir = base.with_name(self.run_id)
        self.run_dir.mkdir(parents=True, exist_ok=False)
        return self

    def __exit__(self, *exc) -> bool:
        return False                                   # 不吞异常
