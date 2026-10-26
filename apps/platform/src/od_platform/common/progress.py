#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : progress.py
# @Project   : ODPlatform
# @Function  : 长循环的进度反馈（让"大数据集执行时用户无感知"成为过去）

"""通用进度反馈组件。

★ 解决什么问题

    D3 的扫描/物化、D4 的逐样本核验在大数据集上会**静默跑很久**, 用户不知道是
    卡死了还是在干活。这几处都需要"跑到哪儿了"的反馈。

★ 为什么不用 tqdm

    本仓库的可见性载体是【日志】—— ``logging/`` 与 ``runs/`` 下的文件, 外加控制台。
    tqdm 往 stderr 刷 ``\\r`` 覆盖行: 落不进日志文件, 在 CI 的日志里也只是一串
    转义字符, 事后无法回答"当时跑到哪儿卡住了"。所以这里做的是**按比例打点**:
    每过一个档位记一行 INFO, 带已完成/总数/百分比/速率/预计剩余。

★ 打点规则（两条都满足才记）

    1. 跨过预设的比例档位（默认每 10%）;
    2. 距上次打点至少 MIN_INTERVAL_SEC 秒 —— 小数据集瞬间跑完时不该刷一屏。

    总量很小（小于档位数）时自动关闭中间打点, 只留收尾那一行。
"""
from __future__ import annotations

import logging
import math
import time
from typing import Optional

# 默认每 10% 打一个点
DEFAULT_STEPS: int = 10
# 两次打点之间的最小间隔: 小数据集不该刷屏
MIN_INTERVAL_SEC: float = 2.0
# 总项数小于该值时不做中间打点（一屏就完事了）
MIN_TOTAL_FOR_INTERMEDIATE: int = 50
# 总量未知时（如"数一数有多少张图"）每处理这么多项打一个点
UNKNOWN_TOTAL_STEP: int = 1000
# 预计剩余时间算不出来时显示的字样
_UNKNOWN_ETA = "--"


class ProgressReporter:
    """按比例 + 时间间隔汇报一次长循环的进度。

    用法::

        progress = ProgressReporter("materialize", total=len(samples), logger=logger)
        for index, sample in enumerate(samples, 1):
            ...
            progress.update(index)
        progress.finish()          # 收尾一行: 总数/总耗时/平均速率

    Args:
        name: 任务名（进日志, 便于在大日志里 grep）
        total: 总项数; <= 0 表示未知总量（此时只在 finish 时报总数）
        logger: 目标 logger; None 时用本模块的 logger
        steps: 分成几个档位打点（默认 10 = 每 10%）
        unit: 量词（"个样本"/"张图"）, 只影响措辞
        min_interval: 两次打点之间的最小秒数
        clock: 取当前时间的函数（测试注入用）
    """

    def __init__(self, name: str, total: int, *, logger: Optional[logging.Logger] = None,
                 steps: int = DEFAULT_STEPS, unit: str = "项",
                 min_interval: float = MIN_INTERVAL_SEC, clock=time.monotonic):
        if steps < 1:
            raise ValueError(f"steps 必须 >= 1, 收到 {steps}")
        self.name = name
        self.total = int(total)
        self.unit = unit
        self.steps = steps
        self.min_interval = min_interval
        self._clock = clock
        self._logger = logger or logging.getLogger(__name__)
        self._started = clock()
        self._last_report = self._started
        self._next_threshold = self._first_threshold()
        # 总量已知且够大 → 按档位打点; 总量未知(total<=0) → 按固定项数打点
        self._enabled = self.total <= 0 or self.total >= MIN_TOTAL_FOR_INTERMEDIATE

    # ---------------------------------------------------------------- 内部
    def _step_size(self) -> int:
        """两次打点之间跨多少项。"""
        if self.total <= 0:
            return UNKNOWN_TOTAL_STEP          # 总量未知: 按固定项数
        return max(1, math.ceil(self.total / self.steps))

    def _first_threshold(self) -> int:
        return self._step_size()

    def _eta(self, done: int) -> str:
        """按当前速率估计剩余时间。"""
        if done <= 0 or self.total <= 0:
            return _UNKNOWN_ETA
        elapsed = self._clock() - self._started
        if elapsed <= 0:
            return _UNKNOWN_ETA
        remaining = max(0, self.total - done)
        return f"{elapsed / done * remaining:.0f}s"

    # ---------------------------------------------------------------- 对外
    def update(self, done: int) -> Optional[str]:
        """到达打点条件时记一行日志并返回该行文本, 否则返回 None。"""
        if not self._enabled:
            return None
        if done < self._next_threshold:
            return None
        now = self._clock()
        if now - self._last_report < self.min_interval:
            return None
        self._last_report = now
        # 跳过档位时一并推进阈值, 避免后续每项都触发
        step = self._step_size()
        while done >= self._next_threshold:
            self._next_threshold += step
        rate = done / max(now - self._started, 1e-9)
        if self.total > 0:
            head = f"{done}/{self.total} {self.unit} ({done / self.total * 100:.0f}%)"
        else:
            head = f"已处理 {done} {self.unit}"       # 总量未知: 不编百分比
        message = f"{self.name}: {head} {rate:.1f}/s 剩余约 {self._eta(done)}"
        self._logger.info(message)
        return message

    def finish(self, *, done: Optional[int] = None) -> str:
        """收尾一行: 总数/总耗时/平均速率。总是记录。"""
        finished = self.total if done is None else int(done)
        elapsed = max(self._clock() - self._started, 0.0)
        rate = finished / elapsed if elapsed > 0 else 0.0
        message = (f"{self.name}: 完成 {finished} {self.unit}, "
                   f"用时 {elapsed:.2f}s ({rate:.1f}/s)")
        self._logger.info(message)
        return message