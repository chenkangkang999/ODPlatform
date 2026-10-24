#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : identity.py
# @Project   : ODPlatform
# @Function  : 数据身份对账 —— 确认"我要吃的这批, 就是上游划过、验过的那一份"
"""数据身份对账。

回答: 【我(D6 训练 / D7 评估)正要用的这批数据, 是不是 D3 划过、D4 验过的那一份?】

三方独立取证, 口径一致 = 信任链闭合:
    ① 我自己算的         从盘上(经 manifest)重算 data_fingerprint   ← 一手证据
    ② D3 的 transform 章  它划分时盖的 data_fingerprint
    ③ D4 的 validate 章   它验证时盖的 data_fingerprint + 结论
三者一致, 且 D4 章为 PASS → 闭合。

★ 本模块不【发明】身份, 只【确认】身份(阶段 4.5)。它自己算一枚指纹, 不是为了造一个
  新真相, 而是拿去和上游的章核对 —— "我摸到的, 是不是你们说的那一个"。

★ 住 data_validation/ 而非 training/: 它依赖的全是 D3/D4 的知识(manifest/章/指纹),
  对 training 一无所知; 而 D7 评估会调同一个函数。知识归属决定归属。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from od_platform.common import stamp
from od_platform.common.lineage import compute_data_fingerprint
from od_platform.data_validation.checks._lineage import load_manifest_from_yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataIdentity:
    """一次数据身份对账的结果(纯数据, 可直接进审计)。frozen: 既成事实不该被改。"""
    ok:                    bool
    reason:                str = ""
    data_fingerprint:      Optional[str] = None      # ① D6 自己从盘算的(一手)
    contract_fingerprint:  Optional[str] = None
    transform_stamp:       Optional[Dict[str, Any]] = None   # ② D3 的章(读回来看, 不抄)
    validate_stamp:        Optional[Dict[str, Any]] = None   # ③ D4 的章
    notes:                 List[str] = field(default_factory=list)

    @property
    def reconciled(self) -> bool:
        """三方指纹是否一致(都在场且相等)。"""
        mine = self.data_fingerprint
        d3 = (self.transform_stamp or {}).get("data_fingerprint")
        d4 = (self.validate_stamp or {}).get("data_fingerprint")
        return bool(mine) and mine == d3 and mine == d4

    @property
    def validated_pass(self) -> bool:
        """D4 的章在, 且结论 PASS。"""
        return bool(self.validate_stamp) and \
            self.validate_stamp.get("overall_severity") == "PASS"

    def to_dict(self) -> Dict[str, Any]:
        """给 odp_audit.json —— 回答判据第 4 问: 数据是哪一份、出身与验证清不清楚。"""
        return {
            "data_fingerprint":     self.data_fingerprint,     # 我确认的身份(给 D7 用)
            "contract_fingerprint": self.contract_fingerprint,
            "reconciled":           self.reconciled,           # 三方指纹一致?
            "validated_pass":       self.validated_pass,       # D4 章在且 PASS?
            # 只记"我确认了这两枚章存在及其关键字段", 不整份转抄它们的内容
            "transform_stamp_seen": self.transform_stamp is not None,
            "validate_stamp_seen":  self.validate_stamp is not None,
            "validate_tool_version": (self.validate_stamp or {}).get("tool_version"),
            "notes":                self.notes,
            "ok":                   self.ok,
            "reason":               self.reason,
        }


def _recompute_data_fingerprint(yaml_path: Path) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """从 yaml → manifest → 自己算 data_fingerprint(一手证据)。

    Returns: (data_fingerprint, contract_fingerprint, error)
    """
    yaml_data: Dict[str, Any] = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    manifest, err, _odp = load_manifest_from_yaml(yaml_data)
    if manifest is None:
        return None, None, err
    # ★ 自己算, 不读任何章 —— 这是"确认"的前提: 手里得先有一手证据
    data_fp = compute_data_fingerprint(manifest.samples)
    return data_fp, manifest.contract_fingerprint, None


def reconcile_data_identity(yaml_path: Path, *, require_validated: bool = True) -> DataIdentity:
    """确认"我要用的这批数据"的身份, 并与 D3/D4 的章三方对账。

    Args:
        yaml_path:         数据集 yaml 的绝对路径(调用方用 refs 解析好)
        require_validated: True(默认)= D4 章缺失/非 PASS 则 ok=False(拒绝开训);
                           False = 只对账、不因未验证而拒绝(给"明知故犯"留口子)

    Returns:
        DataIdentity。ok=False 时 reason 是人话, 调用方据此拒绝开训。

    ★ D6 在这里做的是【确认】: 先自己算出 data_fingerprint(一手), 再去流转卡目录
      读 D3/D4 的章, 比对三方是否一致。它记下的是"我确认了这个共同身份", 不是"我造了
      一个身份"(阶段 4.5)。
    """
    notes: List[str] = []

    # ① 一手证据: 自己从盘上算
    data_fp, contract_fp, err = _recompute_data_fingerprint(yaml_path)
    if data_fp is None:
        return DataIdentity(ok=False, reason=f"拿不到 manifest, 无法确定数据身份: {err}")

    # ②③ 读回 D3、D4 的章(读回来是为了【确认它在、看它说了什么】, 不抄进自己的结论)
    d3 = stamp.read_stamp(data_fp, stamp.STAMP_TRANSFORM)
    d4 = stamp.read_stamp(data_fp, stamp.STAMP_VALIDATE)

    identity = DataIdentity(
        ok=True, data_fingerprint=data_fp, contract_fingerprint=contract_fp,
        transform_stamp=d3, validate_stamp=d4, notes=notes,
    )

    # —— 判定 ——
    # D3 章不在: 缺证据, 不是罪证。警告不阻断(可能是老数据集, 盖章能力上线前划的)。
    if d3 is None:
        notes.append("未找到 D3 的 transform 章 —— 这批数据可能在盖章能力上线前划分, "
                     "或流转卡目录被清理。出身链不完整, 但不阻断。")
        logger.warning(notes[-1])

    # D4 章: 这是"验过没"的答案。缺失或非 PASS, 默认拒绝(require_validated)。
    if d4 is None:
        msg = "未找到 D4 的 validate 章 —— 这批数据没有经过验证。"
        if require_validated:
            return DataIdentity(ok=False, reason=msg + " 请先 odp-validate, 或 --skip-validate 明知故犯。",
                                data_fingerprint=data_fp, contract_fingerprint=contract_fp,
                                transform_stamp=d3, validate_stamp=None, notes=notes)
        notes.append(msg + " (--skip-validate: 继续, 但审计将标记未验证)")
        logger.warning(notes[-1])
    elif d4.get("overall_severity") != "PASS":
        msg = f"D4 的 validate 章结论为 {d4.get('overall_severity')}, 非 PASS。"
        if require_validated:
            return DataIdentity(ok=False, reason=msg + " 数据未通过验证, 拒绝开训。",
                                data_fingerprint=data_fp, contract_fingerprint=contract_fp,
                                transform_stamp=d3, validate_stamp=d4, notes=notes)
        notes.append(msg + " (--skip-validate: 继续)")
        logger.warning(notes[-1])

    # 三方对账: 指纹是否闭合(不阻断, 但记进审计与日志)
    if identity.reconciled:
        logger.info("数据身份闭合 ✅ 三方指纹一致 data_fingerprint=%s", data_fp[:16])
    else:
        notes.append(f"三方指纹未完全闭合(reconciled=false): "
                     f"我算={data_fp[:16]} / D3章={(d3 or {}).get('data_fingerprint', '无')} / "
                     f"D4章={(d4 or {}).get('data_fingerprint', '无')}。")
        logger.warning(notes[-1])

    return identity
