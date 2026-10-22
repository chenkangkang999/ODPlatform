"""数据身份对账。
确认"我(D6/D7)正要用的这批数据, 是不是 D3 划过、D4 验过的那一份"。
三方独立取证, 口径一致 = 信任链闭合。
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
    """一次数据身份对账的结果(纯数据, 可直接进审计)。"""
    ok:                    bool
    reason:                str = ""
    data_fingerprint:      Optional[str] = None
    contract_fingerprint: Optional[str] = None
    transform_stamp:       Optional[Dict[str, Any]] = None
    validate_stamp:        Optional[Dict[str, Any]] = None
    notes:                 List[str] = field(default_factory=list)

    @property
    def reconciled(self) -> bool:
        mine = self.data_fingerprint
        d3 = (self.transform_stamp or {}).get("data_fingerprint")
        d4 = (self.validate_stamp or {}).get("data_fingerprint")
        return bool(mine) and mine == d3 and mine == d4

    @property
    def validated_pass(self) -> bool:
        return bool(self.validate_stamp) and \
            self.validate_stamp.get("overall_severity") == "PASS"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_fingerprint":     self.data_fingerprint,
            "contract_fingerprint": self.contract_fingerprint,
            "reconciled":           self.reconciled,
            "validated_pass":       self.validated_pass,
            "transform_stamp_seen": self.transform_stamp is not None,
            "validate_stamp_seen":  self.validate_stamp is not None,
            "validate_tool_version": (self.validate_stamp or {}).get("tool_version"),
            "notes":                self.notes,
            "ok":                   self.ok,
            "reason":               self.reason,
        }


def _recompute_data_fingerprint(yaml_path: Path):
    yaml_data: Dict[str, Any] = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    manifest, err, _odp = load_manifest_from_yaml(yaml_data)
    if manifest is None:
        return None, None, err
    data_fp = compute_data_fingerprint(manifest.samples)
    return data_fp, manifest.contract_fingerprint, None


def reconcile_data_identity(yaml_path: Path, *,
                            require_validated: bool = True) -> DataIdentity:
    """确认"我要用的这批数据"的身份, 并与 D3/D4 的章三方对账。"""
    notes: List[str] = []

    data_fp, contract_fp, err = _recompute_data_fingerprint(yaml_path)
    if data_fp is None:
        return DataIdentity(ok=False, reason=f"拿不到 manifest, 无法确定数据身份: {err}")

    d3 = stamp.read_stamp(data_fp, stamp.STAMP_TRANSFORM)
    d4 = stamp.read_stamp(data_fp, stamp.STAMP_VALIDATE)

    identity = DataIdentity(
        ok=True, data_fingerprint=data_fp, contract_fingerprint=contract_fp,
        transform_stamp=d3, validate_stamp=d4, notes=notes,
    )

    if d3 is None:
        notes.append("未找到 D3 的 transform 章 —— 这批数据可能在盖章能力上线前划分, "
                     "或流转卡目录被清理。出身链不完整, 但不阻断。")
        logger.warning(notes[-1])

    if d4 is None:
        msg = "未找到 D4 的 validate 章 —— 这批数据没有经过验证。"
        if require_validated:
            return DataIdentity(ok=False,
                reason=msg + " 请先 odp-validate, 或 --skip-validate 明知故犯。",
                data_fingerprint=data_fp, contract_fingerprint=contract_fp,
                transform_stamp=d3, validate_stamp=None, notes=notes)
        notes.append(msg + " (--skip-validate: 继续, 但审计将标记未验证)")
        logger.warning(notes[-1])
    elif d4.get("overall_severity") != "PASS":
        msg = f"D4 的 validate 章结论为 {d4.get('overall_severity')}, 非 PASS。"
        if require_validated:
            return DataIdentity(ok=False,
                reason=msg + " 数据未通过验证, 拒绝开训。",
                data_fingerprint=data_fp, contract_fingerprint=contract_fp,
                transform_stamp=d3, validate_stamp=d4, notes=notes)
        notes.append(msg + " (--skip-validate: 继续)")
        logger.warning(notes[-1])

    if identity.reconciled:
        logger.info("数据身份闭合 ✅ 三方指纹一致 data_fingerprint=%s", data_fp[:16])
    else:
        notes.append(f"三方指纹未完全闭合(reconciled=false): "
                     f"我算={data_fp[:16]} / D3章={(d3 or {}).get('data_fingerprint', '无')} / "
                     f"D4章={(d4 or {}).get('data_fingerprint', '无')}。")
        logger.warning(notes[-1])

    return identity
