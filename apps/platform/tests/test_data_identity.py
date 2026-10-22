"""数据身份对账测试: 确认而非发明; 三方独立取证。"""
from __future__ import annotations

from od_platform.data_validation.identity import DataIdentity


def _di(mine, d3, d4_fp, d4_sev="PASS"):
    return DataIdentity(
        ok=True, data_fingerprint=mine,
        transform_stamp={"data_fingerprint": d3} if d3 else None,
        validate_stamp=({"data_fingerprint": d4_fp, "overall_severity": d4_sev}
                        if d4_fp else None))


def test_reconciled_when_three_agree():
    assert _di("X", "X", "X").reconciled is True


def test_not_reconciled_when_disk_differs():
    assert _di("Y", "X", "X").reconciled is False


def test_not_reconciled_when_a_stamp_missing():
    assert _di("X", None, "X").reconciled is False
    assert _di("X", "X", None).reconciled is False


def test_validated_pass_needs_pass():
    assert _di("X", "X", "X", "PASS").validated_pass is True
    assert _di("X", "X", "X", "ERROR").validated_pass is False


def test_audit_records_confirmation_not_copy():
    d = _di("X", "X", "X").to_dict()
    assert d["transform_stamp_seen"] is True and d["validate_stamp_seen"] is True
    assert d["data_fingerprint"] == "X" and d["reconciled"] is True
    assert "transform_stamp" not in d and "validate_stamp" not in d
