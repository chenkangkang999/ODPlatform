"""check: fingerprint_match —— 契约指纹三方对账。
① manifest 自洽(重算 == 记录):内容没被改;② yaml 指纹 == manifest 指纹:没张冠李戴。"""
from __future__ import annotations

from od_platform.common.lineage import compute_contract_fingerprint
from od_platform.data_validation.checks._lineage import load_manifest_from_yaml
from od_platform.data_validation.registry import (
    CheckContext, CheckResult, CheckSeverity, check,
)

_NAME = "fingerprint_match"


@check(_NAME)
def validate_fingerprint_match(ctx: CheckContext) -> CheckResult:
    manifest, err, odp_meta = load_manifest_from_yaml(ctx.snapshot.yaml_data)
    if err:
        return CheckResult(_NAME, CheckSeverity.ERROR, f"拿不到 D3 划分契约: {err}", {})
    recomputed = compute_contract_fingerprint(
        manifest.dataset, manifest.strategy, manifest.seed, manifest.ratios,
        manifest.names, manifest.samples, manifest.tool_version)
    if recomputed != manifest.contract_fingerprint:
        return CheckResult(_NAME, CheckSeverity.ERROR,
                           "manifest 自身指纹对不上 —— 内容被改过",
                           {"expected": manifest.contract_fingerprint,
                            "recomputed": recomputed})
    yaml_fp = odp_meta.get("contract_fingerprint")
    if yaml_fp != manifest.contract_fingerprint:
        return CheckResult(_NAME, CheckSeverity.ERROR,
                           "yaml 指纹与 manifest 不一致 —— yaml 张冠李戴",
                           {"yaml_fingerprint": yaml_fp,
                            "manifest_fingerprint": manifest.contract_fingerprint})
    return CheckResult(_NAME, CheckSeverity.PASS, "契约指纹三方一致",
                       {"fingerprint": manifest.contract_fingerprint,
                        "n_samples": len(manifest.samples),
                        "strategy": manifest.strategy, "seed": manifest.seed})
