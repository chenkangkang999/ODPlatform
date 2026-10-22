"""质检报告:纯数据 + 三种出口(logger / json / csv)。报告不含判断逻辑。"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from od_platform.data_validation.registry import CheckResult, CheckSeverity

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    run_id: str
    snapshot: Any                                    # DatasetSnapshot(避免循环 import)
    results: List[CheckResult] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def overall_severity(self) -> str:
        """全场最严:按 rank 取最大;空结果视为 PASS。"""
        if not self.results:
            return CheckSeverity.PASS
        return max((r.severity for r in self.results), key=CheckSeverity.rank)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "yaml_path": str(self.snapshot.yaml_path),
            "overall_severity": self.overall_severity,
            "duration_seconds": round(self.duration_seconds, 3),
            "results": [
                {"name": r.name, "severity": r.severity,
                 "summary": r.summary, "details": r.details}
                for r in self.results
            ],
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8")

    def write_results_csv(self, path: Path) -> None:
        """utf-8-sig 带 BOM:Excel 直接打开中文不乱码。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run_id", "check", "severity", "summary"])
            for r in self.results:
                w.writerow([self.run_id, r.name, r.severity, r.summary])

    def render_to_logger(self, log=None) -> None:
        log = log or logger
        log.info("=" * 60)
        log.info("数据质检报告 run_id=%s  总评: %s(%.2fs)",
                 self.run_id, self.overall_severity, self.duration_seconds)
        log.info("=" * 60)
        for r in self.results:
            log.info("[%-7s] %s: %s", r.severity, r.name, r.summary)
        fp = next((r for r in self.results if r.name == "fingerprint_match"), None)
        if fp and fp.details.get("fingerprint"):
            log.info("-" * 60)
            log.info("契约指纹: %s", fp.details["fingerprint"])
            log.info("样本数: %s  策略: %s  种子: %s",
                     fp.details.get("n_samples"), fp.details.get("strategy"),
                     fp.details.get("seed"))
