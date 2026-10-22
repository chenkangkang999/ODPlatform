"""划分契约:DatasetPipeline 的 lineage 记录与指纹模块。
谁负责:写 SplitManifest + 算 contract_fingerprint。
谁调用:orchestrator(产出)、materializer/yaml_writer(消费)、D4 质检(验账)。
输入:划分结果 + 每个样本的标签字节。输出:manifest.json + 契约指纹。
不变量:manifest 写完即不可变;指纹对"配方+全部样本sha256"敏感、对时间戳免疫。

为什么放在 common:D3 生产这份契约、D4 核验这份契约,两个子系统互不 import,
契约必须住在公共层 —— 它是跨子系统的"语言",不是任何一边的内部实现。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

TOOL_VERSION = "odp-transform/0.4"       # 契约版本:指纹配方变动时升号


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical_sha256(obj) -> str:
    """规范化哈希的唯一实现。任何"对一份结构算哈希"的需求都走这里。"""
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(blob)


def compute_data_fingerprint(samples: List[SampleLineage]) -> str:
    """数据指纹: 只看"有哪些样本 + 内容", 不含 split 归属。
    改划分比例不动它; 加/删样本、改标签字节才动它。"""
    return canonical_sha256(sorted((s.stem, s.sha256) for s in samples))


@dataclass(frozen=True)
class SampleLineage:
    """单样本 lineage:stem 属于哪个 split、内容 sha256 是多少。"""
    stem: str
    split: str
    sha256: str


def compute_contract_fingerprint(
    dataset: str,
    strategy: str,
    seed: int,
    ratios: Tuple[float, float, float],
    names: List[str],
    samples: List[SampleLineage],
    tool_version: str = TOOL_VERSION,
) -> str:
    """对'配方 + 排序后的样本指纹行'做规范化 sha256。"""
    canonical = {
        "dataset": dataset,
        "strategy": strategy,
        "seed": seed,
        "ratios": [round(float(r), 6) for r in ratios],
        "names": list(names),
        "samples": sorted((s.stem, s.split, s.sha256) for s in samples),
        "tool_version": tool_version,
    }
    return canonical_sha256(canonical)


@dataclass
class SplitManifest:
    """划分契约文件:配方 + 全量样本 lineage + 指纹。写完即不可变。"""
    dataset: str
    strategy: str
    seed: int
    ratios: Tuple[float, float, float]
    names: List[str]
    samples: List[SampleLineage]
    contract_fingerprint: str = ""
    created_at: str = ""
    tool_version: str = TOOL_VERSION

    @classmethod
    def build(
        cls,
        *,
        dataset: str,
        strategy: str,
        seed: int,
        ratios: Tuple[float, float, float],
        names: List[str],
        splits: Dict[str, List[str]],
        label_bytes: Dict[str, bytes],
        created_at: str = "",
    ) -> "SplitManifest":
        """从划分结果 + 标签字节构建契约。label_bytes 以 stem 为键。"""
        samples = [
            SampleLineage(stem, split, _sha256_bytes(label_bytes[stem]))
            for split, stems in splits.items()
            for stem in stems
        ]
        samples.sort(key=lambda s: s.stem)
        m = cls(dataset=dataset, strategy=strategy, seed=seed, ratios=ratios,
                names=list(names), samples=samples,
                created_at=created_at or datetime.now().isoformat(timespec="seconds"))
        m.contract_fingerprint = compute_contract_fingerprint(
            dataset, strategy, seed, ratios, list(names), samples, m.tool_version)
        return m

    def stems_of(self, split: str) -> List[str]:
        return [s.stem for s in self.samples if s.split == split]

    def by_stem(self) -> Dict[str, SampleLineage]:
        return {s.stem: s for s in self.samples}

    # ---- 序列化 ----

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> "SplitManifest":
        d = json.loads(path.read_text(encoding="utf-8"))
        d["ratios"] = tuple(d["ratios"])
        d["samples"] = [SampleLineage(**s) for s in d["samples"]]
        return cls(**d)
