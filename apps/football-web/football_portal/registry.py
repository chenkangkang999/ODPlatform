import hashlib
import json
from pathlib import Path

from django.conf import settings

_cache = {}
SAFE_METADATA = ("run_id", "model_version", "class_names", "data_fingerprint", "content_fingerprint",
                 "contract_fingerprint", "dataset_fingerprint", "development_only", "dataset_limitations", "purpose", "execution")


def _digest(path):
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _cache:
        with path.open("rb") as stream:
            _cache[key] = hashlib.file_digest(stream, "sha256").hexdigest()
    return _cache[key]


def _entry(path):
    digest = _digest(path)
    metadata, metadata_path, warning = {}, None, "模型未提供可核验元信息；不显示虚构精度。"
    explicit_model = getattr(settings, "FOOTBALL_MODEL_PATH", "")
    is_explicit = bool(explicit_model) and Path(explicit_model).expanduser().resolve() == path
    configured_metadata = getattr(settings, "FOOTBALL_MODEL_METADATA", "") if is_explicit else ""
    candidates = [Path(configured_metadata).expanduser().resolve()] if configured_metadata else []
    candidates.extend((path.with_suffix(".metadata.json"), path.parent / "model_metadata.json"))
    for candidate in candidates:
        if candidate.is_file():
            metadata_path = candidate
            try:
                raw = json.loads(candidate.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    raise ValueError("模型元信息必须是对象")
                metadata = {key: raw[key] for key in SAFE_METADATA if key in raw}
                recorded = raw.get("model_sha256", raw.get("weight_sha256"))
                if recorded and recorded != digest:
                    return {"id": f"model_{digest[:24]}", "name": path.stem, "version": digest[:12],
                            "metadata": {}, "ready": False, "warning": "权重与元信息摘要不一致。", "_path": path}
                warning = "" if recorded else "元信息没有权重摘要，来源未完整核验。"
            except (ValueError, OSError, TypeError):
                warning = "元信息无法解析，来源未完整核验。"
            break
    name = getattr(settings, "FOOTBALL_MODEL_LABEL", "") if is_explicit else ""
    return {"id": f"model_{digest[:24]}", "name": name or path.stem, "version": metadata.get("run_id", digest[:12]),
            "sha256": digest, "metadata": metadata, "ready": path.stat().st_size > 0,
            "warning": warning, "_path": path, "_metadata_path": metadata_path}


def model_entries():
    paths = []
    configured = getattr(settings, "FOOTBALL_MODEL_PATH", "")
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.is_file() and path.suffix.lower() == ".pt":
            paths.append(path)
    directory = getattr(settings, "FOOTBALL_MODEL_DIR", "")
    if directory:
        root = Path(directory).expanduser().resolve()
        if root.is_dir():
            paths.extend(p.resolve() for p in sorted(root.glob("*.pt"))[:30] if p.resolve().is_relative_to(root))
    unique = {}
    for path in paths:
        try:
            entry = _entry(path)
            unique.setdefault(entry["id"], entry)
        except OSError:
            continue
    return list(unique.values())


def public_models():
    return [{key: value for key, value in entry.items() if not key.startswith("_")} for entry in model_entries()]


def find_model(model_id):
    for entry in model_entries():
        if entry["id"] == model_id and entry["ready"]:
            return entry
    raise ValueError("模型不可用，请配置同学提供的可信足球权重后刷新。")
