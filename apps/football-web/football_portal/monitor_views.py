"""Session-scoped operations and evidence readiness; never simulated telemetry."""
from collections import Counter
import json
from pathlib import Path
import time

from django.conf import settings
from django.db.models import Count
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.http import require_GET

from .models import AnalysisJob, AssistantRecord, VarReviewCase
from .event_catalog import taxonomy
from .views import _owner

_STARTED = time.monotonic()
DATASET_RELATIVE_DIR = "datasets/football_events"


def dataset_inventory():
    """Check private manifest entries and sizes, without serving private raw data."""
    root = Path(getattr(settings, "FOOTBALL_EVENT_DATA_DIR", Path(settings.FOOTBALL_DATA_DIR) / DATASET_RELATIVE_DIR)).resolve()
    valid_ids = {event["id"] for event in taxonomy()["events"]}
    project = Path(__file__).resolve().parent.parent
    display_directory = root.relative_to(project).as_posix() if root.is_relative_to(project) else "由FOOTBALL_EVENT_DATA_DIR指定的服务端目录"
    manifest = root / "manifest.json"
    result = {"present": False, "directory": display_directory,
              "files": [], "counts": {}, "total_bytes": 0, "missing_files": 0,
              "invalid_entries": 0, "blocked_sources": [], "sources": [],
              "download_failures": [], "download_failures_count": 0,
              "verification": "local_manifest_and_file_sizes"}
    if not manifest.is_file():
        return result
    try:
        if manifest.stat().st_size > 8 * 1024**2:
            raise ValueError("manifest too large")
        value = json.loads(manifest.read_text(encoding="utf-8-sig"))
        entries = value.get("files", [])
        if not isinstance(entries, list) or len(entries) > 10000:
            raise ValueError("invalid manifest")
    except (OSError, ValueError, AttributeError):
        result["invalid_entries"] = 1
        return result
    result.update(present=True, created_at=value.get("created_at"),
                  blocked_sources=value.get("blocked_sources", []), sources=value.get("sources", []))
    failures = value.get("failed_downloads", value.get("failures", []))
    if isinstance(failures, list):
        result["download_failures"] = [{"path": str(item.get("path", "待下载文件"))[:200],
                                         "error": str(item.get("error", "下载失败，请重试"))[:400]}
                                        for item in failures if isinstance(item, dict)]
        result["download_failures_count"] = len(failures)
    counts = Counter()
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            result["invalid_entries"] += 1
            continue
        candidate = (root / entry["path"]).resolve()
        if not candidate.is_relative_to(root) or candidate in seen:
            result["invalid_entries"] += 1
            continue
        seen.add(candidate)
        expected = entry.get("bytes")
        if not isinstance(expected, int) or isinstance(expected, bool) or expected < 1:
            result["invalid_entries"] += 1
            continue
        if not candidate.is_file() or candidate.stat().st_size != expected:
            result["missing_files"] += 1
            continue
        kind = entry.get("kind")
        kind = kind if kind in {"video", "image", "annotations", "metadata", "license"} else "metadata"
        ids = entry.get("coverage_event_ids", [])
        ids = sorted({number for number in ids if type(number) is int and number in valid_ids}) if isinstance(ids, list) else []
        result["files"].append({"path": entry["path"], "bytes": expected, "kind": kind,
                                "source_id": entry.get("source_id", "unknown"),
                                "sha256": entry.get("sha256"), "coverage_event_ids": ids,
                                "license": entry.get("license", "待核验")})
        counts[kind] += 1
        result["total_bytes"] += expected
    result["counts"] = dict(counts)
    return result


def coverage(events, inventory):
    rows = []
    for event in events:
        files = [item for item in inventory["files"] if event["id"] in item["coverage_event_ids"]]
        kinds = Counter(item["kind"] for item in files)
        visual = kinds["image"] + kinds["video"]
        state = "visual_samples" if visual else "annotations_only" if kinds["annotations"] else "pending"
        rows.append({**event, "data_state": state, "image_files": kinds["image"],
                     "video_files": kinds["video"], "annotation_files": kinds["annotations"],
                     "model_status": event.get("model_status", "unknown"),
                     "automated_recognition": event.get("automated_available", False) is True})
    return rows


@require_GET
def events_view(request):
    return JsonResponse(taxonomy(), json_dumps_params={"ensure_ascii": False})


@require_GET
def datasets_view(request):
    return JsonResponse(dataset_inventory(), json_dumps_params={"ensure_ascii": False})


@require_GET
def monitor_view(request):
    owner = _owner(request)
    jobs = AnalysisJob.objects.filter(owner_session=owner)
    reviews = VarReviewCase.objects.filter(owner_session=owner)
    counts = dict(jobs.values_list("status").annotate(total=Count("id")))
    review_counts = dict(reviews.values_list("status").annotate(total=Count("id")))
    inventory = dataset_inventory()
    rows = coverage(taxonomy()["events"], inventory)
    recent = list(jobs.values("id", "input_name", "status", "progress", "error", "created_at", "updated_at")[:8])
    alerts = []
    if counts.get("failed", 0):
        alerts.append({"level": "warning", "message": f"本会话有{counts['failed']}个失败任务，请查看任务历史。"})
    if inventory["missing_files"] or inventory["invalid_entries"]:
        alerts.append({"level": "warning", "message": "部分下载清单与本地文件不一致，请重新校验下载。"})
    if inventory["download_failures_count"]:
        alerts.append({"level": "warning", "message": f"下载计划有{inventory['download_failures_count']}项未成功，已落盘文件仍可使用；请查看文件清单下的失败记录。"})
    if not inventory["counts"].get("video"):
        alerts.append({"level": "info", "message": "本地尚无已登记的真实视频样本；事件JSON不能替代视觉训练视频。"})
    alerts.append({"level": "info", "message": "42项为能力目标，新增事件模型尚未训练；手工复核和数据样本不代表自动识别已完成。"})
    return JsonResponse({"generated_at": timezone.now().isoformat(), "scope": "current_session",
                         "service": {"framework": "Django", "process_uptime_seconds": int(time.monotonic() - _STARTED),
                                     "worker_concurrency": 1, "refresh_seconds": settings.FOOTBALL_MONITOR_REFRESH_SECONDS,
                                     "gpu_telemetry": None, "accuracy": None},
                         "jobs": {"total": jobs.count(), "counts": counts, "recent": recent},
                         "reviews": {"total": reviews.count(), "counts": review_counts,
                                     "keyframes": sum(len(value) for value in reviews.values_list("keyframes", flat=True))},
                         "assistant": {"real_calls": AssistantRecord.objects.filter(owner_session=owner, source="llm").count()},
                         "datasets": inventory, "event_coverage": rows,
                         "coverage_summary": {"targets": len(rows), "visual_sample_events": sum(row["data_state"] == "visual_samples" for row in rows),
                                              "annotation_only_events": sum(row["data_state"] == "annotations_only" for row in rows),
                                              "trained_event_models": sum(row["automated_recognition"] for row in rows)}, "alerts": alerts},
                        json_dumps_params={"ensure_ascii": False})
