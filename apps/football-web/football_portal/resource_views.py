"""Read-only, reviewed primary-source catalog. No media is fetched for visitors."""
import json
from pathlib import Path

from django.http import JsonResponse
from django.views.decorators.http import require_GET
from .monitor_views import dataset_inventory


@require_GET
def var_resources(request):
    path = Path(__file__).resolve().parent / "data" / "var_sources.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    inventory = dataset_inventory()
    known = {item["id"] for item in value["datasets"]}
    for source in inventory["sources"]:
        if not isinstance(source, dict) or source.get("source_id") in known:
            continue
        files = [item for item in inventory["files"] if item["source_id"] == source.get("source_id")]
        counts = {kind: sum(item["kind"] == kind for item in files) for kind in ("video", "image", "annotations")}
        value["datasets"].insert(0, {"id": source.get("source_id"), "title": source.get("title", "本地下载来源"),
                                    "source_url": source.get("url"), "event_types": source.get("event_types", []),
                                    "media_type": "本项目已下载的研究子集", "labels": source.get("annotation_quality", "结构化事件记录"),
                                    "scale": f"视频{counts['video']} / 图片{counts['image']} / 标注文件{counts['annotations']}",
                                    "access": "已存入项目本地数据目录，详细文件见运行监控。",
                                    "notes": source.get("not_covered"), "license": source.get("license"),
                                    "verified_on": inventory.get("created_at", value["as_of"])[:10],
                                    "local_download_note": "已下载子集，不代表完整数据集或42项自动识别已完成。"})
    value["media_downloaded"] = bool(inventory["counts"].get("video") or inventory["counts"].get("image"))
    return JsonResponse(value, json_dumps_params={"ensure_ascii": False})
