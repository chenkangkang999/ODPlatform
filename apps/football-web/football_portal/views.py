import json
import math
import mimetypes
from pathlib import Path
import shutil
import uuid

from django.conf import settings
from django.http import FileResponse, Http404, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_http_methods, require_POST
from PIL import Image, ImageOps, UnidentifiedImageError

from . import jobs, registry
from .models import AnalysisJob, AssistantRecord


def _owner(request):
    if not request.session.session_key:
        request.session.create()
    return request.session.session_key


def _job(request, job_id):
    try:
        job_id = uuid.UUID(str(job_id))
    except (ValueError, TypeError) as exc:
        raise Http404 from exc
    return get_object_or_404(AnalysisJob, pk=job_id, owner_session=_owner(request))


def _error(message, *, status=400, code="INVALID_REQUEST"):
    return JsonResponse({"error": message, "code": code}, status=status)


def _json_body(request, maximum=2 * 1024 * 1024):
    if int(request.META.get("CONTENT_LENGTH") or 0) > maximum:
        raise ValueError("JSON内容过大，请使用精简的逐帧结果")
    value = json.loads(request.body.decode("utf-8-sig") or "{}")
    if not isinstance(value, dict):
        raise ValueError("请求必须是JSON对象")
    return value


def _artifact_path(job, key):
    directory = jobs.job_directory(job)
    if key == "original":
        name = job.parameters.get("_input_filename")
        candidate = directory / name if name else None
        base = directory
    else:
        allowed = {"annotated", "video", "frames", "counts", "result", "report", "preview"}
        if key not in allowed:
            return None
        name = job.result.get("artifacts", {}).get(key)
        if key == "result":
            name = "result.json"
        if not isinstance(name, str):
            return None
        base = directory / "output"
        candidate = base / name
    if candidate is None:
        return None
    candidate, base = candidate.resolve(), base.resolve()
    if not candidate.is_relative_to(base) or not candidate.is_file():
        return None
    if candidate.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".mp4", ".webm", ".avi", ".mov", ".mkv", ".json", ".jsonl", ".csv", ".md", ".txt"}:
        return None
    return candidate


def serialize_job(job, *, detailed=False):
    urls = {}
    for key in ("original", "annotated", "video", "frames", "counts", "result", "report", "preview"):
        if _artifact_path(job, key):
            urls[key] = reverse("football_portal:artifact", kwargs={"job_id": job.pk, "key": key})
    result = {"id": str(job.pk), "status": job.status, "progress": job.progress,
              "input_name": job.input_name, "kind": job.kind, "model_id": job.model_id,
              "parameters": {k: v for k, v in job.parameters.items() if not k.startswith("_")},
              "error": job.error, "cancel_requested": job.cancel_requested,
              "created_at": job.created_at.isoformat(), "updated_at": job.updated_at.isoformat(),
              "artifact_urls": urls}
    if detailed:
        payload = dict(job.result)
        payload["artifacts"] = {key: value for key, value in urls.items() if key != "original"}
        result["result"] = payload
    elif job.result:
        result["summary"] = {k: job.result.get("analysis", {}).get(k) for k in ("processed_frames", "total_detections", "per_class")}
    return result


@ensure_csrf_cookie
@require_GET
def shell(request):
    _owner(request)
    return render(request, "football_portal/index.html", {
        "carousel_interval_ms": settings.FOOTBALL_CAROUSEL_SECONDS * 1000,
        "monitor_interval_ms": settings.FOOTBALL_MONITOR_REFRESH_SECONDS * 1000,
    })


@require_GET
def status_view(request):
    from . import llm_service
    _owner(request)
    demo_path = getattr(settings, "FOOTBALL_DEMO_IMAGE_PATH", "")
    return JsonResponse({"models": registry.public_models(), "llm": llm_service.status(),
                         "demo_available": bool(demo_path) and Path(demo_path).is_file(),
                         "limits": {"image_bytes": int(getattr(settings, "FOOTBALL_MAX_IMAGE_BYTES", 15 * 1024**2)),
                                    "video_bytes": int(getattr(settings, "FOOTBALL_MAX_VIDEO_BYTES", 200 * 1024**2)),
                                    "max_frames": int(getattr(settings, "FOOTBALL_MAX_FRAMES", 1500))},
                         "capabilities": {"image": True, "video": True, "import": True,
                                          "team_assignment": False, "tracking": False, "possession": False},
                         "notice": "YOLO负责检测，LLM仅解释检测记录；无真值评估时不显示准确率。"})


@require_GET
def models_view(request):
    _owner(request)
    return JsonResponse({"models": registry.public_models()})


def _parameters(post):
    if any(isinstance(post.get(name), bool) for name in ("conf", "imgsz", "stride")):
        raise ValueError("数值参数不能使用布尔值")
    try:
        conf = float(post.get("conf", 0.25))
        imgsz = int(post.get("imgsz", 960))
        stride = int(post.get("stride", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("置信度、输入尺寸和采样步长应为有效数字") from exc
    if not math.isfinite(conf) or not 0.001 <= conf <= 0.99:
        raise ValueError("置信度范围为0.001到0.99")
    if not 320 <= imgsz <= 1536 or imgsz % 32 or not 1 <= stride <= 30:
        raise ValueError("输入尺寸需为320到1536的32倍数，步长范围1到30")
    return {"conf": conf, "imgsz": imgsz, "stride": stride}


@require_http_methods(["GET", "POST"])
def jobs_view(request):
    owner = _owner(request)
    if request.method == "GET":
        return JsonResponse({"jobs": [serialize_job(job) for job in AnalysisJob.objects.filter(owner_session=owner)[:100]]})
    upload = request.FILES.get("file")
    if not upload:
        return _error("请选择图片或视频文件")
    if AnalysisJob.objects.filter(status__in=("queued", "running")).count() >= int(getattr(settings, "FOOTBALL_MAX_QUEUE", 6)):
        return _error("本机任务队列已满，请等待当前任务结束", status=429, code="QUEUE_FULL")
    try:
        model = registry.find_model(request.POST.get("model_id", ""))
        parameters = _parameters(request.POST)
        name = Path(upload.name).name
        suffix = Path(name).suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            kind, limit = "image", int(getattr(settings, "FOOTBALL_MAX_IMAGE_BYTES", 15 * 1024**2))
        elif suffix in {".mp4", ".webm", ".avi", ".mov", ".mkv"}:
            kind, limit = "video", int(getattr(settings, "FOOTBALL_MAX_VIDEO_BYTES", 200 * 1024**2))
        else:
            raise ValueError("仅支持常见图片和MP4/AVI/MOV/MKV视频，不接收模型或可执行文件")
        if not 0 < upload.size <= limit:
            raise ValueError("文件为空或超出页面提示的大小限制")
        if kind == "image":
            try:
                with Image.open(upload) as image:
                    if image.width * image.height > 40_000_000:
                        raise ValueError("图片像素过大，请缩小后上传")
                    image.verify()
                upload.seek(0)
            except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
                raise ValueError("文件无法解码为有效图片") from exc
        parameters.update(_input_filename="input" + suffix, model_sha256=model.get("sha256"))
        job = AnalysisJob.objects.create(owner_session=owner, input_name=name[:255], kind=kind,
                                         model_id=model["id"], parameters=parameters)
        directory = jobs.job_directory(job)
        directory.mkdir(parents=True, exist_ok=False)
        with (directory / parameters["_input_filename"]).open("wb") as stream:
            for chunk in upload.chunks():
                stream.write(chunk)
        jobs.submit_job(job.pk)
        job.refresh_from_db()
        return JsonResponse({"job": serialize_job(job, detailed=True)}, status=201)
    except (ValueError, OSError) as exc:
        return _error(str(exc))


@require_GET
def job_detail(request, job_id):
    return JsonResponse({"job": serialize_job(_job(request, job_id), detailed=True)})


@require_POST
def demo_view(request):
    configured = getattr(settings, "FOOTBALL_DEMO_IMAGE_PATH", "")
    if not configured or not Path(configured).is_file():
        return _error("尚未配置本机示例图片；可直接上传自己的图片。", status=404, code="DEMO_NOT_CONFIGURED")
    if AnalysisJob.objects.filter(status__in=("queued", "running")).count() >= int(getattr(settings, "FOOTBALL_MAX_QUEUE", 6)):
        return _error("本机任务队列已满，请稍后重试", status=429, code="QUEUE_FULL")
    try:
        body = _json_body(request, maximum=16 * 1024)
        model = registry.find_model(body.get("model_id", ""))
        parameters = _parameters(body)
        source = Path(configured).resolve()
        suffix = source.suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            raise ValueError("本机示例配置不是支持的图片格式")
        if source.stat().st_size > int(getattr(settings, "FOOTBALL_MAX_IMAGE_BYTES", 15 * 1024**2)):
            raise ValueError("本机示例图片超出大小限制")
        with Image.open(source) as image:
            image.verify()
        parameters.update(_input_filename="input" + suffix, model_sha256=model.get("sha256"), sample_origin="local_integration_example")
        job = AnalysisJob.objects.create(owner_session=_owner(request), input_name="本机联调示例" + suffix,
                                         kind="image", model_id=model["id"], parameters=parameters)
        directory = jobs.job_directory(job)
        directory.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(source, directory / parameters["_input_filename"])
        jobs.submit_job(job.pk)
        job.refresh_from_db()
        return JsonResponse({"job": serialize_job(job, detailed=True)}, status=201)
    except (ValueError, OSError) as exc:
        return _error(str(exc))


@require_POST
def cancel_job(request, job_id):
    job = _job(request, job_id)
    if job.status in ("queued", "running"):
        job.cancel_requested = True
        if job.status == "queued":
            job.status = "cancelled"
        job.save(update_fields=["cancel_requested", "status", "updated_at"])
    return JsonResponse({"job": serialize_job(job, detailed=True)})


@require_GET
def artifact(request, job_id, key):
    job = _job(request, job_id)
    path = _artifact_path(job, key)
    if path is None:
        raise Http404
    if key == "annotated" and job.kind == "image" and job.status == "completed":
        _refresh_image_annotation(job, path)
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    response = FileResponse(path.open("rb"), content_type=content_type,
                            as_attachment=path.suffix.lower() in {".json", ".jsonl", ".csv", ".md", ".txt"}, filename=path.name)
    response["X-Content-Type-Options"] = "nosniff"
    response["Cache-Control"] = "private, no-store"
    return response


def _refresh_image_annotation(job, path):
    """Restyle saved image boxes on demand; never re-run or alter YOLO results."""
    from .annotations import STYLE_VERSION, render_annotations

    marker = path.parent / "annotation_style.json"
    try:
        if json.loads(marker.read_text(encoding="utf-8")).get("version") == STYLE_VERSION:
            return
    except (OSError, ValueError, AttributeError):
        pass
    original = _artifact_path(job, "original")
    preview = job.result.get("preview", {})
    boxes = preview.get("boxes", job.result.get("boxes"))
    if original is None or not isinstance(boxes, list):
        return
    temporary = path.with_name(f"annotation-{uuid.uuid4().hex}.tmp")
    try:
        with Image.open(original) as source:
            oriented = ImageOps.exif_transpose(source)
            rendered = render_annotations(oriented, boxes)
        rendered.save(temporary, format="PNG")
        temporary.replace(path)
        marker.write_text(json.dumps({"version": STYLE_VERSION}), encoding="utf-8")
    finally:
        temporary.unlink(missing_ok=True)


@require_POST
def import_view(request):
    job = None
    try:
        if request.FILES.get("file"):
            uploaded = request.FILES["file"]
            if uploaded.size > 4 * 1024 * 1024:
                raise ValueError("导入JSON上限4MB，请精简记录")
            payload = json.loads(uploaded.read().decode("utf-8-sig"))
        else:
            body = _json_body(request, maximum=4 * 1024 * 1024)
            payload = body.get("payload", body)
        from .results import normalize_payload
        normalized = normalize_payload(payload)
        if normalized["analysis"]["processed_frames"] > int(getattr(settings, "FOOTBALL_MAX_FRAMES", 1500)):
            raise ValueError("导入帧数超出本机处理上限")
        job = AnalysisJob.objects.create(owner_session=_owner(request), input_name="导入检测结果.json", kind="import",
                                         parameters={"imported": True})
        jobs.write_import(job, payload)
        job.refresh_from_db()
        return JsonResponse({"job": serialize_job(job, detailed=True)}, status=201)
    except (ValueError, UnicodeError, OSError) as exc:
        if job is not None:
            job.delete()
        return _error(str(exc))


def _record(record):
    return {"id": record.pk, "job_id": str(record.job_id) if record.job_id else None,
            "mode": record.mode, "question": record.question, "answer": record.answer,
            "source": record.source, "model": record.model, "usage": record.usage,
            "status": record.status, "created_at": record.created_at.isoformat()}


@require_POST
def assistant(request):
    from . import llm_service
    try:
        body = _json_body(request, maximum=64 * 1024)
        mode = body.get("mode", "chat")
        question = str(body.get("question", "")).strip()
        if mode not in {"report", "diagnosis", "chat"} or len(question) > 2000:
            raise ValueError("模式无效或问题超过2000字")
        job = _job(request, body["job_id"]) if body.get("job_id") else None
        if mode in {"report", "diagnosis"} and (job is None or job.status != "completed"):
            raise ValueError("请先选择一个已完成的检测任务")
        if not question and mode == "chat":
            raise ValueError("请输入问题")
        state = llm_service.status()
        if not state.get("configured"):
            return _error("尚未配置大模型API密钥。检测功能不受影响，可先使用本地统计摘要。", status=503, code="LLM_NOT_CONFIGURED")
        history = body.get("history", [])
        if not isinstance(history, list):
            raise ValueError("history必须为列表")
        clean_history = [{"role": item["role"], "content": str(item.get("content", ""))[:3000]}
                         for item in history[-12:] if isinstance(item, dict) and item.get("role") in {"user", "assistant"}]
        context = llm_service.build_context(job.result) if job and job.status == "completed" else {}
        owner = _owner(request)
        ledger = Path(settings.FOOTBALL_DATA_DIR) / "usage" / (owner + ".jsonl")
        result = llm_service.answer(context=context, question=question, mode=mode, history=clean_history, ledger_path=ledger)
        if not isinstance(result, dict) or not isinstance(result.get("answer"), str):
            raise RuntimeError("助手返回格式无效")
        record = AssistantRecord.objects.create(owner_session=owner, job=job, mode=mode,
                                                question=llm_service._redact(question, 2000),
                                                answer=result["answer"], source=result.get("source", "llm"),
                                                model=result.get("model", ""), usage=result.get("usage", {}))
        if job is not None and mode == "report":
            report_path = jobs.job_directory(job) / "output" / "assistant_report.md"
            report_path.write_text(result["answer"], encoding="utf-8")
            job.result.setdefault("artifacts", {})["report"] = "assistant_report.md"
            job.save(update_fields=["result", "updated_at"])
        return JsonResponse({**result, "record_id": record.pk})
    except (ValueError, UnicodeError) as exc:
        return _error(str(exc))
    except Http404:
        raise
    except Exception as exc:
        public_message = getattr(exc, "public_message", "智能助手暂不可用，请检查本地API配置或稍后重试。")
        status_code = getattr(exc, "status_code", 502)
        return _error(public_message, status=status_code, code="LLM_ERROR")


@require_POST
def local_summary_view(request):
    from . import llm_service
    try:
        body = _json_body(request)
        job = _job(request, body.get("job_id"))
        if job.status != "completed":
            raise ValueError("请先完成检测")
        text = llm_service.local_summary(job.result)
        record = AssistantRecord.objects.create(owner_session=_owner(request), job=job, mode="report", answer=text,
                                                source="local", model="local-rules", usage={})
        return JsonResponse({"answer": text, "source": "local", "model": "local-rules", "usage": {}, "record_id": record.pk})
    except (ValueError, UnicodeError) as exc:
        return _error(str(exc))


@require_GET
def assistant_history(request):
    records = AssistantRecord.objects.filter(owner_session=_owner(request))
    if request.GET.get("job_id"):
        records = records.filter(job=_job(request, request.GET["job_id"]))
    return JsonResponse({"records": [_record(record) for record in records[:50]]})


@require_GET
def usage(request):
    records = AssistantRecord.objects.filter(owner_session=_owner(request))[:100]
    rows = [_record(record) for record in records]
    total = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "unknown_usage_calls": 0}
    for row in rows:
        if row["source"] != "llm" or row["status"] != "completed":
            continue
        total["calls"] += 1
        if row["usage"].get("complete") is False:
            total["unknown_usage_calls"] += 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            alias = {"prompt_tokens": "input_tokens", "completion_tokens": "output_tokens", "total_tokens": "total_tokens"}[key]
            number = row["usage"].get(key, row["usage"].get(alias))
            if isinstance(number, (int, float)) and not isinstance(number, bool) and math.isfinite(number) and number >= 0:
                total[key] += int(number)
    return JsonResponse({"records": rows, "totals": total, "note": "仅展示服务端返回的usage；缺失值不猜测，费用记录以提供方账单为准。"})
