"""Session-private human VAR review APIs. No detector or external API is called."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

from django.conf import settings
from django.db import transaction
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_http_methods, require_POST

from .jobs import job_directory
from .models import AnalysisJob
from .var_models import VarReviewCase
from . import var_services as service


def _owner(request):
    if not request.session.session_key:
        request.session.create()
    return request.session.session_key


def _case(request, case_id, *, lock=False):
    query = VarReviewCase.objects.select_for_update() if lock else VarReviewCase.objects
    return get_object_or_404(query, pk=case_id, owner_session=_owner(request))


def _error(message, status=400, code="VAR_INVALID_REQUEST"):
    return JsonResponse({"error": message, "code": code}, status=status)


def _body(request, maximum=128 * 1024):
    if int(request.META.get("CONTENT_LENGTH") or 0) > maximum:
        raise ValueError("复核JSON内容过大")
    result = json.loads(request.body.decode("utf-8-sig") or "{}")
    if not isinstance(result, dict):
        raise ValueError("请求需要JSON对象")
    return result


def _event(value):
    if not isinstance(value, str) or value not in service.EVENTS:
        raise ValueError("event_type需要使用事件目录中已登记的code")
    return value


def _duration(value, media_kind):
    if media_kind == "image":
        if value not in (None, "") and service.number(value, "duration_seconds") != 0:
            raise ValueError("图片的时长必须为0")
        return 0.0
    return None if value in (None, "") else service.number(value, "duration_seconds", minimum=.001)


def _source_job(request, identity):
    try:
        identity = uuid.UUID(str(identity))
    except (ValueError, TypeError) as exc:
        raise ValueError("job_id格式无效") from exc
    job = get_object_or_404(AnalysisJob, pk=identity, owner_session=_owner(request))
    if job.kind not in ("image", "video"):
        raise ValueError("该任务没有可复核的原始图片或视频")
    base = job_directory(job).resolve()
    filename = job.parameters.get("_input_filename")
    if not isinstance(filename, str):
        raise ValueError("任务缺少原始素材")
    source = (base / filename).resolve()
    if source.parent != base or not source.is_file() or source.suffix.lower() not in service.IMAGE_EXTENSIONS | service.VIDEO_EXTENSIONS:
        raise ValueError("任务原始素材不存在或不可访问")
    maximum = int(getattr(settings, "FOOTBALL_MAX_IMAGE_BYTES" if job.kind == "image" else "FOOTBALL_MAX_VIDEO_BYTES",
                          (15 if job.kind == "image" else 200) * 1024**2))
    if not 0 < source.stat().st_size <= maximum:
        raise ValueError("原始素材为空或超过当前大小限制")
    return job, source


@ensure_csrf_cookie
@require_http_methods(["GET", "POST"])
def cases(request):
    owner = _owner(request)
    if request.method == "GET":
        return JsonResponse({"cases": [service.serialize(case) for case in VarReviewCase.objects.filter(owner_session=owner)[:100]],
                             "event_types": service.EVENTS, "automated_decision": False,
                             "limits": {"keyframes": service.MAX_KEYFRAMES, "lines_per_frame": 12,
                                        "snapshot_bytes": min(int(getattr(settings, "FOOTBALL_MAX_IMAGE_BYTES", 15 * 1024**2)), 8 * 1024**2),
                                        "media_seconds": service.MAX_SECONDS}})
    created = None
    source_path = None
    try:
        if VarReviewCase.objects.filter(owner_session=owner).count() >= 100:
            return _error("当前会话最多保留100个复核案例", 429, "VAR_CASE_LIMIT")
        upload = request.FILES.get("file")
        data = request.POST if request.content_type == "multipart/form-data" else _body(request)
        allowed = {"job_id", "title", "event_type", "duration_seconds", "csrfmiddlewaretoken"}
        if set(data) - allowed:
            raise ValueError("创建复核时包含不支持的字段")
        event = _event(data.get("event_type", "offside"))
        linked, original = None, None
        if upload is not None:
            if data.get("job_id"):
                raise ValueError("直接上传与job_id二选一")
            input_name, kind, suffix = service.validate_media(upload)
        elif data.get("job_id"):
            linked, original = _source_job(request, data["job_id"])
            input_name, kind, suffix = linked.input_name, linked.kind, original.suffix.lower()
        else:
            raise ValueError("请上传图片/视频，或提供当前会话的job_id")
        title = service.text(data.get("title", Path(input_name).stem[:160]), "复核名称", 160, required=True)
        duration = _duration(data.get("duration_seconds"), kind)
        created = VarReviewCase(owner_session=owner, source_job=linked, title=title, event_type=event,
                                input_name=input_name, media_kind=kind, source_filename="source" + suffix,
                                duration_seconds=duration)
        directory = service.directory(created)
        directory.mkdir(parents=True, exist_ok=False)
        source_path = directory / created.source_filename
        if original is not None:
            shutil.copyfile(original, source_path)
        else:
            with source_path.open("wb") as stream:
                for chunk in upload.chunks():
                    stream.write(chunk)
        created.save()
        return JsonResponse({"case": service.serialize(created)}, status=201)
    except (ValueError, UnicodeError) as exc:
        return _error(str(exc))
    except OSError:
        return _error("复核素材保存失败，请检查本机存储空间", 500, "VAR_STORAGE_ERROR")
    finally:
        if created is not None and created._state.adding and source_path is not None:
            source_path.unlink(missing_ok=True)
            try:
                source_path.parent.rmdir()
            except OSError:
                pass


@require_http_methods(["GET", "PATCH"])
def case_detail(request, case_id):
    if request.method == "GET":
        return JsonResponse({"case": service.serialize(_case(request, case_id))})
    try:
        data = _body(request)
        allowed = {"title", "event_type", "status", "observations", "verdict", "duration_seconds"}
        if not data or set(data) - allowed:
            raise ValueError("仅可更新复核名称、类型、状态、人工观察、人工意见或浏览器时长")
        with transaction.atomic():
            case = _case(request, case_id, lock=True)
            if "title" in data:
                case.title = service.text(data["title"], "复核名称", 160, required=True)
            if "event_type" in data:
                case.event_type = _event(data["event_type"])
            if "observations" in data:
                case.observations = service.text(data["observations"], "人工观察", 4000)
            if "verdict" in data:
                case.verdict = service.text(data["verdict"], "人工复核意见", 2000)
            if "duration_seconds" in data:
                case.duration_seconds = _duration(data["duration_seconds"], case.media_kind)
                if case.duration_seconds is not None and any(frame["media_time"] > case.duration_seconds for frame in case.keyframes):
                    raise ValueError("浏览器时长小于已有关键帧时间，不能覆盖")
            if "status" in data:
                if not isinstance(data["status"], str) or data["status"] not in service.STATUSES:
                    raise ValueError("status只能是pending/needs_evidence/reviewed")
                case.status = data["status"]
            if case.status == "reviewed" and (not case.keyframes or not case.verdict):
                raise ValueError("已复核需要至少一张关键帧和人工复核意见")
            case.save()
        return JsonResponse({"case": service.serialize(case)})
    except (ValueError, UnicodeError) as exc:
        return _error(str(exc))


@require_POST
def keyframes(request, case_id):
    saved = None
    try:
        _case(request, case_id)
        upload = request.FILES.get("file")
        if upload is None:
            raise ValueError("请以multipart file提交关键帧图片")
        media_time = service.number(request.POST.get("media_time", 0), "media_time")
        notes = service.text(request.POST.get("notes", ""), "截图备注", 2000)
        lines = service.validate_lines(request.POST.get("lines", "[]"))
        image = service.validate_image(upload, min(int(getattr(settings, "FOOTBALL_MAX_IMAGE_BYTES", 15 * 1024**2)), 8 * 1024**2))
        with transaction.atomic():
            case = _case(request, case_id, lock=True)
            if len(case.keyframes) >= service.MAX_KEYFRAMES:
                raise ValueError(f"每个案例最多{service.MAX_KEYFRAMES}张关键帧")
            if case.media_kind == "image" and media_time != 0:
                raise ValueError("图片关键帧的media_time必须为0")
            if case.duration_seconds is not None and media_time > case.duration_seconds:
                raise ValueError("关键帧时间超过浏览器报告的素材时长")
            identity = str(uuid.uuid4())
            filename = "frame-" + identity + image["suffix"]
            saved = service.directory(case) / filename
            with saved.open("wb") as stream:
                for chunk in upload.chunks():
                    stream.write(chunk)
            frame = {"id": identity, "filename": filename, "media_time": media_time,
                     "notes": notes, "lines": lines, "width": image["width"], "height": image["height"],
                     "created_at": timezone.now().isoformat()}
            case.keyframes = [*case.keyframes, frame]
            case.save(update_fields=["keyframes", "updated_at"])
        return JsonResponse({"case": service.serialize(case)}, status=201)
    except (ValueError, UnicodeError) as exc:
        if saved is not None:
            saved.unlink(missing_ok=True)
        return _error(str(exc))
    except OSError:
        if saved is not None:
            saved.unlink(missing_ok=True)
        return _error("关键帧保存失败，请检查本机存储空间", 500, "VAR_STORAGE_ERROR")


@require_http_methods(["DELETE"])
def delete_keyframe(request, case_id, frame_id):
    try:
        with transaction.atomic():
            case = _case(request, case_id, lock=True)
            frame = next((item for item in case.keyframes if item.get("id") == str(frame_id)), None)
            if frame is None:
                raise Http404
            path = service.artifact_path(case, "frame-" + str(frame_id))
            # If a Windows stream still holds the file, retain its evidence record
            # and let the user retry instead of silently leaving an orphaned file.
            if path is not None:
                path.unlink(missing_ok=True)
            case.keyframes = [item for item in case.keyframes if item["id"] != str(frame_id)]
            if not case.keyframes and case.status == "reviewed":
                case.status = "needs_evidence"
            case.save(update_fields=["keyframes", "status", "updated_at"])
    except OSError:
        return _error("关键帧文件暂时无法删除，请等待图片读取结束后重试", 409, "VAR_ARTIFACT_BUSY")
    return JsonResponse({"case": service.serialize(case)})


@require_http_methods(["GET", "HEAD"])
def artifact(request, case_id, key):
    case = _case(request, case_id)
    path = service.artifact_path(case, key)
    if path is None:
        raise Http404
    return service.file_response(request, path, case.input_name if key == "original" else path.name)


@require_http_methods(["GET"])
def report_json(request, case_id):
    case = _case(request, case_id)
    response = JsonResponse(service.report(case), json_dumps_params={"ensure_ascii": False, "indent": 2, "allow_nan": False})
    response["Content-Disposition"] = f'attachment; filename="var-review-{case.pk}.json"'
    response["Cache-Control"] = "private, no-store"
    return response


@require_http_methods(["GET"])
def report_md(request, case_id):
    case = _case(request, case_id)
    response = HttpResponse(service.report_markdown(case), content_type="text/markdown; charset=utf-8")
    response["Content-Disposition"] = f'attachment; filename="var-review-{case.pk}.md"'
    response["Cache-Control"] = "private, no-store"
    response["X-Content-Type-Options"] = "nosniff"
    return response
