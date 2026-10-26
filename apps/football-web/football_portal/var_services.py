"""Validation, private artifacts and human-evidence exports for VAR review."""
from __future__ import annotations

import html
import json
import math
import mimetypes
from pathlib import Path
import re
import uuid

from django.conf import settings
from django.http import FileResponse, HttpResponse, StreamingHttpResponse
from django.urls import reverse
from django.utils.http import content_disposition_header
from PIL import Image, UnidentifiedImageError
from .event_catalog import event_types


EVENTS = event_types()
STATUSES = {"pending", "needs_evidence", "reviewed"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".avi", ".mov", ".mkv"}
MAX_SECONDS = 6 * 60 * 60
MAX_KEYFRAMES = 30
LIMITATIONS = [
    "这是人工VAR辅助复核记录，不是模型自动判罚，也不是官方裁判决定。",
    "截图时间来自浏览器显示时间，视频时长来自浏览器元数据；不保证专业逐帧或踢球接触时刻精度。",
    "手绘线为归一化图片参考线，未进行相机或球场标定，不能当作正式越位线。",
    "四类目标检测不能独立判断越位、犯规、球体整体过门线或红黄牌；这些事件模型与多视角证据尚待接入。",
]


def text(value, name, maximum, *, required=False):
    if not isinstance(value, str):
        raise ValueError(f"{name}必须为文本")
    result = value.strip()
    if len(result) > maximum or (required and not result):
        raise ValueError(f"{name}不能为空且不超过{maximum}字" if required else f"{name}不能超过{maximum}字")
    if any(ord(character) < 32 and character not in "\n\r\t" for character in result):
        raise ValueError(f"{name}不能包含控制字符")
    return result


def number(value, name, *, minimum=0, maximum=MAX_SECONDS):
    if isinstance(value, bool):
        raise ValueError(f"{name}必须为有限数值")
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name}必须为有限数值") from exc
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name}范围为{minimum}到{maximum}")
    return result


def validate_lines(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError) as exc:
            raise ValueError("lines必须是有效JSON列表") from exc
    if not isinstance(value, list) or len(value) > 12:
        raise ValueError("每张截图最多12条参考线")
    result = []
    for line in value:
        if not isinstance(line, dict):
            raise ValueError("每条参考线必须包含x1/y1/x2/y2")
        normalized = {key: number(line.get(key), key, maximum=1) for key in ("x1", "y1", "x2", "y2")}
        if normalized["x1"] == normalized["x2"] and normalized["y1"] == normalized["y2"]:
            raise ValueError("参考线的两个端点不能相同")
        if "label" in line:
            normalized["label"] = text(line["label"], "参考线标签", 32)
        if "color" in line:
            color = line["color"]
            if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
                raise ValueError("参考线color必须为六位十六进制颜色")
            normalized["color"] = color
        result.append(normalized)
    return result


def validate_image(upload, maximum):
    if not 0 < upload.size <= maximum:
        raise ValueError("图片为空或超出大小限制")
    try:
        with Image.open(upload) as picture:
            width, height, image_format = picture.width, picture.height, picture.format
            if width * height > 40_000_000:
                raise ValueError("图片不能超过4000万像素")
            picture.verify()
        suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp"}.get(image_format)
        if suffix is None:
            raise ValueError("仅支持JPEG/PNG/WebP/BMP图片")
        upload.seek(0)
        return {"suffix": suffix, "width": width, "height": height}
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError("截图或图片不能解码为有效图片") from exc


def validate_media(upload):
    name = Path(upload.name).name
    suffix = Path(name).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        image = validate_image(upload, int(getattr(settings, "FOOTBALL_MAX_IMAGE_BYTES", 15 * 1024**2)))
        return name[:255], "image", image["suffix"]
    if suffix in VIDEO_EXTENSIONS:
        maximum = int(getattr(settings, "FOOTBALL_MAX_VIDEO_BYTES", 200 * 1024**2))
        if not 0 < upload.size <= maximum:
            raise ValueError("视频为空或超出大小限制")
        return name[:255], "video", suffix
    raise ValueError("只接受常见图片或视频，不接受模型、程序或任意文件")


def directory(case):
    return Path(settings.FOOTBALL_DATA_DIR).resolve() / "var_reviews" / str(case.pk)


def artifact_path(case, key):
    base = directory(case).resolve()
    if key == "original":
        name = case.source_filename
    elif key.startswith("frame-"):
        try:
            identity = str(uuid.UUID(key[6:]))
        except (ValueError, TypeError):
            return None
        frame = next((item for item in case.keyframes if item.get("id") == identity), None)
        name = frame.get("filename") if frame else None
    else:
        return None
    if not isinstance(name, str):
        return None
    candidate = (base / name).resolve()
    if not candidate.is_relative_to(base) or candidate.parent != base or not candidate.is_file():
        return None
    if candidate.suffix.lower() not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
        return None
    return candidate


def _artifact_url(case, key):
    if artifact_path(case, key) is None:
        return None
    return reverse("football_var:artifact", kwargs={"case_id": case.pk, "key": key})


def serialize(case):
    keyframes = []
    for index, item in enumerate(case.keyframes):
        keyframes.append({key: item.get(key) for key in ("id", "media_time", "notes", "lines", "width", "height", "created_at")})
        keyframes[-1].update(sequence=index + 1, image_url=_artifact_url(case, "frame-" + item["id"]),
                             source="browser_snapshot", time_source="browser_display_time", calibrated=False)
    return {"id": str(case.pk), "title": case.title, "event_type": case.event_type,
            "event_label": EVENTS[case.event_type], "status": case.status, "input_name": case.input_name,
            "media_kind": case.media_kind, "duration_seconds": case.duration_seconds,
            "duration_source": "image_static" if case.media_kind == "image" else "browser_reported" if case.duration_seconds is not None else "unknown",
            "observations": case.observations, "verdict": case.verdict,
            "source_job_id": str(case.source_job_id) if case.source_job_id else None,
            "keyframes": keyframes, "original_url": _artifact_url(case, "original"),
            "report_urls": {"json": reverse("football_var:report-json", kwargs={"case_id": case.pk}),
                            "markdown": reverse("football_var:report-md", kwargs={"case_id": case.pk})},
            "source": "manual_review", "automated_decision": False, "calibration": False,
            "limitations": LIMITATIONS, "evidence_status": "has_keyframes" if keyframes else "needs_keyframes",
            "capabilities": {"automatic_offside": False, "automatic_foul": False,
                             "goal_line_detection": False, "automatic_card_recognition": False},
            "created_at": case.created_at.isoformat(), "updated_at": case.updated_at.isoformat()}


def report(case):
    payload = serialize(case)
    payload["schema_version"] = "football.var.manual-review.v1"
    payload["lines_note"] = "线条单独保存，没有烘焙进原始截图；未标定，不是官方越位线。"
    return payload


def report_markdown(case):
    payload = report(case)

    def clean(value):
        return html.escape(str(value), quote=False)

    parts = ["# VAR辅助复核记录（人工）", "", f"复核名称：{clean(case.title)}", f"事件类型：{EVENTS[case.event_type]}",
             f"记录状态：{case.status}", "结论来源：人工复核；自动判罚：否；场地标定：否。", "",
             "## 人工观察", "", clean(case.observations or "尚未填写"), "", "## 人工复核意见", "",
             clean(case.verdict or "待补充证据，尚无意见"), "", "## 关键帧证据", ""]
    if not payload["keyframes"]:
        parts.append("尚未保存关键帧。")
    for item in payload["keyframes"]:
        parts.extend([f"- 关键帧{item['sequence']}：浏览器显示时间 {item['media_time']:.3f} 秒；尺寸{item['width']}×{item['height']}。",
                      f"  人工备注：{clean(item['notes'] or '未填写')}；参考线{len(item['lines'])}条（未校准）。"])
    parts.extend(["", "## 使用限制", "", *[f"- {item}" for item in LIMITATIONS], "", payload["lines_note"], ""])
    return "\n".join(parts)


def _range_chunks(path, start, count):
    with path.open("rb") as stream:
        stream.seek(start)
        remaining = count
        while remaining:
            block = stream.read(min(64 * 1024, remaining))
            if not block:
                break
            remaining -= len(block)
            yield block


def file_response(request, path, filename):
    """Single byte ranges allow seeking without reading the complete video into RAM."""
    size = path.stat().st_size
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    header = request.headers.get("Range")
    start, end = 0, max(0, size - 1)
    if header:
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", header, re.IGNORECASE) if len(header) <= 200 else None
        valid = bool(match and size > 0 and (match[1] or match[2]))
        if valid:
            if match[1]:
                start = int(match[1])
                end = min(int(match[2]), size - 1) if match[2] else size - 1
                valid = start < size and start <= end
            else:
                length = int(match[2])
                valid = length > 0
                start, end = max(0, size - length), size - 1
        if not valid:
            response = HttpResponse(status=416)
            response["Content-Range"] = f"bytes */{size}"
            response["Content-Length"] = "0"
        else:
            response = HttpResponse(status=206, content_type=mime) if request.method == "HEAD" else StreamingHttpResponse(
                _range_chunks(path, start, end - start + 1), status=206, content_type=mime)
            response["Content-Range"] = f"bytes {start}-{end}/{size}"
            response["Content-Length"] = str(end - start + 1)
    elif request.method == "HEAD":
        response = HttpResponse(content_type=mime)
        response["Content-Length"] = str(size)
    else:
        response = FileResponse(path.open("rb"), content_type=mime)
    response["Accept-Ranges"] = "bytes"
    response["Content-Disposition"] = content_disposition_header(False, filename)
    response["Cache-Control"] = "private, no-store"
    response["X-Content-Type-Options"] = "nosniff"
    return response
