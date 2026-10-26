#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : views.py
# @Project   : ODPlatform / demo_ui
# @Function  : 两个视图（总览 / 推理演示）

"""演示站点的视图。

视图只做 HTTP 那点事（读请求、挑错误、渲染）; 能力汇总在 ``capabilities``、
检测与绘制在 ``engine`` —— 那两处不 import Django, 可以脱离 Web 直接测。
"""
from __future__ import annotations

import logging

from django.conf import settings
from django.shortcuts import render

from showcase import capabilities, engine

logger = logging.getLogger(__name__)

# 字节 → 兆字节的换算（写成常量, 免得 1024 散在算式里）
_BYTES_PER_MB = 1024 * 1024


def overview(request):
    """总览页: 如实展示四层能力的当前状态与参数。"""
    return render(request, "showcase/overview.html",
                  {"report": capabilities.build_report()})


def _upload_limits() -> dict:
    """上传限制的展示文案（模板不写数字）。"""
    limit_bytes = settings.DEMO_UPLOAD_MAX_BYTES
    return {
        "max_mb": round(limit_bytes / _BYTES_PER_MB, 1),
        "suffixes": "、".join(engine.demo_config.ALLOWED_IMAGE_SUFFIXES),
    }


def _read_upload(request) -> bytes:
    """取出并校验上传的图片字节。

    Raises:
        engine.UploadError: 没选文件 / 后缀不在白名单 / 超过大小上限。
    """
    upload = request.FILES.get("image")
    if upload is None:
        raise engine.UploadError("请先选择一张图片, 再点开始检测")
    if not engine.suffix_allowed(upload.name):
        allowed = "、".join(engine.demo_config.ALLOWED_IMAGE_SUFFIXES)
        raise engine.UploadError(f"只支持这几种格式: {allowed}")
    if upload.size > settings.DEMO_UPLOAD_MAX_BYTES:
        size_mb = round(upload.size / _BYTES_PER_MB, 1)
        limit_mb = round(settings.DEMO_UPLOAD_MAX_BYTES / _BYTES_PER_MB, 1)
        raise engine.UploadError(f"图片 {size_mb} MB, 超过 {limit_mb} MB 上限")
    return upload.read()


def infer(request):
    """推理演示页: GET 出表单, POST 出结果。

    任何一种失败都渲染回本页并显示原因 —— 演示现场不该出现 500 白页。
    """
    context = {"result": None, "error": None, "limits": _upload_limits(),
               "device": settings.DEMO_DEVICE}

    if request.method == "POST":
        try:
            payload = _read_upload(request)
            image = engine.decode_image(payload)
            context["result"] = engine.analyse(image, device=settings.DEMO_DEVICE)
        except engine.UploadError as exc:
            context["error"] = str(exc)
        except Exception as exc:  # noqa: BLE001 —— 权重缺失/显存不足等都要落回页面
            logger.exception("推理演示失败")
            context["error"] = f"推理过程出错: {exc}"

    return render(request, "showcase/infer.html", context)