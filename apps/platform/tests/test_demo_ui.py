#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""演示站点（apps/platform/demo_ui）的集成测试。

两处接入动作, 别的什么都不用改:

  1. 把 ``apps/platform/demo_ui`` 插到 sys.path —— 演示站点是独立 Django 工程,
     不在已安装的包里, 所以测试要自己把它挂上;
  2. ``DJANGO_SETTINGS_MODULE`` 指向它自己的 ``demo_site.settings``。

★ 不做真实推理。检测那一层用 ``engine.build_ultralytics_detector`` 的注入点换掉
  （与队友的 ``worker_checks/check_bridge.py`` 用 ``detector=`` 注入是同一手法）,
  于是测试既不需要加载 torch, 也不需要那份权重。绘制走的仍然是真实路径 ——
  被测的是"页面 + 绘制 + 统计", 只有模型被替掉。
"""
from __future__ import annotations

import csv
import os
import sys
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

# ---------------------------------------------------------------- 演示工程接入点
DEMO_ROOT = Path(__file__).resolve().parents[1] / "demo_ui"
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "demo_site.settings")

import django  # noqa: E402  —— 必须在 sys.path 挂好之后才能 import

django.setup()

from django.core.files.uploadedfile import SimpleUploadedFile  # noqa: E402
from django.test import Client  # noqa: E402

import manage  # noqa: E402  —— 演示工程自己的入口（测 runserver 默认端口）
from showcase import capabilities, demo_config, engine  # noqa: E402


# ============================================================
# 夹具
# ============================================================

@pytest.fixture()
def client():
    """Django 测试客户端（ALLOWED_HOSTS 里已含 testserver）。"""
    return Client()


def image_bytes(size=(64, 64), color="green") -> bytes:
    """自造一张小图 —— 不依赖磁盘上的任何真实数据集。"""
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


def upload(payload: bytes, name="frame.jpg") -> SimpleUploadedFile:
    return SimpleUploadedFile(name, payload, content_type="image/jpeg")


@pytest.fixture()
def fake_detector(monkeypatch):
    """替掉真实检测器: 返回 3 个目标（2 球员 + 1 足球）。"""
    def detector(image):
        return [
            {"cls": "player", "conf": 0.91, "xyxy": [5, 5, 30, 45]},
            {"cls": "player", "conf": 0.83, "xyxy": [35, 5, 58, 45]},
            {"cls": "football", "conf": 0.72, "xyxy": [20, 48, 28, 56]},
        ]

    monkeypatch.setattr(engine, "build_ultralytics_detector", lambda *a, **k: detector)
    return detector


# ============================================================
# 页面 1: 总览
# ============================================================

def test_overview_returns_200(client):
    assert client.get("/").status_code == 200


def test_overview_states_capabilities_honestly(client):
    """答辩会被追问的四个点, 页面上必须都能看到。"""
    html = client.get("/").content.decode("utf-8")
    # ① 检测权重只训了 3 轮 → 必须标成冒烟级, 不能当可用模型
    assert "冒烟级" in html
    # ② BoT-SORT 跟踪已接入, 且参数文件位置可见
    assert "BoT-SORT" in html
    assert "botsort.yaml" in html
    # ③ 场地标定: 磁盘上没有标定文件 → 必须显示"未标定", 并说明几何事件不产出
    assert "未标定" in html
    assert "几何事件" in html
    # ④ 动作识别: 只有 VideoMAE 已实现, SlowFast 必须写成"没有实现"
    assert "VideoMAE" in html
    assert "SlowFast 没有实现" in html
    assert "Kinetics-400" in html


def test_overview_shows_training_metrics_read_from_artifacts(client):
    """指标与轮数是从 results.csv 读出来的, 不是写死在模板里。"""
    html = client.get("/").content.decode("utf-8")
    summary = capabilities.training_summary()
    assert "mAP@50" in html
    if not summary.get("available"):
        pytest.skip(f"训练产物不在本机: {summary.get('reason')}")
    assert f"{summary['epochs_done']}" in html


def test_overview_shows_no_detection_parameters_hardcoded(client):
    """页面上出现的 imgsz/conf 必须等于配置层的值（改了配置页面就跟着变）。"""
    from od_platform.detection.settings import DetectionSettings

    detection = DetectionSettings.from_app_config(capabilities.load_config())
    html = client.get("/").content.decode("utf-8")
    assert str(detection.imgsz) in html
    assert str(detection.conf) in html


# ============================================================
# 页面 2: 推理演示
# ============================================================

def test_infer_get_renders_the_upload_form(client):
    response = client.get("/infer/")
    assert response.status_code == 200
    html = response.content.decode("utf-8")
    assert 'name="image"' in html
    assert "开始检测" in html


def test_infer_post_runs_detection_and_shows_result(client, fake_detector):
    response = client.post("/infer/", {"image": upload(image_bytes())})
    assert response.status_code == 200
    html = response.content.decode("utf-8")
    assert 'id="detection-result"' in html
    assert "检出 3 个目标" in html
    # 每类计数表（中文显示名来自本演示的展示映射）
    assert "球员" in html and "足球" in html
    # 原图与带框图都以 data URI 内联
    assert html.count("data:image/jpeg;base64,") == 2


def test_infer_draws_boxes_through_the_real_visualizer(client, fake_detector):
    """页面上的带框图必须真的画了框（visualization 此前是桩, 原图直出）。

    这里不去读 ``response.context``: 那需要 Django 的测试环境instrumentation,
    在纯 pytest 下是 None。改成直接从渲染出来的 HTML 里取两个内联图比较 ——
    测的正是"用户能看到什么"。
    """
    import re

    response = client.post("/infer/", {"image": upload(image_bytes())})
    assert response.status_code == 200
    html = response.content.decode("utf-8")
    rendered = re.findall(r'src="(data:image/jpeg;base64,[^"]+)"', html)
    assert len(rendered) == 2, "应当同时有原图与带框图"
    assert rendered[0] != rendered[1], "两张图一模一样 = 绘制又退回桩了"


def test_infer_counts_use_display_names(client, fake_detector):
    """页面表格用中文显示名; 机器可读的原始类别名另外给一份。"""
    from showcase import engine as engine_module

    import numpy as np

    result = engine_module.analyse(np.zeros((64, 64, 3), dtype=np.uint8),
                                   detector=fake_detector, settings=None)
    assert result["counts"] == {"球员": 2, "足球": 1}
    assert result["counts_raw"] == {"player": 2, "football": 1}
    assert result["total"] == 3


def test_infer_post_with_zero_detections_is_not_an_error(client, monkeypatch):
    monkeypatch.setattr(engine, "build_ultralytics_detector", lambda *a, **k: (lambda image: []))
    response = client.post("/infer/", {"image": upload(image_bytes())})
    assert response.status_code == 200
    html = response.content.decode("utf-8")
    assert "没有检测到任何目标" in html
    assert "没能处理这张图" not in html          # 零检测不是错误


def test_infer_without_file_reports_a_friendly_error(client):
    response = client.post("/infer/", {})
    assert response.status_code == 200
    assert "请先选择一张图片" in response.content.decode("utf-8")


def test_infer_rejects_unsupported_suffix(client):
    response = client.post("/infer/", {"image": upload(b"whatever", name="notes.txt")})
    assert response.status_code == 200
    assert "只支持这几种格式" in response.content.decode("utf-8")


def test_infer_rejects_undecodable_payload(client):
    response = client.post("/infer/", {"image": upload(b"this is not an image", name="broken.jpg")})
    assert response.status_code == 200
    assert "解不开" in response.content.decode("utf-8")


def test_infer_rejects_oversized_upload(client, monkeypatch):
    """超限要在"解码/推理"之前被挡住。"""
    monkeypatch.setattr(
        "django.conf.settings.DEMO_UPLOAD_MAX_BYTES", 16, raising=False)
    response = client.post("/infer/", {"image": upload(image_bytes((256, 256)))})
    assert response.status_code == 200
    assert "超过" in response.content.decode("utf-8")


# ============================================================
# 支撑逻辑（脱离 Web 直接测）
# ============================================================

def test_calibration_status_is_honest_about_absence():
    status = capabilities.calibration_status({})
    assert status["calibrated"] is False
    assert "没有任何标定" in status["reason"]


def test_calibration_status_accepts_inline_points():
    status = capabilities.calibration_status(
        {"homography": {"points": [{"landmark": "centre", "pixel": [1, 2]}]}})
    assert status["calibrated"] is True
    assert "内联" in status["reason"]


def test_training_summary_reads_columns_by_name(tmp_path):
    """按列名取值 —— ultralytics 增删列也不会读错。"""
    run = tmp_path / "some_run"
    run.mkdir()
    header = ["epoch", "time", "metrics/precision(B)", "metrics/recall(B)",
              "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    with (run / demo_config.TRAIN_RESULTS_FILENAME).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerow([1, 60, 0.4, 0.3, 0.35, 0.18])
        writer.writerow([2, 120, 0.8, 0.5, 0.56, 0.30])
    (run / demo_config.TRAIN_ARGS_FILENAME).write_text(
        "epochs: 2\nimgsz: 1280\nbatch: 4\ndevice: '0'\nmodel: yolo11n.pt\n", encoding="utf-8")

    summary = capabilities.training_summary(run)
    assert summary["epochs_done"] == 2
    assert summary["precision"] == 0.8 and summary["map50"] == 0.56
    assert summary["planned_epochs"] == 2 and summary["imgsz"] == 1280
    assert summary["is_smoke"] is True            # 2 轮 ≤ 冒烟阈值
    assert summary["weights_exists"] is False     # 假的 run 里没有权重


def test_training_summary_reports_missing_artifacts_instead_of_inventing(tmp_path):
    summary = capabilities.training_summary(tmp_path / "nothing_here")
    assert summary["available"] is False
    assert "没有训练记录文件" in summary["reason"]


def test_smoke_threshold_flags_the_real_three_epoch_run():
    """真机上那份 3 轮权重必须被判为冒烟级（这是页面上最要紧的一句实话）。"""
    summary = capabilities.training_summary()
    if not summary.get("available"):
        pytest.skip("本机没有训练产物")
    assert summary["is_smoke"] is True
    assert summary["epochs_done"] <= demo_config.SMOKE_EPOCH_THRESHOLD


def test_render_changes_pixels_and_returns_a_data_uri():
    import numpy as np

    canvas = np.zeros((80, 120, 3), dtype=np.uint8)
    drawn = engine.render(canvas, [{"cls": "player", "conf": 0.9, "xyxy": [10, 10, 60, 70]}])
    assert drawn.shape == canvas.shape
    assert drawn.any(), "画完之后不该还是全黑"
    assert not canvas.any(), "原图必须保持不动（页面要同时显示两张）"
    assert engine.to_data_uri(drawn).startswith("data:image/jpeg;base64,")


def test_manage_defaults_runserver_to_the_demo_port():
    """runserver 不带地址时补 8100 —— 免得顺手占了 Django 默认的 8000。"""
    assert manage._with_default_addr(["manage.py", "runserver"]) == [
        "manage.py", "runserver", f"127.0.0.1:{demo_config.DEMO_PORT}"]
    # 用户显式给了地址就一个字不动
    assert manage._with_default_addr(["manage.py", "runserver", "0.0.0.0:9000"]) == [
        "manage.py", "runserver", "0.0.0.0:9000"]
    # 非 runserver 的命令不受影响
    assert manage._with_default_addr(["manage.py", "check"]) == ["manage.py", "check"]


def test_demo_port_avoids_the_neighbours():
    """8000 是 Django 默认, 8001 是队友的足球门户。"""
    assert demo_config.DEMO_PORT not in (8000, 8001)