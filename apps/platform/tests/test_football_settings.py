#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""检测层 / 动作识别层的配置化测试。

这一组测试专门守"配置写了但不生效"这类事故:
    · football.yaml 的键名必须与 dataclass 字段对得上（拼错要能被发现）;
    · 默认值只在 settings 里声明一次, request.json 能逐项覆盖且会重新校验;
    · 被中途废弃的旧键名（conf_threshold / iou_threshold / num_classes 等）
      不许再出现在 yaml 里 —— 它们曾经让人误以为改了配置就生效。
"""
from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import pytest

from od_platform.action_recognition.classifier import ActionRecognizer
from od_platform.action_recognition.settings import (
    PLANNED_FAMILIES,
    SUPPORTED_FAMILIES,
    ActionRecognitionSettings,
)
from od_platform.common import paths
from od_platform.common.config_utils import override_settings
from od_platform.common.football_config import (
    CONFIG_NAME,
    coerce_config,
    default_config_path,
    load_config,
)
from od_platform.detection.settings import DetectionSettings

REAL_CONFIG = paths.RUNTIME_CONFIGS_DIR / CONFIG_NAME


# ============================================================
# 单一来源: 配置路径
# ============================================================

def test_config_locator_is_defined_once():
    """四个子系统必须指向同一个 football.yaml。"""
    from od_platform.geometric.settings import default_config_path as geometric_path

    assert default_config_path() == REAL_CONFIG
    assert geometric_path() == REAL_CONFIG
    assert REAL_CONFIG.is_file(), "football.yaml 应当存在; 不存在则本组测试失去意义"


def test_coerce_config_accepts_content_path_or_nothing(tmp_path):
    assert coerce_config({"detection": {"conf": 0.5}}) == {"detection": {"conf": 0.5}}
    assert coerce_config(str(REAL_CONFIG))["detection"]["conf"] == 0.25
    assert coerce_config(None)["detection"]["conf"] == 0.25
    assert coerce_config(tmp_path / "missing.yaml") == {}
    broken = tmp_path / "broken.yaml"
    broken.write_text("just a string", encoding="utf-8")
    assert coerce_config(broken) == {}


def test_load_config_tolerates_missing_and_broken_files(tmp_path):
    assert load_config(tmp_path / "nope.yaml") == {}
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [unclosed", encoding="utf-8")
    assert load_config(bad) == {}


# ============================================================
# DetectionSettings
# ============================================================

def test_detection_defaults_live_in_the_dataclass():
    settings = DetectionSettings()
    assert (settings.conf, settings.iou, settings.imgsz) == (0.25, 0.7, 960)
    assert (settings.max_det, settings.stride, settings.max_frames) == (300, 1, 1500)
    assert settings.tracking is True and settings.classes == (0, 1)


def test_real_config_matches_the_dataclass_without_unknown_keys(caplog):
    """这条最要紧: yaml 的键名必须真的能被 dataclass 接住（拼错就会在这里炸）。"""
    with caplog.at_level(logging.WARNING):
        settings = DetectionSettings.load()
    assert "无法识别的配置键" not in caplog.text, caplog.text
    # 与 football.yaml 里写的值一一对应（改了 yaml 就要改这里, 逼着人保持同步）
    # 注意与 dataclass 默认值(960)不同: 本项目配置把推理尺寸调到了 1280 以匹配训练。
    assert (settings.conf, settings.iou, settings.imgsz) == (0.25, 0.7, 1280)
    assert (settings.max_det, settings.stride, settings.max_frames) == (300, 1, 1500)
    assert settings.tracker == "botsort.yaml"


def test_dead_key_names_are_gone_from_the_config():
    """旧的死键名不许复活 —— 它们曾经让 iou=0.7 的硬编码被 0.45 的配置掩盖。"""
    section = load_config()["detection"]
    for dead in ("conf_threshold", "iou_threshold", "num_classes"):
        assert dead not in section, f"{dead} 已废弃, 别再写进配置"


def test_detection_accepts_partial_config_and_ignores_unknown(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        settings = DetectionSettings.from_config({"detection": {"conf": 0.4, "typo_key": 1}})
    assert settings.conf == 0.4
    assert settings.iou == DetectionSettings().iou       # 未给的字段走默认值
    assert "typo_key" in caplog.text                     # 但拼错的键要被报出来


@pytest.mark.parametrize("kwargs", [
    {"conf": 0.0}, {"conf": 1.0}, {"iou": 0.0}, {"iou": 1.2},
    {"imgsz": 300}, {"imgsz": 1000}, {"stride": 0}, {"stride": 31},
    {"max_frames": 0}, {"max_det": 0}, {"tracker": ""},
])
def test_detection_rejects_out_of_range_values(kwargs):
    with pytest.raises(ValueError):
        DetectionSettings(**kwargs)


def test_detection_from_app_config_accepts_section_or_whole_file():
    whole = load_config()
    assert DetectionSettings.from_app_config(whole).conf == 0.25
    assert DetectionSettings.from_app_config(whole["detection"]).conf == 0.25
    assert DetectionSettings.from_app_config(None).conf == 0.25


def test_resolved_tracker_prefers_the_in_repo_file():
    """本仓库放的 botsort.yaml 要优先于 ultralytics 内置的那份。"""
    settings = DetectionSettings()
    resolved = Path(settings.resolved_tracker())
    assert resolved.is_file() and resolved.parent == paths.RUNTIME_CONFIGS_DIR
    assert resolved.name == "botsort.yaml"
    # 名字在本仓库找不到时原样交给 ultralytics, 不能拼出一个不存在的绝对路径
    assert replace(settings, tracker="bytetrack.yaml").resolved_tracker() == "bytetrack.yaml"


def test_in_repo_botsort_file_carries_the_expected_parameters():
    import yaml

    data = yaml.safe_load((paths.RUNTIME_CONFIGS_DIR / "botsort.yaml").read_text(encoding="utf-8"))
    assert data["tracker_type"] == "botsort"
    assert data["track_high_thresh"] == 0.25 and data["track_low_thresh"] == 0.1
    assert data["track_buffer"] == 30 and data["match_thresh"] == 0.8
    assert data["gmc_method"] == "sparseOptFlow" and data["with_reid"] is False


# ============================================================
# override_settings（运行期覆盖）
# ============================================================

def test_override_applies_only_matching_keys():
    base = DetectionSettings()
    other = override_settings(base, {"conf": 0.5, "input_path": "/tmp/x.mp4",
                                     "output_dir": "/tmp/out", "unknown": 1})
    assert other.conf == 0.5
    assert other.iou == base.iou
    assert not hasattr(other, "input_path")


def test_override_skips_none_and_coerces_types():
    other = override_settings(DetectionSettings(), {"conf": "0.4", "imgsz": "512", "device": None})
    assert other.conf == 0.4 and isinstance(other.conf, float)
    assert other.imgsz == 512 and isinstance(other.imgsz, int)
    assert other.device == "cpu"                       # None 不算覆盖


def test_override_returns_same_instance_when_nothing_changes():
    base = DetectionSettings()
    assert override_settings(base, {}) is base
    assert override_settings(base, None) is base
    assert override_settings(base, {"nope": 1}) is base


def test_override_revalidates_so_bad_request_values_are_rejected():
    with pytest.raises(ValueError):
        override_settings(DetectionSettings(), {"conf": 5})
    with pytest.raises(ValueError):
        override_settings(DetectionSettings(), {"imgsz": 333})


def test_worker_style_flow_reads_yaml_then_request(monkeypatch):
    """模拟 Worker 的装配顺序: 先 football.yaml, 再 request.json 覆盖。"""
    request = {"conf": 0.31, "imgsz": 512, "stride": 2, "max_frames": 40, "device": "0",
               "input_path": "x.mp4", "output_dir": "out", "model_path": "w.pt"}
    params = override_settings(DetectionSettings.from_app_config(coerce_config(None)), request)
    assert (params.conf, params.imgsz, params.stride) == (0.31, 512, 2)
    assert (params.max_frames, params.device) == (40, "0")
    assert params.iou == 0.7          # 没传就用配置里的默认, 不再是硬编码


# ============================================================
# ActionRecognitionSettings
# ============================================================

def test_action_defaults_and_capability_flags():
    settings = ActionRecognitionSettings()
    assert (settings.model_name, settings.clip_length) == ("videomae", 16)
    assert (settings.confidence_threshold, settings.top_k) == (0.3, 5)
    assert settings.implemented is True and settings.family == "videomae"


def test_action_real_config_matches_the_dataclass_without_unknown_keys(caplog):
    with caplog.at_level(logging.WARNING):
        settings = ActionRecognitionSettings.load()
    assert "无法识别的配置键" not in caplog.text, caplog.text
    assert settings.model_name == "videomae"
    assert settings.clip_length == 16 and settings.confidence_threshold == 0.3
    assert settings.checkpoint.endswith("videomae-base-kinetics")


def test_action_config_no_longer_claims_a_wrong_class_count():
    """num_classes 曾经写 10, 而加载的模型是 400 类 —— 这类谎话不许回到配置里。"""
    assert "num_classes" not in load_config()["action_recognition"]


def test_videomae_v2_label_still_dispatches_to_the_implemented_family():
    """历史配置里的 videomae_v2_base 仍要能跑（按前缀分派）。"""
    settings = ActionRecognitionSettings.from_config(
        {"action_recognition": {"model_name": "videomae_v2_base"}})
    assert settings.family == "videomae" and settings.implemented is True


@pytest.mark.parametrize("name", PLANNED_FAMILIES)
def test_planned_families_are_explicitly_not_implemented(name):
    """PLANNED_FAMILIES 里的预留位必须在代码层面表现为"未实现", 不能假装支持。"""
    settings = ActionRecognitionSettings.from_config(
        {"action_recognition": {"model_name": name, "checkpoint": "whatever"}})
    assert settings.family == "" and settings.implemented is False
    recognizer = ActionRecognizer(settings=settings)
    assert recognizer.ready is False
    assert recognizer.num_classes == 0


@pytest.mark.parametrize("kwargs", [
    {"model_name": ""}, {"clip_length": 0}, {"top_k": 0},
    {"confidence_threshold": -0.1}, {"confidence_threshold": 1.5},
])
def test_action_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        ActionRecognitionSettings(**kwargs)


def test_recognizer_without_checkpoint_degrades_quietly():
    recognizer = ActionRecognizer(settings=ActionRecognitionSettings(checkpoint=""))
    assert recognizer.ready is False
    assert recognizer.predict(None) == []
    assert recognizer.num_classes == 0


def test_recognizer_accepts_legacy_dict_and_new_settings_equally():
    section = {"checkpoint": "", "clip_length": 8, "confidence_threshold": 0.45, "top_k": 3}
    from_dict = ActionRecognizer(section)
    from_settings = ActionRecognizer(settings=ActionRecognitionSettings.from_app_config(section))
    assert from_dict.settings == from_settings.settings
    assert from_dict.clip_length == 8 and from_dict.conf_threshold == 0.45


def test_recognizer_audit_reports_honest_capability():
    audit = ActionRecognizer(settings=ActionRecognitionSettings(checkpoint="")).to_audit()
    assert audit["implemented"] is True and audit["ready"] is False
    assert audit["num_classes"] == 0 and audit["top_k"] == 5


def test_supported_families_are_the_ones_actually_implemented():
    assert SUPPORTED_FAMILIES == ("videomae",)
    assert set(PLANNED_FAMILIES).isdisjoint(SUPPORTED_FAMILIES)