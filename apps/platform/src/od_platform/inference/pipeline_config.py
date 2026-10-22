"""帧源捕获 + 美化 的配置读取 helper."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _to_bgr_tuple(value: Any) -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return (int(value[0]), int(value[1]), int(value[2]))
    raise ValueError(f"颜色必须是 3 元素的 [B, G, R], 收到: {value!r}")


@dataclass
class PipelineConfig:
    camera_raw: dict[str, Any] = field(default_factory=dict)
    viz_enabled: bool = True
    use_label_mapping: bool = True
    label_mapping: dict[str, str] = field(default_factory=dict)
    color_mapping: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    default_color: tuple[int, int, int] = (0, 255, 0)
    font_path: str | None = None
    style_overrides: dict[str, Any] = field(default_factory=dict)

    def build_camera_config(self):
        from od_platform.frame_source import CameraConfig
        if not self.camera_raw:
            return None
        try:
            return CameraConfig(**self.camera_raw)
        except Exception as e:
            logger.warning(f"camera 配置无效, 走默认: {e}")
            return None

    def to_audit(self) -> dict[str, Any]:
        return {
            "viz_enabled": self.viz_enabled,
            "use_label_mapping": self.use_label_mapping,
            "label_mapping_n": len(self.label_mapping),
            "color_mapping_n": len(self.color_mapping),
            "default_color": list(self.default_color),
            "font_path": self.font_path,
            "camera": dict(self.camera_raw),
            "style_overrides_n": len(self.style_overrides),
        }


def load_pipeline_config(yaml_path: str | Path | None) -> PipelineConfig:
    if yaml_path is None:
        yaml_path = Path("configs/runtime/infer_pipeline.yaml")
    else:
        yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        logger.warning(f"pipeline yaml 不存在, 用默认配置: {yaml_path}")
        return PipelineConfig()

    try:
        with yaml_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"pipeline yaml 解析失败, 用默认配置: {e}")
        return PipelineConfig()

    pc = PipelineConfig()
    fs = raw.get("frame_source", {}) or {}
    pc.camera_raw = fs.get("camera", {}) or {}

    vz = raw.get("visualization", {}) or {}
    pc.viz_enabled = bool(vz.get("enabled", True))
    pc.use_label_mapping = bool(vz.get("use_label_mapping", True))
    pc.label_mapping = dict(vz.get("label_mapping", {}) or {})

    raw_colors = vz.get("color_mapping", {}) or {}
    pc.color_mapping = {k: _to_bgr_tuple(v) for k, v in raw_colors.items()}

    if "default_color" in vz:
        pc.default_color = _to_bgr_tuple(vz["default_color"])

    pc.font_path = vz.get("font_path")
    pc.style_overrides = dict(vz.get("style", {}) or {})

    logger.info(f"pipeline 配置加载: {yaml_path}, "
                f"美化={pc.viz_enabled}, 标签映射={len(pc.label_mapping)}, "
                f"颜色映射={len(pc.color_mapping)}")
    return pc
