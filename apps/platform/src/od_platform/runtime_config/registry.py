"""配置注册表: "名字 → 配置类" 的唯一真相."""
from __future__ import annotations

from od_platform.runtime_config.infer import YOLOInferConfig
from od_platform.runtime_config.train import YOLOTrainConfig
from od_platform.runtime_config.val   import YOLOValConfig

# 名字 → (配置类, 模板标题). 加新模式 = 加一行.
CONFIG_REGISTRY: dict[str, tuple[type, str]] = {
    "train": (YOLOTrainConfig, "YOLO 训练配置"),
    "val":   (YOLOValConfig,   "YOLO 验证配置"),
    "infer": (YOLOInferConfig, "YOLO 推理配置"),
}

__all__ = ["CONFIG_REGISTRY"]
