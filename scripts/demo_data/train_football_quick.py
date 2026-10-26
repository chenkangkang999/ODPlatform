"""快速训练入口（联调用）：超参只来自 configs/runtime/train.yaml。

与 ``odp-train`` 的分工
    ``odp-train``      : 走 D1-D6 全链路, 开训前做【数据身份核验】—— 数据集必须有
                         D3(odp-transform) 产出的 manifest, 否则直接拒绝开训
                         （见 data_validation/identity.py）。原始/自备数据集过不了。
    本脚本             : 给"还没走 D3 的数据集"用的快速通道。**只省掉核验**,
                         超参仍然只有 train.yaml 一个来源 —— 本文件不写死任何超参。

为什么不再自己带一套参数
    历史上这里的默认值与 football.yaml、train.yaml 各写一份, 结果"改了配置不生效"。
    现在临时覆盖只开放最常用的几个（--epochs/--imgsz/--batch/--workers/--device）,
    其余一律去改 ``configs/runtime/train.yaml``（CLI > YAML > DEFAULT）。

用法::

    python scripts/demo_data/train_football_quick.py                 # 用 train.yaml 的全部超参
    python scripts/demo_data/train_football_quick.py --epochs 1      # 临时覆盖轮数（冒烟）
"""
from __future__ import annotations

import argparse
from pathlib import Path

from od_platform.common.config_utils import parse_batch_arg
from od_platform.common.paths import (
    FOOTBALL_DATA_YAML,
    PRETRAINED_MODELS_DIR,
    RUNTIME_CONFIGS_DIR,
    TRAINED_MODELS_DIR,
)
from od_platform.runtime_config import build_train_config
from od_platform.training.autobatch import resolve_batch

DEFAULT_PRESET: Path = RUNTIME_CONFIGS_DIR / "train.yaml"

# 只开放这几个临时覆盖; 其余超参改 train.yaml, 避免同一份参数有两个写处。
_OVERRIDE_KEYS = ("epochs", "imgsz", "batch", "workers", "device")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", type=Path, default=DEFAULT_PRESET,
                        help="训练配置（默认 configs/runtime/train.yaml）")
    parser.add_argument("--data", type=Path, default=FOOTBALL_DATA_YAML,
                        help="数据集 yaml（默认足球检测数据集）")
    parser.add_argument("--model", type=Path, default=PRETRAINED_MODELS_DIR / "yolo11n.pt")
    parser.add_argument("--project", type=Path, default=TRAINED_MODELS_DIR)
    parser.add_argument("--name", default="football_yolo")
    parser.add_argument("--exist-ok", action="store_true", help="明确允许复用同名实验目录")
    for key, kind, help_text in (
        ("epochs", int, "临时覆盖轮数"), ("imgsz", int, "临时覆盖输入尺寸"),
        ("batch", parse_batch_arg, "临时覆盖批大小（整数, 或 (0,1] 的显存利用率, -1=自动）"),
        ("workers", int, "临时覆盖加载线程数"),
        ("device", str, "cpu / 0 / 0,1"),
    ):
        parser.add_argument(f"--{key}", type=kind, default=None, help=help_text)
    return parser


def main(argv=None, *, model_factory=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.data.is_file():
        parser.error("找不到数据YAML，请先转换数据或用--data指定已有配置")
    if not args.preset.is_file():
        parser.error(f"找不到训练配置: {args.preset}")

    overrides = {key: getattr(args, key) for key in _OVERRIDE_KEYS
                 if getattr(args, key) is not None}
    config, _merger = build_train_config(yaml_path=str(args.preset), cli_args=overrides or None)

    options = config.to_ultralytics_kwargs()
    options.pop("model", None)
    options.pop("data", None)
    # batch=-1 表示"按机器自动": 用本仓库的估算规则解析（ultralytics 的 AutoBatch
    # 会按 2×imgsz 探测显存, 8GB 消费级卡在 imgsz=1280 时直接 OOM）。
    options["batch"] = resolve_batch(options.get("batch"), options.get("imgsz", 640),
                                     options.get("device"))
    options.update(project=str(args.project.resolve()), name=args.name, exist_ok=args.exist_ok)

    if model_factory is None:
        from ultralytics import YOLO
        model_factory = YOLO
    model = model_factory(str(args.model.resolve()))
    model.train(data=str(args.data.resolve()), **options)

    best = Path(model.trainer.save_dir) / "weights/best.pt"
    if not best.is_file():
        raise RuntimeError(f"训练没有产出最佳权重: {best}")
    print(f"训练完成，最佳权重: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())