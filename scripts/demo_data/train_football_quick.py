"""快速训练入口（联调用，非最终模型）；路径、设备与参数均可配置。"""
import argparse
from pathlib import Path

from od_platform.common.paths import PRETRAINED_MODELS_DIR, RAW_DATA_DIR, TRAINED_MODELS_DIR


def main(argv=None, *, model_factory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=RAW_DATA_DIR / "soccer/football-yolo/data.yaml")
    parser.add_argument("--model", type=Path, default=PRETRAINED_MODELS_DIR / "yolo11n.pt")
    parser.add_argument("--project", type=Path, default=TRAINED_MODELS_DIR)
    parser.add_argument("--name", default="football_yolo")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="如cpu或0；未指定时使用Ultralytics默认设备")
    parser.add_argument("--exist-ok", action="store_true", help="明确允许复用同名实验目录")
    args = parser.parse_args(argv)
    if not args.data.is_file():
        parser.error("找不到数据YAML，请先转换数据或用--data指定已有配置")
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1 or args.workers < 0:
        parser.error("训练轮数、尺寸和batch必须为正数，workers不能为负数")
    if model_factory is None:
        from ultralytics import YOLO
        model_factory = YOLO
    model = model_factory(str(args.model.resolve()))
    options = {"data": str(args.data.resolve()), "epochs": args.epochs, "imgsz": args.imgsz,
               "batch": args.batch, "workers": args.workers, "project": str(args.project.resolve()),
               "name": args.name, "exist_ok": args.exist_ok, "verbose": True}
    if args.device is not None:
        options["device"] = args.device
    model.train(**options)
    best = Path(model.trainer.save_dir) / "weights/best.pt"
    if not best.is_file():
        raise RuntimeError(f"训练没有产出最佳权重: {best}")
    print(f"训练完成，最佳权重: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
