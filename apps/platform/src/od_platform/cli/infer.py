"""odp-infer CLI 入口."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from od_platform.common.logging_utils import get_logger
from od_platform.inference import infer_yolo


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="odp-infer",
                                description="ODPlatform YOLO 推理 CLI")
    p.add_argument("--source", type=str, default=None,
                   help="输入源: 摄像头号/视频/图片/RTSP")
    p.add_argument("--model", type=str, default=None, help="模型文件")
    p.add_argument("--yaml", type=str, default=None, help="D5 infer.yaml 路径")
    p.add_argument("--pipeline-yaml", type=str, default=None,
                   help="帧源+美化 yaml 路径")
    p.add_argument("--name", type=str, default=None, dest="experiment_name",
                   help="输出子目录名")
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--iou", type=float, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--max-det", type=int, default=None, dest="max_det")
    p.add_argument("--classes", type=int, nargs="+", default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch", type=int, default=None)
    g_show = p.add_mutually_exclusive_group()
    g_show.add_argument("--show", action="store_true", default=None)
    g_show.add_argument("--no-show", dest="show", action="store_false")
    g_save = p.add_mutually_exclusive_group()
    g_save.add_argument("--save", action="store_true", default=None)
    g_save.add_argument("--no-save", dest="save", action="store_false")
    p.add_argument("--no-viz", action="store_true")
    p.add_argument("--no-hud", action="store_true")
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def _ns_to_cli_args(ns: argparse.Namespace) -> dict:
    keys = ("source", "model", "experiment_name",
            "conf", "iou", "imgsz", "max_det", "classes", "device", "batch",
            "show", "save")
    return {k: v for k in keys if (v := getattr(ns, k, None)) is not None}


def main(argv=None) -> int:
    ns = _build_parser().parse_args(argv)
    from od_platform.common.paths import LOGGING_DIR
    get_logger(base_path=LOGGING_DIR, log_type="infer",
               log_level=getattr(logging, ns.log_level))
    cli_args = _ns_to_cli_args(ns)
    result = infer_yolo(
        yaml_path=ns.yaml, pipeline_yaml=ns.pipeline_yaml, cli_args=cli_args,
        beautify=(not ns.no_viz), show_info=(not ns.no_hud),
        warmup_frames=ns.warmup)
    if result.success:
        return 0
    sys.stderr.write(f"\n推理失败: {result.error}\n")
    if result.log_path:
        sys.stderr.write(f"详细日志见: {result.log_path}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
