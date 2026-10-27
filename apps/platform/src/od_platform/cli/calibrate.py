#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : calibrate.py
# @Project   : ODPlatform
# @Function  : odp-calibrate —— 生成/校验球场标定（把像素与球场米对应起来）

"""odp-calibrate: 球场标定工具。

为什么需要它: 几何层的每一条结论（球员间距、出界、定位球）都以"米"为单位,
而"米"只有靠标定才能从像素里算出来。没有标定的系统只能输出像素, 不能输出
任何场上判断 —— 所以标定是几何层的第一道工序。

三种用法::

    # 1) 看看有哪些可用的关键点名字（表由 football.yaml 的 pitch 节推导）
    odp-calibrate --list-landmarks

    # 2) 用一帧图 + 你肉眼点出来的关键点生成标定
    odp-calibrate --image frame.jpg \
        --landmark penalty_area_left_top:512,402 \
        --landmark penalty_area_left_bottom:498,655 \
        --landmark halfway_bottom:960,604 \
        --landmark goal_left_post_bottom:371,470 \
        --save default

    # 3) 校验一份已有标定（可同时核对分辨率）
    odp-calibrate --verify default --image frame.jpg

退出码: 0=成功; 1=标定非法（点数不足/共线/重投影误差过大）; 2=用法错误。

★ 所有阈值与场地尺寸都来自 football.yaml（pitch / calibration 两节）,
  本文件里不出现任何几何常量。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from od_platform.common import paths
from od_platform.common.logging_utils import get_logger
from od_platform.geometric.calibration import (
    CALIBRATION_SUFFIX, CalibrationError, PitchCalibration, default_calibration_dir,
)
from od_platform.geometric.settings import GeometrySettings

EXIT_OK = 0
EXIT_CALIBRATION_ERR = 1
EXIT_USAGE = 2

logger = get_logger(base_path=paths.LOGGING_DIR, log_type="calibrate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odp-calibrate",
        description="生成或校验球场标定（像素 ↔ 球场米）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path,
                        help="football.yaml 路径; 默认 configs/runtime/football.yaml")
    parser.add_argument("--image", type=Path,
                        help="一帧画面, 用于记录分辨率并核对标定")
    parser.add_argument("--landmark", action="append", default=[],
                        metavar="NAME:X,Y",
                        help="具名关键点, 例如 penalty_area_left_top:512,402（可重复）")
    parser.add_argument("--image-size", metavar="W,H",
                        help="不提供 --image 时直接声明分辨率, 如 1920,1080")
    parser.add_argument("--save", metavar="NAME_OR_PATH",
                        help="保存标定; 只给名字时存到标定目录 <名字>.json")
    parser.add_argument("--verify", metavar="NAME_OR_PATH",
                        help="校验一份已有标定并打印质量报告")
    parser.add_argument("--notes", default="", help="备注（写进标定文件, 便于追溯机位）")
    parser.add_argument("--list-landmarks", action="store_true",
                        help="列出所有可用的场地关键点名称")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        settings = GeometrySettings.load(args.config)
    except (ValueError, OSError) as exc:
        logger.error("读取几何配置失败: %s", exc)
        return EXIT_USAGE

    if args.list_landmarks:
        _print_landmarks(settings)
        return EXIT_OK

    if args.verify:
        return _verify(args, settings)

    if not args.landmark:
        logger.error("至少要给 %d 个 --landmark（或用 --list-landmarks 查看可用名字）",
                     settings.calibration.min_points)
        return EXIT_USAGE

    try:
        points = _parse_landmarks(args.landmark)
        image_size = _resolve_image_size(args.image, args.image_size)
        calibration = PitchCalibration.from_landmarks(
            points, image_size=image_size, source="odp-calibrate", notes=args.notes,
            pitch=settings.pitch, settings=settings.calibration)
    except CalibrationError as exc:
        logger.error("标定不合法: %s", exc)
        return EXIT_CALIBRATION_ERR
    except ValueError as exc:
        logger.error("%s", exc)
        return EXIT_USAGE

    _report(calibration)
    if args.save:
        target = _resolve_target(args.save)
        calibration.save(target)
        logger.info("已保存标定: %s", target)
        logger.info("提示: 把它放在素材同目录并改名为 <素材名>%s, 或用 "
                    "football.yaml 的 homography.calibration_file 指定",
                    CALIBRATION_SUFFIX)
    return EXIT_OK


# ============================================================
# 实现
# ============================================================

def _print_landmarks(settings: GeometrySettings) -> None:
    pitch = settings.pitch
    print("可用场地关键点（名称 → 球场坐标 米）:")
    for name, (x, y) in sorted(pitch.landmarks().items()):
        print(f"  {name:<32} ({x:>7.2f}, {y:>7.2f})")
    print(f"\n场地尺寸: {pitch.length} x {pitch.width} m; "
          f"球门宽 {pitch.goal_width} m（可在 football.yaml 的 pitch 节覆盖）")


def _parse_landmarks(items: List[str]) -> Dict[str, Tuple[float, float]]:
    points: Dict[str, Tuple[float, float]] = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"--landmark 格式应为 NAME:X,Y, 收到 {item!r}")
        name, _, coords = item.partition(":")
        name = name.strip()
        parts = coords.split(",")
        if len(parts) != 2:
            raise ValueError(f"--landmark {name} 的坐标应为 X,Y, 收到 {coords!r}")
        try:
            points[name] = (float(parts[0]), float(parts[1]))
        except ValueError:
            raise ValueError(f"--landmark {name} 的坐标不是数字: {coords!r}") from None
    return points


def _resolve_image_size(image: Optional[Path],
                        declared: Optional[str]) -> Optional[Tuple[int, int]]:
    if image is not None:
        if not image.is_file():
            raise ValueError(f"图片不存在: {image}")
        import cv2
        import numpy as np
        decoded = cv2.imdecode(np.fromfile(str(image), dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f"图片无法解码: {image}")
        height, width = decoded.shape[:2]
        logger.info("参考帧 %s: %dx%d", image.name, width, height)
        return int(width), int(height)
    if declared:
        parts = declared.split(",")
        if len(parts) != 2:
            raise ValueError(f"--image-size 应为 W,H, 收到 {declared!r}")
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            raise ValueError(f"--image-size 不是整数: {declared!r}") from None
    return None


def _resolve_target(value: str) -> Path:
    path = Path(value)
    if path.suffix.lower() == ".json" or path.is_absolute() or len(path.parts) > 1:
        return path
    return default_calibration_dir() / f"{path.name}.json"


def _resolve_source(value: str) -> Path:
    candidates = [Path(value)]
    if Path(value).suffix.lower() != ".json":
        candidates.append(default_calibration_dir() / f"{value}.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise CalibrationError(f"找不到标定文件: {value}（已尝试 {[str(c) for c in candidates]}）")


def _verify(args: argparse.Namespace, settings: GeometrySettings) -> int:
    try:
        source = _resolve_source(args.verify)
        calibration = PitchCalibration.load(source, pitch=settings.pitch,
                                            settings=settings.calibration)
    except CalibrationError as exc:
        logger.error("%s", exc)
        return EXIT_CALIBRATION_ERR

    if args.image is not None:
        try:
            size = _resolve_image_size(args.image, None)
        except ValueError as exc:
            logger.error("%s", exc)
            return EXIT_USAGE
        if calibration.image_size and tuple(calibration.image_size) != tuple(size):
            logger.error("标定记录的分辨率 %s 与 %s 的实际分辨率 %s 不一致",
                         calibration.image_size, args.image.name, size)
            return EXIT_CALIBRATION_ERR

    logger.info("标定文件: %s", source)
    _report(calibration)
    return EXIT_OK


def _report(calibration: PitchCalibration) -> None:
    quality = calibration.quality()
    logger.info("标定点数: %d", quality["points"])
    logger.info("重投影误差: %.4f m", quality["reprojection_error_m"])
    logger.info("标定来源: %s", quality["source"])
    logger.info("记录分辨率: %s", quality["image_size"] or "未记录")
    if quality["full_frame_suspected"]:
        logger.warning("像素点就是整幅画面四角 —— 俯拍机位可能成立, "
                       "转播机位请确认是否点错了")
    logger.info("已按标定校验通过（几何事件可以产出）")


if __name__ == "__main__":
    sys.exit(main())