"""Web推理子进程：只读取可信权重，不训练，不修改ODPlatform。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from football_portal.results import CLASSES, build_analysis, normalize_frame
from football_portal.annotations import ROLE_COLORS, STYLE_VERSION, render_bgr

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SAFE_METADATA = ("run_id", "class_names", "data_fingerprint", "content_fingerprint", "contract_fingerprint",
                 "model_version", "development_only", "purpose", "execution", "dataset_limitations")


def dump(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    # Windows briefly denies replace while the Web process reads progress.json.
    # Keep writes atomic and allow that short-lived reader to close its handle.
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.02 * (attempt + 1))


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _reader(path: Path, stride: int, platform_root: str):
    source = None
    root = Path(platform_root) / "apps/platform/src" if platform_root else None
    if root and root.is_dir():
        sys.path.insert(0, str(root))
        try:
            from od_platform.frame_source import create_frame_source
            # Read all frames to validate the complete file. Apply sampling here:
            # upstream stride>1 can misclassify EOF after a final skipped frame.
            source = create_frame_source(str(path), stride=1)
        except ImportError:
            source = None
    if source is not None:
        with source:
            for frame in source:
                info = frame.info
                if int(info.frame_index) % stride:
                    continue
                fps = getattr(info, "fps", None)
                yield frame.image, {"frame_index": int(info.frame_index),
                    "timestamp": float(info.timestamp) if fps and fps > 0 else None,
                    "fps": float(fps) if fps and fps > 0 else None,
                    "total_frames": getattr(info, "total_frames", None), "reader": "ODPlatform.FrameSource"}
            reason = str(source.end_reason).lower()
            if "error" in reason or "not_started" in reason:
                raise ValueError("ODPlatform帧源未能完整解码输入，请检查素材")
        return
    import cv2
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        import numpy as np
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("图片无法解码")
        yield image, {"frame_index": 0, "timestamp": None, "fps": None, "total_frames": 1,
                      "reader": "opencv_compatible_adapter"}
        return
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("视频无法打开")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("视频没有可靠帧率，不能生成媒体时间")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        index = 0
        while True:
            ok, image = capture.read()
            if not ok:
                if total > 0 and index < total:
                    raise ValueError("视频提前解码终止，结果不完整")
                break
            if index % stride == 0:
                yield image, {"frame_index": index, "timestamp": index / fps, "fps": fps,
                              "total_frames": total, "reader": "opencv_compatible_adapter"}
            index += 1
    finally:
        capture.release()


def run_request(request: dict, *, detector=None) -> dict:
    import cv2

    path, output, weight = (Path(request[key]).resolve() for key in ("input_path", "output_dir", "model_path"))
    if not path.is_file() or not weight.is_file():
        raise ValueError("输入素材或配置的可信模型不存在")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError("结果已存在，拒绝覆盖")
    conf, iou = float(request.get("conf", .25)), .7
    imgsz, stride = int(request.get("imgsz", 960)), int(request.get("stride", 1))
    max_frames = int(request.get("max_frames", 1500))
    if not math.isfinite(conf) or not 0.001 <= conf <= .99 or not 320 <= imgsz <= 1536 or imgsz % 32 or not 1 <= stride <= 30 or max_frames < 1:
        raise ValueError("推理参数超出允许范围")
    metadata = {}
    metadata_path = request.get("metadata_path")
    if metadata_path and Path(metadata_path).is_file():
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    digest = sha256(weight)
    if metadata.get("model_sha256", metadata.get("weight_sha256", digest)) != digest:
        raise ValueError("模型摘要与元信息不一致")
    metadata = {key: metadata[key] for key in SAFE_METADATA if key in metadata}
    os.environ["YOLO_CONFIG_DIR"] = str(output / "ultralytics-settings")
    is_image = path.suffix.lower() in IMAGE_EXTENSIONS
    start = time.perf_counter()
    injected = detector is not None
    if detector is None:
        from ultralytics import YOLO

        model = YOLO(str(weight))
        names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
        if set(names) != set(CLASSES):
            raise ValueError("该权重不是足球/门将/球员/裁判四类模型，请同学提供正确的足球权重")

        def detector(image):
            result = model.predict(image, imgsz=imgsz, conf=conf, iou=iou,
                                   max_det=300, device=request.get("device", "cpu"), verbose=False)[0]
            boxes = []
            if result.boxes is not None:
                for coords, score, raw_id in zip(result.boxes.xyxy.cpu().tolist(),
                                                result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
                    boxes.append({"cls": result.names[int(raw_id)], "conf": float(score), "xyxy": coords})
            return boxes, image

    frames, writer, first_info = [], None, None
    codec, video_name, truncated = None, None, False
    stream = _reader(path, stride, str(request.get("platform_root", "")))
    try:
        with (output / "frames.jsonl").open("w", encoding="utf-8") as handle:
            for image, info in stream:
                if len(frames) >= max_frames:
                    truncated = True
                    break
                height, width = image.shape[:2]
                boxes, _ = detector(image)
                frame = normalize_frame({"frame_index": info["frame_index"], "timestamp": info["timestamp"],
                                         "image_size": {"width": width, "height": height}, "boxes": boxes})
                annotated = render_bgr(image, frame["boxes"])
                frames.append(frame)
                handle.write(json.dumps(frame, ensure_ascii=False, allow_nan=False) + "\n")
                if first_info is None:
                    first_info = {**info, "width": width, "height": height}
                    ok, encoded = cv2.imencode(".png", annotated)
                    if not ok:
                        raise RuntimeError("标注图片编码失败")
                    encoded.tofile(str(output / "annotated.png"))
                    ok, original = cv2.imencode(".png", image)
                    if not ok:
                        raise RuntimeError("预览原帧编码失败")
                    original.tofile(str(output / "preview.png"))
                    dump(output / "annotation_style.json", {"version": STYLE_VERSION})
                if not is_image:
                    if writer is None:
                        fps = info.get("fps")
                        if not fps or fps <= 0:
                            raise ValueError("视频缺少可靠帧率")
                        video_name, codec = "annotated.webm", "VP8"
                        writer = cv2.VideoWriter(str(output / video_name), cv2.VideoWriter_fourcc(*"VP80"), fps / stride, (width, height))
                        if not writer.isOpened():
                            writer.release()
                            video_name, codec = "annotated.mp4", "MPEG4"
                            writer = cv2.VideoWriter(str(output / video_name), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (width, height))
                        if not writer.isOpened():
                            raise RuntimeError("视频编码器不可用；可查看首帧及逐帧数据")
                    writer.write(annotated)
                total = info.get("total_frames")
                expected = min(max_frames, max(1, math.ceil(total / stride))) if total else max_frames
                dump(output / "progress.json", {"progress": min(95, int(len(frames) / expected * 95)),
                                                "processed_frames": len(frames), "frame_index": frame["frame_index"]})
    finally:
        stream.close()
        if writer is not None:
            writer.release()
    if not frames:
        raise ValueError("输入中没有可读取的帧")
    analysis = build_analysis(frames)
    if truncated:
        analysis["limitations"].append(f"本次仅处理前{max_frames}个采样帧，未覆盖全部视频。")
    preview = frames[0]
    best = max(preview["boxes"], key=lambda box: box["conf"], default=None)
    artifacts = {"frames": "frames.jsonl", "counts": "counts.csv", "annotated": "annotated.png", "preview": "preview.png", "result": "result.json"}
    if video_name:
        artifacts["video"] = video_name
    result = {"schema_version": "football.portal.result.v1", "success": True,
              "source": {"kind": "image" if is_image else "video", "width": first_info["width"],
                         "height": first_info["height"], "fps": first_info["fps"], "reader": first_info["reader"],
                         "truncated": truncated},
              "model": {"id": request.get("model_id"), "name": request.get("model_label") or weight.stem, "sha256": digest, "metadata": metadata},
              "parameters": {"confidence": conf, "imgsz": imgsz, "stride": stride, "iou": iou},
              "frames": frames, "preview": preview, "boxes": preview["boxes"], "best_class": best["cls"] if best else None,
              "best_conf": best["conf"] if best else None, "analysis": analysis, "artifacts": artifacts,
              "execution": "injected_program_test" if injected else "trusted_weight_inference",
              "duration_seconds": round(time.perf_counter()-start, 3), "video_codec": codec,
              "annotation_style": {"version": STYLE_VERSION, "confidence_only": True, "colors": ROLE_COLORS, "ball_highlight": True},
              "capabilities": {"team_assignment": False, "tracking": False, "calibrated_pitch": False, "possession": False}}
    with (output / "counts.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["frame_index", "timestamp", *CLASSES, "team_A", "team_B", "team_unknown"]
        csv_writer = csv.DictWriter(handle, fieldnames=fields)
        csv_writer.writeheader()
        csv_writer.writerows(analysis["counts_series"])
    dump(output / "result.json", result)
    dump(output / "progress.json", {"progress": 100, "processed_frames": len(frames)})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.request.read_text(encoding="utf-8"))
    try:
        run_request(config)
    except Exception as exc:
        output = Path(config["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        dump(output / "worker_error.json", {"error": str(exc), "type": type(exc).__name__})
        print(f"Inference failed: {type(exc).__name__}", file=sys.stderr)
        raise SystemExit(2)
