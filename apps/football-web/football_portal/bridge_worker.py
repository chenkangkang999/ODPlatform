"""Web推理子进程：只读取可信权重，不训练，不修改ODPlatform。

能力分层：
- YOLO 检测：球员/球（门将/裁判有数据时自动支持）
- BoT-SORT 跟踪：视频逐帧给球员/球分配稳定 ID（ultralytics 内置）
- 几何规则：Homography 投影 + 冲撞/球出界判断（od_platform.geometric）
- 动作识别：VideoMAE 片段识别（od_platform.action_recognition，无权重时降级）
"""
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
from football_portal.results import CLASSES, ALIASES, build_analysis, normalize_frame
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


# ----------------------------------------------------------------------------
# 高级分析：几何规则 + 动作识别
# ----------------------------------------------------------------------------
class PitchAnalyzer:
    """兼容壳: 真实实现在 od_platform.geometric.analyzer（算法侧维护）。

    本类只负责把 ODPlatform 的源码目录加进 sys.path, 然后把活交给算法包。
    标定 / 规则 / 跟踪的全部逻辑都不在 Web 层 —— 这样算法侧改判据时,
    网页这一侧一行都不用动。

    ★ 与历史版本的差别: 不再用"整幅画面四角 ↔ 球场四角"伪造单应性。
      没有真实标定时不产出任何几何事件, 并通过 .calibrated 如实上报
      （决定了 capabilities.calibrated_pitch 的真假）。
    """

    def __new__(cls, width, height, platform_root="", config=None, media_path=None):
        root = Path(platform_root) / "apps/platform/src" if platform_root else None
        if root and root.is_dir():
            sys.path.insert(0, str(root))
        from od_platform.geometric.analyzer import PitchAnalyzer as _Implementation
        return _Implementation(width, height, platform_root, config, media_path=media_path)


def _load_action_recognizer(platform_root: str, request: dict):
    """按需加载动作识别器；无 checkpoint 时返回 None（降级）。"""
    checkpoint = request.get("action_checkpoint") or os.environ.get("FOOTBALL_ACTION_CKPT", "")
    if not checkpoint:
        return None
    try:
        root = Path(platform_root) / "apps/platform/src" if platform_root else None
        if root and root.is_dir():
            sys.path.insert(0, str(root))
        from od_platform.action_recognition.classifier import ActionRecognizer

        recognizer = ActionRecognizer({
            "model_name": "videomae_v2_base",
            "checkpoint": checkpoint,
            "clip_length": int(request.get("action_clip_length", 16)),
            "confidence_threshold": float(request.get("action_conf", 0.3)),
        })
        return recognizer if recognizer.ready else None
    except Exception as exc:  # noqa: BLE001
        print(f"[Action] 动作识别不可用，降级: {exc}", file=sys.stderr)
        return None


def _extract_boxes(result, tracking: bool) -> list[dict]:
    """把 ultralytics 结果转成 [{cls, conf, xyxy, id?}]，cls 用模型原始类别名。"""
    boxes = []
    if result.boxes is None:
        return boxes
    ids = result.boxes.id if tracking else None
    id_list = None if ids is None else ids.cpu().tolist()
    for i, (coords, score, raw_id) in enumerate(zip(result.boxes.xyxy.cpu().tolist(),
                                                    result.boxes.conf.cpu().tolist(),
                                                    result.boxes.cls.cpu().tolist())):
        box = {"cls": result.names[int(raw_id)], "conf": float(score), "xyxy": coords}
        if id_list is not None and i < len(id_list):
            box["id"] = int(id_list[i])
        boxes.append(box)
    return boxes


def _draw_ids(image, boxes):
    """在 BGR 帧上为球员/球叠加跟踪 ID（英文避免中文字体问题）。"""
    import cv2
    for box in boxes:
        tid = box.get("track_id")
        if tid is None:
            continue
        x, y = int(box["x1"]), max(12, int(box["y1"]) - 6)
        cv2.rectangle(image, (x - 1, y - 11), (x + 42, y + 2), (0, 0, 0), -1)
        cv2.putText(image, f"ID {tid}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 255), 1, cv2.LINE_AA)
    return image


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
    use_tracking = (not is_image) and bool(request.get("tracking", True))
    start = time.perf_counter()
    injected = detector is not None
    if detector is None:
        from ultralytics import YOLO

        model = YOLO(str(weight))
        names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
        # 类别归一化：模型类别经 ALIASES 映射后必须是足球角色集合的子集，且至少含球员和球
        std_names = {ALIASES.get(str(n).strip().lower(), str(n).strip().lower()) for n in names}
        if not std_names.issubset(set(CLASSES)) or not ({"player", "ball"} & std_names):
            raise ValueError("该权重不是足球检测模型（需要球员/球类别），请检查 FOOTBALL_MODEL_PATH")
        device = request.get("device", "cpu")

        def detector(image):
            if use_tracking:
                result = model.track(image, imgsz=imgsz, conf=conf, iou=iou, max_det=300,
                                     device=device, persist=True, tracker="botsort.yaml",
                                     verbose=False)[0]
            else:
                result = model.predict(image, imgsz=imgsz, conf=conf, iou=iou,
                                       max_det=300, device=device, verbose=False)[0]
            return _extract_boxes(result, use_tracking), image

    frames, writer, first_info = [], None, None
    codec, video_name, truncated = None, None, False
    events, unique_tracks = [], set()
    pitch, recognizer = None, None
    clip_buffer = []
    stream = _reader(path, stride, str(request.get("platform_root", "")))
    try:
        with (output / "frames.jsonl").open("w", encoding="utf-8") as handle:
            for image, info in stream:
                if len(frames) >= max_frames:
                    truncated = True
                    break
                height, width = image.shape[:2]
                if pitch is None and not is_image:
                    pitch = PitchAnalyzer(width, height, str(request.get("platform_root", "")),
                                          request.get("football_config"), media_path=path)
                if recognizer is None and not is_image and not injected:
                    recognizer = _load_action_recognizer(str(request.get("platform_root", "")), request)

                raw_boxes, _ = detector(image)
                frame = normalize_frame({"frame_index": info["frame_index"], "timestamp": info["timestamp"],
                                         "image_size": {"width": width, "height": height}, "boxes": raw_boxes})
                # 回填跟踪 ID（raw_boxes 与归一化框顺序一致：类别都是合法足球角色）
                for std_box, raw_box in zip(frame["boxes"], raw_boxes):
                    if "id" in raw_box:
                        std_box["track_id"] = raw_box["id"]
                        unique_tracks.add(raw_box["id"])

                annotated = render_bgr(image, frame["boxes"])
                if use_tracking:
                    annotated = _draw_ids(annotated, frame["boxes"])

                # 几何规则（仅视频）
                if pitch is not None:
                    frame_events = pitch.analyze(frame)
                    for ev in frame_events:
                        events.append(ev)

                # 动作识别（仅视频、识别器可用）：按 clip_length 采样一个片段
                if recognizer is not None:
                    clip_buffer.append(image)
                    if len(clip_buffer) >= int(request.get("action_clip_length", 16)):
                        import numpy as np
                        acts = recognizer.predict(np.asarray(clip_buffer))
                        for a in acts:
                            events.append({"type": "action", "frame_index": frame["frame_index"],
                                           "timestamp": frame.get("timestamp"), "action": a["cls"],
                                           "confidence": a["conf"]})
                        clip_buffer = clip_buffer[len(clip_buffer) // 2:]

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
                        # 优先 mp4(H.264 优先、回退 MPEG-4)，浏览器/Windows 兼容性最好；再回退 webm
                        video_name, codec = "annotated.mp4", "H.264"
                        writer = cv2.VideoWriter(str(output / video_name), cv2.VideoWriter_fourcc(*"avc1"), fps / stride, (width, height))
                        if not writer.isOpened():
                            writer.release()
                            video_name, codec = "annotated.mp4", "MPEG4"
                            writer = cv2.VideoWriter(str(output / video_name), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (width, height))
                        if not writer.isOpened():
                            writer.release()
                            video_name, codec = "annotated.webm", "VP8"
                            writer = cv2.VideoWriter(str(output / video_name), cv2.VideoWriter_fourcc(*"VP80"), fps / stride, (width, height))
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
    if not is_image:
        geometry_note = ("未提供球场标定，本次未产出越位/出界/进球/接触候选"
                         if not (pitch and pitch.calibrated)
                         else "几何事件基于球场标定，均为待人工复核的候选")
        analysis["limitations"].append(
            f"跟踪 ID 为自动分配，跨视频不连续；{geometry_note}。")
    preview = frames[0]
    best = max(preview["boxes"], key=lambda box: box["conf"], default=None)
    artifacts = {"frames": "frames.jsonl", "counts": "counts.csv", "annotated": "annotated.png", "preview": "preview.png", "result": "result.json"}
    if video_name:
        artifacts["video"] = video_name
    if events:
        artifacts["events"] = "events.json"
        (output / "events.json").write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    if injected:
        capabilities = {"team_assignment": False, "tracking": False,
                        "calibrated_pitch": False, "possession": False, "action_recognition": False}
    else:
        capabilities = {"team_assignment": False,
                        "tracking": use_tracking,
                        "calibrated_pitch": bool(pitch and pitch.ok),
                        "possession": False,
                        "action_recognition": recognizer is not None}
    result = {"schema_version": "football.portal.result.v1", "success": True,
              "source": {"kind": "image" if is_image else "video", "width": first_info["width"],
                         "height": first_info["height"], "fps": first_info["fps"], "reader": first_info["reader"],
                         "truncated": truncated},
              "model": {"id": request.get("model_id"), "name": request.get("model_label") or weight.stem, "sha256": digest, "metadata": metadata},
              "parameters": {"confidence": conf, "imgsz": imgsz, "stride": stride, "iou": iou},
              "frames": frames, "preview": preview, "boxes": preview["boxes"], "best_class": best["cls"] if best else None,
              "best_conf": best["conf"] if best else None, "analysis": analysis, "artifacts": artifacts,
              "events": events,
              "tracks": {"unique_count": len(unique_tracks), "ids": sorted(unique_tracks)} if use_tracking else None,
              "execution": "injected_program_test" if injected else "trusted_weight_inference",
              "duration_seconds": round(time.perf_counter()-start, 3), "video_codec": codec,
              "annotation_style": {"version": STYLE_VERSION, "confidence_only": True, "colors": ROLE_COLORS, "ball_highlight": True},
              "capabilities": capabilities}
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
