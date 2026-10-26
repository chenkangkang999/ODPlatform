"""纯Python结果契约，Web和推理子进程使用相同统计口径。"""
import math

CLASSES = ("ball", "goalkeeper", "player", "referee")
ALIASES = {"足球": "ball", "球员": "player", "运动员": "player", "门将": "goalkeeper", "裁判": "referee",
           "football": "ball", "players": "player", "goalkeepers": "goalkeeper", "referees": "referee"}


def _number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name}必须为有限数值")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}必须为有限数值") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name}必须为有限数值")
    return result


def _image_size(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        value = {"width": value[0], "height": value[1]}
    if not isinstance(value, dict):
        raise ValueError("image_size应包含width和height")
    result = {}
    for key in ("width", "height"):
        number = _number(value.get(key), key)
        if number <= 0 or number != int(number):
            raise ValueError("图片尺寸必须是正整数")
        result[key] = int(number)
    return result


def normalize_frame(frame, *, default_index=0, class_names=None):
    if not isinstance(frame, dict):
        raise ValueError("每帧必须是JSON对象")
    size = _image_size(frame.get("image_size"))
    raw_index = frame.get("frame_index", frame.get("frame_idx", default_index))
    index = _number(raw_index, "frame_index")
    if index < 0 or index != int(index):
        raise ValueError("frame_index必须是非负整数")
    timestamp = frame.get("timestamp")
    if timestamp is not None:
        timestamp = _number(timestamp, "timestamp")
        if timestamp < 0:
            raise ValueError("timestamp不能为负")
    boxes = frame.get("boxes", frame.get("detections", []))
    if not isinstance(boxes, list) or len(boxes) > 1000:
        raise ValueError("boxes必须是列表，每帧最多1000个框")
    normalized = []
    for box in boxes:
        if not isinstance(box, dict):
            raise ValueError("每个检测框必须是对象")
        label = box.get("cls", box.get("label", box.get("class_name")))
        if isinstance(label, int) and not isinstance(label, bool):
            if not isinstance(class_names, (list, tuple)) or not 0 <= label < len(class_names):
                raise ValueError("数字类别需要显式class_names，不能猜测编号")
            label = class_names[label]
        label = ALIASES.get(str(label).strip().lower(), str(label).strip().lower())
        if label not in CLASSES:
            raise ValueError(f"不支持的足球类别: {label}")
        confidence = _number(box.get("conf", box.get("confidence")), "conf")
        if not 0 <= confidence <= 1:
            raise ValueError("置信度必须在0到1之间")
        coords = box.get("xyxy")
        if coords is None:
            coords = [box.get(k) for k in ("x1", "y1", "x2", "y2")]
        if not isinstance(coords, (list, tuple)) or len(coords) != 4:
            raise ValueError("框需要4个原图xyxy坐标")
        x1, y1, x2, y2 = [_number(v, "xyxy") for v in coords]
        if frame.get("coordinate_space") == "normalized":
            if size is None:
                raise ValueError("归一化坐标必须提供图片尺寸")
            x1, x2 = x1 * size["width"], x2 * size["width"]
            y1, y2 = y1 * size["height"], y2 * size["height"]
        if not (0 <= x1 < x2 and 0 <= y1 < y2):
            raise ValueError("框必须有正面积且坐标非负")
        if size and (x2 > size["width"] + 1e-6 or y2 > size["height"] + 1e-6):
            raise ValueError("检测框越出原图范围")
        team = box.get("team") if label == "player" else None
        if label == "player":
            team = team if team in ("A", "B") else "unknown"
        normalized.append({"cls": label, "class_id": CLASSES.index(label), "conf": confidence,
                           "x1": x1, "y1": y1, "x2": x2, "y2": y2, "cx": (x1+x2)/2,
                           "cy": (y1+y2)/2, "w": x2-x1, "h": y2-y1, "team": team})
    if "n_detections" in frame and frame["n_detections"] != len(normalized):
        raise ValueError("n_detections与实际框数不符")
    return {"frame_index": int(index), "timestamp": timestamp, "image_size": size, "boxes": normalized,
            "n_detections": len(normalized)}


def build_analysis(frames):
    if not isinstance(frames, list) or len(frames) > 20000:
        raise ValueError("frames必须为列表且不超过20000帧")
    totals = dict.fromkeys(CLASSES, 0)
    confidences = {name: [] for name in CLASSES}
    positions, series, seen = [], [], set()
    ball_frames = 0
    for ordinal, original in enumerate(frames):
        frame = normalize_frame(original, default_index=ordinal)
        key = frame["frame_index"]
        if key in seen:
            raise ValueError("同一任务存在重复源帧号")
        seen.add(key)
        counts = dict.fromkeys(CLASSES, 0)
        teams = {"A": 0, "B": 0, "unknown": 0}
        for box in frame["boxes"]:
            label = box["cls"]
            counts[label] += 1
            totals[label] += 1
            confidences[label].append(box["conf"])
            if label == "player":
                teams[box["team"]] += 1
            if len(positions) < 2000 and frame["image_size"]:
                size = frame["image_size"]
                positions.append({"cls": label, "x": box["cx"] / size["width"],
                                  "y": box["cy"] / size["height"], "frame_index": key})
        ball_frames += int(counts["ball"] > 0)
        series.append({"frame_index": key, "timestamp": frame["timestamp"], **counts,
                       "team_A": teams["A"], "team_B": teams["B"], "team_unknown": teams["unknown"]})
    count = len(frames)
    return {"processed_frames": count, "total_detections": sum(totals.values()), "per_class": totals,
            "mean_per_frame": {name: totals[name] / count if count else None for name in CLASSES},
            "ball_frames": ball_frames, "ball_frame_fraction": ball_frames / count if count else None,
            "confidence_by_class": {name: {"count": len(values), "mean": sum(values) / len(values) if values else None,
                                           "min": min(values) if values else None, "max": max(values) if values else None}
                                    for name, values in confidences.items()},
            "counts_series": series, "positions": positions,
            "position_sampled": sum(totals.values()) > len(positions),
            "definitions": {"total_detections": "逐帧检测框累计；同一球员跨帧会重复计数，不是独立人数。",
                            "ball_frame_fraction": "有足球检测的处理帧比例，不是召回率或控球率。",
                            "positions": "原图归一化框中心，不是球场坐标和跑动热力图。",
                            "confidence": "模型置信度，不是经过真值评测的准确率。"},
            "limitations": ["没有稳定球员ID、场地校准或比赛事件识别，不能推算控球率、速度、距离或战术。",
                            "类别数量与阈值、采样步长、画面场景有关；A/B仅在来源提供队别时展示。"]}


def normalize_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("导入内容必须是JSON对象")
    class_names = payload.get("class_names")
    raw_frames = payload.get("frames")
    if raw_frames is None:
        reported = payload.get("analysis", {}).get("processed_frames") if isinstance(payload.get("analysis"), dict) else None
        if reported is not None and _number(reported, "processed_frames") > 1:
            raise ValueError("多帧视频必须提供完整frames；不能用首帧预览替代整段统计")
        preview = payload.get("preview")
        candidate = preview if isinstance(preview, dict) else payload
        if "boxes" not in candidate and "detections" not in candidate:
            raise ValueError("需要提供原始boxes或完整frames，不接受只有汇总数字的报告")
        raw_frames = [candidate]
    if not isinstance(raw_frames, list) or not raw_frames or len(raw_frames) > 20000:
        raise ValueError("至少提供一帧原始检测记录，不接受只有汇总数字的报告")
    frames = [normalize_frame(frame, default_index=i, class_names=class_names) for i, frame in enumerate(raw_frames)]
    analysis = build_analysis(frames)
    if any(frame["image_size"] is None for frame in frames):
        analysis["limitations"].append("导入结果未提供全部原图尺寸，不能恢复对应图片或归一化位置。")
    preview = frames[0]
    best = max(preview["boxes"], key=lambda box: box["conf"], default=None)
    return {"schema_version": "football.portal.result.v1", "preview": preview,
            "boxes": preview["boxes"], "best_class": best["cls"] if best else None,
            "best_conf": best["conf"] if best else None, "analysis": analysis,
            "source": {"kind": "import", "imported": True},
            "model": {"name": "导入结果", "metadata": {"identity_status": "unverified_import"}},
            "frames": frames, "capabilities": {"team_assignment": any(box.get("team") in ("A", "B") for f in frames for box in f["boxes"]),
                                                   "tracking": False, "possession": False}}
