"""Verify downloaded videos and extract three unannotated preview frames each.

Run with the algorithm Python containing OpenCV. Frames retain clip-level source
labels only; they are not independent samples or bounding-box ground truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / ".local/datasets/football_events")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = [row for row in manifest["files"] if not row.get("generated_by") == "extract_event_previews.py"]
    previews = []
    errors = []
    for row in files:
        if row["kind"] != "video":
            continue
        capture = cv2.VideoCapture(str(root / row["path"]))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        inspection = {"opencv_opened": capture.isOpened(), "reported_frame_count": total,
                      "fps": fps, "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), "decoded_samples": 0}
        if not capture.isOpened() or fps <= 0 or total < 3:
            errors.append(row["path"])
            capture.release()
            row["media_inspection"] = inspection
            continue
        inspection["duration_seconds"] = round(total / fps, 3)
        image_hashes = []
        for numerator in (1, 2, 3):
            index = min(total - 1, int(total * numerator / 4))
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            valid, frame = capture.read()
            if not valid:
                errors.append(f"{row['path']}:frame:{index}")
                continue
            inspection["decoded_samples"] += 1
            image_hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
            ok, image = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                errors.append(f"{row['path']}:encode:{index}")
                continue
            relative = f"infactory/preview_frames/{Path(row['path']).stem}_{index:06d}.jpg"
            output = root / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(image.tobytes())
            previews.append({"path": relative, "url": row["url"], "bytes": output.stat().st_size,
                             "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "kind": "image",
                             "source_id": row["source_id"], "license": row["license"],
                             "coverage_event_ids": [], "clip_coverage_event_ids": row["coverage_event_ids"],
                             "derived_from": row["path"], "source_frame_index": index,
                             "source_timestamp_seconds": round(index / fps, 3),
                             "annotation_origin": "Unannotated preview; clip label must not be assumed visible in this frame.",
                             "generated_by": "extract_event_previews.py"})
        capture.release()
        inspection["unique_sample_pixel_hashes"] = len(set(image_hashes))
        row["media_inspection"] = inspection
    manifest["files"] = sorted(files + previews, key=lambda row: row["path"])
    manifest["counts"] = {kind: sum(row["kind"] == kind for row in manifest["files"])
                          for kind in ("video", "image", "annotations", "metadata", "license")}
    manifest["total_bytes"] = sum(row["bytes"] for row in manifest["files"])
    manifest["preview_frames_note"] = "Derived quarter/mid/three-quarter frames from the same videos, not independently labeled training images. Split by original match/video to prevent leakage."
    manifest["media_verification_errors"] = errors
    queue = root / "infactory/manual_review_queue.csv"
    previous_rows = {}
    if queue.is_file():
        with queue.open(encoding="utf-8-sig", newline="") as stream:
            previous_rows = {row["source_video_path"]: row for row in csv.DictReader(stream)}
    columns = ["source_video_path", "match_date", "home_team", "away_team", "source_event_category",
               "source_event_subtype", "source_url", "source_label_origin", "review_status",
               "review_event_category", "review_start_seconds", "review_end_seconds",
               "recipient_player", "recipient_jersey", "goal_validity", "reviewer", "review_notes"]
    with queue.with_suffix(".csv.tmp").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in files:
            if row["kind"] != "video":
                continue
            values = {key: "" for key in columns}
            values.update({"source_video_path": row["path"], "match_date": row.get("match_date", ""),
                           "home_team": row.get("home_team", ""), "away_team": row.get("away_team", ""),
                           "source_event_category": row.get("event_category", ""), "source_event_subtype": row.get("event_subtype", ""),
                           "source_url": row["url"], "source_label_origin": "provider_auto_annotation_unverified",
                           "review_status": "pending"})
            values.update(previous_rows.get(row["path"], {}))
            writer.writerow(values)
    queue.with_suffix(".csv.tmp").replace(queue)
    manifest["manual_review_queue"] = "infactory/manual_review_queue.csv"
    temporary = root / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    print(json.dumps({"verified_video_files": manifest["counts"]["video"], "derived_unannotated_images": len(previews), "errors": errors}), flush=True)


if __name__ == "__main__":
    main()
