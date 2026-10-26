import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from football_portal.bridge_worker import PitchAnalyzer, _reader, dump, run_request
from football_portal.results import normalize_frame, normalize_payload


def request_for(tmp_path, source):
    weight = tmp_path / "test-only.pt"
    weight.write_bytes(b"injected program test, not a trained weight")
    return {"input_path": str(source), "output_dir": str(tmp_path / "output"), "model_path": str(weight),
            "stride": 1, "conf": .25, "imgsz": 640, "max_frames": 100, "platform_root": ""}


def fake_detector(image):
    return [{"cls": "ball", "conf": .8, "xyxy": [10, 10, 18, 18]},
            {"cls": "player", "conf": .9, "xyxy": [30, 10, 45, 60]}], image.copy()


def test_geometry_without_calibration_and_track_ids_does_not_fabricate_events():
    analyzer = PitchAnalyzer(120, 80, "")
    frame = normalize_frame({"frame_index": 0, "timestamp": 0,
                             "image_size": {"width": 120, "height": 80},
                             "boxes": [{"cls": "player", "conf": .9, "xyxy": [10, 10, 20, 30]},
                                       {"cls": "player", "conf": .9, "xyxy": [11, 10, 21, 30]}]})
    events = analyzer.analyze(frame)
    assert events == []
    assert analyzer.calibrated is False
    assert analyzer.skip_reason


def test_image_all_boxes_and_actual_statistics(tmp_path):
    source = tmp_path / "image.png"
    cv2.imencode(".png", np.zeros((80, 120, 3), np.uint8))[1].tofile(str(source))
    request = request_for(tmp_path, source)
    request["imgsz"] = 512  # The worker accepts the same valid dimensions as the API.
    result = run_request(request, detector=fake_detector)
    assert result["execution"] == "injected_program_test"
    assert result["analysis"]["total_detections"] == 2
    assert result["analysis"]["per_class"]["ball"] == 1
    assert result["preview"]["image_size"] == {"width": 120, "height": 80}
    assert result["capabilities"]["team_assignment"] is False
    assert (tmp_path / "output/annotated.png").is_file()
    with pytest.raises(FileExistsError):
        run_request(request, detector=fake_detector)


def test_video_stride_retains_original_frame_time_and_writes_playable_media(tmp_path):
    source = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (120, 80))
    assert writer.isOpened()
    for i in range(6):
        writer.write(np.full((80, 120, 3), i * 25, np.uint8))
    writer.release()
    request = request_for(tmp_path, source)
    request["stride"] = 2
    result = run_request(request, detector=fake_detector)
    frames = [json.loads(line) for line in (tmp_path / "output/frames.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [f["frame_index"] for f in frames] == [0, 2, 4]
    assert [f["timestamp"] for f in frames] == [0, .2, .4]
    assert result["analysis"]["processed_frames"] == 3
    assert result["analysis"]["total_detections"] == 6
    imported = normalize_payload(result)
    assert imported["analysis"]["processed_frames"] == 3
    assert imported["analysis"]["total_detections"] == 6
    cap = cv2.VideoCapture(str(tmp_path / "output" / result["artifacts"]["video"]))
    try:
        assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    finally:
        cap.release()


def test_frame_limit_is_reported_without_pretending_full_video(tmp_path):
    source = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (120, 80))
    for _ in range(5):
        writer.write(np.zeros((80, 120, 3), np.uint8))
    writer.release()
    request = request_for(tmp_path, source)
    request["max_frames"] = 2
    result = run_request(request, detector=lambda image: ([], image))
    assert result["source"]["truncated"] is True
    assert result["analysis"]["processed_frames"] == 2
    assert any("未覆盖全部视频" in text for text in result["analysis"]["limitations"])


def test_corrupt_image_does_not_create_fake_empty_success(tmp_path):
    source = tmp_path / "broken.jpg"
    source.write_bytes(b"not image")
    with pytest.raises(ValueError, match="解码"):
        run_request(request_for(tmp_path, source), detector=fake_detector)
    assert not (tmp_path / "output/result.json").exists()


def test_platform_source_decodes_all_frames_before_sampling(tmp_path, monkeypatch):
    platform = tmp_path / "ODPlatform"
    (platform / "apps/platform/src").mkdir(parents=True)
    requested = []

    class Source:
        end_reason = "completed"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            for index in range(6):
                yield SimpleNamespace(image=np.zeros((8, 8, 3), np.uint8),
                    info=SimpleNamespace(frame_index=index, timestamp=index / 10, fps=10, total_frames=6))

    def create(path, stride):
        requested.append(stride)
        return Source()

    monkeypatch.setitem(sys.modules, "od_platform.frame_source", SimpleNamespace(create_frame_source=create))
    frames = list(_reader(tmp_path / "video.mp4", 2, str(platform)))
    assert requested == [1]
    assert [info["frame_index"] for _, info in frames] == [0, 2, 4]
    assert [info["timestamp"] for _, info in frames] == [0, .2, .4]


def test_progress_write_survives_a_transient_windows_reader_lock(tmp_path, monkeypatch):
    replace = Path.replace
    calls = []

    def transient(source, target):
        calls.append(target)
        if len(calls) <= 2:
            raise PermissionError("simulated short-lived reader lock")
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", transient)
    monkeypatch.setattr("football_portal.bridge_worker.time.sleep", lambda _: None)
    path = tmp_path / "progress.json"
    dump(path, {"progress": 95})
    assert json.loads(path.read_text()) == {"progress": 95}
    assert len(calls) == 3
