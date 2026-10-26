"""The demo generator must create decodable files even under Unicode paths."""
import runpy

import cv2
import numpy as np
import pytest

from od_platform.common import paths


def test_demo_images_are_written_under_unicode_path(tmp_path, monkeypatch):
    target = tmp_path / "中文演示数据"
    monkeypatch.setattr(paths, "RAW_DATA_DIR", target)
    runpy.run_path(str(paths.ROOT_DIR / "scripts/demo_data/make_demo.py"))
    images = sorted((target / "demo/images").glob("*.jpg"))
    assert len(images) == 63
    for image in images:
        decoded = cv2.imdecode(np.frombuffer(image.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None and decoded.shape == (480, 640, 3)


def test_demo_encoding_failure_is_not_reported_as_success(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(cv2, "imencode", lambda *_: (False, None))
    with pytest.raises(OSError, match="编码失败"):
        runpy.run_path(str(paths.ROOT_DIR / "scripts/demo_data/make_demo.py"))
