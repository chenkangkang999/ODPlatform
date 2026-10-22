"""frame_source 包桩 — 帧源抽象(图片目录/视频/摄像头)."""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class SourceType(enum.Enum):
    IMAGE = "image"
    VIDEO = "video"
    CAMERA = "camera"
    RTSP = "rtsp"


@dataclass
class FrameInfo:
    frame_index: int = 0
    filename: str = ""
    fps: float = 30.0


@dataclass
class Frame:
    image: object = None
    info: FrameInfo = field(default_factory=FrameInfo)


@dataclass
class CameraConfig:
    camera_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    backend: str = "auto"


class _ImageDirSource:
    def __init__(self, path: str):
        self._path = Path(path)
        self._files = sorted(p for p in self._path.iterdir()
                             if p.suffix.lower() in IMAGE_EXTENSIONS)
        self._idx = 0

    def get_source_type(self) -> SourceType:
        return SourceType.IMAGE

    def __iter__(self) -> Iterator[Frame]:
        import cv2
        for i, f in enumerate(self._files):
            img = cv2.imread(str(f))
            if img is not None:
                yield Frame(image=img, info=FrameInfo(frame_index=i, filename=f.name))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def create_frame_source(source, *, stride: int = 1):
    """工厂: 根据 source 字符串返回帧源. 桩实现只支持图片目录."""
    s = str(source)
    p = Path(s)
    if p.is_dir():
        return _ImageDirSource(s)
    # 视频/摄像头/RTSP 桩: 返回空迭代器
    class _EmptySource:
        def get_source_type(self): return SourceType.VIDEO
        def __iter__(self): return iter([])
        def __enter__(self): return self
        def __exit__(self, *a): pass
    return _EmptySource()
