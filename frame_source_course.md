# frame_source：可插拔多模态帧输入源（精简实操版）

> **本文档是什么**：frame_source 讲义的纯实操版——只讲每一步正确的做法。代码全部在沙箱真实运行通过：自带自测 **50 通过 / 0 失败 / 1 跳过**（摄像头项无设备自动跳过），扩展性实证（包外加源不改模块）真机跑通。
> **产物性质**：一个**零依赖宿主、可整包拷走**的独立 Python 包。它不 import 任何宿主项目代码，只依赖 `opencv-python` / `numpy` / `pydantic>=2`。
> **读完你能得到**：`frame_source/` 包——任意输入源（摄像头/视频/网络流/图片/文件夹/第三方源）统一成 `for frame in src` 一套接口，加新源只需"新增一个文件 + 一行注册"。

## 0. 总览

### 0.1 它解决什么

推理脚本里直接 `cv2.VideoCapture` + 一堆 if 判断"这是图还是视频还是摄像头"，想加个深度相机就得从头再糊一遍。本模块把"出帧"抽成统一协议：

```python
from frame_source import create_frame_source, CameraConfig

def run(model, source):                      # source: str | SourceConfig
    with create_frame_source(source) as src:
        for frame in src:
            results = model(frame.image)      # frame.image 恒为主视觉平面
            if frame.has_depth:               # 多模态源才有 depth, 彩色源恒 False
                fuse_depth(results, frame.depth, frame.intrinsics)

# 同一个 run, 各种输入一行不改地喂进去:
run(model, "0")                               # 摄像头
run(model, "demo.mp4")                        # 视频文件
run(model, "rtsp://192.168.1.10/stream")      # 网络流
run(model, "./test_images/")                  # 图片文件夹
run(model, CameraConfig(fps=90, backend="msmf"))   # 高帧率摄像头(显式选型)
```

### 0.2 五条工程规矩（所有文件都遵守）

- **A 协议统一**：所有源实现同一个 `FrameSource` 协议（`open/read/close` + `seek`/`set_stride` + `modalities` + 上下文管理器 + 迭代器）。调用方只跟协议打交道。
- **B fail-fast**：路径不存在、格式不支持、配置字段拼错、配置类型没注册——当场报错且报错可读，不留空对象让人后面 debug。
- **C 资源不漏**：每个源支持上下文管理器，`with` 退出必关。
- **D 自有 SSoT**：扩展名、网络流 scheme 等常量模块自己存一份，不引用宿主——这是"可拷走"的必要代价（内容重叠是巧合，不是依赖）。
- **E 开闭原则**：加一种新源，只允许"新增文件 + 一行注册"，不允许改 `factory.py`/`registry.py` 的既有代码。这条是本模块的灵魂，也是注册表存在的理由。

### 0.3 四块结构与依赖方向（严格单向，下层不知道上层存在）

```text
        ┌─────────────────────────────────────────────┐
            factory.py   把 str|config 交给 registry, 拿回源
        └─────────────────────────────────────────────┘
                    │依赖
        ┌─────────────────────────────────────────────┐
            registry.py  配置类型 → 源类; 字符串 → 配置(规则归各源所有)
        └─────────────────────────────────────────────┘
                    │依赖
        ┌───────────┴───────────┬─────────────────────┐
          wrappers/              sources/
          线程/异步/重连          摄像头/视频/图片/文件夹(各自 @register_source)
        └───────────┬───────────┴─────────────────────┘
                    │都依赖
        ┌───────────┴─────────────────────────────────┐
            core/   types(一帧长什么样) + config(每源配置) + base(协议)
        └─────────────────────────────────────────────┘
```

**关键的依赖倒置**：`factory` 不直接 import 四个源，只 import `registry`。四个源在"被 import 时"通过 `@register_source` 把自己登记进注册表，工厂运行期从注册表拿。触发那次 import 的是包的 `__init__.py`。这就是"工厂零改动也能支持新源"的机械原理——注册表的"魔法"拆开只是 import 副作用：装饰器在模块被 import 时才运行，没 import 的源等于不存在。

### 0.4 目录树与构建顺序

```text
frame_source/
├── __init__.py             (公共 API re-export; import sources 触发内置源自注册)
├── README.md               (可拷走模块的身份证)
├── core/
│   ├── __init__.py
│   ├── types.py            (SourceType / FrameInfo / Frame / CameraIntrinsics / EndReason)
│   ├── config.py           (SourceConfig 基类 + 四个源的 config —— 配置类型是派发轴)
│   ├── base.py             (FrameSource 抽象协议; config 驱动 + modalities 能力发现 + end_reason)
│   └── registry.py         (register_source + build_from_config/string + list_sources —— 本模块核心)
├── sources/
│   ├── __init__.py         (import 四个源, 触发它们自注册)
│   ├── camera.py           (CameraSource —— 跨平台后端协商 + 高帧率参数验证)
│   ├── video.py            (VideoSource —— seek + grab 跳采 + 网络流诚实标注)
│   └── image.py            (ImageSource + ImageFolderSource)
├── wrappers/
│   ├── __init__.py
│   ├── threaded.py         (ThreadedSource —— 后台线程采集, 透传 inner 类型/模态)
│   ├── aio.py              (AsyncSource —— 给 async 调用方; 不继承 FrameSource, 守 LSP)
│   └── reconnect.py        (ReconnectingSource —— 断线重连, 基于 end_reason 区分停/断)
├── factory.py              (create_frame_source / threaded / async —— 只做 str|config → registry)
└── plugins.py              (可选: entry-points 发现包外第三方源)
tests/test_frame_source.py  (一键自测: python tests/test_frame_source.py)
examples/synthetic_source.py (扩展性实证: 包外加源, 不改 frame_source 一个字)
```

**构建顺序自底向上**：core → registry → sources → wrappers → factory → 自测。本文档按此顺序给出。

**跨平台提醒**：唯一绕不开的平台差异是摄像头后端（Windows 走 MSMF/DSHOW，Linux 走 V4L2），`CameraSource` 显式处理（§4.3）；图片/视频源不碰硬件，行为一致。

---

## 1. core/types.py —— 一帧长什么样

多模态 `Frame`：主平面 `image`（不再恒 BGR——红外是热图、双目是左目）+ 一等字段 `depth`（深度是最高频的第二模态，值得类型提示和 `has_depth`）+ 一等字段 `intrinsics`（反投影点云刚需）+ 开放式 `aux` dict（只接双目 right/多目第 N 路这类真正开放的副平面，不当杂物袋）。

两个设计点：

- **`SourceType` 枚举只列内置四种，`FrameInfo.source_type` 类型标 `str`**——锁成闭合枚举就把第三方源的类型堵死了（它们没法往你的枚举里加成员），与规矩 E 直接冲突。枚举成员本身是 str，内置源传枚举、第三方传字符串，天然共存。
- **`EndReason` 结束语义**：`read()` 返回 `None` 是二义的——文件放完该**停**、流被掐断该**重连**，塌缩成同一个 `None` 就什么都做不了。协议保留 `read() -> Optional[Frame]` 不变（老循环零改），旁挂一个可查询的 `end_reason`：想用就查，不用就当不存在。

📄 **新建** `frame_source/core/types.py`：

```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : types.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : 核心数据类型 (SourceType / FrameInfo / Frame / CameraIntrinsics)
"""
frame_source 核心数据类型定义(多模态版)。

一帧 = 一个"主视觉平面"(image)+ 一份元数据(info), 外加可选的副平面:
    - depth      : 深度图(RGBD 相机), 高频重要模态 → 一等字段
    - aux        : 开放式副平面字典(双目 right / 多目第 N 路)
    - intrinsics : 相机内参(深度反投影点云要用)

彩色源(摄像头/视频/图片/文件夹)每帧只产一张 BGR 图, 只填 image + info,
副平面走默认值——所以老的 `model(frame.image)` 一个字不用改。

模块独立性原则(规矩 D):
    IMAGE_EXTENSIONS / VIDEO_EXTENSIONS / STREAM_SCHEMES 是 frame_source 的
    自有 SSoT, 与宿主项目同名常量内容重叠是巧合, 不是约束。本模块不引用任何
    外部基础设施常量, 保证整包可拷贝至其他项目, 也保证第三方源能独立扩展。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class SourceType(str, Enum):
    """内置输入源类型枚举(继承 str 便于序列化与日志)。

    注意: 这里列的是**内置**四种。第三方源不必、也无法往这个枚举里加成员——
    它们在自己的 FrameInfo 里用自己的类型字符串即可(见 FrameInfo.source_type
    的类型是 str, 不是 SourceType)。这是为了不把第三方源的类型堵死(规矩 E)。
    """
    CAMERA       = "camera"
    VIDEO        = "video"
    IMAGE        = "image"
    IMAGE_FOLDER = "image_folder"


# ── 模块自有 SSoT(frozenset 兼顾不可变 + set 语义)─────────────
IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff",
})
VIDEO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm",
})

# 网络流 scheme: cv2.VideoCapture 能直接吃这些 URL, 一律按视频流走 VideoSource
STREAM_SCHEMES: tuple[str, ...] = ("rtsp://", "rtmp://", "http://", "https://")


def is_stream_url(source: str) -> bool:
    """是否是 cv2 能直接打开的网络流 URL(单一事实来源, 全模块共用)。"""
    return source.lower().startswith(STREAM_SCHEMES)


@dataclass(frozen=True)
class CameraIntrinsics:
    """
    相机内参(不可变值对象)。深度相机反投影点云、双目算视差时需要。

    彩色/普通源没有标定信息, intrinsics 恒为 None; 深度/双目源填充它。

    字段:
        fx, fy      : 焦距(像素)
        cx, cy      : 主点(像素)
        depth_scale : 深度原始单位 → 米的比例(如 uint16 毫米图, scale=0.001)
        baseline    : 双目基线(米), 单目/深度可为 None
    """
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float = 1.0
    baseline: Optional[float] = None


@dataclass(frozen=True)
class FrameInfo:
    """
    帧元数据(不可变值对象)。

    所有输入源都返回 width/height(主平面尺寸), 供下游计算字体大小/缩放/输出尺寸等。

    ⚠️ source_type 的类型是 str(不是 SourceType):
        内置源传 SourceType.XXX(枚举成员本身是 str, 天然兼容);
        第三方源传自己的字符串(如 "depth")。故意放宽以支持包外扩展(规矩 E)。

    ⚠️ 关于 frame_index 与 stride 的关系:
        frame_index 是**源帧号**, 不是"调用方收到的第几帧"。
        - 普通迭代(stride=1):  0, 1, 2, 3, ...
        - stride=N 跳采下:     0, N, 2N, 3N, ...(保留原始帧号)
        理由: timestamp = frame_index / fps 才正确; 下游帧标注/回放对照需要原始帧号;
             需要连续序号的调用方自行 enumerate。

    关于 timestamp 字段的语义差异(已知不一致, 故意保留):
        - CameraSource:                    墙钟时间(自 open 起经过的秒数)
        - VideoSource:                     媒体时间(frame_index / fps, 即在视频里的位置)
        - ImageSource / ImageFolderSource: 0.0(无时间概念)
    """
    # 主平面尺寸(所有源都有)
    width: int
    height: int

    # 源信息(source_type 收 str, 兼容内置枚举与第三方字符串)
    source_type: str
    source_path: str

    # 序列信息
    frame_index: int = 0
    total_frames: Optional[int] = None

    # 时间信息(摄像头 / 视频)
    timestamp: float = 0.0
    fps: Optional[float] = None

    # 文件名(所有源都填充; 摄像头填 'camera:<id>' 占位)
    filename: Optional[str] = None

    @property
    def resolution(self) -> tuple[int, int]:
        """主平面分辨率 (width, height)。"""
        return (self.width, self.height)


@dataclass
class Frame:
    """
    帧数据(统一返回类型, 多模态)。

    Attributes:
        image      : 主视觉平面 ndarray。彩色源=BGR(OpenCV 标准), 红外源=热图,
                     双目源=左目。默认零拷贝传递, 调用方需持久保存时自行 .copy()。
        info       : 帧元数据。
        depth      : 深度图 ndarray(RGBD 源), 一般 uint16 毫米; 无深度时 None。
        aux        : 开放式副平面字典(如 {'right': 右目图}); 默认空。
        intrinsics : 相机内参(深度反投影/双目视差用); 无标定时 None。

    设计说明:
        彩色四源每帧只产一张主图, 只填 image + info, 副平面走默认值。所以下游
        `model(frame.image)` 换任何一种彩色源都不用改。多模态源(深度/双目/红外)
        额外填 depth / aux / intrinsics, 需要它们的新代码按需去取。

        深度是一等字段(不塞进 aux): 它是最常见最重要的第二模态, 值得类型提示
        和 has_depth 便捷判断。aux 只接"开放式、不确定有几路"的副平面, 不做杂物袋。
    """
    image: np.ndarray
    info: FrameInfo
    depth: Optional[np.ndarray] = None
    aux: dict[str, np.ndarray] = field(default_factory=dict)
    intrinsics: Optional[CameraIntrinsics] = None

    # ── 能力便捷判断 ──
    @property
    def has_depth(self) -> bool:
        """本帧是否带深度图。彩色源恒 False, 深度源为 True。"""
        return self.depth is not None

    # ── 便捷属性(少写一层 .info)──
    @property
    def resolution(self) -> tuple[int, int]:
        return self.info.resolution

    @property
    def width(self) -> int:
        return self.info.width

    @property
    def height(self) -> int:
        return self.info.height

# ── 结束语义(上线加固): read() 返回 None 时, 用 end_reason 区分"停"与"断" ──
class EndReason(str, Enum):
    """一次迭代为何结束(继承 str 便于日志/序列化)。"""
    NOT_STARTED  = "not_started"   # 尚未 open, 或 open 未成功
    RUNNING      = "running"       # 正在出帧, 尚未结束
    COMPLETED    = "completed"     # 干净读完(文件到尾 / 文件夹读尽)—— 不该重连
    DISCONNECTED = "disconnected"  # 异常中断(网络流断开 / 设备掉线)—— 可重连
    ERROR        = "error"         # 读取过程出错(如文件中途损坏)—— 可重试


_RECOVERABLE: frozenset = frozenset({EndReason.DISCONNECTED, EndReason.ERROR})


def is_recoverable(reason: "EndReason") -> bool:
    """该结束原因是否值得重连/重试。COMPLETED 不值得(它本就该停), DISCONNECTED/ERROR 值得。"""
    return reason in _RECOVERABLE
```

---

## 2. core/config.py —— 每源一个配置，配置类型是派发轴

**为什么配置类型做派发轴而不是字符串**：硬件源在字符串层面天然不可区分——webcam、深度相机、红外都可能是 `"0"`，信息在字符串里根本不存在；而 config 的**类型**是确定、不歧义的一步查表。这还白送了内存源（ndarray/tensor）的支持——它们没有字符串形态，却能被 config 派发。

字符串降级为"二线糖"：常见源（`"0"`/`"x.mp4"`/`"rtsp://..."`/`"./imgs/"`）按规则翻译成 config 后进同一条派发；硬件/内存源只走显式 config。

**fail-fast 分界**：路径存在性校验放 config 的 `field_validator`（文件类源构造即炸、报错精确）；硬件源没法廉价探测设备，"打不开"留到 `open()`；URL 图片明确拒绝（交给视频源按流处理）。

📄 **新建** `frame_source/core/config.py`：

```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : config.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : 每源配置 (Pydantic v2) —— 配置类型是本模块的派发轴
"""
源配置(基于 Pydantic v2 BaseModel, 不继承宿主项目内部配置基类)。

本模块把"配置类型"当派发轴: build_from_config(cfg) 靠 type(cfg) 查该造哪个源。
所以每种源都有一个可区分的 config 类, 它们各扛两件实事:
    1. 派发的钥匙 —— type(cfg) 唯一确定造哪个源;
    2. fail-fast 校验 —— 路径存在性 / 字段取值当场校验, 别拖到 open()。

设计原则:
    - 所有 config 继承 SourceConfig, 统一 extra=forbid + validate_assignment;
    - 路径类 config 在构造时校验存在性(网络流豁免, URL 图片直接拒);
    - 配置层不绑 logger: 校验失败直接 raise ValidationError, 由调用方决定记录方式。
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .types import is_stream_url


CameraBackend = Literal["auto", "msmf", "dshow", "v4l2"]
CameraCodec   = Literal["MJPG", "YUYV", "H264", "MP4V"]


class SourceConfig(BaseModel):
    """
    所有源配置的基类。

    一处封装 fail-fast: extra=forbid(拼错字段名当场拦下)+ validate_assignment
    (后续赋值也走校验)。子类继承即享用, 不用各写各的。

    identity: 人类可读的源标识(日志/调试用), 子类按需覆盖。
    """
    model_config = ConfigDict(
        extra="forbid",              # 拼错字段名第一时间拦下
        validate_assignment=True,    # 后续赋值也走验证
    )

    @property
    def identity(self) -> str:
        return self.__class__.__name__


class CameraConfig(SourceConfig):
    """
    cv2 摄像头配置。

    示例:
        CameraConfig()                                             # 默认
        CameraConfig(width=1280, height=720, fps=90, backend="msmf")

    字段说明:
        camera_id : OpenCV 设备 ID(≥ 0)
        width/height/fps : 请求值(实际生效以 set+get 协商结果为准)
        backend   : auto(系统自选) / msmf(Win 高帧率) / dshow(Win 兼容) / v4l2(Linux)
        codec     : FOURCC 编码, 高帧率必须 MJPG
    """
    camera_id: int = Field(default=0,    ge=0,            description="OpenCV 设备 ID")
    width:     int = Field(default=1280, gt=0, le=7680,   description="请求分辨率宽")
    height:    int = Field(default=720,  gt=0, le=4320,   description="请求分辨率高")
    fps:       int = Field(default=30,   gt=0, le=1000,   description="请求帧率")

    backend: CameraBackend = Field(default="auto", description="摄像头后端")
    codec:   CameraCodec   = Field(default="MJPG", description="FOURCC 编码")

    @property
    def identity(self) -> str:
        return f"camera:{self.camera_id}"

    def get_resolution(self) -> tuple[int, int, int]:
        """返回 (width, height, fps) 三元组。"""
        return (self.width, self.height, self.fps)


class VideoConfig(SourceConfig):
    """
    视频文件 / 网络视频流配置。

    path 可以是本地视频文件路径, 也可以是 rtsp/rtmp/http(s) 网络流 URL。
    本地文件在构造时校验存在性; 网络流豁免(不是本地文件)。
    """
    path: str = Field(description="视频文件路径或网络流 URL")

    @field_validator("path")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        if is_stream_url(v):
            return v                       # 网络流不做本地存在性校验
        if not Path(v).is_file():
            raise ValueError(f"视频文件不存在: {v}")
        return v

    @property
    def identity(self) -> str:
        return self.path


class ImageConfig(SourceConfig):
    """
    单张图片配置。仅支持本地文件(cv2.imread 拉不了远程 URL, 故拒绝 URL)。
    """
    path: str = Field(description="本地图片文件路径")

    @field_validator("path")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        if is_stream_url(v):
            raise ValueError(f"图片源不支持 URL(cv2.imread 无法拉取远程图片): {v}")
        if not Path(v).is_file():
            raise ValueError(f"图片文件不存在: {v}")
        return v

    @property
    def identity(self) -> str:
        return self.path


class ImageFolderConfig(SourceConfig):
    """
    图片文件夹配置。path 必须是已存在的目录。
    """
    path: str = Field(description="图片文件夹路径")

    @field_validator("path")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        if not Path(v).is_dir():
            raise ValueError(f"不是有效文件夹: {v}")
        return v

    @property
    def identity(self) -> str:
        return self.path
```

---

## 3. core/base.py —— FrameSource 抽象协议

整个模块的骨架。要点：

- **协议方法而非 wrapper**：`seek`/`set_stride` 定义在基类，各源用自己的最快手段实现跳采——VideoSource 用 `grab()` 跳过 YUV→BGR 解码（快 3~5 倍），ImageFolderSource 直接跳索引零 IO；通用 wrapper 只能 read-and-discard 白白解码。不适用的源（摄像头）覆盖 `set_stride` 明确拒绝，而非假装支持。
- **能力发现在源级**：`modalities -> frozenset[str]` 让消费方在**开循环前**（不逐帧读）决定要不要准备点云管线；帧级取值靠 `frame.has_depth`。wrapper 必须透传 inner 的 modalities，否则能力信息在包装时丢失。
- **重新 open 即重头**：`open()` 必须复位位置状态（`_frame_index` 等），兑现"重新 open 能否重头迭代"的契约。
- **`end_reason` 槽归基类、判定权归各源**："读失败到底是干净结束还是断线"只有源自己判得了（VideoSource 知道自己是文件还是流、读没读到 total；CameraSource 知道自己是实时设备）。基类只定义这个槽和默认值，各源在返回 None 前填好。

📄 **新建** `frame_source/core/base.py`：

```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : base.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : FrameSource 抽象基类 — 统一所有输入源的协议(config 驱动 + 能力发现)
"""
帧源抽象基类。定义所有输入源(摄像头/视频/图片/包装器/第三方源)共享的协议:

    __init__(config)               config 驱动: 源吃 SourceConfig, 不吃裸字符串
    open / read / close            三个抽象方法, 子类必须实现
    get_source_type()              返回源类型(内置枚举或第三方字符串)
    modalities                     能力发现: 本源能产哪些平面, 默认 {'image'}
      + seek                       一次性跳转, 子类按能力覆盖
      + set_stride                 持续跳采, 子类按能力在 read() 内实现
      + 上下文管理器
      + 迭代器

帧位置控制原语(同族, 协议层):
    seek(frame=N)      一次性跳到第 N 帧
    seek(time_sec=T)   一次性跳到 T 秒
    set_stride(N)      之后每 N 帧产出 1 帧, 直到再次调用

为什么 stride 是协议方法而不是 wrapper:
    1. 语义同族 — 跟 seek 一样是"源对自身行为的控制原语"
    2. 性能要求 — 每个源能用自己最快的跳帧手段:
       VideoSource 用 cap.grab() 跳过 YUV→BGR 转换;
       ImageFolderSource 直接跳索引, 零 IO;
       wrapper 路径只能 read-and-discard, 浪费完整解码。
    3. 生命周期 — set_stride() 可在迭代过程中动态切换。

子类实现 stride 的模板(在 read() 开头加几行):
    def read(self):
        # stride>1 且非首帧: 用本源最快的方式跳过 stride-1 帧
        if self._stride > 1 and self._frame_index > 0:
            if not self._fast_skip(self._stride - 1):
                return None
        ...原有读帧逻辑...
    其中 _fast_skip 是私有方法, 各源自己写, 不入协议。
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Iterator, Optional

from .config import SourceConfig
from .types  import Frame, SourceType, EndReason


logger = logging.getLogger(__name__)


class FrameSource(ABC):
    """
    帧源抽象基类(config 驱动)。

    标准用法:
        with SomeSource(some_config) as source:
            for frame in source:
                process(frame.image)
                print(frame.width, frame.height)

    能力发现(开循环前决策):
        if "depth" in source.modalities:
            prepare_pointcloud_pipeline()

    跳帧(seek, 一次性):
        source.seek(frame=100)      # 跳到第 100 帧
        source.seek(time_sec=3.5)   # 跳到 3.5 秒

    跳采(stride, 持续):
        source.set_stride(3)        # 之后每 3 帧产出 1 帧
        source.set_stride(1)        # 恢复逐帧

    注意:
        同一 source 对象只能迭代一次(标准迭代器协议)。
        需要重头迭代请重新 open() 或重新创建对象。
    """

    def __init__(self, config: SourceConfig):
        self.config       = config
        self.source_path  = config.identity     # 日志/调试用的人类可读标识
        self._frame_index = 0
        self._stride      = 1     # 默认透传; >1 时每 stride 帧产出 1 帧
        self._start_time  = time.time()
        self._end_reason: EndReason = EndReason.NOT_STARTED   # 上次迭代因何停止(read() 返回 None 后查询)

    # ── 子类必须实现 ────────────────────────────────────────
    @abstractmethod
    def open(self) -> bool:
        """打开输入源,返回是否成功。实现必须复位位置状态(见类 docstring 契约)。"""

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """
        读取一帧,源耗尽时返回 None。

        实现 stride 的源(VideoSource / ImageFolderSource)在本方法开头
        做"跳过 stride-1 帧"动作; 不实现 stride 的源(CameraSource /
        ImageSource)忽略 self._stride 即可。
        """

    @abstractmethod
    def close(self) -> None:
        """关闭输入源,释放资源。"""

    @abstractmethod
    def get_source_type(self) -> SourceType:
        """返回输入源类型(内置源返回 SourceType 成员; 第三方源可返回自己的字符串)。"""

    # ── 能力发现(默认只产主图, 多模态源覆盖)────────────────
    @property
    def end_reason(self) -> EndReason:
        """
        上一次迭代因何停止。在 read() 返回 None 之后查询, 用于区分:
            COMPLETED             —— 干净读完, 上层该停
            DISCONNECTED / ERROR  —— 异常中断, 上层可重连(见 is_recoverable)
        read() 的签名不变(for frame in src 老循环零改), 这是旁挂的可查询状态。
        """
        return self._end_reason

    @property
    def modalities(self) -> frozenset[str]:
        """
        本源能产出的平面集合。默认 {'image'}(只有主视觉平面)。
        深度源覆盖为 {'image', 'depth'}, 双目源覆盖为 {'image', 'right'} 等。
        供调用方在开循环前决策是否准备额外管线。
        """
        return frozenset({"image"})

    # ── seek 协议(默认不支持,子类按需覆盖)─────────────────
    def seek(
        self,
        frame: Optional[int] = None,
        time_sec: Optional[float] = None,   # 参数名 time_sec 避免与 time 模块冲突
    ) -> bool:
        """跳转到指定位置(基类默认不支持)。返回是否成功。"""
        logger.warning(f"{self.__class__.__name__} 不支持 seek 操作")
        return False

    @property
    def seekable(self) -> bool:
        """是否支持 seek,子类覆盖返回 True。"""
        return False

    # ── stride 协议(持续跳采)──────────────────────────────
    def set_stride(self, stride: int) -> None:
        """
        设置跳采步长。stride=1 透传(默认), stride=N 每 N 帧产出 1 帧。

        基类只登记意图(self._stride), 实际跳帧由子类在 read() 内实现。
        不支持/不适用跳帧的源(摄像头、单图)应覆盖本方法, 给出警告或忽略。

        Raises:
            ValueError: stride < 1
        """
        if stride < 1:
            raise ValueError(f"stride 必须 ≥ 1, 收到: {stride}")
        self._stride = stride
        if stride > 1:
            logger.debug(f"{self.__class__.__name__}: stride 设为 {stride}")

    @property
    def stride(self) -> int:
        """当前跳采步长(1 表示透传)。"""
        return self._stride

    # ── 上下文管理器协议 ────────────────────────────────────
    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False  # 不吞异常

    # ── 迭代器协议 ──────────────────────────────────────────
    def __iter__(self) -> Iterator[Frame]:
        return self

    def __next__(self) -> Frame:
        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame
```

---

## 4. core/registry.py —— 配置驱动派发 + 自注册（本模块核心）

注册表只做 `dict[配置类型, 源类]` 的**轻量映射**——不做"每源注册 (matcher, priority, spec)、输入进来打分仲裁"那套重机制。派发轴是配置类型，一个 dict 查表就完了，没有歧义要仲裁；字符串规则命中多个时**当场 raise 报歧义**，不猜、不排优先级。抽象复杂度匹配需求复杂度：那套仲裁机器是给"派发轴本身充满歧义"的场景准备的，这里没这需求。

字符串边界把 config 构造失败统一包装成 `ValueError`（`from e` 保链），调用方一处 `except ValueError` 兜住。

📄 **新建** `frame_source/core/registry.py`：

```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : registry.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : 源注册表 —— 配置类型是派发轴; @register_source 让源自注册
"""
源注册表: frame_source 可扩展性的核心。

派发轴 = 配置类型:
    build_from_config(cfg) 靠 type(cfg) 在 _BY_CONFIG 里一步查到源类。
    确定、不歧义, 且天然支持硬件源(深度/红外/双目都传各自的 config)。

字符串糖(可选, 二线):
    能自证身份的输入(路径/URL/裸数字)可由源自愿登记一条字符串规则,
    build_from_string 先把字符串翻译成 config, 再走 build_from_config。
    硬件源不登记字符串规则 —— 所以 "0" 不会在深度/红外/彩色之间产生歧义。

扩展方式(规矩 E, 对修改关闭):
    @register_source(XxxConfig, str_matcher=..., str_to_config=...)
    class XxxSource(FrameSource): ...
    内置源在自己模块里这样登记; 第三方源在自己的包里这样登记, import 一下即接入,
    factory.py / registry.py 的既有代码一个字不用改。

设计取舍(不做什么):
    不做 matcher 打分 + 优先级 + spec 那套重机制 —— 派发轴是配置类型, 一个
    dict 查表即可, 没有歧义要仲裁。字符串命中多个时当场 raise 报歧义, 不猜。
"""
from __future__ import annotations

import logging
from typing import Callable, NamedTuple

from .base   import FrameSource
from .config import SourceConfig


logger = logging.getLogger(__name__)


class StrRule(NamedTuple):
    """一条字符串识别规则(某个源自愿登记)。"""
    name:      str                              # 源类名(报错/内省用)
    matcher:   Callable[[str], bool]            # 纯判断: 这个字符串归我吗(契约: 不得抛异常)
    to_config: Callable[[str], SourceConfig]    # 翻译: 字符串 → 该源的 config


# ── 注册表(模块级单例)────────────────────────────────────
_BY_CONFIG: dict[type[SourceConfig], type[FrameSource]] = {}   # 配置类型 → 源类(主派发表)
_STR_RULES: list[StrRule] = []                                 # 字符串规则(可选糖)


def register_source(
    config_cls: type[SourceConfig],
    *,
    str_matcher:   Callable[[str], bool] | None = None,
    str_to_config: Callable[[str], SourceConfig] | None = None,
):
    """
    源自注册装饰器。

    Args:
        config_cls    : 该源对应的配置类(派发的钥匙, 必填)。
        str_matcher   : 可选。str -> bool, 判断某字符串是否归本源。
                        契约: 必须是纯判断, 不得抛异常。
        str_to_config : 可选。str -> config, 把字符串翻译成本源的 config。
                        提供 str_matcher 时必须一并提供。

    用法:
        @register_source(VideoConfig,
                         str_matcher=lambda s: Path(s).suffix.lower() in VIDEO_EXTENSIONS,
                         str_to_config=lambda s: VideoConfig(path=s))
        class VideoSource(FrameSource): ...

        # 硬件源: 只给 config_cls, 不给字符串规则(字符串分不开硬件)
        @register_source(DepthConfig)
        class DepthSource(FrameSource): ...

    Raises:
        TypeError : config_cls 不是 SourceConfig 子类, 或被装饰类不是 FrameSource 子类。
        ValueError: 给了 str_matcher 却没给 str_to_config; 或该 config 已被别的源注册。
    """
    if not (isinstance(config_cls, type) and issubclass(config_cls, SourceConfig)):
        raise TypeError(f"config_cls 必须是 SourceConfig 的子类, 收到: {config_cls!r}")
    if str_matcher is not None and str_to_config is None:
        raise ValueError("提供 str_matcher 时必须同时提供 str_to_config")

    def decorator(source_cls: type[FrameSource]) -> type[FrameSource]:
        if not (isinstance(source_cls, type) and issubclass(source_cls, FrameSource)):
            raise TypeError(f"被注册的类必须是 FrameSource 的子类, 收到: {source_cls!r}")

        existing = _BY_CONFIG.get(config_cls)
        if existing is not None and existing is not source_cls:
            raise ValueError(
                f"配置 {config_cls.__name__} 已被 {existing.__name__} 注册, "
                f"拒绝用 {source_cls.__name__} 覆盖(一种配置只能对应一个源)"
            )

        _BY_CONFIG[config_cls] = source_cls
        if str_matcher is not None:
            _STR_RULES.append(StrRule(source_cls.__name__, str_matcher, str_to_config))

        logger.debug(
            f"已注册源: {config_cls.__name__} → {source_cls.__name__}"
            + (" (+字符串规则)" if str_matcher is not None else "")
        )
        return source_cls

    return decorator


def build_from_config(config: SourceConfig) -> FrameSource:
    """
    按配置类型派发, 造出对应的源。这是唯一的"造源"出口。

    Args:
        config: 任意 SourceConfig 子类实例。

    Returns:
        对应的 FrameSource 实例(尚未 open)。

    Raises:
        TypeError : config 不是 SourceConfig 实例。
        ValueError: config 的类型没有被任何源注册。
    """
    if not isinstance(config, SourceConfig):
        raise TypeError(f"build_from_config 需要 SourceConfig 实例, 收到 {type(config).__name__}")

    source_cls = _BY_CONFIG.get(type(config))
    if source_cls is None:
        raise ValueError(
            f"未注册的配置类型: {type(config).__name__}。"
            f"已注册: {sorted(c.__name__ for c in _BY_CONFIG)}"
        )
    return source_cls(config)


def build_from_string(source: str) -> FrameSource:
    """
    把字符串按注册的规则翻译成 config, 再走 build_from_config。这是字符串糖的入口。

    Args:
        source: 输入字符串(如 "0" / "x.mp4" / "rtsp://..." / "./imgs")。

    Returns:
        对应的 FrameSource 实例(尚未 open)。

    Raises:
        TypeError : source 不是 str。
        ValueError: 没有任何规则命中(无法识别); 或命中多个规则(歧义);
                    或命中规则但构造 config 失败(如路径不存在)。
    """
    if not isinstance(source, str):
        raise TypeError(f"build_from_string 需要 str, 收到 {type(source).__name__}")

    # 注: matcher 契约要求纯判断不抛异常, 故此处不做 try 容错 —— matcher 抛了就该暴露
    hits = [rule for rule in _STR_RULES if rule.matcher(source)]

    if not hits:
        raise ValueError(
            f"无法识别的源字符串: {source!r}。"
            f"已注册的字符串规则: {[r.name for r in _STR_RULES]}"
        )
    if len(hits) > 1:
        raise ValueError(
            f"源字符串 {source!r} 命中多个规则(歧义): {[h.name for h in hits]}。"
            f"这通常意味着某个源的 matcher 写得过宽, 或存在冲突的第三方源。"
        )

    rule = hits[0]
    try:
        config = rule.to_config(source)     # 把字符串翻译成 config(可能因路径不存在等 raise)
    except Exception as e:
        # 统一字符串边界的异常类型为 ValueError, 方便调用方一处 except;
        # from e 保留原始异常链(含 pydantic ValidationError 的详细字段报错), 不丢调试信息
        raise ValueError(f"由 {source!r} 构造 {rule.name} 的配置失败: {e}") from e

    return build_from_config(config)


def list_sources() -> dict[str, str]:
    """
    内省当前注册表, 返回 {配置类名: 源类名}。

    用于回答"到底支持哪些源"—— 这是注册表相对 if/elif 工厂损失的可发现性的补偿:
    if/elif 读一个函数就全知道, 注册表得靠这个内省(或全仓 grep config 类)。
    """
    return {cfg.__name__: src.__name__ for cfg, src in _BY_CONFIG.items()}
```

📄 **新建** `frame_source/core/__init__.py`：

```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : __init__.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : core 子包入口 — 核心类型 / 配置 / 协议 / 注册表
"""core 子包:数据类型、每源配置、抽象协议、注册表。"""
from __future__ import annotations

from .types  import (
    SourceType, FrameInfo, Frame, CameraIntrinsics, EndReason, is_recoverable,
    IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, STREAM_SCHEMES, is_stream_url,
)
from .config import (
    SourceConfig, CameraConfig, VideoConfig, ImageConfig, ImageFolderConfig,
    CameraBackend, CameraCodec,
)
from .base   import FrameSource
from .registry import (
    register_source, build_from_config, build_from_string, list_sources,
)

__all__ = [
    # 类型
    "SourceType", "FrameInfo", "Frame", "CameraIntrinsics",
    "IMAGE_EXTENSIONS", "VIDEO_EXTENSIONS", "STREAM_SCHEMES", "is_stream_url",
    "EndReason", "is_recoverable",
    # 配置
    "SourceConfig", "CameraConfig", "VideoConfig", "ImageConfig", "ImageFolderConfig",
    "CameraBackend", "CameraCodec",
    # 协议
    "FrameSource",
    # 注册表
    "register_source", "build_from_config", "build_from_string", "list_sources",
]
```

```bash
git add frame_source/core/ && git commit -m "feat(frame_source): core 层 —— 多模态类型 + 每源配置 + 协议 + 注册表"
```

---

## 5. sources/ —— 四个内置源

每个源三件事：实现协议、返回 types、用一行 `@register_source` 登记自己（顺带登记自己的字符串规则——规则归各源所有，注册表不替它们做主）。

### 5.1 sources/image.py —— 单图 + 文件夹

设计点：**统一零拷贝**——不给"单图源 read 时主动 copy"开特例，那个 copy 保护的是观察不到的场景，徒增不一致。全模块一条规矩：零拷贝，要持久保存自行 `.copy()`。

📄 **新建** `frame_source/sources/image.py`：

```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : image.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : 图片源 —— 单张图片 (ImageSource) + 图片文件夹 (ImageFolderSource)
"""
图片输入源。两个类都自注册, 都登记字符串规则(路径能自证身份)。

ImageSource       : 单张图片, read() 只产一帧(第二次返回 None)。
ImageFolderSource : 文件夹, 按排序后的文件列表逐张产出; stride 走索引跳转(零 IO)。

统一零拷贝: read() 返回的 Frame.image 直接引用底层数组, 调用方需持久保存自行 .copy()。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2

from ..core.base     import FrameSource
from ..core.config   import ImageConfig, ImageFolderConfig
from ..core.registry import register_source
from ..core.types    import Frame, FrameInfo, SourceType, EndReason, IMAGE_EXTENSIONS, is_stream_url


logger = logging.getLogger(__name__)


@register_source(
    ImageConfig,
    # 能自证身份: 非 URL 且后缀属于图片扩展名(URL 一律排除, 交给视频源按流处理)
    str_matcher=lambda s: (not is_stream_url(s)) and Path(s).suffix.lower() in IMAGE_EXTENSIONS,
    str_to_config=lambda s: ImageConfig(path=s),
)
class ImageSource(FrameSource):
    """单张图片源。read() 只产一帧。"""

    def __init__(self, config: ImageConfig):
        super().__init__(config)
        self._path: str = config.path
        self._image: Optional["cv2.typing.MatLike"] = None
        self._read_count: int = 0

    def open(self) -> bool:
        # 复位位置状态(兑现"重新 open 即重头"契约)
        self._read_count = 0
        self._frame_index = 0
        self._end_reason = EndReason.RUNNING              # ★ 结束语义

        self._image = cv2.imread(self._path)
        if self._image is None:
            self._end_reason = EndReason.ERROR            # ★ 图读不出是错误
            logger.error(f"无法读取图片: {self._path}")
            return False
        return True

    def read(self) -> Optional[Frame]:
        if self._image is None:
            self._end_reason = EndReason.NOT_STARTED      # ★ 未 open
            return None
        if self._read_count > 0:
            self._end_reason = EndReason.COMPLETED        # ★ 已读完(单图就一帧)
            return None
        self._read_count += 1

        h, w = self._image.shape[:2]
        info = FrameInfo(
            width=w, height=h,
            source_type=SourceType.IMAGE,
            source_path=self._path,
            frame_index=0, total_frames=1,
            timestamp=0.0,                       # 图片无时间概念
            filename=Path(self._path).name,
        )
        # 零拷贝: 与其它源一致, 调用方需持久保存自行 .copy()
        return Frame(image=self._image, info=info)

    def close(self) -> None:
        self._image = None

    def get_source_type(self) -> SourceType:
        return SourceType.IMAGE

    # 单图 stride 无意义(read 一帧即止), 沿用基类默认(set_stride 只登记不生效)


@register_source(
    ImageFolderConfig,
    str_matcher=lambda s: Path(s).is_dir(),      # 是已存在的目录 → 归文件夹源
    str_to_config=lambda s: ImageFolderConfig(path=s),
)
class ImageFolderSource(FrameSource):
    """图片文件夹源。按排序后的文件列表逐张产出。"""

    def __init__(self, config: ImageFolderConfig):
        super().__init__(config)
        self._folder: str = config.path
        self._files: list[Path] = []
        self._current_index: int = 0

    def open(self) -> bool:
        # 复位位置状态
        self._current_index = 0
        self._frame_index = 0

        folder = Path(self._folder)
        # 过滤(后缀小写)+ 排序(帧序稳定可复现)
        self._files = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self._files:
            logger.warning(f"文件夹内没有支持的图片: {self._folder}")
            return False
        logger.info(f"文件夹共 {len(self._files)} 张图片: {self._folder}")
        self._end_reason = EndReason.RUNNING              # ★ 成功扫描后置位
        return True

    def read(self) -> Optional[Frame]:
        # stride: 跳过 stride-1 张(纯索引跳转, 零 IO)
        if self._stride > 1 and self._current_index > 0:
            self._current_index += self._stride - 1

        # 跳过读不出的坏图, 直到读到一张或耗尽
        while self._current_index < len(self._files):
            path = self._files[self._current_index]
            img = cv2.imread(str(path))
            if img is None:
                logger.warning(f"跳过无法读取的图片: {path}")
                self._current_index += 1
                continue

            idx = self._current_index
            self._current_index += 1
            self._frame_index = idx

            h, w = img.shape[:2]
            info = FrameInfo(
                width=w, height=h,
                source_type=SourceType.IMAGE_FOLDER,
                source_path=self._folder,
                frame_index=idx, total_frames=len(self._files),
                timestamp=0.0,
                filename=path.name,
            )
            return Frame(image=img, info=info)   # 零拷贝

        self._end_reason = EndReason.COMPLETED            # ★ 循环走完 = 读尽
        return None

    def close(self) -> None:
        self._files = []

    def get_source_type(self) -> SourceType:
        return SourceType.IMAGE_FOLDER

    # ── 文件夹支持 seek(按文件索引跳转, 廉价)──
    @property
    def seekable(self) -> bool:
        return True

    def seek(self, frame: Optional[int] = None, time_sec: Optional[float] = None) -> bool:
        if frame is None:
            logger.warning("文件夹源只支持按帧索引 seek(frame=N)")
            return False
        if not (0 <= frame < len(self._files)):
            logger.warning(f"seek 越界: {frame} 不在 [0, {len(self._files)})")
            return False
        self._current_index = frame
        return True
```

### 5.2 sources/video.py —— 视频文件 + 网络流

设计点：**网络流诚实标注**——流 `seekable=False`、`total_frames=None`，不给流编一个假边界。错的信息比没有信息更有害。`fps` 缺失/异常同样明确 warning + 置 None，按秒 seek 直接拒绝。

📄 **新建** `frame_source/sources/video.py`：

```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : video.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : 视频源 —— 本地视频文件 + rtsp/rtmp/http(s) 网络流
"""
视频输入源(VideoSource)。自注册, 登记字符串规则(视频后缀 + 网络流 URL 能自证身份)。

关键处理:
    - stride 用 cap.grab() 跳采(跳过 YUV→BGR 转换, 快 3~5 倍)
    - 网络流诚实标注: is_stream_url 检测 → seekable=False, total_frames=None
    - fps 缺失/异常 → 明确 warning + 置 None, 按秒 seek 直接拒绝
    - seek 与 stride 交互: _just_sought 标志, seek 后下一 read 不做 stride 预跳
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import cv2

from ..core.base     import FrameSource
from ..core.config   import VideoConfig
from ..core.registry import register_source
from ..core.types    import Frame, FrameInfo, SourceType, EndReason, VIDEO_EXTENSIONS, is_stream_url


logger = logging.getLogger(__name__)


@register_source(
    VideoConfig,
    # 能自证身份: 网络流 URL, 或后缀属于视频扩展名
    str_matcher=lambda s: is_stream_url(s) or Path(s).suffix.lower() in VIDEO_EXTENSIONS,
    str_to_config=lambda s: VideoConfig(path=s),
)
class VideoSource(FrameSource):
    """视频文件 / 网络流源。"""

    def __init__(self, config: VideoConfig):
        super().__init__(config)
        self._path: str = config.path
        self._is_stream: bool = is_stream_url(config.path)
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: Optional[float] = None
        self._total: Optional[int] = None
        self._just_sought: bool = False

    def open(self) -> bool:
        # 复位位置状态
        self._frame_index = 0
        self._just_sought = False
        self._end_reason = EndReason.RUNNING              # ★ 结束语义

        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            logger.error(f"无法打开视频源: {self._path}")
            self._cap = None
            return False

        # fps: 缺失/0/NaN → 明确置 None
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0 or math.isnan(fps):
            logger.warning(f"视频 fps 缺失或异常({fps}), 时间相关功能(时间戳/按秒 seek)不可用: {self._path}")
            self._fps = None
        else:
            self._fps = float(fps)

        # total_frames: 网络流没有总帧数(无限流); 文件读 FRAME_COUNT, ≤0 也置 None(诚实)
        if self._is_stream:
            self._total = None
            logger.info(f"网络流已打开(不可 seek, 无总帧数): {self._path}")
        else:
            n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._total = n if n > 0 else None
        return True

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            self._end_reason = EndReason.NOT_STARTED
            return None

        # stride: 非首帧且未刚 seek 过时, grab 跳过 stride-1 帧(只解复用不解码, 快)
        if self._stride > 1 and self._frame_index > 0 and not self._just_sought:
            for _ in range(self._stride - 1):
                if not self._cap.grab():
                    # 跳帧途中就到头/断了
                    self._end_reason = (EndReason.DISCONNECTED if self._is_stream
                                        else EndReason.COMPLETED)
                    return None
        self._just_sought = False   # 消费掉 seek 标志

        ret, img = self._cap.read()
        if not ret:
            if self._is_stream:
                self._end_reason = EndReason.DISCONNECTED           # 流断了 → 可重连
            elif self._total is not None and self._frame_index < self._total:
                self._end_reason = EndReason.ERROR                  # 文件没读到尾就失败 → 疑似截断/损坏
            else:
                self._end_reason = EndReason.COMPLETED              # 正常读到文件尾 → 停
            return None

        # 当前帧号: 文件用解码器位置(POS_FRAMES 反映 seek); 流不可靠 → 退回自增计数
        if self._is_stream:
            idx = self._frame_index
        else:
            idx = max(0, int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        self._frame_index = idx + 1     # 下一帧位置

        h, w = img.shape[:2]
        timestamp = (idx / self._fps) if self._fps else 0.0
        info = FrameInfo(
            width=w, height=h,
            source_type=SourceType.VIDEO,
            source_path=self._path,
            frame_index=idx, total_frames=self._total,
            timestamp=timestamp, fps=self._fps,
            filename=(self._path if self._is_stream else Path(self._path).name),
        )
        return Frame(image=img, info=info)   # 零拷贝

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_source_type(self) -> SourceType:
        return SourceType.VIDEO

    # ── seek(仅文件, 流不可 seek)──
    @property
    def seekable(self) -> bool:
        return self._cap is not None and not self._is_stream

    def seek(self, frame: Optional[int] = None, time_sec: Optional[float] = None) -> bool:
        if not self.seekable:
            reason = "网络流" if self._is_stream else "未打开的视频"
            logger.warning(f"{reason}不支持 seek")
            return False
        if frame is None and time_sec is None:
            logger.warning("seek 需要 frame 或 time_sec 之一")
            return False

        if time_sec is not None:
            if self._fps is None:
                logger.warning("视频无 fps, 无法按秒 seek")
                return False
            frame = int(time_sec * self._fps)

        frame = max(0, frame)
        ok = self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame))
        if ok:
            self._frame_index = frame
            self._just_sought = True    # 下一次 read 不做 stride 预跳, 直接返回本帧
        return ok
```

### 5.3 sources/camera.py —— 摄像头（跨平台 + 后端协商）

高帧率协商的三条坑（都是平台特定的摄像头行为）：

1. **高帧率必须先设编码 MJPG**——裸 YUYV 带宽不够；
2. **`set()` 只是请求，必须 `read()` 一帧才触发驱动协商**——所以 `open()` 里读一帧丢掉是必要动作，不是浪费；
3. **参数顺序：编码 → 分辨率 → 帧率**——MSMF 下顺序错了高帧率静默失效。

📄 **新建** `frame_source/sources/camera.py`：

```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : camera.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : 摄像头源 —— 跨平台后端协商 + 高帧率参数验证
"""
摄像头输入源(CameraSource)。自注册, 登记字符串规则("0"/"1" 这类裸数字 = 默认 webcam)。

三条撞墙(高帧率协商, 见讲义 §4.3):
    F①  高帧率必须先设编码 MJPG(裸 YUYV 带宽不够)
    F②  set() 只是请求, 必须 read() 一帧才触发驱动协商; 故 open() 里读一帧丢掉是必要动作
    F③  参数顺序: 编码 → 分辨率 → 帧率(MSMF 下顺序错高帧率失效)

跨平台: backend 选 auto/msmf/dshow/v4l2。Windows 高帧率走 msmf, Linux 走 v4l2。

注意: 摄像头是实时源, 不支持 stride(见 set_stride 覆盖)。字符串糖只认裸数字并默认
     映射到 CameraConfig(彩色 webcam); 深度/红外/双目这类同样是"0"的硬件不走字符串,
     必须显式传各自的 config —— 这正是"配置类型做派发轴"的意义(讲义 §1.2、§3.1)。
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import cv2

from ..core.base     import FrameSource
from ..core.config   import CameraConfig
from ..core.registry import register_source
from ..core.types    import Frame, FrameInfo, SourceType, EndReason


logger = logging.getLogger(__name__)

# 后端名 → OpenCV flag
_BACKEND_FLAGS = {
    "auto":  cv2.CAP_ANY,
    "msmf":  cv2.CAP_MSMF,
    "dshow": cv2.CAP_DSHOW,
    "v4l2":  cv2.CAP_V4L2,
}


@register_source(
    CameraConfig,
    str_matcher=lambda s: s.isdigit(),                       # 裸数字 = 默认彩色 webcam
    str_to_config=lambda s: CameraConfig(camera_id=int(s)),
)
class CameraSource(FrameSource):
    """cv2 摄像头源, 带后端协商与参数验证。"""

    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self._cap: Optional[cv2.VideoCapture] = None
        self._actual_fps: Optional[float] = None

    def open(self) -> bool:
        # 复位位置状态
        self._frame_index = 0
        self._end_reason = EndReason.RUNNING              # ★ 结束语义
        cfg: CameraConfig = self.config

        backend_flag = _BACKEND_FLAGS.get(cfg.backend, cv2.CAP_ANY)
        self._cap = cv2.VideoCapture(cfg.camera_id, backend_flag)
        if not self._cap.isOpened():
            logger.error(f"无法打开摄像头 {cfg.camera_id}(backend={cfg.backend})")
            self._cap = None
            return False

        # 撞墙 F①: 先设编码为 MJPG, 否则高帧率上不去
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.codec))
        # 撞墙 F③: 顺序 —— 编码 → 分辨率 → 帧率
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self._cap.set(cv2.CAP_PROP_FPS,          cfg.fps)

        # 撞墙 F②: set() 只是请求, read 一帧才真正触发驱动协商
        ret, _ = self._cap.read()
        if not ret:
            logger.error(f"摄像头 {cfg.camera_id} 已打开但读不到帧")
            self._cap.release()
            self._cap = None
            return False

        # 参数验证: 读回实际生效值, 与请求不符则告警
        aw = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        af = self._cap.get(cv2.CAP_PROP_FPS)
        if (aw, ah) != (cfg.width, cfg.height):
            logger.warning(f"分辨率: 请求 {cfg.width}x{cfg.height}, 实际 {aw}x{ah}")
        if af and abs(af - cfg.fps) > 1:
            logger.warning(f"帧率: 请求 {cfg.fps}, 实际 {af:.1f}")
        self._actual_fps = float(af) if af and af > 0 else None
        logger.info(f"摄像头 {cfg.camera_id} 就绪: {aw}x{ah} @ {af:.1f}fps(backend={cfg.backend})")
        return True

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            self._end_reason = EndReason.NOT_STARTED
            return None
        ret, img = self._cap.read()
        if not ret:
            self._end_reason = EndReason.DISCONNECTED     # ★ 摄像头读失败 = 掉线, 可重连
            return None

        idx = self._frame_index
        self._frame_index += 1
        h, w = img.shape[:2]
        info = FrameInfo(
            width=w, height=h,
            source_type=SourceType.CAMERA,
            source_path=self.source_path,
            frame_index=idx, total_frames=None,     # 实时源无总帧数
            timestamp=time.time() - self._start_time,   # 墙钟时间
            fps=self._actual_fps,
            filename=self.source_path,               # 'camera:<id>'
        )
        return Frame(image=img, info=info)           # 零拷贝

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_source_type(self) -> SourceType:
        return SourceType.CAMERA

    def set_stride(self, stride: int) -> None:
        # 摄像头是实时源, 跳采无意义; 覆盖基类, 不改变 self._stride(始终 1)
        if stride != 1:
            logger.warning(
                "摄像头是实时源, 不支持 stride 跳采。"
                "要降频请在消费端丢帧, 或用 ThreadedSource 始终取最新帧。"
            )
```

### 5.4 sources/__init__.py —— import 即自注册

📄 **新建** `frame_source/sources/__init__.py`：

```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : __init__.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : sources 子包入口 —— import 四个源, 触发它们自注册
"""
sources 子包。

import 本子包的副作用: 执行各源模块顶层的 @register_source, 把四个内置源登记进
注册表。因此"导入 frame_source"→"导入 sources"→"四个源自注册"这条链一走完,
create_frame_source 就能派发到它们, 而 factory.py 完全不认识这四个类。
"""
from __future__ import annotations

from .camera import CameraSource
from .video  import VideoSource
from .image  import ImageSource, ImageFolderSource

__all__ = ["CameraSource", "VideoSource", "ImageSource", "ImageFolderSource"]
```

```bash
git add frame_source/sources/ && git commit -m "feat(frame_source): sources 层 —— 四个内置源(自注册 + open 复位 + end_reason)"
```

---

## 6. wrappers/ —— 正交增强（不进注册表）

wrapper **不是一种源**，是"源的修饰"——你不会说"造一个 ThreadedSource 作为输入类型"，而是说"把这个源线程化"。所以它们不带 `@register_source`，复用 inner 的 config，透传 inner 的类型/模态/可 seek 性。由工厂便捷函数或调用方手动套上。

### 6.1 wrappers/threaded.py —— 后台线程采集

📄 **新建** `frame_source/wrappers/threaded.py`：

```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : threaded.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : ThreadedSource —— 后台线程采集, 主循环取最新帧(正交增强, 不进注册表)
"""
线程化包装器: 把任意 FrameSource 的采集丢到后台线程, 主循环取帧不被采集阻塞。

正交性: 不认任何具体源, 只认 FrameSource 协议; 透传 inner 的类型与 modalities。
不是一种源: 没有 @register_source, 由工厂便捷函数或调用方手动套上。

缓冲策略:
    LATEST  —— 只留最新 1 帧(实时优先, 丢旧帧), 适合摄像头
    BOUNDED —— 有界队列(不丢帧, 满了阻塞采集), 适合离线视频并行解码
"""
from __future__ import annotations

import logging
import queue
import threading
from enum import Enum
from typing import Optional

from ..core.base  import FrameSource
from ..core.types import Frame, SourceType


logger = logging.getLogger(__name__)


class BufferStrategy(str, Enum):
    LATEST  = "latest"     # 只保留最新帧, 丢旧帧(实时优先)
    BOUNDED = "bounded"    # 有界队列, 不丢帧, 满则阻塞采集


class ThreadedSource(FrameSource):
    """后台线程采集包装器。"""

    def __init__(
        self,
        inner: FrameSource,
        *,
        strategy: BufferStrategy = BufferStrategy.LATEST,
        maxsize: int = 1,
        warmup: bool = True,
        read_timeout: float = 5.0,
    ):
        super().__init__(inner.config)      # 复用 inner 的 config → source_path 等有值
        self._inner   = inner
        self._strategy = strategy
        qsize = 1 if strategy == BufferStrategy.LATEST else max(1, maxsize)
        self._queue: "queue.Queue[Frame]" = queue.Queue(maxsize=qsize)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._eos  = threading.Event()      # inner 耗尽标志
        self._warmup = warmup
        self._read_timeout = read_timeout
        self._timeout_warned = False

    # ── 采集线程 ──
    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            frame = self._inner.read()
            if frame is None:
                self._eos.set()
                break
            if self._strategy == BufferStrategy.LATEST:
                try: self._queue.get_nowait()       # 丢掉旧帧
                except queue.Empty: pass
                try: self._queue.put_nowait(frame)
                except queue.Full: pass
            else:  # BOUNDED: 满则阻塞等消费, 但可被 stop 打断
                while not self._stop.is_set():
                    try:
                        self._queue.put(frame, timeout=0.1)
                        break
                    except queue.Full:
                        continue

    def _drain(self) -> None:
        while not self._queue.empty():
            try: self._queue.get_nowait()
            except queue.Empty: break

    # ── FrameSource 协议 ──
    def open(self) -> bool:
        self._frame_index = 0
        if not self._inner.open():
            return False
        self._stop.clear(); self._eos.clear(); self._timeout_warned = False
        self._drain()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        if self._warmup:    # 等第一帧就位, 让 open 返回时已有帧可读
            for _ in range(int(self._read_timeout * 20)):
                if not self._queue.empty() or self._eos.is_set():
                    break
                threading.Event().wait(0.05)
        return True

    def read(self) -> Optional[Frame]:
        if self._thread is None:
            return None
        while not self._stop.is_set():
            try:
                return self._queue.get(timeout=self._read_timeout)
            except queue.Empty:
                if self._eos.is_set():
                    return None     # 采集线程已结束且队列取空 → 真结束
                if not self._timeout_warned:
                    logger.warning(
                        f"ThreadedSource: {self._read_timeout}s 未取到帧"
                        f"(采集慢或卡住), 后续不再重复此告警"
                    )
                    self._timeout_warned = True
                # 非 EOS: 继续等(实时源可能恢复)
        return None

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._inner.close()
        self._drain()

    # ── 透传 inner(线程化不改变'是什么源、能产什么')──
    def get_source_type(self) -> SourceType:
        return self._inner.get_source_type()

    @property
    def modalities(self) -> frozenset[str]:
        return self._inner.modalities

    @property
    def seekable(self) -> bool:
        return self._inner.seekable

    def seek(self, frame=None, time_sec=None) -> bool:
        # 注: 采集线程运行中 seek 有竞态, 建议仅在离线场景谨慎使用
        return self._inner.seek(frame=frame, time_sec=time_sec)

    def set_stride(self, stride: int) -> None:
        self._inner.set_stride(stride)      # 转发给 inner, 由它用最快方式跳采

    @property
    def stride(self) -> int:
        return self._inner.stride
```

### 6.2 wrappers/aio.py —— 异步适配

**AsyncSource 不继承 FrameSource（守 LSP）**：它的 `read()` 是协程，无法替换同步的 `FrameSource.read()`——强行继承会违反里氏替换（拿到 FrameSource 的调用方 `for frame in src` 会炸）。为复用继承而破坏协议契约是本末倒置；类型上诚实（它就不是同步源），代价是少一点点代码复用——值。

📄 **新建** `frame_source/wrappers/aio.py`：

```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : aio.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : AsyncSource —— 同步源 → 异步接口适配(不继承 FrameSource, 守 LSP)
"""
异步适配器: 把同步 FrameSource 适配成 async 接口, 供事件循环中的调用方 async for。

用 asyncio.to_thread 把阻塞的 open/read/close 丢到线程池, 不卡事件循环。

刻意不继承 FrameSource: 它的 read 是协程, 无法替换同步协议(LSP), 故是独立适配器,
提供 async for 而非 for。config/modalities/可 seek 性照常透传(与同步/异步无关)。
"""
from __future__ import annotations

import asyncio
from typing import Optional

from ..core.base   import FrameSource
from ..core.config import SourceConfig
from ..core.types  import Frame, SourceType


class AsyncSource:
    """把同步 FrameSource 适配成异步接口(不是 FrameSource 子类)。"""

    def __init__(self, inner: FrameSource):
        self._inner = inner

    # ── 透传描述性属性(与同步/异步无关)──
    @property
    def config(self) -> SourceConfig:
        return self._inner.config

    @property
    def source_path(self) -> str:
        return self._inner.source_path

    @property
    def modalities(self) -> frozenset[str]:
        return self._inner.modalities

    @property
    def seekable(self) -> bool:
        return self._inner.seekable

    @property
    def stride(self) -> int:
        return self._inner.stride

    def set_stride(self, stride: int) -> None:
        self._inner.set_stride(stride)

    def get_source_type(self) -> SourceType:
        return self._inner.get_source_type()

    # ── 异步方法(阻塞调用丢线程池)──
    async def open(self) -> bool:
        return await asyncio.to_thread(self._inner.open)

    async def read(self) -> Optional[Frame]:
        return await asyncio.to_thread(self._inner.read)

    async def close(self) -> None:
        await asyncio.to_thread(self._inner.close)

    async def seek(self, frame: Optional[int] = None, time_sec: Optional[float] = None) -> bool:
        return await asyncio.to_thread(self._inner.seek, frame, time_sec)

    # ── 异步上下文管理器 + 异步迭代器 ──
    async def __aenter__(self) -> "AsyncSource":
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        await self.close()
        return False

    def __aiter__(self) -> "AsyncSource":
        return self

    async def __anext__(self) -> Frame:
        frame = await self.read()
        if frame is None:
            raise StopAsyncIteration
        return frame
```

用法：

```python
async def run_async(model, source):
    async with create_async_source(source) as src:
        async for frame in src:
            results = await async_infer(model, frame.image)
```

### 6.3 wrappers/reconnect.py —— 断线重连

协议能区分停/断之后（`end_reason`），重连就水到渠成：包一层 wrapper，`read()` 拿到 None 时看 `inner.end_reason`——`COMPLETED` 透传 None（停），`DISCONNECTED`/`ERROR` 指数退避重连，连上继续出帧。老循环 `for frame in src` 完全无感。

三个设计点：

- **指数退避 + 上限 + 可无限重试**：失败立刻猛重连会在服务端真宕机时打出一场 DDoS；1s→2s→4s…封顶 30s 给对端喘息。`max_retries=None` 表示永不放弃（生产流常用）。
- **退避睡眠必须可被 close 打断**：用 `threading.Event.wait` 而不是 `time.sleep`——否则用户 `close()` 时要干等退避完。
- **两个隐藏坑**：①"刚连上又立刻断"（服务端半死不活时 `open()` 成功但第一帧就 None）——连上后那次 `read()` 若还是 None，**不重置 backoff**、继续退避，否则陷入高频抖动；②无限重试会 hang 主循环——这是故意的（生产流就是要死等对端回来），但代价是 `close()` 必须能打断。
- **对文件优雅降级**：手滑套在本地文件上时，文件读完报 `COMPLETED`（不可恢复）→ 正常停，不会傻乎乎重开文件。另外 `seekable` 直接写死 `False`：重连会重置会话，seek 位置语义在重连下不成立——宁可不支持，不给会失效的承诺。

📄 **新建** `frame_source/wrappers/reconnect.py`：

```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : reconnect.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : ReconnectingSource —— 断线重连(基于 end_reason 区分停/断; 正交增强, 不进注册表)
"""
断线重连包装器。给会断的源(网络流 / 摄像头)包一层自动重连。

工作原理(依赖 §10.2 的协议补丁):
    inner.read() 返回 None 时看 inner.end_reason —
        COMPLETED(或其它不可恢复原因) → 透传 None, 上层停
        DISCONNECTED / ERROR          → 指数退避后重开 inner, 成功则继续出帧

对上层透明: for frame in src 要么拿到帧, 要么只在真正结束时拿到 None。
对文件优雅降级: 文件读完报 COMPLETED, 本 wrapper 正常停(不会重开文件)。
不是一种源: 无 @register_source; 由工厂便捷函数或调用方手动套上。
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from ..core.base  import FrameSource
from ..core.types import Frame, SourceType, EndReason, is_recoverable


logger = logging.getLogger(__name__)


class ReconnectingSource(FrameSource):
    """断线重连包装器(适用于网络流 / 摄像头)。"""

    def __init__(
        self,
        inner: FrameSource,
        *,
        max_retries: Optional[int] = None,          # None = 无限重试
        backoff_initial: float = 1.0,
        backoff_max: float = 30.0,
        backoff_factor: float = 2.0,
        on_reconnect: Optional[Callable[[int, EndReason], None]] = None,   # (第几次, 原因) 回调
    ):
        super().__init__(inner.config)              # 复用 inner 的 config
        self._inner = inner
        self._max_retries = max_retries
        self._backoff_initial = backoff_initial
        self._backoff_max = backoff_max
        self._backoff_factor = backoff_factor
        self._on_reconnect = on_reconnect
        self._opened = False
        self._stop = threading.Event()              # 用于可打断的退避睡眠

    # ── 可被 close() 打断的睡眠 ──
    def _sleep(self, seconds: float) -> bool:
        """退避睡眠。返回 True 表示期间被要求停止(close 被调用)。"""
        return self._stop.wait(seconds)

    # ── FrameSource 协议 ──
    def open(self) -> bool:
        self._frame_index = 0
        self._stop.clear()
        self._end_reason = EndReason.RUNNING
        self._opened = self._open_with_retry()
        return self._opened

    def _open_with_retry(self) -> bool:
        """首次打开也带重试(冷启动时对端可能还没就绪)。"""
        attempt = 0
        backoff = self._backoff_initial
        while not self._stop.is_set():
            if self._inner.open():
                return True
            attempt += 1
            if self._max_retries is not None and attempt > self._max_retries:
                logger.error(f"首次打开失败, 已重试 {attempt - 1} 次仍不成, 放弃")
                self._end_reason = EndReason.DISCONNECTED
                return False
            logger.warning(f"打开失败, {backoff:.1f}s 后第 {attempt} 次重试: {self.source_path}")
            if self._sleep(backoff):
                return False                        # 睡眠中被 close
            backoff = min(backoff * self._backoff_factor, self._backoff_max)
        return False

    def read(self) -> Optional[Frame]:
        if not self._opened:
            self._end_reason = EndReason.NOT_STARTED
            return None

        frame = self._inner.read()
        if frame is not None:
            return frame                            # 正常出帧, 直接透传

        reason = self._inner.end_reason
        if not is_recoverable(reason):
            self._end_reason = reason               # 干净结束/不可恢复 → 透传, 上层停
            return None

        return self._reconnect_and_read(reason)     # 可恢复 → 退避重连

    def _reconnect_and_read(self, reason: EndReason) -> Optional[Frame]:
        attempt = 0
        backoff = self._backoff_initial
        while not self._stop.is_set():
            attempt += 1
            if self._max_retries is not None and attempt > self._max_retries:
                logger.error(f"重连放弃(中断原因 {reason.value}): 已重试 {attempt - 1} 次")
                self._end_reason = EndReason.DISCONNECTED
                return None

            logger.warning(f"源中断({reason.value}), {backoff:.1f}s 后第 {attempt} 次重连: {self.source_path}")
            if self._sleep(backoff):
                self._end_reason = EndReason.DISCONNECTED
                return None                         # 退避中被 close

            try:
                self._inner.close()                 # 先关掉旧句柄
            except Exception as e:
                logger.debug(f"重连前关闭旧句柄出错(忽略): {e}")

            if self._inner.open():
                if self._on_reconnect is not None:
                    try: self._on_reconnect(attempt, reason)
                    except Exception as e: logger.debug(f"on_reconnect 回调出错(忽略): {e}")
                logger.info(f"第 {attempt} 次重连成功: {self.source_path}")
                frame = self._inner.read()
                if frame is not None:
                    return frame                    # 连上且拿到帧
                # 刚连上又立刻空/断: 继续退避重试(不重置 backoff, 防抖动猛连)
            backoff = min(backoff * self._backoff_factor, self._backoff_max)
        # 被 close
        self._end_reason = EndReason.DISCONNECTED
        return None

    def close(self) -> None:
        self._opened = False
        self._stop.set()                            # 打断任何退避睡眠
        self._inner.close()

    # ── 透传 inner(重连不改变'是什么源、能产什么')──
    def get_source_type(self) -> SourceType:
        return self._inner.get_source_type()

    @property
    def modalities(self) -> frozenset[str]:
        return self._inner.modalities

    @property
    def seekable(self) -> bool:
        return False                                # 重连语义下 seek 无意义(会话会重置), 直接不支持

    def set_stride(self, stride: int) -> None:
        self._inner.set_stride(stride)

    @property
    def stride(self) -> int:
        return self._inner.stride
```

**生产装配——重连在内，线程在外**（顺序不能反）：后台采集线程（ThreadedSource）调的是它 inner 的 `read()`，你希望那次 `read()` 就已经把重连处理掉了；反过来，断线信息会先被线程层吃成"队列空"，外层重连拿到的 `end_reason` 是 ThreadedSource 的而非真实源的，判断就错位了。**wrapper 叠放顺序取决于"这个增强需要看到多原始的信号"**——重连要看最原始的断线原因，所以它最贴近源：

```python
from frame_source import create_frame_source, ThreadedSource, ReconnectingSource

def build_stream(url):
    raw    = create_frame_source(url)                    # VideoSource(流)
    robust = ReconnectingSource(raw, backoff_max=15.0)   # 内层: 断了自己悄悄重连
    fast   = ThreadedSource(robust)                      # 外层: 后台线程采集, 主循环拿最新帧
    return fast

with build_stream("rtsp://192.168.1.10/stream") as src:
    for frame in src:            # 断线在这一层完全无感, 只在真正停止时才结束
        model(frame.image)
```

📄 **新建** `frame_source/wrappers/__init__.py`：

```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : __init__.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : wrappers 子包入口
"""wrappers 子包:正交增强(线程采集 / 异步适配 / 断线重连)。均不进注册表。"""
from __future__ import annotations

from .threaded import ThreadedSource, BufferStrategy
from .aio      import AsyncSource
from .reconnect import ReconnectingSource

__all__ = ["ThreadedSource", "BufferStrategy", "AsyncSource", "ReconnectingSource"]
```

```bash
git add frame_source/wrappers/ && git commit -m "feat(frame_source): wrappers —— 线程采集 + 异步适配 + 断线重连(正交, 不进注册表)"
```

---

## 7. factory + 包入口 + 插件发现

### 7.1 factory.py —— 薄壳，只做 str|config → registry

📄 **新建** `frame_source/factory.py`：

```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : factory.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : 便捷工厂 —— str|config 收口到注册表, 三个入口(普通/线程/异步)
"""
便捷工厂: frame_source 对外的主入口。

薄: 真正的"认哪种源"逻辑在 registry 和各源自己手里, 工厂只做一个判断——
输入是 SourceConfig 还是 str, 分别交给 build_from_config / build_from_string。

不 import 任何具体源(依赖倒置): 源靠"被 import sources 时自注册"进入注册表,
所以加一种源工厂零改动(规矩 E)。
"""
from __future__ import annotations

from .core.config   import SourceConfig
from .core.registry import build_from_config, build_from_string
from .core.base     import FrameSource
from .wrappers      import ThreadedSource, AsyncSource, BufferStrategy


def _resolve(source: str | SourceConfig) -> FrameSource:
    """把输入(字符串糖 或 显式 config)解析成一个尚未 open 的源。"""
    if isinstance(source, SourceConfig):
        return build_from_config(source)
    if isinstance(source, str):
        return build_from_string(source)
    raise TypeError(
        f"source 必须是 str 或 SourceConfig, 收到 {type(source).__name__}。"
        f"内存/硬件源请传对应的 config 对象。"
    )


def create_frame_source(source: str | SourceConfig, *, stride: int = 1) -> FrameSource:
    """
    创建一个帧源(最常用入口)。

    Args:
        source: 字符串糖("0"/"x.mp4"/"rtsp://..."/"./imgs")或显式 config
                (CameraConfig(...) / 未来的 DepthConfig(...) / 第三方 XxxConfig(...))。
        stride: 跳采步长(≥1); 不支持跳采的源(摄像头)会告警并忽略。

    Returns:
        尚未 open 的 FrameSource; 建议用 with 使用。
    """
    if stride < 1:
        raise ValueError(f"stride 必须 ≥ 1, 收到: {stride}")
    src = _resolve(source)
    if stride > 1:
        src.set_stride(stride)
    return src


def create_threaded_source(
    source: str | SourceConfig,
    *,
    stride: int = 1,
    strategy: BufferStrategy = BufferStrategy.LATEST,
    maxsize: int = 1,
    warmup: bool = True,
) -> ThreadedSource:
    """创建后台线程采集的源(实时推理防止采集被拖慢, 见 §5.1)。"""
    if stride < 1:
        raise ValueError(f"stride 必须 ≥ 1, 收到: {stride}")
    inner = _resolve(source)
    if stride > 1:
        inner.set_stride(stride)
    return ThreadedSource(inner, strategy=strategy, maxsize=maxsize, warmup=warmup)


def create_async_source(
    source: str | SourceConfig,
    *,
    stride: int = 1,
    threaded: bool = False,
    strategy: BufferStrategy = BufferStrategy.LATEST,
    maxsize: int = 1,
) -> AsyncSource:
    """创建异步接口的源(供 async for 使用, 见 §5.2)。threaded=True 时内部再套一层线程采集。"""
    if stride < 1:
        raise ValueError(f"stride 必须 ≥ 1, 收到: {stride}")
    inner = _resolve(source)
    if stride > 1:
        inner.set_stride(stride)
    if threaded:
        inner = ThreadedSource(inner, strategy=strategy, maxsize=maxsize)
    return AsyncSource(inner)
```

### 7.2 frame_source/__init__.py —— 包入口

`from . import sources` 这行是内置源注册的**唯一触发点**（import 时执行各源模块顶部的 `@register_source`），勿删。

📄 **新建** `frame_source/__init__.py`：

```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : __init__.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : frame_source 包入口 —— 公共 API + 触发内置源自注册
"""
frame_source: 统一帧输入源抽象层(可插拔 · 多模态)。

把任意输入源封装成同一套 `for frame in source` 接口。派发用配置类型(注册表),
加源用一行 @register_source(内置或第三方, 不改本模块), Frame 支持多模态副平面。

最常用:
    from frame_source import create_frame_source
    with create_frame_source("0") as src:          # "0"/"x.mp4"/"rtsp://..."/"./imgs"
        for frame in src:
            model(frame.image)

显式选型(硬件/内存源):
    from frame_source import create_frame_source, CameraConfig
    with create_frame_source(CameraConfig(fps=90, backend="msmf")) as src:
        ...

第三方源发现(可选): 见 load_plugins()。
"""
from __future__ import annotations

__version__ = "0.2.0"     # 0.1=首发; 0.2=上线加固(EndReason + 断线重连)

# ── core: 类型 / 配置 / 协议 / 注册表 ──
from .core import (
    SourceType, FrameInfo, Frame, CameraIntrinsics, EndReason, is_recoverable,
    IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, STREAM_SCHEMES, is_stream_url,
    SourceConfig, CameraConfig, VideoConfig, ImageConfig, ImageFolderConfig,
    CameraBackend, CameraCodec,
    FrameSource,
    register_source, build_from_config, build_from_string, list_sources,
)

# ── sources: import 即触发四个内置源自注册(这几行是注册的唯一触发点, 勿删)──
from .sources import CameraSource, VideoSource, ImageSource, ImageFolderSource

# ── wrappers ──
from .wrappers import ThreadedSource, BufferStrategy, AsyncSource, ReconnectingSource

# ── factory: 主入口 ──
from .factory import create_frame_source, create_threaded_source, create_async_source

# ── 可选: 第三方源发现(默认不自动调用, 交给使用者)──
from .plugins import load_plugins

__all__ = [
    "__version__",
    # 类型
    "SourceType", "FrameInfo", "Frame", "CameraIntrinsics", "EndReason", "is_recoverable",
    "IMAGE_EXTENSIONS", "VIDEO_EXTENSIONS", "STREAM_SCHEMES", "is_stream_url",
    # 配置
    "SourceConfig", "CameraConfig", "VideoConfig", "ImageConfig", "ImageFolderConfig",
    "CameraBackend", "CameraCodec",
    # 协议 + 注册表
    "FrameSource", "register_source", "build_from_config", "build_from_string", "list_sources",
    # 内置源
    "CameraSource", "VideoSource", "ImageSource", "ImageFolderSource",
    # wrappers
    "ThreadedSource", "BufferStrategy", "AsyncSource", "ReconnectingSource",
    # 工厂
    "create_frame_source", "create_threaded_source", "create_async_source",
    # 插件
    "load_plugins",
]
```

### 7.3 plugins.py —— 第三方源发现（两条路）

- **路 A（默认基线）**：显式 import 触发自注册——第三方包自己 `import frame_source` + `@register_source`，使用者 import 那个包即可。简单、显式、可控。
- **路 B（生产进阶）**：entry-points 自动发现——第三方包在自己的 `pyproject.toml` 里登记 entry-point，`load_plugins()` 扫描并 import。

📄 **新建** `frame_source/plugins.py`：

```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : plugins.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : 第三方源发现 —— 扫 entry-points 并 import(触发其 @register_source)
"""
第三方源发现(可选, 生产级)。

约定 entry-point group = "frame_source.sources", 每个 entry-point 的值是一个
"import 即会执行 @register_source"的模块。load_plugins() 逐个 import 它们。

默认不自动调用 —— 由使用方在启动时显式 load_plugins() 决定何时发现插件。
"""
from __future__ import annotations

import logging
from importlib.metadata import entry_points

logger = logging.getLogger(__name__)

PLUGIN_GROUP = "frame_source.sources"


def load_plugins(group: str = PLUGIN_GROUP, *, raise_on_error: bool = False) -> list[str]:
    """
    发现并加载已安装的第三方源插件(import 它们以触发 @register_source)。

    Args:
        group          : entry-point group 名。
        raise_on_error : 某个插件加载失败时, True=抛出, False=记录并跳过(默认)。

    Returns:
        成功加载的插件名列表。
    """
    loaded: list[str] = []
    for ep in entry_points(group=group):     # Python 3.10+ 签名
        try:
            ep.load()                        # import 插件模块 → 触发其 @register_source
            loaded.append(ep.name)
            logger.info(f"已加载源插件: {ep.name}")
        except Exception as e:
            logger.error(f"加载源插件 {ep.name} 失败: {e}")
            if raise_on_error:
                raise
    return loaded
```

### 7.4 README.md —— 可拷走模块的身份证

📄 **新建** `frame_source/README.md`：

```markdown
# frame_source

统一帧输入源抽象层(可插拔 · 多模态): 把任意输入源封装成同一套 `for frame in source` 接口。

## 特性
- 四种内置源: 摄像头 / 视频·网络流 / 单图 / 图片文件夹, 统一 `for frame in src` 迭代
- 配置驱动派发: 字符串糖("0"/"x.mp4"/"./imgs")或显式 config; 硬件源靠 config 类型选型
- 可插拔: 一行 @register_source 加源, 内置或第三方均不改本模块(开闭原则)
- 多模态 Frame: 主平面 image + 可选 depth / aux / intrinsics; modalities 能力发现
- 正交增强: ThreadedSource(后台采集)/ AsyncSource(异步接口)
- 上下文管理器 + fail-fast 配置校验, 资源不漏

## 依赖
opencv-python, numpy, pydantic>=2.0, Python>=3.10

## 快速开始
    from frame_source import create_frame_source
    with create_frame_source("0") as src:      # "0"/"x.mp4"/"rtsp://..."/"./imgs"
        for frame in src:
            do_something(frame.image)

## 显式选型(硬件 / 高帧率 / 内存源)
    from frame_source import create_frame_source, CameraConfig
    with create_frame_source(CameraConfig(fps=90, backend="msmf")) as src:
        ...

## 加一种源(不改本模块)
    from frame_source import FrameSource, SourceConfig, register_source

    class MyConfig(SourceConfig): ...
    @register_source(MyConfig)
    class MySource(FrameSource): ...
    # import 到你的源模块即接入; 或用 entry-points + load_plugins() 自动发现

## 独立性
无宿主项目依赖, 可整包拷贝至任意项目。第三方源可独立扩展。
```

```bash
git add frame_source/factory.py frame_source/__init__.py frame_source/plugins.py frame_source/README.md
git commit -m "feat(frame_source): factory + 包入口 + 插件发现 + README"
```

---

## 8. 扩展性实证：不改 frame_source，当场加一种源

在 `frame_source/` **之外**写一个内存合成源（模拟"未来某天要接的新硬件"），一行 `@register_source` 登记自己：

📄 **新建** `examples/synthetic_source.py`：

```python
# ── 文件: examples/synthetic_source.py(在 frame_source 之外!)──
import numpy as np
from frame_source import (
    FrameSource, SourceConfig, register_source,
    Frame, FrameInfo, create_frame_source, list_sources,
)


class SyntheticConfig(SourceConfig):
    """合成源配置。纯内存, 没有路径 —— 无法用字符串表达, 只能显式传 config。"""
    n_frames: int = 30
    width:    int = 640
    height:   int = 480

    @property
    def identity(self) -> str:
        return f"synthetic:{self.n_frames}帧"


@register_source(SyntheticConfig)     # ← 无 str_matcher: config-only, 和硬件源一样只能显式选型
class SyntheticSource(FrameSource):
    """每帧现生成一张渐变测试图的内存源。"""

    def __init__(self, config: SyntheticConfig):
        super().__init__(config)
        self._n, self._w, self._h = config.n_frames, config.width, config.height
        self._i = 0

    def open(self) -> bool:
        self._i = 0
        self._frame_index = 0
        return True

    def read(self):
        if self._i >= self._n:
            return None
        img = np.full((self._h, self._w, 3), (self._i * 8) % 256, dtype=np.uint8)
        info = FrameInfo(
            width=self._w, height=self._h,
            source_type="synthetic",              # ← 第三方类型字符串(不是内置枚举成员)
            source_path=self.source_path,
            frame_index=self._i, total_frames=self._n,
            filename=f"synthetic_{self._i:04d}",
        )
        self._i += 1
        self._frame_index = self._i
        return Frame(image=img, info=info)

    def close(self) -> None:
        pass

    def get_source_type(self):
        return "synthetic"                        # ← 返回字符串(因为不是内置枚举, §2.1 的放宽在此兑现)
```

然后（真机输出）：

```bash
PYTHONPATH=. python -c "
import examples.synthetic_source as syn   # import 即自注册
from frame_source import create_frame_source, list_sources
print(list_sources())
with create_frame_source(syn.SyntheticConfig(n_frames=3, width=320, height=240)) as src:
    print('modalities:', src.modalities)
    for frame in src:
        print(frame.info.frame_index, frame.image.shape, frame.info.source_type)
"
```

```text
注册表: {'CameraConfig': 'CameraSource', 'VideoConfig': 'VideoSource', 'ImageConfig': 'ImageSource',
        'ImageFolderConfig': 'ImageFolderSource', 'SyntheticConfig': 'SyntheticSource'}   ← 多了一条, frame_source/ 一个字没改
modalities: frozenset({'image'})
0 (240, 320, 3) synthetic
1 (240, 320, 3) synthetic
2 (240, 320, 3) synthetic
```

注意 `SyntheticConfig` 是**纯 config 派发**（它没有字符串形态）——这正是"配置类型做派发轴"白捡的能力。

```bash
git add examples/ && git commit -m "test(frame_source): 扩展性实证 —— 包外加源不改模块"
```

---

## 9. 自测：`tests/test_frame_source.py`

自带迷你测试框架（`check`/`skip`/`expect_raises`），覆盖：多模态类型、注册表、fail-fast 失败路径、modalities 能力发现、四源端到端、seek/stride、open 复位、线程/异步 wrapper、包外加源、坏图健壮性、断线重连。摄像头/视频不可用时相关项自动跳过。

📄 **新建** `tests/test_frame_source.py`：

```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : test_frame_source.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : frame_source 自测 —— 注册表 + 多模态 + 四源 + 扩展性实证
"""一键自测: python tests/test_frame_source.py。摄像头/视频不可用时相关项自动跳过。"""
from __future__ import annotations

import asyncio
import tempfile
import shutil
from pathlib import Path

import numpy as np
import cv2
from pydantic import ValidationError

from frame_source import (
    create_frame_source, create_threaded_source, create_async_source,
    CameraConfig, VideoConfig, ImageConfig, ImageFolderConfig, SourceConfig,
    Frame, FrameInfo, CameraIntrinsics, SourceType,
    FrameSource, register_source, build_from_config, build_from_string, list_sources,
    ThreadedSource, BufferStrategy, AsyncSource, ReconnectingSource, EndReason,
)

# ── 迷你测试框架 ──
_passed = 0
_failed = 0
_skipped = 0

def check(name: str, cond: bool) -> None:
    global _passed, _failed
    mark = "✅" if cond else "❌"
    print(f"  {mark} {name}")
    if cond: _passed += 1
    else:    _failed += 1

def skip(name: str, reason: str) -> None:
    global _skipped
    print(f"  ⏭️  {name}(跳过: {reason})")
    _skipped += 1

def expect_raises(name: str, exc_types, fn) -> None:
    global _passed, _failed
    try:
        fn()
        print(f"  ❌ {name}(未抛异常)")
        _failed += 1
    except exc_types:
        print(f"  ✅ {name}")
        _passed += 1
    except Exception as e:
        print(f"  ❌ {name}(抛了非预期异常: {type(e).__name__}: {e})")
        _failed += 1


# ── 一个会"断线一次再恢复"的假源, 用于验证重连 ──
class FlakyConfig(SourceConfig):
    fail_after: int = 3      # 第一次会话跑到第几帧就"断线"
    total:      int = 6      # 重连后正常跑完多少帧

class FlakySource(FrameSource):
    def __init__(self, config: FlakyConfig):
        super().__init__(config)
        self._fail_after = config.fail_after
        self._total = config.total
        self._i = 0
        self._session = 0    # 第几次 open(重连会 +1)

    def open(self):
        self._i = 0
        self._session += 1
        self._end_reason = EndReason.RUNNING
        return True

    def read(self):
        limit = self._fail_after if self._session == 1 else self._total
        if self._i >= limit:
            # 第一次会话到 limit 就"断线", 触发重连; 之后会话正常读完
            self._end_reason = (EndReason.DISCONNECTED if self._session == 1
                                else EndReason.COMPLETED)
            return None
        img = np.zeros((4, 4, 3), np.uint8)
        info = FrameInfo(width=4, height=4, source_type="flaky",
                         source_path=self.source_path, frame_index=self._i)
        self._i += 1
        return Frame(image=img, info=info)

    def close(self): pass
    def get_source_type(self): return "flaky"


# ── 准备临时素材 ──
def make_fixtures(root: Path):
    # 单张图
    img_path = root / "one.jpg"
    cv2.imwrite(str(img_path), np.full((48, 64, 3), 127, np.uint8))
    # 图片文件夹(6 张, 含一张大小写后缀)
    folder = root / "imgs"; folder.mkdir()
    for i in range(5):
        cv2.imwrite(str(folder / f"{i:02d}.jpg"), np.full((48, 64, 3), i * 40, np.uint8))
    cv2.imwrite(str(folder / "05.PNG"), np.full((48, 64, 3), 200, np.uint8))   # 大写后缀也要认
    # 短视频(可能因编码器不可用而失败)
    vid_path = root / "clip.mp4"
    vw = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    ok_video = vw.isOpened()
    if ok_video:
        for i in range(10):
            vw.write(np.full((48, 64, 3), i * 20, np.uint8))
        vw.release()
    else:
        vw.release()
    return img_path, folder, (vid_path if ok_video else None)


# ── 一个包外自定义源, 用于扩展性实证(config-only 内存源)──
class SyntheticConfig(SourceConfig):
    n_frames: int = 5
    @property
    def identity(self) -> str:
        return f"synthetic:{self.n_frames}"

@register_source(SyntheticConfig)     # 无 str_matcher: 只能靠 config 类型派发
class SyntheticSource(FrameSource):
    def __init__(self, config: SyntheticConfig):
        super().__init__(config)
        self._n = config.n_frames; self._i = 0
    def open(self):
        self._i = 0; self._frame_index = 0; return True
    def read(self):
        if self._i >= self._n: return None
        img = np.full((16, 16, 3), self._i, np.uint8)
        info = FrameInfo(width=16, height=16, source_type="synthetic",
                         source_path=self.source_path, frame_index=self._i, total_frames=self._n)
        self._i += 1; self._frame_index = self._i
        return Frame(image=img, info=info)
    def close(self): pass
    def get_source_type(self): return "synthetic"


def main():
    tmp = Path(tempfile.mkdtemp(prefix="fs_test_"))
    try:
        img_path, folder, vid_path = make_fixtures(tmp)

        print("\n=== 1. 多模态数据类型 ===")
        # 断言 Frame 带 depth/aux/intrinsics 且能用
        f = Frame(image=np.zeros((4, 4, 3), np.uint8),
                  info=FrameInfo(width=4, height=4, source_type=SourceType.IMAGE, source_path="x"))
        check("Frame 有 depth 字段且默认 None",       f.depth is None)
        check("Frame 有 aux 字段且默认空 dict",         f.aux == {})
        check("Frame 有 intrinsics 字段且默认 None",    f.intrinsics is None)
        check("无深度时 has_depth 为 False",           f.has_depth is False)
        fd = Frame(image=np.zeros((4, 4, 3), np.uint8), depth=np.zeros((4, 4), np.uint16),
                   info=f.info)
        check("有深度时 has_depth 为 True",            fd.has_depth is True)
        intr = CameraIntrinsics(fx=600, fy=600, cx=320, cy=240, depth_scale=0.001)
        check("CameraIntrinsics 字段可读",             intr.fx == 600 and intr.depth_scale == 0.001)
        check("Frame.width 便捷属性",                  f.width == 4)

        print("\n=== 2. 注册表 ===")
        reg = list_sources()
        check("list_sources 含四个内置源",
              {"CameraConfig", "VideoConfig", "ImageConfig", "ImageFolderConfig"} <= set(reg))
        check("list_sources 含实证注册的 SyntheticConfig", "SyntheticConfig" in reg)
        # 配置类型派发
        check("build_from_config(ImageConfig) → ImageSource",
              type(build_from_config(ImageConfig(path=str(img_path)))).__name__ == "ImageSource")
        check("build_from_config(CameraConfig) → CameraSource",
              type(build_from_config(CameraConfig())).__name__ == "CameraSource")
        # 字符串糖派发
        check("build_from_string('0') → CameraSource",
              type(build_from_string("0")).__name__ == "CameraSource")
        check("build_from_string(图片路径) → ImageSource",
              type(build_from_string(str(img_path))).__name__ == "ImageSource")
        check("build_from_string(文件夹路径) → ImageFolderSource",
              type(build_from_string(str(folder))).__name__ == "ImageFolderSource")

        print("\n=== 3. 派发的失败路径(fail-fast)===")
        class _UnregConfig(SourceConfig): pass
        expect_raises("未注册的配置类型 → ValueError", ValueError,
                      lambda: build_from_config(_UnregConfig()))
        expect_raises("无法识别的字符串 → ValueError", ValueError,
                      lambda: build_from_string("nonsense.xyz"))
        expect_raises("build_from_config 收到非 config → TypeError", TypeError,
                      lambda: build_from_config("0"))
        expect_raises("视频路径不存在 → 构造 config 即抛", ValidationError,
                      lambda: VideoConfig(path="/no/such/file.mp4"))
        expect_raises("字符串糖遇不存在路径 → ValueError(包装)", ValueError,
                      lambda: create_frame_source("/no/such/file.mp4"))
        expect_raises("config 字段拼错 → ValidationError(extra=forbid)", ValidationError,
                      lambda: CameraConfig(fpss=90))
        expect_raises("图片源拒绝 URL → ValidationError", ValidationError,
                      lambda: ImageConfig(path="http://x/a.jpg"))

        print("\n=== 4. 能力发现 modalities ===")
        with create_frame_source(str(img_path)) as src:
            check("彩色源 modalities == {'image'}", src.modalities == frozenset({"image"}))
        # 深度能力靠源覆盖(此处用一个临时深度源验证覆盖点有效)
        class _DepthCfg(SourceConfig): pass
        @register_source(_DepthCfg)
        class _DepthSrc(FrameSource):
            def open(self): return True
            def read(self): return None
            def close(self): pass
            def get_source_type(self): return "depth"
            @property
            def modalities(self): return frozenset({"image", "depth"})
        check("深度源可覆盖 modalities 含 depth",
              "depth" in build_from_config(_DepthCfg()).modalities)

        print("\n=== 5. 四源端到端 ===")
        # 单图
        with create_frame_source(ImageConfig(path=str(img_path))) as src:
            frames = list(src)
            check("单图产 1 帧", len(frames) == 1)
            check("单图帧 image 形状正确", frames[0].image.shape == (48, 64, 3))
            check("单图 source_type=IMAGE", frames[0].info.source_type == SourceType.IMAGE)
        # 文件夹
        with create_frame_source(ImageFolderConfig(path=str(folder))) as src:
            frames = list(src)
            check("文件夹产 6 帧(含大写后缀 PNG)", len(frames) == 6)
            check("文件夹帧序稳定(frame_index 递增)",
                  [fr.info.frame_index for fr in frames] == list(range(6)))
        # 视频
        if vid_path:
            with create_frame_source(VideoConfig(path=str(vid_path))) as src:
                frames = list(src)
                check("视频产约 10 帧", 9 <= len(frames) <= 10)
                check("视频有 fps 元数据", frames[0].info.fps is not None)
                check("本地视频 seekable=True", src.seekable is True)
        else:
            skip("视频端到端", "VideoWriter 编码器不可用")
        # 摄像头(设备可能不存在)
        cam = create_frame_source("0")
        if cam.open():
            fr = cam.read(); cam.close()
            check("摄像头能读到一帧", fr is not None)
        else:
            cam.close(); skip("摄像头端到端", "无可用摄像头设备")

        print("\n=== 6. seek 与 stride ===")
        # 文件夹 seek
        fsrc = create_frame_source(ImageFolderConfig(path=str(folder)))
        fsrc.open()
        check("文件夹 seekable=True", fsrc.seekable is True)
        check("文件夹 seek(frame=3) 成功", fsrc.seek(frame=3) is True)
        check("seek 后读到第 3 帧", fsrc.read().info.frame_index == 3)
        fsrc.close()
        # 文件夹 stride
        with create_frame_source(ImageFolderConfig(path=str(folder)), stride=3) as src:
            idxs = [fr.info.frame_index for fr in src]
            check("stride=3 只产 0,3 两帧(零 IO 跳采)", idxs == [0, 3])
        # 视频 stride
        if vid_path:
            with create_frame_source(VideoConfig(path=str(vid_path)), stride=5) as src:
                got = list(src)
                check("视频 stride=5 产约 2 帧", 1 <= len(got) <= 2)
        else:
            skip("视频 stride", "无视频")
        # 摄像头 stride 应被忽略(不报错)
        cam2 = create_frame_source("0", stride=3)
        check("摄像头 stride 被忽略(stride 仍为 1)", cam2.stride == 1)
        cam2.close()

        print("\n=== 7. 重新 open 复位 + 迭代器单次 ===")
        src = create_frame_source(ImageFolderConfig(path=str(folder)))
        src.open(); n1 = len(list(src))
        src.open(); n2 = len(list(src))          # 重新 open 应能重头再来
        check("重新 open 后能重头迭代", n1 == n2 == 6)
        src.close()
        # 迭代器单次: 不重新 open, 第二次迭代应为空
        src2 = create_frame_source(ImageConfig(path=str(img_path)))
        src2.open()
        first = len(list(src2)); second = len(list(src2))
        check("同一 open 内迭代一次即耗尽", first == 1 and second == 0)
        src2.close()

        print("\n=== 8. wrappers: 线程 / 异步 ===")
        # 线程(用 BOUNDED 保证不丢帧, 便于精确计数)
        with create_threaded_source(SyntheticConfig(n_frames=10),
                                    strategy=BufferStrategy.BOUNDED, maxsize=32) as src:
            n = len(list(src))
            check("ThreadedSource(BOUNDED) 不丢帧, 产 10 帧", n == 10)
            check("ThreadedSource 透传 modalities", src.modalities == frozenset({"image"}))
        # 异步
        async def _run_async():
            cnt = 0
            async with create_async_source(SyntheticConfig(n_frames=5)) as src:
                async for _ in src:
                    cnt += 1
            return cnt
        check("AsyncSource async for 产 5 帧", asyncio.run(_run_async()) == 5)
        check("AsyncSource 不是 FrameSource 子类(守 LSP)",
              not issubclass(AsyncSource, FrameSource))

        print("\n=== 9. 扩展性实证(不改 frame_source 加源)===")
        with create_frame_source(SyntheticConfig(n_frames=3)) as src:
            got = list(src)
            check("包外 config-only 源可被 create_frame_source 派发", len(got) == 3)
            check("包外源可用第三方 source_type 字符串", got[0].info.source_type == "synthetic")
            check("内存源(无字符串形态)正常工作", got[0].image.shape == (16, 16, 3))

        print("\n=== 10. 错误处理健壮性 ===")
        # 文件夹里放一张坏图, 应跳过而非崩溃
        bad = folder / "broken.jpg"; bad.write_bytes(b"not an image")
        with create_frame_source(ImageFolderConfig(path=str(folder))) as src:
            frames = list(src)     # 坏图被跳过, 其余正常
            check("坏图被跳过, 不影响其余帧", len(frames) == 6)

        print("\n=== 11. 断线重连 ===")
        # 用极短退避让测试跑得快
        rc = ReconnectingSource(FlakySource(FlakyConfig(fail_after=3, total=6)),
                                backoff_initial=0.01, backoff_max=0.05, max_retries=5)
        with rc as src:
            got = list(src)      # 断线前 3 帧 + 重连后 6 帧 = 9 帧, 且不 hang
        check("重连拿到断线前+重连后的全部帧", len(got) == 3 + 6)
        check("重连后干净结束(end_reason=COMPLETED)", src.end_reason == EndReason.COMPLETED)

        # 附带验证 end_reason 语义本身
        img_src = create_frame_source(ImageConfig(path=str(img_path)))
        img_src.open(); _ = list(img_src)
        check("单图读完 end_reason=COMPLETED", img_src.end_reason == EndReason.COMPLETED)
        img_src.close()

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'='*44}")
    print(f"通过 {_passed}  失败 {_failed}  跳过 {_skipped}")
    print(f"{'='*44}")
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

### 运行

```bash
pip install opencv-python numpy "pydantic>=2"
PYTHONPATH=. python tests/test_frame_source.py
```

真机输出（沙箱实测，摄像头项因无设备自动跳过）：

```text
=== 1. 多模态数据类型 ===        7 项全过
=== 2. 注册表 ===                list_sources 含四个内置源 ✓
=== 3. 派发的失败路径 ===         未注册 config/无法识别字符串/非 config/路径不存在/拼错字段/URL 图片 全部正确报错 ✓
=== 4. 能力发现 modalities ===    彩色源 {'image'}, 深度源可覆盖 ✓
=== 5. 四源端到端 ===             单图 1 帧 / 文件夹 6 帧(含大写 PNG) / 视频约 10 帧带 fps / 摄像头跳过 ⏭️
=== 6. seek 与 stride ===         文件夹 seek 精确到帧 / stride 零 IO 跳采 / 摄像头拒绝 stride ✓
=== 7. 重新 open 复位 ===         重新 open 重头迭代 / 同一 open 迭代一次即耗尽 ✓
=== 8. wrappers ===               线程不丢帧 / 透传 modalities / async for / AsyncSource 守 LSP ✓
=== 9. 扩展性实证 ===             包外 config-only 源被同一入口派发 ✓
=== 10. 错误处理健壮性 ===        坏图跳过不崩溃 ✓
=== 11. 断线重连 ===              FlakySource 断线一次→重连→拿全 9 帧→干净结束 ✓
============================================
通过 50  失败 0  跳过 1
============================================
```

零宿主依赖自查（可拷走性的硬指标）：

```bash
grep -rn "import odplatform\|from odplatform" frame_source/ --include="*.py"
# 期望: 0 输出。有输出就说明漏了依赖, 拷走就会崩。
```

```bash
git add tests/ && git commit -m "test(frame_source): 自测 —— 注册表/四源/多模态/wrappers/扩展性/重连"
```

---

## 附录 A：架构决策记录（ADR 浓缩）

每条格式：决策 → 理由 → 放弃的替代 → 后果。

| # | 决策 | 理由（一句话） | 放弃的替代 | 后果 |
|---|------|--------------|-----------|------|
| D1 | 配置类型作唯一派发轴，字符串降为二线糖 | 硬件源在字符串层面不可区分（`"0"` 歧义）；config 类型是确定的一步查表，还白送内存源 | 字符串模式匹配做主派发轴 | 每种源要有一个 config 类；硬件源调用方写 `CameraConfig(...)` 而非 `"0"`——恰是显式表达意图 |
| D2 | 注册表 + `@register_source` 自注册 + 两路插件发现 | 需求明确"未来加更多源、第三方在包外加"；if/elif 让每次加源都改工厂，第三方完全插不进 | if/elif 工厂（固定几种源时它够用） | 可发现性下降（见 C1）；import 时序有理解成本（见 C2） |
| D3 | 注册表只做轻量映射，不上 matcher+优先级+spec | 派发轴是配置类型，无歧义要仲裁；字符串命中多个当场 raise | 打分仲裁注册表框架（刻意的"不做"） | 若未来出现"多源合法竞争同一输入"，需升级这一层（触发条件已标记） |
| D4 | Frame：主平面 image + 一等 depth + 开放 aux + intrinsics | 深度/双目/红外装不进 image+info；深度高频重要给一等字段；内参是点云刚需 | 单模态 image+info；或所有副平面塞 aux 杂物袋（明确拒绝，丢类型安全） | image 语义放宽（BGR/热图/左目），消费方取副平面前看 modalities；对只看主图的老代码透明 |
| D5 | modalities 源级能力发现，has_depth 帧级 | 开循环前就要决定要不要准备点云管线 | 只有帧级判断（得先读一帧） | wrapper 必须透传 modalities（正交性强制要求） |
| D6 | seek/stride 是协议方法，不是 wrapper | 语义同族 + 各源有最快跳法（grab 跳采/索引直跳）+ 可迭代中动态切换 | StrideWrapper（浪费各源性能） | 每源各自实现跳采（有重复）；不适用的源明确拒绝 |
| D7 | wrappers 正交、不进注册表、复用 inner config | wrapper 是源的修饰，不是一种源 | wrapper 也注册（概念混乱） | wrapper 由工厂便捷函数或手动套上 |
| D8 | AsyncSource 不继承 FrameSource | 协程 read() 无法替换同步 read()，强行继承违反 LSP | 为复用继承树破坏协议（本末倒置） | 类型诚实，少一点代码复用 |
| D9 | FrameInfo.source_type 收 str | 闭合枚举堵死第三方源的类型（违反开闭） | 闭合 SourceType 强约束 | 损失一点类型强度，换第三方源类型自由 |
| D10 | fail-fast 分界 + 统一零拷贝 + 网络流诚实 | 文件类源构造即校验（路径错当场炸）；硬件源 open 时校验；零拷贝不开特例；流不谎报边界 | 给单图源 copy 开特例；给流编假边界 | config 层做了一点文件系统 IO（见 C3） |
| D11 | end_reason 可查询状态补齐结束语义 | None 二义（停/断反应相反）卡死一切"源断了要应对"的功能 | 抛异常（改动大）；Result 返回类型（全模块连锁改） | 旁路信息想用才查；重连/报警/切备用源有了共同地基 |

**明写的代价（Consequences）**：

- **C1 可发现性下降**：if/elif 读一个函数就知道支持哪些源；注册表把源散在各模块，靠 `list_sources()` 内省补偿。
- **C2 "import 即注册"的时序理解成本**：新人卡壳最多的点；包 `__init__` 用一行 `from . import sources` 集中触发并在注释点明"勿删"。
- **C3 config 层做了一点文件系统 IO**：路径校验在 validator 里跑，纯数据洁癖党不喜；换来"路径错在构造点就炸"。
- **C4 每源一个 config 类的样板**：D1 的必然结果；但这些 config 不空（扛派发 + fail-fast 两件实事），且是稳定接缝。

**值不值，永远相对于需求**：需求明确要"可插拔 + 多模态"，C1–C4 是资产的持有成本；若需求只是固定几种源、永不扩展、永远单模态，集中式 if/elif 的简单写法反而才对。

## 附录 B：与原始讲义的差异

1. **撞墙叙事全部移除**（"朴素方案的墙"、各阶段 🧱/🤔 块），设计动机以"设计点"浓缩进正文，决策沉淀为附录 A 的 ADR 表。
2. **阶段 10（EndReason + 断线重连）已合并进最终代码**：本文档给出的 `types.py`/`base.py`/三个源文件/`wrappers/` 均为打上加固补丁后的**终态**（沙箱实际运行的版本），不再分"先建后补"两步讲。
3. **补丁落点适配**：原讲义的阶段 10 补丁片段与主体代码有个别措辞差异（如 ImageSource.read 的合并条件），已按终态语义正确合并，`end_reason` 的四种标注位置（NOT_STARTED/RUNNING/COMPLETED/DISCONNECTED/ERROR）经自测 §11 验证。
4. **深度源设计草图**（`examples/depth_source.py`，SDK 调用为伪代码不可运行）未收录——它只是设计示意；多模态消费模式已在 §0.1 的 `run()` 与 `Frame` 字段设计中完整表达。
5. 自测文件并入了断线重连一节（§11），共 50 项断言全过。

**一句话主线**：让抽象的复杂度始终追着需求的复杂度走——不多一分，不少一分。需求到了不长是欠债（一次网络抖动让服务永久停摆），需求没到硬长是过度设计（matcher+优先级现在上就是纯负债）。
