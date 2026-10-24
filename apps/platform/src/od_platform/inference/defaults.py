"""inference 子系统的私有默认值 —— 有名字的工程常量, 不是业务决策。

为什么放这里而不是 ``common/constants.py``: 按那里的入库规则, "两个以上子系统都要
引用的业务决策"才进公共 constants; 这些值只被 inference 内部使用, 属于子系统私有,
所以留在子系统 —— 但必须【有名字】, 不许在代码里当魔法数字。

用户可调的部分不放这里:

* 模型 / 批量 / 置信度 / 设备 → ``configs/runtime/infer.yaml``(runtime_config)
* 美化与输出(文件名、编码器、兜底帧率) → ``configs/runtime/infer_pipeline.yaml``
"""
from __future__ import annotations

# ---------------------------------------------------------------- 模型与运行命名
DEFAULT_MODEL_NAME: str = "yolo11n.pt"      # 未指定模型时的兜底(与 infer.yaml 示例一致)
DEFAULT_RUN_NAME: str = "predict"           # 未指定实验名时的输出子目录名
DEFAULT_BATCH_SIZE: int = 16                # batch 配置无效时的兜底批大小

# ---------------------------------------------------------------- 输出视频
DEFAULT_VIDEO_FILENAME: str = "output.mp4"  # 视频源的结果文件名
DEFAULT_VIDEO_FOURCC: str = "mp4v"          # OpenCV 编码器四字符码
DEFAULT_VIDEO_FPS: float = 30.0             # 源视频读不到 fps 时的兜底帧率

# ---------------------------------------------------------------- 流水线队列与超时
LIVE_QUEUE_SIZE: int = 1                    # 直播/流: 只保最新一帧
RENDER_QUEUE_SIZE: int = 2                  # 渲染线程 → 显示线程
READER_QUEUE_FACTOR: int = 2                # 读帧队列容量 = 批大小 × 该系数
READER_QUEUE_MIN: int = 8                   # ...且不小于该值
READER_READ_TIMEOUT_SEC: float = 2.0        # 主循环等一帧的上限
QUEUE_PUT_TIMEOUT_SEC: float = 1.0          # 阻塞入队的单次等待(用于可中断循环)
RENDER_POLL_SEC: float = 0.1                # 渲染线程空轮询间隔
DISPLAY_POLL_SEC: float = 0.03              # 显示线程空轮询间隔
PAUSE_POLL_SEC: float = 0.02                # 暂停状态下的轮询间隔
RENDERER_JOIN_SEC: float = 3.0              # 收尾时等渲染线程的上限
DISPLAY_JOIN_SEC: float = 1.0               # 收尾时等显示线程的上限
DISPLAY_DRAIN_SEC: float = 0.05             # 关闭窗前留给显示线程的时间
