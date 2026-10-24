"""推理服务编排器."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from od_platform.common.config_log import log_effective_config, log_override_chains
from od_platform.common.log_rename import rename_log_to_save_dir
from od_platform.common.model_path import resolve_model_path
from od_platform.common.paths import CHECKPOINTS_DIR, PRETRAINED_MODELS_DIR, RUNS_DIR
from od_platform.common.system_utils import log_device_info
from od_platform.runtime_config import build_infer_config
from od_platform.visualization import BeautifyVisualizer, DrawStyle

from .cancel import CancelToken
from .defaults import DEFAULT_BATCH_SIZE, DEFAULT_MODEL_NAME, DEFAULT_RUN_NAME
from .hooks import InferHooks
from .pipeline_config import PipelineConfig, load_pipeline_config
from .sinks import LocalFileSink, NullSink, OutputSink

logger = logging.getLogger(__name__)

_PREDICT_KEYS: tuple[str, ...] = (
    "conf", "iou", "imgsz", "max_det", "classes",
    "agnostic_nms", "augment", "device", "retina_masks",
)


def _find_project_log_path() -> Path | None:
    root = logging.getLogger("od_platform")
    for h in root.handlers:
        if isinstance(h, logging.FileHandler):
            return Path(h.baseFilename)
    return None


def _resolve_output_dir(base: Path, name: str, *, exist_ok: bool) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    candidate = base / name
    if exist_ok or not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    i = 2
    while (base / f"{name}{i}").exists():
        i += 1
    out = base / f"{name}{i}"
    out.mkdir(parents=True, exist_ok=True)
    return out


@dataclass
class InferStats:
    frames: int = 0
    detections: int = 0
    per_class: dict[str, int] = field(default_factory=dict)
    infer_time_sec: float = 0.0
    capture_fps: float = 0.0
    infer_fps: float = 0.0
    render_fps: float = 0.0
    loop_fps: float = 0.0
    current_fps: float = 0.0
    speed_ms: dict[str, float] = field(default_factory=dict)

    @property
    def avg_fps(self) -> float:
        return self.frames / self.infer_time_sec if self.infer_time_sec > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return (self.infer_time_sec / self.frames * 1000.0) if self.frames else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "detections": self.detections,
            "per_class": dict(self.per_class),
            "infer_time_sec": round(self.infer_time_sec, 4),
            "avg_fps": round(self.avg_fps, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "fps": {"capture": self.capture_fps, "infer": self.infer_fps,
                    "render": self.render_fps, "loop": self.loop_fps,
                    "current": self.current_fps},
            "speed_ms": dict(self.speed_ms),
        }


def log_infer_stats(stats: InferStats, *, logger: logging.Logger = logger) -> None:
    logger.info(f"处理帧数:   {stats.frames}")
    logger.info(f"检测总数:   {stats.detections}")
    logger.info(f"平均延迟:   {stats.avg_latency_ms:.2f} ms/帧")
    logger.info("帧率(FPS):  捕获 %.1f | 推理 %.1f | 渲染 %.1f | loop %.1f | 当前 %.1f" % (
        stats.capture_fps, stats.infer_fps, stats.render_fps,
        stats.loop_fps, stats.current_fps))
    if stats.speed_ms:
        logger.info("模型 speed(ms): 预处理 %.2f | 推理 %.2f | 后处理 %.2f" % (
            stats.speed_ms.get("preprocess", 0.0),
            stats.speed_ms.get("inference", 0.0),
            stats.speed_ms.get("postprocess", 0.0)))
    if stats.per_class:
        logger.info("各类别检测数:")
        for name, cnt in sorted(stats.per_class.items(), key=lambda kv: -kv[1]):
            logger.info(f"    {name:<20} {cnt}")


@dataclass(frozen=True)
class InferResult:
    success:    bool
    output_dir: Path
    stats:      dict[str, Any] = field(default_factory=dict)
    infer_time: float | None = None
    saved:      bool = False
    error:      str | None = None
    audit_path: Path | None = None
    log_path:   Path | None = None


class InferService:
    def __init__(self) -> None:
        pass

    def predict(self, yaml_path=None, pipeline_yaml=None, cli_args=None, *,
                beautify: bool = True, rename_log: bool = True,
                threaded: bool = False, warmup_frames: int = 0,
                window_name: str = "odp-infer", show_info: bool = True,
                output_sink: OutputSink | None = None,
                hooks: InferHooks | None = None,
                cancel_token: CancelToken | None = None) -> InferResult:
        if hooks is None:
            hooks = InferHooks()
        start = datetime.now()
        output_dir: Path | None = None
        try:
            config, merger = build_infer_config(
                yaml_path=yaml_path or "infer.yaml", cli_args=cli_args)
            pipe: PipelineConfig = load_pipeline_config(pipeline_yaml)

            logger.info("=" * 60)
            logger.info(f"开始 YOLO 推理 (task={config.task})".center(60))
            logger.info("=" * 60)
            raw_model = config.model or DEFAULT_MODEL_NAME
            raw_source = config.source
            logger.info(f"任务类型:    {config.task}")
            logger.info(f"输入源(声明): {raw_source!r}")
            logger.info(f"模型(声明):  {raw_model}")
            log_device_info()
            log_effective_config(config, merger, logger=logger)
            log_override_chains(config, merger, logger=logger)

            if raw_source is None:
                raise RuntimeError(
                    "未指定推理输入源. 请在 infer.yaml 写 source, 或用 "
                    "`odp-infer --source <图/视频/目录/摄像头号>` 传入.")
            model_path = resolve_model_path(
                raw_model, search_dirs=[CHECKPOINTS_DIR, PRETRAINED_MODELS_DIR])
            logger.info(f"模型(解析):  {model_path}")

            from ultralytics import YOLO
            model = YOLO(str(model_path))
            class_names: list[str] = list(model.names.values())
            do_beautify = beautify and pipe.viz_enabled
            visualizer = None
            if do_beautify:
                visualizer = BeautifyVisualizer(
                    labels=class_names, label_mapping=pipe.label_mapping or None,
                    color_mapping=pipe.color_mapping or None,
                    default_color=pipe.default_color, font_path=pipe.font_path)
            else:
                logger.info("美化已关闭, 使用 YOLO 原生 plot() 绘制.")

            run_name = config.experiment_name or DEFAULT_RUN_NAME
            output_dir = _resolve_output_dir(
                RUNS_DIR / f"{config.task}_infer", run_name,
                exist_ok=bool(getattr(config, "exist_ok", False)))
            logger.info(f"输出目录:    {output_dir}")

            predict_kwargs = {k: getattr(config, k) for k in _PREDICT_KEYS
                              if getattr(config, k, None) is not None}
            predict_kwargs["verbose"] = False
            want_save = bool(getattr(config, "save", True))
            want_show = bool(getattr(config, "show", False))

            if output_sink is None:
                output_sink = LocalFileSink(
                    filename=pipe.video_filename, fourcc=pipe.video_fourcc,
                    default_fps=pipe.default_fps) if want_save else NullSink()
                _sink_owned = True
            else:
                _sink_owned = False
                logger.info(f"使用调用方提供的 sink: {output_sink.__class__.__name__}")

            logger.info("=" * 60)
            logger.info("启动推理".center(60))
            logger.info("=" * 60)

            stats = InferStats()
            camera_cfg = pipe.build_camera_config()
            processor = _FrameProcessor(
                model=model, predict_kwargs=predict_kwargs, do_beautify=do_beautify,
                visualizer=visualizer, use_label_mapping=pipe.use_label_mapping,
                style_overrides=pipe.style_overrides, names=model.names)
            raw_batch = getattr(config, "batch", 16)
            batch_size = raw_batch if isinstance(raw_batch, int) and raw_batch >= 1 else DEFAULT_BATCH_SIZE

            from .pipeline import ThreadedPipeline
            logger.info(f"执行引擎: 多线程流水线 (batch={batch_size})")
            pipeline = ThreadedPipeline(
                processor=processor, source=str(raw_source), camera_config=camera_cfg,
                output_dir=output_dir, output_sink=output_sink, batch_size=batch_size,
                save=want_save, show=want_show, show_info=show_info,
                window_name=window_name, warmup_frames=warmup_frames,
                hooks=hooks, cancel_token=cancel_token)
            interrupted = pipeline.run(stats)
            if interrupted:
                logger.warning("推理被用户提前结束.")

            logger.info("=" * 60)
            logger.info("推理完成".center(60))
            logger.info("=" * 60)
            log_infer_stats(stats, logger=logger)

            model_stem = Path(raw_model).stem
            if rename_log:
                rename_log_to_save_dir(output_dir, model_stem)

            audit_path: Path | None = output_dir / "odp_audit.json"
            log_path = _find_project_log_path()
            try:
                audit_payload = {
                    "mode": "infer",
                    "config": config.to_audit_snapshot(),
                    "merger": merger.to_audit_log(),
                    "pipeline": pipe.to_audit(),
                    "stats": stats.to_dict(),
                    "result_summary": {
                        "output_dir": str(output_dir), "saved": want_save,
                        "beautified": do_beautify,
                        "infer_time_sec": (datetime.now() - start).total_seconds(),
                        "log_path": str(log_path) if log_path else None,
                    }}
                audit_path.write_text(json.dumps(audit_payload, ensure_ascii=False,
                                                 indent=2), encoding="utf-8")
                logger.info(f"审计快照:   {audit_path}")
            except OSError as e:
                logger.warning(f"写审计快照失败 (不影响推理结果): {e}")
                audit_path = None

            infer_time = (datetime.now() - start).total_seconds()
            logger.info("=" * 60)
            logger.info(f"推理总耗时: {infer_time:.2f} 秒")
            logger.info(f"输出目录:   {output_dir}")
            if want_save:
                logger.info("结果已保存到上面的目录.")
            if log_path:
                logger.info(f"本次日志:   {log_path}")
            logger.info("=" * 60)

            result = InferResult(success=True, output_dir=output_dir,
                                  stats=stats.to_dict(), infer_time=infer_time,
                                  saved=want_save, audit_path=audit_path, log_path=log_path)
            hooks.fire_complete(result)
            return result
        except Exception as e:
            logger.error(f"推理失败: {e}", exc_info=True)
            infer_time = (datetime.now() - start).total_seconds()
            hooks.fire_error(e)
            return InferResult(success=False,
                               output_dir=output_dir or Path("unknown"),
                               stats={}, infer_time=infer_time,
                               error=str(e), log_path=_find_project_log_path())


@dataclass
class _FrameProcessor:
    model: Any
    predict_kwargs: dict[str, Any]
    do_beautify: bool
    visualizer: Any
    use_label_mapping: bool
    style_overrides: dict[str, Any]
    names: dict[int, str]
    _style: DrawStyle | None = None

    def infer_batch(self, images: list):
        t0 = time.perf_counter()
        results = self.model(images, **self.predict_kwargs)
        batch_dt = time.perf_counter() - t0
        labels_list: list[list[str]] = []
        n_list: list[int] = []
        for result in results:
            boxes = result.boxes
            n = 0 if boxes is None else len(boxes)
            n_list.append(n)
            labels_list.append(
                [self.names[i] for i in boxes.cls.int().cpu().tolist()] if n else [])
        return results, labels_list, n_list, batch_dt

    def draw(self, image, result, labels, n):
        if self.do_beautify and self.visualizer is not None:
            if self._style is None:
                h, w = image.shape[:2]
                self._style = DrawStyle.from_image_size(h, w, **self.style_overrides)
            boxes = result.boxes
            dets = BeautifyVisualizer.from_yolo_results(
                boxes=(boxes.xyxy.cpu().numpy() if n else _empty_boxes()),
                confidences=(boxes.conf.cpu().numpy() if n else _empty_conf()),
                labels=labels)
            return self.visualizer.draw(image, dets, style=self._style,
                                        use_label_mapping=self.use_label_mapping)
        return result.plot()


def _empty_boxes():
    import numpy as np
    return np.zeros((0, 4), dtype=float)


def _empty_conf():
    import numpy as np
    return np.zeros((0,), dtype=float)


def infer_yolo(yaml_path=None, pipeline_yaml=None, cli_args=None, *,
               beautify: bool = True, rename_log: bool = True,
               threaded: bool = False, warmup_frames: int = 0,
               window_name: str = "odp-infer", show_info: bool = True,
               output_sink=None, hooks=None, cancel_token=None) -> InferResult:
    service = InferService()
    return service.predict(
        yaml_path=yaml_path, pipeline_yaml=pipeline_yaml, cli_args=cli_args,
        beautify=beautify, rename_log=rename_log, threaded=threaded,
        warmup_frames=warmup_frames, window_name=window_name, show_info=show_info,
        output_sink=output_sink, hooks=hooks, cancel_token=cancel_token)
