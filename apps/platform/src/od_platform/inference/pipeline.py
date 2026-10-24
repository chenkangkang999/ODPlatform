"""多线程推理流水线."""
from __future__ import annotations

import logging
import time
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread

from od_platform.frame_source import create_frame_source
from od_platform.frame_source.core.types import is_stream_url

from .cancel import CancelToken
from .defaults import (
    DISPLAY_DRAIN_SEC,
    DISPLAY_JOIN_SEC,
    DISPLAY_POLL_SEC,
    LIVE_QUEUE_SIZE,
    PAUSE_POLL_SEC,
    QUEUE_PUT_TIMEOUT_SEC,
    READER_QUEUE_FACTOR,
    READER_QUEUE_MIN,
    READER_READ_TIMEOUT_SEC,
    RENDER_POLL_SEC,
    RENDER_QUEUE_SIZE,
    RENDERER_JOIN_SEC,
)
from .hooks import FrameEvent, InferHooks, ProgressEvent
from .overlay import Metrics, draw_hud, draw_pause
from .sinks import NullSink, OutputSink

logger = logging.getLogger(__name__)

_SENTINEL = object()


def _put_latest(q: Queue, item) -> None:
    try:
        q.put_nowait(item)
    except Full:
        try:
            q.get_nowait()
        except Empty:
            pass
        try:
            q.put_nowait(item)
        except Full:
            pass


def _put_block(q: Queue, item) -> None:
    while True:
        try:
            q.put(item, timeout=QUEUE_PUT_TIMEOUT_SEC)
            return
        except Full:
            continue


class _Controller:
    def __init__(self) -> None:
        self._paused = Event()
    def toggle(self) -> None:
        self._paused.clear() if self._paused.is_set() else self._paused.set()
    def is_paused(self) -> bool:
        return self._paused.is_set()


class _Reader(Thread):
    def __init__(self, source, camera_config, *, live: bool, capacity: int, capture_fps) -> None:
        super().__init__(daemon=True)
        self._source = source
        self._camera_config = camera_config
        self._live = live
        self._capture_fps = capture_fps
        self.q: Queue = Queue(maxsize=LIVE_QUEUE_SIZE if live else capacity)
        self._stop_evt = Event()
        self.source_type = None
        self.error: Exception | None = None

    def _resolve_source(self):
        if self._camera_config is not None and str(self._source).isdigit():
            return self._camera_config.model_copy(update={"camera_id": int(self._source)})
        return self._source

    def run(self) -> None:
        try:
            with create_frame_source(self._resolve_source()) as src:
                self.source_type = src.get_source_type()
                t_prev = time.perf_counter()
                for frame in src:
                    if self._stop_evt.is_set():
                        break
                    now = time.perf_counter()
                    self._capture_fps.update((now - t_prev) * 1000)
                    t_prev = now
                    if self._live:
                        _put_latest(self.q, frame)
                    else:
                        _put_block(self.q, frame)
        except Exception as e:
            self.error = e
        finally:
            _put_block(self.q, _SENTINEL)

    def get(self, timeout: float):
        try:
            return self.q.get(timeout=timeout)
        except Empty:
            return None

    def get_nowait(self):
        try:
            return self.q.get_nowait()
        except Empty:
            return None

    def stop(self) -> None:
        self._stop_evt.set()


class _Renderer(Thread):
    def __init__(self, processor, in_q: Queue, out_q: Queue, *,
                 drop: bool, output_sink: OutputSink, show: bool, show_info: bool,
                 recording: bool, metrics: Metrics, hooks: InferHooks) -> None:
        super().__init__(daemon=True)
        self._proc = processor
        self._in = in_q
        self._out = out_q
        self._drop = drop
        self._sink = output_sink
        self._show = show
        self._show_info = show_info
        self._recording = recording
        self._m = metrics
        self._hooks = hooks
        self._stop_evt = Event()
        self._frame_idx = 0

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                item = self._in.get(timeout=RENDER_POLL_SEC)
            except Empty:
                continue
            if item is _SENTINEL:
                _put_block(self._out, _SENTINEL)
                break
            frame, result, labels, n = item
            t0 = time.perf_counter()
            try:
                annotated = self._proc.draw(frame.image, result, labels, n)
            except Exception as e:
                logger.warning(f"渲染单帧失败, 跳过: {e}")
                continue
            self._m.render.update((time.perf_counter() - t0) * 1000)
            self._sink.write(frame, annotated)
            if self._hooks.on_frame is not None:
                self._hooks.fire_frame(FrameEvent(
                    frame_idx=self._frame_idx, image=frame.image,
                    annotated=annotated, n_detections=n, detections=None))
            self._frame_idx += 1
            if self._show:
                disp = annotated.copy()
                draw_hud(disp, self._m, n_dets=n, recording=self._recording,
                         show_info=self._show_info)
                _put_latest(self._out, disp) if self._drop else _put_block(self._out, disp)


class _Display(Thread):
    def __init__(self, out_q: Queue, window_name: str, controller: _Controller) -> None:
        super().__init__(daemon=True)
        self._out = out_q
        self._win = window_name
        self._ctrl = controller
        self._stop_evt = Event()
        self._key_lock = Lock()
        self._key = -1
        self._last = None

    def stop(self) -> None:
        self._stop_evt.set()

    def get_key(self) -> int:
        with self._key_lock:
            k, self._key = self._key, -1
            return k

    def run(self) -> None:
        import cv2
        poll = cv2.pollKey if hasattr(cv2, "pollKey") else (lambda: cv2.waitKey(1))
        while not self._stop_evt.is_set():
            frame = None
            try:
                item = self._out.get(timeout=DISPLAY_POLL_SEC)
                if item is _SENTINEL:
                    pass
                else:
                    frame = item
                    self._last = frame
            except Empty:
                if self._ctrl.is_paused() and self._last is not None:
                    frame = self._last.copy()
                    draw_pause(frame)
            if frame is not None:
                cv2.imshow(self._win, frame)
            key = poll() & 0xFF
            if key != 255:
                with self._key_lock:
                    self._key = key
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


class ThreadedPipeline:
    def __init__(self, *, processor, source, camera_config, output_dir,
                 output_sink: OutputSink, batch_size, save, show, show_info,
                 window_name, warmup_frames,
                 hooks: InferHooks | None = None,
                 cancel_token: CancelToken | None = None) -> None:
        self.proc = processor
        self.source = str(source)
        self.camera_config = camera_config
        self.output_dir = output_dir
        self.sink = output_sink
        self.batch_size = max(1, batch_size)
        self.save = save
        self.show = show
        self.show_info = show_info
        self.window_name = window_name
        self.warmup_frames = warmup_frames
        self.hooks = hooks if hooks is not None else InferHooks()
        self.cancel_token = cancel_token

    def _is_cancelled(self) -> bool:
        return self.cancel_token is not None and self.cancel_token.is_cancelled()

    def run(self, stats) -> bool:
        m = Metrics()
        s = self.source
        live = s.isdigit() or is_stream_url(s)
        eff_batch = 1 if live else self.batch_size
        render_drop = not self.save
        reader = _Reader(s, self.camera_config, live=live,
                         capacity=max(eff_batch * READER_QUEUE_FACTOR, READER_QUEUE_MIN), capture_fps=m.capture)
        in_q: Queue = Queue(maxsize=max(eff_batch * 2, 4))
        out_q: Queue = Queue(maxsize=RENDER_QUEUE_SIZE)
        controller = _Controller()
        renderer = None
        display = None
        interrupted = False
        warmed = 0
        last_batch_end_t = None
        start_time = time.perf_counter()
        sink_opened = False
        reader.start()
        try:
            while True:
                if controller.is_paused():
                    if self._is_cancelled():
                        interrupted = True
                        break
                    if self._handle_key(display, controller):
                        interrupted = True
                        break
                    time.sleep(PAUSE_POLL_SEC)
                    continue
                if self._is_cancelled():
                    interrupted = True
                    break
                first = reader.get(timeout=READER_READ_TIMEOUT_SEC)
                if first is None:
                    if reader.error:
                        raise reader.error
                    continue
                if first is _SENTINEL:
                    break
                batch = [first]
                ended = False
                for _ in range(eff_batch - 1):
                    nxt = reader.get_nowait()
                    if nxt is None:
                        break
                    if nxt is _SENTINEL:
                        ended = True
                        break
                    batch.append(nxt)
                if warmed < self.warmup_frames:
                    warmed += len(batch)
                    if ended:
                        break
                    continue
                if renderer is None:
                    try:
                        self.sink.open(self.output_dir, reader.source_type)
                        sink_opened = True
                    except Exception as e:
                        logger.error(f"sink.open 失败, 退化用 NullSink: {e}")
                        self.sink = NullSink()
                        self.sink.open(self.output_dir, reader.source_type)
                        sink_opened = True
                    renderer = _Renderer(self.proc, in_q, out_q, drop=render_drop,
                                         output_sink=self.sink, show=self.show,
                                         show_info=self.show_info, recording=self.save,
                                         metrics=m, hooks=self.hooks)
                    renderer.start()
                    if self.show:
                        display = _Display(out_q, self.window_name, controller)
                        display.start()
                images = [f.image for f in batch]
                results, labels_list, n_list, batch_dt = self.proc.infer_batch(images)
                stats.infer_time_sec += batch_dt
                for frame, result, labels, n in zip(batch, results, labels_list, n_list):
                    stats.frames += 1
                    stats.detections += n
                    for name in labels:
                        stats.per_class[name] = stats.per_class.get(name, 0) + 1
                    m.add_speed(getattr(result, "speed", None))
                    if render_drop:
                        _put_latest(in_q, (frame, result, labels, n))
                    else:
                        _put_block(in_q, (frame, result, labels, n))
                    if (self.hooks.on_progress is not None
                            and stats.frames % self.hooks.progress_interval_frames == 0):
                        self.hooks.fire_progress(ProgressEvent(
                            frame_idx=stats.frames, total_frames=None,
                            elapsed_sec=time.perf_counter() - start_time,
                            fps_loop=m.loop.fps, fps_infer=m.infer.fps,
                            detections_total=stats.detections))
                batch_end_t = time.perf_counter()
                if last_batch_end_t is not None:
                    per_frame_loop_ms = (batch_end_t - last_batch_end_t) * 1000 / len(batch)
                    for _ in batch:
                        m.loop.update(per_frame_loop_ms)
                last_batch_end_t = batch_end_t
                if self._is_cancelled():
                    interrupted = True
                    break
                if self._handle_key(display, controller):
                    interrupted = True
                    break
                if ended:
                    break
        finally:
            reader.stop()
            _put_block(in_q, _SENTINEL)
            if renderer is not None:
                renderer.join(timeout=RENDERER_JOIN_SEC)
                renderer.stop()
            if display is not None:
                time.sleep(DISPLAY_DRAIN_SEC)
                display.stop()
                display.join(timeout=DISPLAY_JOIN_SEC)
            if sink_opened:
                try:
                    self.sink.close()
                except Exception as e:
                    logger.warning(f"sink.close 异常 (已吞): {e}")
        _write_fps(stats, m)
        logger.info("流水线收尾: 捕获 %.1f | 推理 %.1f | 渲染 %.1f | loop %.1f FPS"
                    % (m.capture.fps, m.infer.fps, m.render.fps, m.loop.fps))
        return interrupted

    def _handle_key(self, display, controller) -> bool:
        if display is None:
            return False
        key = display.get_key()
        if key in (ord("q"), 27):
            return True
        if key == ord(" "):
            controller.toggle()
        return False


def _write_fps(stats, m: Metrics) -> None:
    snap = m.snapshot()
    stats.capture_fps = snap["capture_fps"]
    stats.infer_fps = snap["infer_fps"]
    stats.render_fps = snap["render_fps"]
    stats.loop_fps = snap["loop_fps"]
    stats.current_fps = snap["current_fps"]
    stats.speed_ms = snap["speed_ms"]
