import csv
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from django.conf import settings
from django.db import close_old_connections
from django.utils import timezone

from .models import AnalysisJob
from .registry import find_model
from .results import build_analysis, normalize_payload

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="football-inference")


def job_directory(job):
    return Path(settings.FOOTBALL_DATA_DIR).resolve() / "jobs" / str(job.pk)


def _update(job_id, **values):
    values["updated_at"] = timezone.now()
    AnalysisJob.objects.filter(pk=job_id).update(**values)


def write_import(job, payload):
    normalized = normalize_payload(payload)
    directory = job_directory(job) / "output"
    directory.mkdir(parents=True, exist_ok=False)
    frames = normalized.pop("frames")
    with (directory / "frames.jsonl").open("w", encoding="utf-8") as stream:
        for frame in frames:
            stream.write(json.dumps(frame, ensure_ascii=False, allow_nan=False) + "\n")
    with (directory / "counts.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        rows = normalized["analysis"]["counts_series"]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    normalized["artifacts"] = {"frames": "frames.jsonl", "counts": "counts.csv", "result": "result.json"}
    (directory / "result.json").write_text(json.dumps(normalized, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    _update(job.pk, result=normalized, status="completed", progress=100)


def _stop_process(process):
    if process.poll() is None:
        if sys.platform == "win32":
            try:
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               capture_output=True, timeout=8, check=False)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if sys.platform != "win32":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait(timeout=5)


def _worker_environment():
    environment = os.environ.copy()
    for key in ("FOOTBALL_LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY", "DJANGO_SECRET_KEY"):
        environment.pop(key, None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONUTF8"] = "1"
    return environment


def _run(job_id):
    close_old_connections()
    process = None
    try:
        job = AnalysisJob.objects.get(pk=job_id)
        if job.cancel_requested or job.status == "cancelled":
            _update(job_id, status="cancelled")
            return
        model = find_model(job.model_id)
        directory = job_directory(job)
        output = directory / "output"
        request_data = {"input_path": str(directory / job.parameters["_input_filename"]),
                        "output_dir": str(output), "model_path": str(model["_path"]),
                        "model_id": model["id"], "model_label": model["name"],
                        "metadata_path": str(model["_metadata_path"]) if model.get("_metadata_path") else None,
                        "conf": job.parameters["conf"], "imgsz": job.parameters["imgsz"],
                        "stride": job.parameters["stride"], "device": str(getattr(settings, "FOOTBALL_DEVICE", "cpu")),
                        "max_frames": int(getattr(settings, "FOOTBALL_MAX_FRAMES", 1500)),
                        "platform_root": str(getattr(settings, "FOOTBALL_PLATFORM_ROOT", ""))}
        request_path = directory / "request.json"
        request_path.write_text(json.dumps(request_data, ensure_ascii=False, indent=2), encoding="utf-8")
        python = str(getattr(settings, "FOOTBALL_INFERENCE_PYTHON", "") or sys.executable)
        command = [python, str(Path(__file__).with_name("bridge_worker.py")), "--request", str(request_path)]
        _update(job_id, status="running", progress=1)
        started = time.monotonic()
        with (directory / "worker.log").open("w", encoding="utf-8") as log:
            kwargs = {"stdout": log, "stderr": subprocess.STDOUT, "shell": False,
                      "cwd": str(directory), "env": _worker_environment()}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            else:
                kwargs["start_new_session"] = True
            process = subprocess.Popen(command, **kwargs)
            while process.poll() is None:
                state = AnalysisJob.objects.values("cancel_requested").get(pk=job_id)
                if state["cancel_requested"]:
                    _stop_process(process)
                    _update(job_id, status="cancelled", error="任务已取消。")
                    return
                if time.monotonic() - started > int(getattr(settings, "FOOTBALL_JOB_TIMEOUT", 600)):
                    _stop_process(process)
                    raise RuntimeError("任务处理超时，请缩短视频或降低输入尺寸。")
                try:
                    progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
                    number = min(99, max(1, int(progress.get("progress", 1))))
                    _update(job_id, progress=number)
                except (OSError, ValueError, TypeError):
                    pass
                time.sleep(0.4)
        job.refresh_from_db()
        if job.cancel_requested:
            _update(job_id, status="cancelled", error="任务已取消。")
            return
        if process.returncode:
            raise RuntimeError("YOLO处理失败，请检查模型类别、图片／视频格式及本地推理环境。")
        result_path = output / "result.json"
        if not result_path.is_file() or result_path.stat().st_size > 64 * 1024 * 1024:
            raise RuntimeError("推理没有生成有效结果。")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        # Keep full frame records in the downloadable artifact, not every polling response.
        export_frames = payload.pop("frames", None)
        if payload.get("success") is False or payload.get("status") in ("failed", "cancelled"):
            raise RuntimeError("推理未成功完成，部分输出不能当作完整统计。")
        frames_path = output / "frames.jsonl"
        if frames_path.is_file():
            frames = []
            with frames_path.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        frames.append(json.loads(line))
                    if len(frames) > int(getattr(settings, "FOOTBALL_MAX_FRAMES", 1500)):
                        raise RuntimeError("实际帧数超出任务限制。")
            payload["analysis"] = build_analysis(frames)
            export_frames = frames
            if payload.get("source", {}).get("truncated"):
                maximum = int(getattr(settings, "FOOTBALL_MAX_FRAMES", 1500))
                payload["analysis"]["limitations"].append(f"仅处理前{maximum}个采样帧，未覆盖全部视频。")
        if not isinstance(payload.get("analysis"), dict):
            raise RuntimeError("结果缺少真实逐帧统计。")
        payload.setdefault("artifacts", {})["result"] = "result.json"
        export_payload = {**payload, "frames": export_frames} if export_frames is not None else payload
        result_path.write_text(json.dumps(export_payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        _update(job_id, status="completed", progress=100, result=payload, error="")
    except Exception:
        if process is not None:
            _stop_process(process)
        _update(job_id, status="failed", error="任务未能完成，请核对模型、文件与独立推理环境；详细日志仅保存在本机。")
    finally:
        close_old_connections()


def submit_job(job_id):
    if getattr(settings, "FOOTBALL_RUN_JOBS_INLINE", False):
        runner = getattr(settings, "FOOTBALL_JOB_RUNNER", None)
        (runner or _run)(job_id)
    else:
        _executor.submit(_run, job_id)
