import io
import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase, override_settings
from PIL import Image

from football_portal import registry
from football_portal.jobs import _run, _stop_process, _worker_environment, job_directory
from football_portal.models import AnalysisJob, AssistantRecord
from football_portal.results import build_analysis, normalize_payload


def image_upload(name="pitch.png"):
    stream = io.BytesIO()
    Image.new("RGB", (100, 60), "green").save(stream, format="PNG")
    return SimpleUploadedFile(name, stream.getvalue(), content_type="image/png")


def box(label="ball", confidence=.8):
    return {"cls": label, "conf": confidence, "x1": 10, "y1": 10, "x2": 20, "y2": 20}


class PortalAPITests(TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.weight = self.directory / "trusted.pt"
        self.weight.write_bytes(b"test-only weight; never loaded")
        self.overrides = override_settings(FOOTBALL_DATA_DIR=self.directory / "private",
                                          FOOTBALL_MODEL_PATH=str(self.weight), FOOTBALL_MODEL_DIR="",
                                          FOOTBALL_MODEL_METADATA="", FOOTBALL_MAX_QUEUE=6,
                                          FOOTBALL_MAX_IMAGE_BYTES=15 * 1024**2,
                                          FOOTBALL_MAX_VIDEO_BYTES=200 * 1024**2)
        self.overrides.enable()
        self.submit = patch("football_portal.jobs.submit_job")
        self.mock_submit = self.submit.start()
        self.client.get("/football/api/jobs/")
        self.model_id = registry.public_models()[0]["id"]

    def tearDown(self):
        self.submit.stop()
        self.overrides.disable()
        self.temporary.cleanup()

    def post_json(self, url, payload, client=None):
        return (client or self.client).post(url, data=json.dumps(payload), content_type="application/json")

    def create_job(self):
        response = self.client.post("/football/api/jobs/", {"file": image_upload(), "model_id": self.model_id})
        self.assertEqual(response.status_code, 201, response.content)
        return response.json()["job"]

    def import_job(self, payload=None):
        response = self.post_json("/football/api/import/", payload or {"image_size": {"width": 100, "height": 60}, "boxes": [box()]})
        self.assertEqual(response.status_code, 201, response.content)
        return response.json()["job"]

    def test_model_registry_never_exposes_trusted_server_paths(self):
        response = self.client.get("/football/api/models/")
        self.assertEqual(response.status_code, 200)
        model = response.json()["models"][0]
        self.assertTrue(model["ready"])
        self.assertNotIn("_path", model)
        self.assertNotIn(str(self.directory), response.content.decode())

    def test_digest_mismatch_marks_model_not_ready(self):
        self.weight.with_suffix(".metadata.json").write_text('{"model_sha256":"incorrect"}', encoding="utf-8")
        self.assertFalse(registry.public_models()[0]["ready"])

    def test_upload_stays_private_and_queues_only_one_trusted_model(self):
        result = self.create_job()
        job = AnalysisJob.objects.get(pk=result["id"])
        self.assertEqual(job.status, "queued")
        self.assertTrue((job_directory(job) / "input.png").is_file())
        self.assertEqual(result["parameters"]["imgsz"], 960)
        self.assertNotIn("_input_filename", result["parameters"])
        self.mock_submit.assert_called_once_with(job.pk)
        artifact = self.client.get(result["artifact_urls"]["original"])
        self.assertEqual(artifact.status_code, 200)
        artifact.close()

    def test_no_path_or_arbitrary_model_upload_is_allowed(self):
        response = self.client.post("/football/api/jobs/", {"file": image_upload(), "model_id": "../../untrusted.pt"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(AnalysisJob.objects.count(), 0)

    def test_other_session_cannot_read_cancel_or_download_job(self):
        job = self.create_job()
        other = Client()
        self.assertEqual(other.get(f"/football/api/jobs/{job['id']}/").status_code, 404)
        self.assertEqual(other.post(f"/football/api/jobs/{job['id']}/cancel/").status_code, 404)
        self.assertEqual(other.get(job["artifact_urls"]["original"]).status_code, 404)
        self.assertEqual(other.get("/football/api/jobs/").json()["jobs"], [])

    def test_cancel_queued_job(self):
        job = self.create_job()
        response = self.client.post(f"/football/api/jobs/{job['id']}/cancel/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["job"]["status"], "cancelled")
        self.assertTrue(response.json()["job"]["cancel_requested"])

    def test_demo_not_configured_is_honest_and_does_not_create_job(self):
        with override_settings(FOOTBALL_DEMO_IMAGE_PATH=""):
            response = self.post_json("/football/api/demo/", {"model_id": self.model_id, "path": "ignored-client-path"})
            self.assertEqual(response.status_code, 404)
            self.assertFalse(self.client.get("/football/api/status/").json()["demo_available"])
        self.assertEqual(AnalysisJob.objects.count(), 0)

    def test_demo_copies_only_configured_image_for_current_owner(self):
        source = self.directory / "configured.png"
        Image.new("RGB", (100, 60), "green").save(source)
        with override_settings(FOOTBALL_DEMO_IMAGE_PATH=str(source)):
            response = self.post_json("/football/api/demo/", {"model_id": self.model_id, "path": "ignored-client-path"})
        self.assertEqual(response.status_code, 201, response.content)
        job = AnalysisJob.objects.get(pk=response.json()["job"]["id"])
        self.assertEqual(job.parameters["sample_origin"], "local_integration_example")
        self.assertEqual((job_directory(job) / "input.png").read_bytes(), source.read_bytes())
        self.assertEqual(job.owner_session, self.client.session.session_key)
        self.mock_submit.assert_called_once_with(job.pk)

    def test_csrf_protection_remains_active(self):
        response = Client(enforce_csrf_checks=True).post("/football/api/import/", data="{}", content_type="application/json")
        self.assertEqual(response.status_code, 403)

    def test_invalid_image_and_threshold_are_rejected(self):
        response = self.client.post("/football/api/jobs/", {"file": SimpleUploadedFile("bad.png", b"not image"), "model_id": self.model_id})
        self.assertEqual(response.status_code, 400)
        response = self.client.post("/football/api/jobs/", {"file": image_upload(), "model_id": self.model_id, "conf": "nan"})
        self.assertEqual(response.status_code, 400)
        response = self.client.post("/football/api/jobs/", {"file": image_upload("model.pt"), "model_id": self.model_id})
        self.assertEqual(response.status_code, 400)

    def test_upload_limit_and_queue_limit(self):
        with override_settings(FOOTBALL_MAX_IMAGE_BYTES=2):
            self.assertEqual(self.client.post("/football/api/jobs/", {"file": image_upload(), "model_id": self.model_id}).status_code, 400)
        self.create_job()
        with override_settings(FOOTBALL_MAX_QUEUE=1):
            self.assertEqual(self.client.post("/football/api/jobs/", {"file": image_upload(), "model_id": self.model_id}).status_code, 429)

    def test_import_recomputes_counts_and_retains_empty_frames(self):
        job = self.import_job({"frames": [
            {"frame_index": 0, "image_size": {"width": 100, "height": 60}, "boxes": [box(), box("player")]},
            {"frame_index": 2, "image_size": {"width": 100, "height": 60}, "boxes": []}],
            "analysis": {"accuracy": 1.0, "possession": .8}})
        result = job["result"]
        self.assertEqual(result["analysis"]["processed_frames"], 2)
        self.assertEqual(result["analysis"]["total_detections"], 2)
        self.assertEqual(result["analysis"]["ball_frame_fraction"], .5)
        self.assertNotIn("accuracy", result["analysis"])
        self.assertNotIn("possession", result["analysis"])
        self.assertTrue(result["source"]["imported"])
        artifact = self.client.get(job["artifact_urls"]["counts"])
        self.assertEqual(artifact.status_code, 200)
        artifact.close()

    def test_import_rejects_corrupt_duplicate_or_nan_records(self):
        for payload in ({"boxes": [box(confidence=float("nan"))]},
                        {"frames": [{"frame_index": 0, "boxes": []}, {"frame_index": 0, "boxes": []}]},
                        {"boxes": [{"cls": "ball", "conf": .8, "xyxy": [20, 20, 10, 10]}]}):
            with self.subTest(payload=payload):
                self.assertEqual(self.post_json("/football/api/import/", payload).status_code, 400)
        self.assertEqual(AnalysisJob.objects.count(), 0)

    def test_artifact_path_traversal_and_private_log_are_not_served(self):
        result = self.import_job()
        job = AnalysisJob.objects.get(pk=result["id"])
        jobs_root = self.directory / "private"
        jobs_root.mkdir(parents=True, exist_ok=True)
        (jobs_root / "outside.txt").write_text("not public", encoding="utf-8")
        job.result["artifacts"]["annotated"] = "../../../outside.txt"
        job.save()
        self.assertEqual(self.client.get(f"/football/api/jobs/{job.pk}/artifact/annotated/").status_code, 404)
        self.assertEqual(self.client.get(f"/football/api/jobs/{job.pk}/artifact/worker.log/").status_code, 404)

    def test_existing_image_preview_is_restyled_without_rerunning_detection(self):
        created = self.create_job()
        job = AnalysisJob.objects.get(pk=created["id"])
        output = job_directory(job) / "output"
        output.mkdir()
        Image.new("RGB", (100, 60), "black").save(output / "annotated.png")
        job.result = normalize_payload({"image_size": {"width": 100, "height": 60}, "boxes": [box()]})
        job.result["artifacts"] = {"annotated": "annotated.png"}
        job.status = "completed"
        job.save()
        previous = json.dumps(job.result, sort_keys=True)
        response = self.client.get(f"/football/api/jobs/{job.pk}/artifact/annotated/")
        self.assertEqual(response.status_code, 200)
        response.close()
        job.refresh_from_db()
        self.assertEqual(json.dumps(job.result, sort_keys=True), previous)
        with Image.open(output / "annotated.png") as rendered:
            self.assertNotEqual(rendered.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(json.loads((output / "annotation_style.json").read_text())["version"], "football.roles.v2")
        self.mock_submit.assert_called_once()

    def test_no_key_is_honest_and_local_summary_is_separate(self):
        job = self.import_job()
        with patch("football_portal.llm_service.status", return_value={"configured": False}):
            response = self.post_json("/football/api/assistant/", {"job_id": job["id"], "mode": "report"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["code"], "LLM_NOT_CONFIGURED")
        with patch("football_portal.llm_service.local_summary", return_value="本地统计摘要，不是LLM输出"):
            response = self.post_json("/football/api/local-summary/", {"job_id": job["id"]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["source"], "local")
        self.assertEqual(self.client.get("/football/api/usage/").json()["totals"]["calls"], 0)

    def test_assistant_uses_server_job_data_and_saves_actual_usage(self):
        job = self.import_job()
        fake = {"answer": "检测到1个足球框；不是控球率。", "source": "llm", "model": "test-model",
                "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}}
        with patch("football_portal.llm_service.status", return_value={"configured": True}), \
             patch("football_portal.llm_service.answer", return_value=fake) as answer:
            response = self.post_json("/football/api/assistant/", {"job_id": job["id"], "mode": "report", "question": "解释结果"})
        self.assertEqual(response.status_code, 200, response.content)
        self.assertEqual(answer.call_args.kwargs["context"]["statistics"]["cumulative_detections_by_class"]["ball"], 1)
        self.assertEqual(AssistantRecord.objects.count(), 1)
        self.assertEqual(self.client.get("/football/api/usage/").json()["totals"]["total_tokens"], 20)
        other = Client()
        self.assertEqual(other.get("/football/api/assistant/history/").json()["records"], [])
        self.assertEqual(other.get("/football/api/usage/").json()["totals"]["total_tokens"], 0)

    def test_foreign_job_is_not_sent_to_llm(self):
        job = self.import_job()
        response = self.post_json("/football/api/assistant/", {"job_id": job["id"], "mode": "report"}, client=Client())
        self.assertEqual(response.status_code, 404)

    def test_assistant_history_redacts_accidentally_pasted_credentials(self):
        job = self.import_job()
        fake = {"answer": "只能根据统计解读。", "source": "llm", "model": "test-model", "usage": {}}
        with patch("football_portal.llm_service.status", return_value={"configured": True}), \
             patch("football_portal.llm_service.answer", return_value=fake):
            response = self.post_json("/football/api/assistant/", {"job_id": job["id"], "mode": "chat",
                                                                 "question": "api_key=private-test-value 请解释"})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("private-test-value", AssistantRecord.objects.get().question)

    def test_windows_cancellation_targets_only_our_subprocess_tree(self):
        process = MagicMock()
        process.poll.return_value = None
        process.pid = 123456
        with patch("football_portal.jobs.sys.platform", "win32"), patch("football_portal.jobs.subprocess.run") as run:
            _stop_process(process)
        self.assertEqual(run.call_args.args[0], ["taskkill", "/PID", "123456", "/T", "/F"])
        process.wait.assert_called_once_with(timeout=5)

    def test_algorithm_process_does_not_inherit_llm_credentials(self):
        with patch.dict("os.environ", {"FOOTBALL_LLM_API_KEY": "private-test-key", "DEEPSEEK_API_KEY": "private-test-key", "OPENAI_API_KEY": "private-test-key"}):
            environment = _worker_environment()
        self.assertNotIn("FOOTBALL_LLM_API_KEY", environment)
        self.assertNotIn("DEEPSEEK_API_KEY", environment)
        self.assertNotIn("OPENAI_API_KEY", environment)
        self.assertEqual(environment["PYTHONDONTWRITEBYTECODE"], "1")

    def test_background_job_recomputes_output_and_preserves_truncation_notice(self):
        created = self.create_job()
        job = AnalysisJob.objects.get(pk=created["id"])

        def worker(command, **options):
            config = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
            output = Path(config["output_dir"])
            output.mkdir()
            frame = {"frame_index": 0, "image_size": {"width": 100, "height": 60}, "boxes": [box()]}
            (output / "frames.jsonl").write_text(json.dumps(frame) + "\n", encoding="utf-8")
            result = {"success": True, "source": {"truncated": True},
                      "analysis": {"total_detections": 999}, "artifacts": {"frames": "frames.jsonl"}}
            (output / "result.json").write_text(json.dumps(result), encoding="utf-8")
            process = MagicMock()
            process.poll.return_value = 0
            process.returncode = 0
            return process

        with patch("football_portal.jobs.subprocess.Popen", side_effect=worker), \
             patch("football_portal.jobs.close_old_connections"):
            _run(job.pk)
        job.refresh_from_db()
        self.assertEqual(job.status, "completed")
        self.assertEqual(job.result["analysis"]["total_detections"], 1)
        self.assertTrue(any("未覆盖全部视频" in text for text in job.result["analysis"]["limitations"]))
        exported = json.loads((job_directory(job) / "output/result.json").read_text(encoding="utf-8"))
        self.assertEqual(len(exported["frames"]), 1)
        self.assertNotIn("frames", job.result)

    def test_cancelled_queued_job_never_starts_algorithm_process(self):
        created = self.create_job()
        job = AnalysisJob.objects.get(pk=created["id"])
        job.cancel_requested = True
        job.save()
        with patch("football_portal.jobs.subprocess.Popen") as popen, patch("football_portal.jobs.close_old_connections"):
            _run(job.pk)
        popen.assert_not_called()
        job.refresh_from_db()
        self.assertEqual(job.status, "cancelled")


class ResultContractTests(TestCase):
    def test_multiframe_preview_only_is_rejected_instead_of_losing_video_statistics(self):
        with self.assertRaisesMessage(ValueError, "完整frames"):
            normalize_payload({"analysis": {"processed_frames": 2}, "preview": {"boxes": [box()]}})

    def test_aggregate_only_report_is_not_an_empty_success(self):
        with self.assertRaisesMessage(ValueError, "原始boxes"):
            normalize_payload({"analysis": {"total_detections": 25}})

    def test_legacy_label_xyxy_and_true_centres(self):
        value = normalize_payload({"image_size": {"width": 100, "height": 60},
                                   "boxes": [{"label": "足球", "confidence": .8, "xyxy": [10, 10, 30, 30]}]})
        self.assertEqual(value["boxes"][0]["cx"], 20)
        self.assertEqual(value["analysis"]["positions"][0]["x"], .2)

    def test_absent_image_dimensions_do_not_invent_positions(self):
        value = normalize_payload({"boxes": [box()]})
        self.assertEqual(value["analysis"]["positions"], [])
        self.assertIsNone(value["preview"]["image_size"])

    def test_no_predictions_does_not_produce_fake_points_or_accuracy(self):
        result = build_analysis([])
        self.assertEqual(result["total_detections"], 0)
        self.assertIsNone(result["ball_frame_fraction"])
        self.assertEqual(result["positions"], [])

    def test_numeric_ids_require_explicit_names(self):
        with self.assertRaises(ValueError):
            normalize_payload({"boxes": [{"cls": 0, "conf": .9, "xyxy": [1, 1, 2, 2]}]})
        value = normalize_payload({"class_names": ["player"], "boxes": [{"cls": 0, "conf": .9, "xyxy": [1, 1, 2, 2]}]})
        self.assertEqual(value["boxes"][0]["cls"], "player")
