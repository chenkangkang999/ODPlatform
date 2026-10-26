import hashlib
import io
import json
from pathlib import Path
import tempfile

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase, override_settings
from PIL import Image

from football_portal.models import AnalysisJob, AssistantRecord
from football_portal.monitor_views import dataset_inventory


class MonitorTests(TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.events = self.root / "events"
        self.events.mkdir()
        self.settings = override_settings(FOOTBALL_DATA_DIR=self.root / "private", FOOTBALL_EVENT_DATA_DIR=self.events)
        self.settings.enable()
        self.client.get("/football/")

    def tearDown(self):
        self.settings.disable()
        self.temporary.cleanup()

    def manifest(self, files):
        (self.events / "manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")

    def test_monitor_counts_and_names_do_not_leak_other_session(self):
        owner = self.client.session.session_key
        AnalysisJob.objects.create(owner_session=owner, status="failed", input_name="my-video.mp4", error="sample failure")
        AnalysisJob.objects.create(owner_session="other-session", status="running", input_name="private-other.mp4")
        AssistantRecord.objects.create(owner_session="other-session", source="llm", answer="private")
        value = self.client.get("/football/api/monitor/").json()
        self.assertEqual(value["jobs"]["total"], 1)
        self.assertEqual(value["jobs"]["counts"], {"failed": 1})
        self.assertEqual(value["assistant"]["real_calls"], 0)
        self.assertNotIn("private-other", json.dumps(value))
        self.assertIsNone(value["service"]["accuracy"])
        self.assertIsNone(value["service"]["gpu_telemetry"])

    def test_annotation_files_are_not_reported_as_visual_training_samples(self):
        content = b'[{"event":"goal"}]'
        (self.events / "event.json").write_bytes(content)
        self.manifest([{"path": "event.json", "bytes": len(content), "kind": "annotations", "coverage_event_ids": [2],
                        "sha256": hashlib.sha256(content).hexdigest()}])
        value = self.client.get("/football/api/monitor/").json()
        goal = next(row for row in value["event_coverage"] if row["code"] == "goal")
        self.assertEqual(goal["data_state"], "annotations_only")
        self.assertEqual(goal["video_files"], 0)
        self.assertFalse(goal["automated_recognition"])
        self.assertEqual(value["coverage_summary"]["trained_event_models"], 0)

    def test_manifest_escape_duplicate_missing_and_wrong_size_excluded(self):
        (self.root / "outside.txt").write_bytes(b"x")
        (self.events / "inside.json").write_bytes(b"[]")
        self.manifest([{"path": "../outside.txt", "bytes": 1}, {"path": "inside.json", "bytes": 2, "kind": "annotations"},
                       {"path": "inside.json", "bytes": 2}, {"path": "missing.mp4", "bytes": 10},
                       {"path": "wrong.mp4", "bytes": 4}])
        (self.events / "wrong.mp4").write_bytes(b"x")
        value = dataset_inventory()
        self.assertEqual(len(value["files"]), 1)
        self.assertEqual(value["total_bytes"], 2)
        self.assertEqual(value["invalid_entries"], 2)
        self.assertEqual(value["missing_files"], 2)

    def test_malformed_manifest_does_not_crash_monitor(self):
        (self.events / "manifest.json").write_text("broken", encoding="utf-8")
        response = self.client.get("/football/api/monitor/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["datasets"]["invalid_entries"], 1)

    def test_download_failures_are_visible_and_never_counted_as_files(self):
        (self.events / "manifest.json").write_text(json.dumps({"files": [], "failed_downloads": [
            {"path": "events/missing.json", "error": "temporary network error"}]}), encoding="utf-8")
        value = self.client.get("/football/api/monitor/").json()
        self.assertEqual(value["datasets"]["download_failures_count"], 1)
        self.assertEqual(value["datasets"]["files"], [])
        self.assertTrue(any(item["level"] == "warning" for item in value["alerts"]))

    def test_taxonomy_is_shared_with_review_validation_and_rule_guide(self):
        events = self.client.get("/football/api/events/").json()["events"]
        self.assertEqual({item["id"] for item in events}, set(range(1, 43)))
        names = self.client.get("/football/api/var/cases/").json()["event_types"]
        self.assertEqual(names["second_yellow_red"], next(item["name"] for item in events if item["code"] == "second_yellow_red"))
        stream = io.BytesIO()
        Image.new("RGB", (30, 30), "green").save(stream, format="PNG")
        response = self.client.post("/football/api/var/cases/", {"file": SimpleUploadedFile("frame.png", stream.getvalue()),
                                                               "title": "Manual smoke review", "event_type": "smoke_pyrotechnics"})
        self.assertEqual(response.status_code, 201)
        self.assertFalse(response.json()["case"]["automated_decision"])

    def test_display_timing_is_rendered_from_configuration(self):
        with override_settings(FOOTBALL_CAROUSEL_SECONDS=13, FOOTBALL_MONITOR_REFRESH_SECONDS=17):
            response = self.client.get("/football/")
            self.assertContains(response, 'data-interval-ms="13000"')
            self.assertContains(response, 'data-monitor-interval-ms="17000"')
            self.assertEqual(self.client.get("/football/api/monitor/").json()["service"]["refresh_seconds"], 17)

    def test_another_session_does_not_share_review_count(self):
        outsider = Client()
        outsider.get("/football/")
        self.assertEqual(outsider.get("/football/api/monitor/").json()["reviews"]["total"], 0)
