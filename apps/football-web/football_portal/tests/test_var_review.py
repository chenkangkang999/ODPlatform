import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase, override_settings
from django.urls import include, path
from PIL import Image

from football_portal.jobs import job_directory
from football_portal.models import AnalysisJob
from football_portal.var_models import VarReviewCase
from football_portal import var_services

urlpatterns = [path("football/api/var/", include("football_portal.var_urls"))]
BASE = "/football/api/var/cases/"


def picture(name="pitch.png"):
    stream = io.BytesIO()
    Image.new("RGB", (120, 80), "green").save(stream, format="PNG")
    return SimpleUploadedFile(name, stream.getvalue(), content_type="image/png")


@override_settings(ROOT_URLCONF=__name__)
class VarReviewTests(TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.settings = override_settings(FOOTBALL_DATA_DIR=self.root / "private",
                                         FOOTBALL_MAX_IMAGE_BYTES=15 * 1024**2,
                                         FOOTBALL_MAX_VIDEO_BYTES=200 * 1024**2)
        self.settings.enable()
        self.client.get(BASE)

    def tearDown(self):
        self.settings.disable()
        self.temporary.cleanup()

    def create(self, *, event="offside", video=False):
        # Test-only bytes for streaming semantics; never represented as a decoded match.
        upload = SimpleUploadedFile("stream-test.mp4", bytes(range(256)) * 512, content_type="video/mp4") if video else picture()
        response = self.client.post(BASE, {"file": upload, "title": "人工复核测试", "event_type": event})
        self.assertEqual(response.status_code, 201, response.content)
        return response.json()["case"]

    def patch_case(self, case, data):
        return self.client.patch(BASE + case["id"] + "/", json.dumps(data), content_type="application/json")

    def add_frame(self, case, **values):
        data = {"file": picture(), "media_time": 0, "notes": "人工观察", "lines": json.dumps([{"x1": .2, "y1": .4, "x2": .8, "y2": .4}])}
        data.update(values)
        return self.client.post(BASE + case["id"] + "/keyframes/", data)

    def test_upload_creates_private_manual_case_without_detector(self):
        with patch("football_portal.jobs.submit_job") as submit:
            case = self.create()
        self.assertEqual(AnalysisJob.objects.count(), 0)
        submit.assert_not_called()
        self.assertFalse(case["automated_decision"])
        self.assertFalse(case["calibration"])
        self.assertEqual(case["source"], "manual_review")
        self.assertEqual(case["keyframes"], [])
        self.assertEqual(case["duration_seconds"], 0)
        self.assertNotIn(str(self.root), json.dumps(case))
        response = self.client.get(case["original_url"])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Cache-Control"], "private, no-store")
        response.close()

    def test_existing_queued_job_media_is_copied_without_mutation_or_inference(self):
        owner = self.client.session.session_key
        job = AnalysisJob.objects.create(owner_session=owner, kind="image", status="queued", input_name="uploaded.png",
                                         parameters={"_input_filename": "input.png"}, result={"unchanged": True})
        directory = job_directory(job)
        directory.mkdir(parents=True)
        original = picture().read()
        (directory / "input.png").write_bytes(original)
        with patch("football_portal.jobs.submit_job") as submit:
            response = self.client.post(BASE, json.dumps({"job_id": str(job.pk), "event_type": "goal"}), content_type="application/json")
        self.assertEqual(response.status_code, 201, response.content)
        case = response.json()["case"]
        self.assertEqual(case["source_job_id"], str(job.pk))
        job.refresh_from_db()
        self.assertEqual(job.status, "queued")
        self.assertEqual(job.result, {"unchanged": True})
        self.assertEqual((directory / "input.png").read_bytes(), original)
        submit.assert_not_called()
        self.assertEqual(var_services.artifact_path(VarReviewCase.objects.get(pk=case["id"]), "original").read_bytes(), original)

    def test_cross_session_cannot_access_change_capture_export_or_download(self):
        case = self.create()
        captured = self.add_frame(case)
        self.assertEqual(captured.status_code, 201)
        frame = captured.json()["case"]["keyframes"][0]
        other = Client()
        for url in (BASE + case["id"] + "/", case["original_url"], frame["image_url"], *case["report_urls"].values()):
            self.assertEqual(other.get(url).status_code, 404)
        self.assertEqual(other.patch(BASE + case["id"] + "/", "{}", content_type="application/json").status_code, 400)
        self.assertEqual(other.patch(BASE + case["id"] + "/", '{"verdict":"fake"}', content_type="application/json").status_code, 404)
        self.assertEqual(other.post(BASE + case["id"] + "/keyframes/", {"file": picture()}).status_code, 404)
        self.assertEqual(other.delete(BASE + case["id"] + "/keyframes/" + frame["id"] + "/").status_code, 404)
        self.assertEqual(other.get(BASE).json()["cases"], [])

    def test_csrf_is_required_for_create_patch_and_delete(self):
        strict = Client(enforce_csrf_checks=True)
        strict.get(BASE)
        self.assertEqual(strict.post(BASE, {"file": picture()}).status_code, 403)
        token = strict.cookies["csrftoken"].value
        created = strict.post(BASE, {"file": picture()}, HTTP_X_CSRFTOKEN=token)
        self.assertEqual(created.status_code, 201)
        case = created.json()["case"]
        self.assertEqual(strict.patch(BASE + case["id"] + "/", '{"verdict":"x"}', content_type="application/json").status_code, 403)

    def test_keyframe_bitmap_and_lines_are_stored_separately(self):
        case = self.create(event="card")
        image = picture()
        original = image.read()
        image.seek(0)
        response = self.add_frame(case, file=image, notes="裁判举牌，接收球员待核实")
        self.assertEqual(response.status_code, 201, response.content)
        frame = response.json()["case"]["keyframes"][0]
        self.assertEqual(frame["sequence"], 1)
        self.assertEqual(frame["source"], "browser_snapshot")
        self.assertEqual(frame["time_source"], "browser_display_time")
        self.assertFalse(frame["calibrated"])
        downloaded = self.client.get(frame["image_url"])
        self.assertEqual(b"".join(downloaded.streaming_content), original)
        downloaded.close()
        self.assertEqual(frame["lines"][0], {"x1": .2, "y1": .4, "x2": .8, "y2": .4})
        self.assertEqual((frame["width"], frame["height"]), (120, 80))

    def test_invalid_coordinates_nan_and_non_image_snapshots_are_rejected(self):
        case = self.create()
        for lines in ([{"x1": -1, "y1": .4, "x2": .8, "y2": .4}],
                      [{"x1": float("nan"), "y1": .4, "x2": .8, "y2": .4}],
                      [{"x1": True, "y1": .4, "x2": .8, "y2": .4}],
                      [{"x1": .2, "y1": .4, "x2": .2, "y2": .4}],
                      [{"x1": .2, "y1": .4, "x2": .8, "y2": .4, "color": "url(evil)"}]):
            self.assertEqual(self.add_frame(case, lines=json.dumps(lines)).status_code, 400)
        self.assertEqual(self.add_frame(case, file=SimpleUploadedFile("fake.png", b"not a picture")).status_code, 400)
        self.assertEqual(VarReviewCase.objects.get(pk=case["id"]).keyframes, [])

    def test_snapshot_times_are_bounded_and_browser_duration_cannot_hide_existing_evidence(self):
        case = self.create(video=True)
        self.assertEqual(self.patch_case(case, {"duration_seconds": 5}).status_code, 200)
        for value in (-1, float("inf"), True, 5.01, 99999):
            self.assertEqual(self.add_frame(case, media_time=value).status_code, 400)
        saved = self.add_frame(case, media_time=4.25)
        self.assertEqual(saved.status_code, 201)
        self.assertEqual(self.patch_case(case, {"duration_seconds": 4}).status_code, 400)
        self.assertEqual(self.patch_case(case, {"duration_seconds": float("nan")}).status_code, 400)
        image_case = self.create()
        self.assertEqual(self.add_frame(image_case, media_time=.1).status_code, 400)

    def test_reviewed_status_requires_evidence_and_verdict_and_is_always_manual(self):
        case = self.create(event="foul")
        self.assertEqual(self.patch_case(case, {"status": "reviewed", "verdict": "人工意见"}).status_code, 400)
        self.assertEqual(self.add_frame(case).status_code, 201)
        self.assertEqual(self.patch_case(case, {"status": "reviewed"}).status_code, 400)
        response = self.patch_case(case, {"status": "reviewed", "observations": "观察到接触", "verdict": "需结合其他角度确认"})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["case"]["automated_decision"])
        self.assertEqual(self.patch_case(case, {"automated_decision": True}).status_code, 400)
        self.assertEqual(self.patch_case(case, {"event_type": {}}).status_code, 400)
        self.assertEqual(self.patch_case(case, {"status": []}).status_code, 400)

    def test_delete_snapshot_removes_file_and_reopens_evidence_state(self):
        case = self.create()
        case = self.add_frame(case).json()["case"]
        self.patch_case(case, {"status": "reviewed", "verdict": "人工意见"})
        frame = case["keyframes"][0]
        record = VarReviewCase.objects.get(pk=case["id"])
        file = var_services.artifact_path(record, "frame-" + frame["id"])
        response = self.client.delete(BASE + case["id"] + "/keyframes/" + frame["id"] + "/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["case"]["status"], "needs_evidence")
        self.assertFalse(file.exists())
        self.assertEqual(self.client.get(frame["image_url"]).status_code, 404)

    def test_reports_remain_explicitly_manual_and_escape_markup_in_markdown(self):
        case = self.create(event="goal")
        self.add_frame(case, notes="<script>test</script>")
        self.patch_case(case, {"observations": "<img src=x>", "verdict": "不能凭球心判断整体过线"})
        report = self.client.get(case["report_urls"]["json"])
        self.assertEqual(report.status_code, 200)
        self.assertEqual(report.json()["source"], "manual_review")
        self.assertFalse(report.json()["automated_decision"])
        self.assertFalse(report.json()["calibration"])
        self.assertNotIn(str(self.root), report.content.decode())
        markdown = self.client.get(case["report_urls"]["markdown"]).content.decode()
        self.assertIn("自动判罚：否", markdown)
        self.assertIn("&lt;img src=x&gt;", markdown)
        self.assertNotIn("<script>", markdown)

    def test_busy_snapshot_file_does_not_silently_drop_its_evidence_record(self):
        case = self.add_frame(self.create()).json()["case"]
        frame = case["keyframes"][0]
        with patch.object(Path, "unlink", side_effect=PermissionError("reader holds snapshot")):
            response = self.client.delete(BASE + case["id"] + "/keyframes/" + frame["id"] + "/")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(len(VarReviewCase.objects.get(pk=case["id"]).keyframes), 1)
        self.assertIsNotNone(var_services.artifact_path(VarReviewCase.objects.get(pk=case["id"]), "frame-" + frame["id"]))

    def test_artifact_allowlist_blocks_filenames_and_parent_paths(self):
        case = self.create()
        for key in ("worker.log", "settings.py", "source.png", "frame-invalid"):
            self.assertEqual(self.client.get(BASE + case["id"] + "/artifact/" + key + "/").status_code, 404)
        record = VarReviewCase.objects.get(pk=case["id"])
        record.source_filename = "../../private.txt"
        record.save()
        self.assertEqual(self.client.get(case["original_url"]).status_code, 404)

    def test_original_video_supports_single_range_without_loading_whole_file(self):
        case = self.create(video=True)
        response = self.client.get(case["original_url"], HTTP_RANGE="bytes=10-19")
        self.assertEqual(response.status_code, 206)
        self.assertTrue(response.streaming)
        self.assertEqual(response["Content-Range"], "bytes 10-19/131072")
        self.assertEqual(response["Content-Length"], "10")
        self.assertEqual(response["Accept-Ranges"], "bytes")
        self.assertEqual(b"".join(response.streaming_content), bytes(range(10, 20)))
        response.close()
        whole = self.client.get(case["original_url"], HTTP_RANGE="bytes=0-131071")
        chunks = list(whole.streaming_content)
        self.assertEqual([len(chunk) for chunk in chunks], [65536, 65536])
        whole.close()

    def test_range_suffix_open_end_head_and_invalid_requests(self):
        case = self.create(video=True)
        for header, expected in (("bytes=-4", bytes(range(252, 256))), ("bytes=131070-", bytes((254, 255))),
                                 ("bytes=131070-999999", bytes((254, 255)))):
            response = self.client.get(case["original_url"], HTTP_RANGE=header)
            self.assertEqual(response.status_code, 206)
            self.assertEqual(b"".join(response.streaming_content), expected)
            response.close()
        head = self.client.head(case["original_url"], HTTP_RANGE="bytes=0-9")
        self.assertEqual(head.status_code, 206)
        self.assertEqual(head["Content-Length"], "10")
        self.assertEqual(head.content, b"")
        for header in ("bytes=999999-", "bytes=20-10", "bytes=-0", "bytes=0-2,4-6", "bytes=-", "invalid"):
            response = self.client.get(case["original_url"], HTTP_RANGE=header)
            self.assertEqual(response.status_code, 416)
            self.assertEqual(response["Content-Range"], "bytes */131072")

    def test_empty_unknown_or_model_upload_cannot_create_case(self):
        self.assertEqual(self.client.post(BASE, {"file": SimpleUploadedFile("weights.pt", b"test")}).status_code, 400)
        self.assertEqual(self.client.post(BASE, {"file": SimpleUploadedFile("bad.png", b"test")}).status_code, 400)
        self.assertEqual(self.client.post(BASE, "{}", content_type="application/json").status_code, 400)
        self.assertEqual(self.client.post(BASE, {"file": picture(), "event_type": "unsupported"}).status_code, 400)
        self.assertEqual(VarReviewCase.objects.count(), 0)
