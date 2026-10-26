"""data_validation 回归测试:干净集 + 九类脏集,守住各 check 的关键语义。"""
from io import BytesIO
from pathlib import Path

from PIL import Image

from od_platform.data_validation.registry import CheckContext, CheckSeverity, get_check
from od_platform.data_validation.service import run_all_checks, validate_dataset
from od_platform.data_validation.snapshot import build_snapshot


def _ctx(yaml_path: Path) -> CheckContext:
    return CheckContext(yaml_path=yaml_path, snapshot=build_snapshot(yaml_path))


def _run(yaml_path: Path, name: str):
    return get_check(name).func(_ctx(yaml_path))


def _jpeg_bytes(size=(32, 32)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, "green").save(buffer, format="JPEG")
    return buffer.getvalue()


def _make_image_dataset(root: Path, *, truncate_one: bool = False,
                        n_good: int = 4) -> Path:
    """造一个最小可质检数据集: images/train + labels/train + data.yaml。"""
    images = root / "images" / "train"
    labels = root / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for index in range(n_good):
        (images / f"good{index}.jpg").write_bytes(_jpeg_bytes())
        (labels / f"good{index}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    if truncate_one:
        payload = _jpeg_bytes()
        (images / "broken.jpg").write_bytes(payload[: len(payload) // 3])
        (labels / "broken.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    yaml_path = root / "data.yaml"
    yaml_path.write_text("path: .\ntrain: images/train\nnc: 2\n"
                         "names: {0: player, 1: football}\n", encoding="utf-8")
    return yaml_path


def test_clean_dataset_all_pass(demo_yaml):
    """干净集:除 D3 故意埋的 c_050 越界框(bbox_within_image WARNING)外,全部 PASS/INFO。"""
    for r in run_all_checks(_ctx(demo_yaml)):
        if r.name == "bbox_within_image":
            assert r.severity == CheckSeverity.WARNING, r  # c_050 埋雷,留给 D4 抓
        else:
            assert r.severity in (CheckSeverity.PASS, CheckSeverity.INFO), \
                f"{r.name}: {r.severity} {r.summary}"


def test_pair_existence_flags_missing(demo_yaml_missing_labels):
    r = _run(demo_yaml_missing_labels, "pair_existence")
    assert r.severity == CheckSeverity.WARNING       # ≈16%:≥5% 且 <50%
    assert r.details["missing"] > 0


def test_fingerprint_match_flags_swapped_yaml(demo_yaml_wrong_fingerprint):
    r = _run(demo_yaml_wrong_fingerprint, "fingerprint_match")
    assert r.severity == CheckSeverity.ERROR         # yaml 指纹与 manifest 不一致


def test_manifest_lineage_flags_tampered_label(demo_yaml_tampered_label):
    r = _run(demo_yaml_tampered_label, "manifest_lineage")
    assert r.severity == CheckSeverity.ERROR
    assert r.details["n_changed"] >= 1


def test_exit_code_blocks_on_error(demo_yaml_tampered_label):
    """端到端:有 ERROR → overall=ERROR(CLI 退出码 2)。"""
    report = validate_dataset(demo_yaml_tampered_label, write_report=False)
    assert report.overall_severity == CheckSeverity.ERROR


def test_bbox_within_image_flags_overflow(demo_yaml_bbox_overflow):
    r = _run(demo_yaml_bbox_overflow, "bbox_within_image")
    assert r.severity in (CheckSeverity.WARNING, CheckSeverity.ERROR)
    assert r.details["n_out"] >= 1


def test_label_format_and_bbox_do_not_double_report(demo_yaml_bad_field_count):
    """结构错误行:label_format 报 ERROR;bbox 跳过它,不重复报(且不升级为 ERROR)。"""
    lf = _run(demo_yaml_bad_field_count, "label_format")
    assert lf.severity == CheckSeverity.ERROR
    bbox = _run(demo_yaml_bad_field_count, "bbox_within_image")
    assert bbox.severity != CheckSeverity.ERROR
    assert bbox.details["skipped_structure_errors"] >= 1


def test_split_uniqueness_hard_error_on_leak(demo_yaml_leaked_sample):
    r = _run(demo_yaml_leaked_sample, "split_uniqueness")
    assert r.severity == CheckSeverity.ERROR          # 泄漏没有分级,直接 ERROR


def test_class_presence_flags_missing_in_train(demo_yaml_class_missing_train):
    r = _run(demo_yaml_class_missing_train, "class_presence")
    assert r.severity == CheckSeverity.WARNING
    kinds = {f["kind"] for f in r.details["findings"]}
    assert "missing_in_train" in kinds


def test_box_geometry_one_kind_per_line(demo_yaml_degenerate_and_extreme):
    """if-chain 优先级:一行只报一个问题 —— 报了退化,就不再重复报极端宽高比。"""
    r = _run(demo_yaml_degenerate_and_extreme, "box_geometry")
    assert r.severity == CheckSeverity.WARNING
    assert r.details["issue_kinds"].get("extreme_aspect_ratio", 0) == 0
    assert sum(r.details["issue_kinds"].values()) >= 1


# ---------------------------------------------------------------- 图片完整性
# 这一项要抓的是"其它检查全都看不见"的失败:图在、大小非零、有标签, 但打不开。

def test_image_decodable_passes_on_intact_images(tmp_path):
    r = _run(_make_image_dataset(tmp_path), "image_decodable")
    assert r.severity == CheckSeverity.PASS
    assert r.details["n_bad"] == 0
    assert r.details["total_images"] == 4


def test_image_decodable_flags_truncated_image(tmp_path):
    """截断的 JPEG: 前面几项检查都认为它正常, 只有这一项能抓出来。"""
    yaml_path = _make_image_dataset(tmp_path, truncate_one=True, n_good=20)
    assert _run(yaml_path, "pair_existence").severity == CheckSeverity.PASS
    assert _run(yaml_path, "annotation_coverage").severity == CheckSeverity.PASS
    r = _run(yaml_path, "image_decodable")
    assert r.severity == CheckSeverity.WARNING        # 1/21 ≈ 4.8% < 10%
    assert r.details["n_bad"] == 1
    assert r.details["bad_preview"][0]["file"] == "broken.jpg"


def test_image_decodable_escalates_to_error_when_widespread(tmp_path):
    """大面积打不开 → ERROR（3/5 = 60% ≥ 10%）。"""
    yaml_path = _make_image_dataset(tmp_path, truncate_one=True, n_good=2)
    for name in ("good0", "good1"):
        payload = _jpeg_bytes()
        (tmp_path / "images" / "train" / f"{name}.jpg").write_bytes(payload[: len(payload) // 4])
    r = _run(yaml_path, "image_decodable")
    assert r.severity == CheckSeverity.ERROR
    assert r.details["n_bad"] == 3


def test_image_decodable_reports_clean_dataset_without_noise(demo_yaml):
    """干净集上它不该产生任何告警（否则每个数据集都会平白多一条 WARNING）。"""
    r = _run(demo_yaml, "image_decodable")
    assert r.severity in (CheckSeverity.PASS, CheckSeverity.INFO)
    assert r.details.get("n_bad", 0) == 0
