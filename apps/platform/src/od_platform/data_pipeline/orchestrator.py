"""数据流水线编排:检查原料 → 转换 → 扫描 → 报告 → 划分 → 契约 → 物化 → yaml。

编排者不实现任何一步的细节,只做接线与次序;每步的失败语义由子系统自己定。
产物分家:processed/(生产)+ runs/(审计 manifest)+ configs/(yaml 合格证)。
"""
from __future__ import annotations

import logging
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from od_platform.common import paths
from od_platform.common.constants import (
    COVERAGE_HARD_THRESHOLD, COVERAGE_SOFT_THRESHOLD, AnnotationFormat,
    DEFAULT_RANDOM_STATE, DEFAULT_SPLIT_STRATEGY, DEFAULT_TRAIN_RATE, DEFAULT_VAL_RATE,
    IMAGE_EXTENSIONS, Task,
)
from od_platform.common.lineage import TOOL_VERSION, compute_data_fingerprint, SplitManifest
from od_platform.common import stamp
from od_platform.common.progress import ProgressReporter
from od_platform.common.run_context import RunContext
from od_platform.data_pipeline.convert.registry import ConvertOptions
from od_platform.data_pipeline.convert.service import convert_data_to_yolo
from od_platform.data_pipeline.report import (
    analyze_balance, format_report, warn_if_rare,
)
from od_platform.data_pipeline.split.materializer import (
    SplitOutputDirs, SplitSourceDirs, materialize,
)
from od_platform.data_pipeline.split.split_service import split_dataset
from od_platform.data_pipeline.split.yaml_writer import write_dataset_yaml

logger = logging.getLogger(__name__)

_SPLITS = ("train", "val", "test")


class DatasetPipeline:
    """一条命令的完整实现:odp-transform 的全部逻辑。"""

    def __init__(
        self,
        dataset: str,
        annotation_format: str,
        *,
        task: str = Task.DETECT,
        train_rate: float = DEFAULT_TRAIN_RATE,
        val_rate: float = DEFAULT_VAL_RATE,
        classes: Optional[List[str]] = None,
        random_state: int = DEFAULT_RANDOM_STATE,
        split_strategy: str = DEFAULT_SPLIT_STRATEGY,
        run: Optional[RunContext] = None,
    ):
        self.dataset = dataset
        self.annotation_format = annotation_format
        self.task = task
        self.train_rate = train_rate
        self.val_rate = val_rate
        self.classes = classes
        self.random_state = random_state
        self.split_strategy = split_strategy
        self._run_ctx = run        # 存成私有名:实例属性不能遮蔽 run() 方法
        self.raw_root = paths.raw_dataset_root(dataset)
        self.raw_images = self.raw_root / "images"
        self.raw_annotations = self.raw_root / "annotations"

    def _annotation_dir(self) -> Path:
        """源标注目录:YOLO 格式的标注在 labels/,其余在 annotations/。"""
        if self.annotation_format == AnnotationFormat.YOLO:
            return self.raw_root / "labels"
        return self.raw_annotations

    # ---- 前置检查 ----

    def _check_raw(self) -> None:
        if not self.raw_root.exists():
            raise FileNotFoundError(f"数据集不存在: {self.raw_root}")
        if not self.raw_images.exists():
            raise FileNotFoundError(f"缺少图像目录: {self.raw_images}")
        ann_dir = self._annotation_dir()
        if not ann_dir.exists():
            raise FileNotFoundError(f"缺少标注目录: {ann_dir}")
        self._check_coverage()

    def _check_coverage(self) -> None:
        """覆盖率闸门:标注数/图像数 过低时直接拒跑。

        这里要先数一遍图像总数, 而总数正是"要数出来的东西"—— 所以走"总量未知"
        模式（每 UNKNOWN_TOTAL_STEP 项打一个点）。大数据集上是几秒到几十秒的
        静默等待, 接入进度反馈后用户能看到它在动。
        """
        progress = ProgressReporter("清点图像", 0, logger=logger, unit="张图")
        n_images = 0
        for path in self.raw_images.rglob("*"):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                n_images += 1
                progress.update(n_images)
        progress.finish(done=n_images)
        if n_images == 0:
            raise FileNotFoundError(f"图像目录为空: {self.raw_images}")
        if self.annotation_format == AnnotationFormat.COCO:
            logger.debug("COCO 为单 json 格式,跳过 stem 级覆盖率检查")
            return
        ann_dir = self._annotation_dir()
        n_annos = len(list(ann_dir.glob("*.*"))) if ann_dir.exists() else 0
        coverage = n_annos / n_images
        if coverage < COVERAGE_HARD_THRESHOLD:
            raise ValueError(
                f"图像-标注覆盖率 {coverage:.1%} 低于硬阈值 "
                f"{COVERAGE_HARD_THRESHOLD:.0%}(图 {n_images} / 标注 {n_annos})。"
                f"疑似标注导出中断或路径配错 —— 请修源头,不要带病训练。")
        if coverage < COVERAGE_SOFT_THRESHOLD:
            logger.warning(
                "图像-标注覆盖率 %.1f%% 低于软阈值 %.0f%%,继续但请确认是有意为之",
                coverage * 100, COVERAGE_SOFT_THRESHOLD * 100)

    # ---- 扫描:转换产物 → 内存结构 ----

    def _scan(self, staging: Path, classes: List[str]):
        image_index: Dict[str, Path] = {}
        candidates = sorted(self.raw_images.rglob("*"))
        progress_images = ProgressReporter("扫描图像目录", len(candidates),
                                           logger=logger, unit="个条目")
        for position, path in enumerate(candidates, 1):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                image_index[path.stem] = path
            progress_images.update(position)
        progress_images.finish(done=len(candidates))

        labels_per_image: Dict[str, List[str]] = defaultdict(list)
        label_bytes: Dict[str, bytes] = {}
        label_files = sorted(staging.glob("*.txt"))
        progress_labels = ProgressReporter("读取标签", len(label_files),
                                           logger=logger, unit="个标签文件")
        for position, txt in enumerate(label_files, 1):
            # 只读一次: 此前 read_bytes() 与 read_text() 把每个文件读了两遍,
            # 实测 1232 个标签因此花了 13.5s, 而复制 1232 对(图+标签)只要 2.9s。
            raw_bytes = txt.read_bytes()
            label_bytes[txt.stem] = raw_bytes
            for line in raw_bytes.decode("utf-8").splitlines():
                line = line.strip()
                if line:
                    idx = int(line.split()[0])
                    labels_per_image[txt.stem].append(classes[idx])
            progress_labels.update(position)
        progress_labels.finish(done=len(label_files))
        return image_index, labels_per_image, label_bytes

    # ---- 主流程 ----

    def run(self) -> Dict[str, object]:
        if self._run_ctx is not None:
            return self._execute(self._run_ctx)
        with RunContext("data_pipeline") as run:
            return self._execute(run)

    def _execute(self, run: RunContext) -> Dict[str, object]:
        self._check_raw()
        out = SplitOutputDirs.for_dataset(self.dataset)

        with tempfile.TemporaryDirectory(prefix="odp_pipe_") as staging_dir:
            staging = Path(staging_dir)
            ann_in = self._annotation_dir()
            options = ConvertOptions(task=self.task, classes=self.classes)
            classes = convert_data_to_yolo(ann_in, staging, self.annotation_format, options)

            image_index, labels_per_image, label_bytes = self._scan(staging, classes)

            report = analyze_balance(
                dict(labels_per_image), classes,
                min_split_rate=min(self.train_rate, self.val_rate,
                                   1 - self.train_rate - self.val_rate))
            for line in format_report(report):
                logger.info(line)
            warn_if_rare(report)

            stems = sorted(set(image_index) & set(label_bytes))
            splits = split_dataset(
                stems, self.train_rate, self.val_rate, self.random_state,
                strategy=self.split_strategy,
                labels_per_image={s: labels_per_image.get(s, []) for s in stems})

            manifest = SplitManifest.build(
                dataset=self.dataset, strategy=self.split_strategy,
                seed=self.random_state,
                ratios=(self.train_rate, self.val_rate,
                        round(1 - self.train_rate - self.val_rate, 6)),
                names=classes, splits=splits, label_bytes=label_bytes,
                created_at=run.created_at)
            manifest_path = run.artifact_path("manifest.json")
            manifest.write(manifest_path)
            manifest_ref = manifest_path.relative_to(paths.ROOT_DIR).as_posix()

            source = SplitSourceDirs(images=self.raw_images, labels=staging)
            counts = materialize(manifest, source, out)

            yaml_path = paths.dataset_yaml_path(self.dataset)
            write_dataset_yaml(
                yaml_path, dataset_root=out.root, classes=classes, manifest=manifest,
                dataset_name=self.dataset, source_format=self.annotation_format,
                task=self.task, manifest_ref=manifest_ref)

        logger.info("数据集 %r 完成: %s", self.dataset, counts)

        # D6 增量: 盖 transform 章(我这一站划过这批数据)
        data_fp = compute_data_fingerprint(manifest.samples)
        stamp.write_stamp(
            stage_file=stamp.STAMP_TRANSFORM, data_fingerprint=data_fp,
            run_id=run.run_id, tool_version=TOOL_VERSION,
            payload={"dataset": self.dataset, "strategy": self.split_strategy,
                     "contract_fingerprint": manifest.contract_fingerprint,
                     "counts": counts},
        )

        return {
            "counts": counts,
            "yaml": str(yaml_path),
            "manifest_path": str(manifest_path),
            "run_dir": str(run.run_dir),
            "contract_fingerprint": manifest.contract_fingerprint,
        }
