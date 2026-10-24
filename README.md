# ODPlatform

通用目标检测开发平台 —— 从原始标注到可部署权重的完整链路：**转换 → 质检 → 训练 → 推理**，
每一步都留指纹与审计快照。

## 当前能力

| 命令 | 阶段 | 作用 |
|---|---|---|
| `odp-init` | 初始化 | 按 `common/paths.py` 的清单建目录、打印环境快照（含 GPU 信息） |
| `odp-transform` | D3 数据流水线 | 原始标注（Pascal VOC / COCO / YOLO）→ 划分 → 物化 → 生成数据集 yaml |
| `odp-validate` | D4 质检 | 对数据集做多维检查（配对、格式、越界、泄漏、类别、血缘…），产出报告 |
| `odp-train` | D5 训练 | 训练前验章（数据身份），训练，归档 best 权重，写审计快照 |
| `odp-infer` | 推理 | 图片/视频/目录/摄像头/RTSP 推理，可保存结果、可视化、HUD |
| `odp-gen-config` | 配置 | 生成/覆盖运行配置模板（自动备份原文件） |
| `odp-reset` | 维护 | 清理运行时产物目录（默认 dry-run，需 `--yes` 才真删） |

## 环境要求

- **Python 3.12**（`requires-python >= 3.10`，实际开发在 3.12 上）
- 依赖：torch、ultralytics、pydantic v2、pyyaml、psutil、colorlog
- 可选 GPU：NVIDIA 显卡 + 匹配的驱动

### ⭐ GPU 用户必读：必须装 CUDA 版 torch

`pyproject.toml` 只写了 `torch>=2.0.0` —— **它不区分 CPU/CUDA 构建**。
如果直接用 `pip install torch` 装了 `+cpu` 构建，GPU 会**静默失效**（不报错，只是慢十倍）。

```powershell
# 先卸掉可能的 +cpu 构建，再按驱动版本装 CUDA 构建
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

- `cu126` 对应驱动 ≥ 560；驱动更老请选 `cu121`/`cu118`。**选高于驱动支持的版本会失败。**
- 装完用 `odp-init`（会打印 `CUDA可用/GPU数量/显存`）或下面一行确认：
  ```powershell
  python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
  ```

## 快速开始

```powershell
# 0) 建目录 + 看环境（首次）
& '.\.venv\Scripts\odp-init.exe'

# 1) 原始数据放 data/raw/<数据集名>/ 之后，转换成可训练成品
#    yolo 格式必须显式给类别名（yaml 里没有类别信息）
& '.\.venv\Scripts\odp-transform.exe' --dataset demo --format pascal_voc
& '.\.venv\Scripts\odp-transform.exe' --dataset demoyolo --format yolo --classes head 反光衣

# 2) 质检（会盖 validate 章，训练前必须通过）
& '.\.venv\Scripts\odp-validate.exe' configs/datasets/demo.yaml

# 3) 训练（--data 可给 'demo' 这种名字，会解析成 configs/datasets/demo.yaml）
& '.\.venv\Scripts\odp-train.exe' --data demo --model yolov8n --epochs 50 --batch 16

# 4) 推理
& '.\.venv\Scripts\odp-infer.exe' --source path\to\video.mp4 --model yolov8n.pt --save
```

## 命令参考（要点）

### odp-transform

| 参数 | 说明 |
|---|---|
| `--dataset` (必填) | 数据集名，读取 `data/raw/<name>/` |
| `--format` (必填) | `pascal_voc` / `coco` / `yolo` |
| `--task` | `detect`（默认）/ `segment` |
| `--train-rate` / `--val-rate` | 划分比例，默认 0.8 / 0.1（其余给 test） |
| `--seed` | 随机种子，默认 42（保证可复现） |
| `--split-strategy` | `random`（默认）/ `stratified_multilabel` |
| `--classes` | 类别白名单，顺序即 class_id；**yolo 格式必填** |

### odp-validate

| 参数 | 说明 |
|---|---|
| `yaml_path` (位置参数) | 数据集 yaml，如 `configs/datasets/demo.yaml` |
| `--task` | 覆盖 yaml 里的任务类型 |
| `--run-id` | 指定运行 ID |
| `--no-report` | 不写 json/csv 报告 |

### odp-train

| 参数 | 说明 |
|---|---|
| `--config` | 运行配置引用，默认 `train` |
| `--data` | 数据集引用：`demo` 或路径 |
| `--model` | 模型引用：`yolov8n` / `yolov8n.pt` / 路径 |
| `--epochs` / `--batch` / `--imgsz` | 训练超参 |
| `--device` | `0` / `0,1` / `cpu`；不传=自动选卡 |
| `--experiment-name` | 实验名（决定 `runs/` 下的子目录） |
| `--skip-validate` | 跳过数据身份核验（**知道有风险时才用**） |
| `--no-archive` | 不归档 best.pt |

### odp-infer

| 参数 | 说明 |
|---|---|
| `--source` | 图片 / 视频 / 目录 / 摄像头号（`0`）/ RTSP / HTTP 流 |
| `--model` | 模型文件；**给不带后缀的名字时必须在搜索目录里同名存在**（见「已知问题」） |
| `--yaml` / `--pipeline-yaml` | 运行配置 / 帧源+美化配置的路径 |
| `--name` | 输出子目录名（`runs/<task>_infer/<name>/`） |
| `--conf` / `--iou` / `--imgsz` / `--max-det` / `--classes` | 推理超参 |
| `--device` / `--batch` | 设备与批大小 |
| `--show` / `--no-show` | 是否弹窗（服务器环境必须关） |
| `--save` / `--no-save` | 是否落盘 |
| `--no-viz` | 关闭美化绘制，改用 ultralytics 原生 `plot()` |
| `--no-hud` | 关掉 HUD 信息面板 |
| `--warmup` | 预热帧数 |

## 目录约定

```
ODPlatform/
├── apps/platform/            # 平台端（唯一的代码端）
│   ├── src/od_platform/      # 包源码（common/data_pipeline/data_validation/training/inference/...）
│   ├── configs/
│   │   ├── datasets/         # 数据集 yaml（odp-transform 产出，进 git）
│   │   └── runtime/          # 运行配置（infer/train/val/infer_pipeline，进 git）
│   ├── tests/                # pytest 测试
│   └── logging/              # 业务工具日志（不进 git，可 reset）
├── data/
│   ├── raw/<name>/           # 原始数据：只读，reset 绝不碰
│   └── processed/<name>/     # 派生数据集：可再生产
├── models/
│   ├── pretrained/           # 下载的预训练权重（输入）
│   ├── checkpoints/          # 推理查找目录
│   └── trained/              # 训练产出（可再生产）
├── runs/                     # 每次运行的现场目录（manifest/audit/输出，不进 git）
├── docs/                     # ADR 与手册
└── .odp-meta/logs/           # 元工具（reset 等）的审计日志
```

路径常量集中在 `common/paths.py`，靠 marker 文件 `.odp-workspace` 定位仓库根 ——
**不写死盘符**，换机器/换目录 clone 下来即可用。

## 配置系统

三层，优先级 **CLI > YAML > DEFAULT**（`odp-infer` 启动时会打印每个字段的来源）：

| 层 | 位置 | 管什么 |
|---|---|---|
| 运行配置 | `configs/runtime/infer.yaml`、`train.yaml`、`val.yaml` | 模型、批量、置信度、设备… |
| 帧源+美化配置 | `configs/runtime/infer_pipeline.yaml` | 相机参数、标签/颜色映射、输出文件名与编码器 |
| 命令行 | `odp-*` 的参数 | 临时覆盖，不落盘 |

约定：**配置里不写机器相关取值**。例如 `device` 保持 `null`（自动选卡：有 CUDA 用 GPU，
否则 CPU），要临时固定就用 `--device 0`；把 `0` 写进 yaml 会让配置在别的机器上失效。

数据集 yaml 的 `path` 一律写**相对仓库根**的路径（如 `data/processed/demo`），
原因与解析规则见 [ADR-002](docs/architecture/ADR-002-dataset-yaml-portable-path.md)。

## 数据身份与审计

- 每次数据转换产出 **契约指纹**（`contract_fingerprint`）与 `manifest.json`，
  写进数据集 yaml 的 `odp_meta` 段（"产品合格证"）。
- `odp-transform` 盖 transform 章、`odp-validate` 盖 validate 章；
  `odp-train` 训练前对账三方（yaml / manifest / 章）一致才开训（可用 `--skip-validate` 强跳）。
- 每次训练/推理都在 `runs/...` 下留 `odp_audit.json`：配置快照 + 各字段来源 + 指标 + 数据身份。
- 训练时会把数据集 yaml 的 `path` 解析成绝对路径另存一份
  `dataset_resolved.yaml`（因为 ultralytics 对相对路径的解析规则与本仓库不同），
  源 yaml 保持可移植。

## 开发流程

```powershell
# 安装（可编辑安装 platform 端）
pip install -e apps/platform[dev]

# 代码风格 / 类型 / 测试（配置都在根 pyproject.toml）
ruff check .            # 风格 + import 排序
mypy apps/platform/src  # 类型
pytest -q               # 测试（testpaths = apps/platform/tests + tests）
```

提交信息用中文约定式前缀：`feat:` / `fix:` / `refactor:` / `chore:` / `docs:`。

### 编码约定（重要）

- **不硬编码**：路径走 `common/paths.py`；名字→路径走 `common/refs.py`；
  阈值/默认值要么进 `common/constants.py`（跨子系统）要么进子系统自己的 `defaults.py`，
  **不许散落成魔法数字**。
- **进 git 的产物必须可移植**：不允许出现盘符路径、机器名、绝对路径。
- **改行为就改文档**：ADR / README / 手册与代码同步更新。

## CI

工作流：`.github/workflows/ci.yml`，在 push 到 `main` 与 PR 时触发。

| 步骤 | 是否阻塞 | 说明 |
|---|---|---|
| `pytest -q` | ✅ 阻塞 | 38 项测试（数据流水线/质检/数据身份/训练/数据集路径） |
| 硬编码守卫 | ✅ 阻塞 | 扫描盘符路径与绝对路径，发现即失败 |
| `ruff check .` | ⚠️ 仅报告 | **存量债务 707 条**，以 UP006（262）/UP045（142）/UP035（109）/I001（50）为主，其中 572 条可 `--fix` 自动修复 |
| `mypy apps/platform/src` | ⚠️ 仅报告 | **存量 86 条**，多为"可能为空"的收窄问题 |

债务清完后请删掉对应步骤的 `continue-on-error`，改成硬门禁。建议分两次做：
先 `ruff check --fix .`（机械修复），再人工处理剩余部分。

## 已知问题（待办）

| 问题 | 影响 | 位置 |
|---|---|---|
| 美化绘制 `BeautifyVisualizer` 是桩：`from_yolo_results()` 丢掉 boxes/conf，`draw()` 直接返回原图 | 默认（美化开启）跑出来的结果**没有框**，且不报错；需加 `--no-viz` 走原生绘制 | `visualization/__init__.py` |
| `resolve_model_path` 不补后缀 | `--model yolov8n` 找不到本地的 `yolov8n.pt`，会尝试联网下载；训练侧用的是 `refs.resolve_ref`（会补 `.pt`），两边行为不一致 | `common/model_path.py` |
| 推理的模型搜索目录不含 `models/trained` | 用自训练权重必须传全路径，否则同样触发联网下载 | `inference/service.py` |
| 推理配置没有 `half` 字段 | `infer.yaml` 里写 `half: true` 会被静默忽略，吃不到 FP16 加速 | `runtime_config/infer.py` |
| `YOLO(...)` 构造后 `model.device` 显示 `cpu` | 只是 ultralytics 的占位值；真正选卡在 `predict` 时发生（`predictor.device` 才是真相），容易被误判成"没走 GPU" | 使用提示 |

## 文档

- 架构决策记录：[`docs/architecture/`](docs/architecture/)（ADR 索引见其 README）
- 使用手册：[`docs/manual.md`](docs/manual.md)
- 帧源设计讲义：[`frame_source_course.md`](frame_source_course.md)

## 相关仓库

- [defect_manage](https://github.com/chenkangkang999/defect_manage) —— 缺陷检测台账系统，
  消费本平台产出的权重与推理结果。