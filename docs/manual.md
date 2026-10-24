# ODPlatform 使用手册

面向使用者的操作手册。环境搭建与设计理由见 [README](../README.md) 与
[ADR](architecture/)；这里只讲"怎么做、产物在哪、出问题怎么办"。

---

## 一、上手三步

### 1.1 准备环境

```powershell
# 依赖(可编辑安装 platform 端 + 开发工具)
pip install -e apps/platform[dev]

# 有 NVIDIA 显卡: 必须换成 CUDA 版 torch, 否则 GPU 静默失效
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 确认
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

### 1.2 建目录、看环境

```powershell
odp-init
```

输出里会打印系统 / CPU / 内存 / **GPU（CUDA可用、GPU数量、显存）**。
`CUDA可用: False` 说明 torch 是 `+cpu` 构建或驱动不匹配，先解决再谈性能。

### 1.3 放数据

把原始数据放到 `data/raw/<数据集名>/`，标注格式决定用哪个转换器：

| 格式 | 目录约定（示例） |
|---|---|
| Pascal VOC | `JPEGImages/*.jpg` + `Annotations/*.xml` |
| COCO | `images/*` + `annotations/*.json` |
| YOLO | `images/*` + `labels/*.txt`（+ 类别名要手动给） |

---

## 二、命令详解

### 2.1 `odp-transform` —— 原始数据 → 可训练成品

```powershell
odp-transform --dataset demo --format pascal_voc
odp-transform --dataset demoyolo --format yolo --classes head 反光衣
odp-transform --dataset demo --format pascal_voc --train-rate 0.7 --val-rate 0.15 --seed 42
```

- **`--classes` 对 yolo 格式必填**：yolo 的 txt 里只有类别号，没有名字，不给就报错退出。
  传的顺序即 `class_id` 的顺序。
- 产物：
  - `data/processed/<name>/`：物化后的 `images/{train,val,test}` 与 `labels/...`
  - `configs/datasets/<name>.yaml`：训练入口 + `odp_meta` 合格证（指纹/manifest/划分）
  - `runs/data_pipeline/<时间戳>/manifest.json`：样本级清单
- **会对覆盖率设闸门**：标注数/图像数低于阈值直接拒跑（疑似导出中断或路径配错）。

### 2.2 `odp-validate` —— 质检（训练前的验收）

```powershell
odp-validate configs/datasets/demo.yaml
odp-validate configs/datasets/demo.yaml --no-report
```

检查项包括：图像与标签配对、标签格式、框是否越界/退化、类别是否存在与失衡、
split 是否泄漏（同一图跨 split）、与 manifest 的血缘是否被篡改等。

- 退出码非 0 = 有 ERROR 级问题，**训练前应当先清零**（`odp-train` 默认也会拦）。
- 报告写在 `runs/data_validation/<run_id>/`（`--no-report` 可关）。
- 通过后会盖 **validate 章**，训练时据此对账数据身份。

### 2.3 `odp-train` —— 训练

```powershell
odp-train --data demo --model yolov8n --epochs 50 --batch 16 --imgsz 640
odp-train --data demo --model yolov8n --device 0 --experiment-name baseline_v1
```

- 开训前做**数据身份核验**：数据集 yaml、manifest、transform 章、validate 章四方对账，
  不一致就拒绝开训（`--skip-validate` 可强跳，慎用）。
- 现场目录：`runs/<task>/<run_id>/`，含 `odp_audit.json`、
  `dataset_resolved.yaml`（给 ultralytics 用的绝对路径副本）、权重与曲线。
- 结束后 `best.pt` 归档到 `models/trained/<stem>.pt`（`--no-archive` 可关）。
- 8 GB 显存的经验值：yolov8n/s + `batch 16` + `imgsz 640` 可行；换 m/l 要降 batch 或 imgsz。

### 2.4 `odp-infer` —— 推理

```powershell
# 图片 / 视频 / 目录 / 摄像头 / 网络流
odp-infer --source test.jpg --model yolov8n.pt --save
odp-infer --source D:\videos\line.mp4 --model yolov8n.pt --save --name line_01
odp-infer --source 0 --model yolov8n.pt --show            # 摄像头弹窗实时看
odp-infer --source rtsp://user:pass@host/stream --model yolov8n.pt --no-show --save
```

- 输出目录 `runs/<task>_infer/<name>/`（`--name` 不传则叫 `predict`），
  视频源产出 `output.mp4`（可用 `infer_pipeline.yaml` 的 `output` 段改文件名与编码器）。
- 服务器/无人值守环境**必须** `--no-show`，否则会卡在弹窗。
- ⚠️ **想要画面里有检测框，请加 `--no-viz`**：美化绘制目前是桩（不出框、不报错），
  `--no-viz` 会走 ultralytics 原生绘制。详见
  [ADR-004](architecture/ADR-004-inference-output-and-viz.md) 与 README「已知问题」。
- `--model` 的解析：给**全路径最稳**；给名字时要求该文件已存在于
  `models/checkpoints/` 或 `models/pretrained/`。注意自训练权重在 `models/trained/`，
  **不在搜索路径里**，必须给全路径。

### 2.5 `odp-gen-config` / `odp-reset`

```powershell
odp-gen-config infer --overwrite        # 重新生成配置模板(默认会备份原文件)
odp-reset                               # 默认 dry-run, 只列出会删什么
odp-reset --yes                         # 真正删除
```

`odp-reset` **只清运行时产物**：`data/processed/`、`runs/`、`models/trained/`、`logging/`。
**绝不动**：`data/raw/`（你的原始数据）、`models/pretrained/`（下载的权重）、
代码/配置/文档/测试。它的审计日志在 `.odp-meta/logs/`（不会被自己清掉）。

---

## 三、产物与追溯

```
runs/
├── data_pipeline/<时间戳>/manifest.json      # D3: 谁、用什么配方切了哪些样本
├── data_validation/<run_id>/report.json|csv  # D4: 质检结论
├── <task>/<run_id>/                          # D5: 训练现场
│   ├── odp_audit.json                        # 配置快照 + 各字段来源 + 指标 + 数据身份
│   ├── dataset_resolved.yaml                 # 喂给 ultralytics 的绝对路径副本
│   └── weights/…
└── <task>_infer/<name>/                      # 推理现场
    ├── odp_audit.json
    └── output.mp4 | <图片名>.jpg
```

**指纹与章**贯穿全链路：

- `contract_fingerprint`：数据集划分的契约指纹（内容决定，同样输入必得同样指纹）；
- transform 章 / validate 章 / train 章（`.odp-meta/`）：记录"这批数据经过了哪几站"；
- `odp-train` 开训前的对账就是查这三者与 manifest 是否一致 —— 防止"训的不是你以为的数据"。

---

## 四、排障对照表

| 症状 | 原因 | 处理 |
|---|---|---|
| `CUDA可用: False` / 推理很慢 | torch 装成了 `+cpu` 构建 | 按 1.1 换 CUDA 构建（注意驱动版本上限） |
| `Download failure ... github.com` / 模型下载失败 | `--model` 给的名字在搜索目录里不存在，ultralytics 转去联网下载，而本机网络/证书不通 | 给全路径，或把权重放到 `models/pretrained/`；自训练权重在 `models/trained/`，需全路径 |
| 默认跑出来的视频**没有框** | 美化绘制是桩 | 加 `--no-viz`（见 ADR-004） |
| 日志说 `pipeline yaml 不存在` | 旧版本配置默认路径写错（已修） | 确认运行的是修复后的版本；自定义路径用 `--pipeline-yaml` |
| 结果视频特别大（几百 MB） | 编码器 `mp4v` 压缩率低 | 改 `infer_pipeline.yaml` 的 `output.fourcc`/`filename` |
| `odp-transform` 报"类别名需显式提供" | yolo 格式无类别信息 | 加 `--classes 名字1 名字2`（顺序 = class_id） |
| 训练被拦"数据身份核验未通过" | 没跑 `odp-validate`，或数据被改过 | 先 `odp-validate`；确实要跳过用 `--skip-validate` |
| 训练时 Windows 卡住/报多进程错 | DataLoader worker 在 Windows 上的已知问题 | 把 `train.yaml` 的 `workers` 设为 `0` 或 `4` |
| 显存不足（OOM） | batch/imgsz/模型过大 | 降 `--batch` 或 `--imgsz`，或换更小的模型 |
| `--device 0` 报错 | 本机无 CUDA 或卡号不对 | 用 `--device cpu` 或去掉该参数（自动选卡） |

---

## 五、日常检查清单

- [ ] `odp-init` 输出里 `CUDA可用: True`、显存正常
- [ ] 训练/推理前 `runs/` 所在盘有足够空间（视频产物动辄数百 MB）
- [ ] 训练前 `odp-validate` 是 PASS
- [ ] 训练后 `models/trained/` 有归档权重，`runs/<task>/<run_id>/odp_audit.json` 存在
- [ ] `odp-reset` 之前先 `odp-reset`（dry-run）确认清单，**确认 `data/raw/` 不在其中**