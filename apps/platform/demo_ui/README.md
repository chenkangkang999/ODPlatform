# 算法能力演示站点（demo_ui）

答辩演示用的**独立 Django 站点**，把 ODPlatform 算法侧的真实能力与参数摊开给人看。
对应任务清单 **2.2「添加和修改 visualization（修改 D7 代码），演示 UI」**。

---

## 为什么是一个独立站点

| 选择 | 原因 |
|---|---|
| 独立在本目录，不复用 `apps/football-web/` | 那个门户是队友的应用（`bridge_worker.py` / `annotations.py` / `views.py`）。本演示**不引用、不修改**它的任何文件，避免互相牵动。 |
| 用 Django，不用 Streamlit / Gradio | 仓库里已经有 Django 6.1.2；换 Django 是任务清单里写明的加分项，而且**零新增依赖**（不用装 streamlit，也不用装 gradio）。 |
| 不接数据库 | 演示只做"上传一张图 → 立刻算 → 立刻显示"，图片在内存里处理、以 base64 内联回页面，没有需要持久化的东西。所以不装 auth/admin/session，没有迁移、没有账号，起来即成。 |

---

## 启动

```powershell
cd D:\ODPlatform
.\.venv\Scripts\python.exe apps\platform\demo_ui\manage.py runserver
```

`manage.py` 在 `runserver` 不带地址时会**自动补上 `127.0.0.1:8100`**，所以直接跑上面这条即可。

> **端口说明**：`8000` 是 Django 默认（容易被顺手占用），`8001` 是队友的足球门户。
> 演示站点固定用 **8100**（常量在 `showcase/demo_config.py` 的 `DEMO_PORT`）。
> 想换端口就显式给地址：`manage.py runserver 127.0.0.1:8200`。

然后打开 <http://127.0.0.1:8100/>。

其他命令：

```powershell
# 自检（不启服务）
.\.venv\Scripts\python.exe apps\platform\demo_ui\manage.py check

# 只跑本演示的测试
.\.venv\Scripts\python.exe -m pytest apps\platform\tests\test_demo_ui.py -q
```

---

## 两个页面

### `/` 能力总览

四张卡片，**每个数值都从配置层与训练产物实时读出**，页面里没有写死的阈值：

| 卡片 | 取值来源 |
|---|---|
| ① YOLO 检测 | `od_platform.detection.settings.DetectionSettings.to_audit()`（= `configs/runtime/football.yaml` 的 `detection` 节）+ `models/trained/<run>/results.csv`、`args.yaml` |
| ② 多目标跟踪 BoT-SORT | `DetectionSettings.resolved_tracker()` / `configs/runtime/botsort.yaml` / football.yaml 的 `tracking` 节 |
| ③ 场地标定 + Homography | `od_platform.geometric.settings.GeometrySettings.to_audit()` + `od_platform.geometric.calibration`（磁盘实况） |
| ④ 动作识别 | `od_platform.action_recognition.settings.ActionRecognitionSettings.to_audit()` + `SUPPORTED_FAMILIES` / `PLANNED_FAMILIES` |

**这一页刻意只讲能做到的事**，三处如实标注：

- 检测权重**只训了 3 轮**，标为**冒烟级（未收敛）**；并说明 recall 偏低意味着漏检。
- 场地标定：磁盘上没有任何标定文件 → 显示**未标定**，并明确写出**接触/出界/角球/点球/任意球五类几何事件目前一条都不会产出**（这是设计行为：没有标定就没有"米"）。
- 动作识别：实际是 **VideoMAE V1 base + Kinetics-400（400 类通用动作）**，不是 "VideoMAE V2"，也没有用足球数据微调过；**SlowFast 已按需求清单删除**（它在代码里从未实现过，且与 VideoMAE 职责重叠）；预留位现在只剩 `custom`，传进去只降级为空结果。

### `/infer/` 推理演示

- GET：上传表单（仅图片，上限 8 MB，见 `demo_config.UPLOAD_MAX_BYTES`）。
- POST：解码 → 用 `ultralytics.YOLO` 跑 `football_yolo/best.pt` → 用 **`od_platform.visualization.BeautifyVisualizer`** 画框 → 页面同时显示**原图**与**带框图**（base64 内联），外加检出总数与每类计数表。
- 任何失败（没选文件 / 格式不对 / 图片解不开 / 超过大小）都渲染回本页并给中文原因，**不出现 500 白页**；「零检测」也正常显示，不当成错误。
- 设备默认 **CPU**（本机显存只有 8 GB，不去抢队友门户和训练的卡）。可用环境变量覆盖：`DEMO_UI_DEVICE`。

---

## 目录结构

```
apps/platform/demo_ui/
├── manage.py                      # 入口；runserver 自动补 8100
├── README.md
├── demo_site/                     # Django 项目
│   ├── settings.py                # 不接数据库、零新增依赖（含取舍说明）
│   ├── urls.py
│   └── wsgi.py
└── showcase/                      # 演示应用
    ├── demo_config.py             # 本演示自己的具名常量（端口/上限/展示映射）
    ├── capabilities.py            # 如实汇总能力与参数（不 import Django，可单测）
    ├── engine.py                  # 解码 → 检测 → 绘制（检测器可注入，便于测试）
    ├── views.py                   # 只做 HTTP
    ├── urls.py
    └── templates/showcase/        # base / overview / infer
```

---

## 测试

`apps/platform/tests/test_demo_ui.py`（21 项）覆盖：总览页 200 与四处诚实文案、
参数确实来自配置层、推理页 GET、注入假检测器后的 POST 结果区与两张内联图、
计数表用中文显示名、零检测不算错误、四类上传校验、标定状态判定、
训练记录按列名解析、真机上 3 轮权重被判为冒烟级、`manage.py` 的默认端口。

测试**不加载真实模型**：检测那一层通过 `engine.build_ultralytics_detector` 的注入点换成
假检测器（与队友的 `worker_checks/check_bridge.py` 用 `detector=` 注入是同一手法）。
绘制走的仍是真实路径，被测的是"页面 + 绘制 + 统计"。

---

## 已知限制（如实列出）

1. **演示用的是 3 轮冒烟级权重**，漏检很多，尤其小目标（足球）。这不是演示故障。
2. **没有场地标定**，所以页面 ③ 只能展示标定的门槛参数，无法演示接触/出界/角球/点球/任意球等几何事件。
   要演示先跑 `odp-calibrate` 生成标定文件。
3. **动作识别没有足球微调**，且没有接进本演示页面（页面只展示其状态与参数）。
4. 上传图在内存里处理、不落盘，所以**没有历史记录**；刷新即丢。