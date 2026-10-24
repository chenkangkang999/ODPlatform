# ADR-002：数据集 yaml 的 path 写相对仓库根，运行期再解析绝对路径

| 项目 | 内容 |
|---|---|
| 状态 | ✅ Accepted |
| 决策日期 | 2026-10-24 |

## 背景

`odp-transform` 产出的数据集 yaml（`configs/datasets/<name>.yaml`）是**进 git** 的：
它同时充当"产品合格证"（`odp_meta` 里有契约指纹与 manifest 引用），D4 质检与 D5 训练
都要读它。而它的 `path` 字段原先由 `yaml_writer` 写成

```yaml
path: D:/ODPlatform/data/processed/demo     # ← 写死盘符
```

后果：仓库 clone 到别的盘符/目录后，所有人（以及 CI）都会因为找不到数据集根目录而失败；
yaml 里还混进了机器相关信息，与"产物可移植"的原则冲突。

难点在于**三个消费方的解析规则不一样**：

| 消费方 | 相对 path 的解析规则 |
|---|---|
| ODPlatform 自己的 D4 快照 | 相对 yaml 所在目录 |
| ultralytics `check_det_dataset` | 相对它全局配置的 `DATASETS_DIR`（不是本仓库！） |
| 人 | 希望是"相对仓库根"这种一眼能懂的写法 |

## 备选方案

- 方案 A：继续写绝对路径，靠"大家都把仓库放同一个位置"约定 —— 不可移植，CI 会挂
- 方案 B：写相对 yaml 目录的路径（`../../../../data/processed/demo`）—— 可移植但难读，
  且 ultralytics 仍然按 `DATASETS_DIR` 解析，等于没修
- 方案 C：yaml 写相对仓库根，**写入端与读取端各自显式解析**，喂 ultralytics 前再落一份
  绝对路径的运行期副本（本决策）
- 方案 D：把数据集 yaml 从 git 里摘掉，改为每次本地生成 —— 可移植性问题消失，
  但丢掉"合格证"的版本留痕能力，且新克隆的人必须先跑 transform 才能训练

## 决定

方案 C：

1. **写**：`yaml_writer` 通过 `common/dataset_yaml.to_repo_relative()` 写
   `data/processed/<name>`（仓库外数据才退回绝对路径，并记 warning）。
2. **读**：`common/dataset_yaml.resolve_dataset_root()` 统一解析 ——
   绝对路径原样 → 相对 yaml 目录（兼容外部 yaml）→ 相对仓库根。
   D4 快照改为调用它，不再自己实现一套。
3. **给 ultralytics**：`training.service.write_resolved_copy()` 在开训前把 `path` 解析成
   绝对路径，落到现场目录 `runs/.../dataset_resolved.yaml`（不进 git），只喂给它；
   源 yaml 保持可移植。审计快照同时记录 `data_yaml` 与 `data_yaml_resolved`。

## 理由

1. 仓库里的产物对人可读、可评审、可 diff，且换机器无痛。
2. 解析规则集中在一处（`common/dataset_yaml.py`），三个消费方不再各自猜。
3. "可移植的源 + 运行期解析副本"是常见模式：源是单一事实，副本是适配层，
   适配层的存在被审计记录，不会变成隐形状态。
4. 不改变 D4/D5 的对外契约（指纹、章、审计字段都不受影响）。

## 后果

- 正面：clone 到任意路径/盘符都能直接 `odp-train`；CI 能在 Linux 上跑数据质检。
- 负面：多了一个概念（`dataset_resolved.yaml`）与一次文件复制；读代码的人需要知道
  "ultralytics 的相对路径语义不同"这件事 —— 已在模块 docstring、README 与本 ADR 写明。
- 负面：`path` 指向仓库外的用户数据集时仍会是绝对路径（这是有意的：那种数据本就不在仓库里）。
- 兼容：旧的外部 yaml 若用"相对 yaml 目录"的写法，仍按原语义解析，不被破坏。

## 撤销条件

- ultralytics 支持"相对 yaml 目录"或允许注入数据集根时，可去掉运行期副本这一层；
- 数据集 yaml 改为不进 git（方案 D 的触发条件出现）时，本 ADR 由新决策取代。