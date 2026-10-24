# docs/architecture —— 架构决策记录

存放所有 ADR(Architecture Decision Record)。一篇 ADR 只回答一个问题：
**当时为什么这么定、代价是什么、什么条件下该重新评估**。

命名：`ADR-<三位序号>-<短横线小写主题>.md`。历史结论不要改写 —— 要推翻就新增一篇，
并在旧的「状态」里标注 `Superseded by ADR-xxx`。

| 编号 | 主题 | 状态 |
|---|---|---|
| [ADR-001](ADR-001-monorepo.md) | 采用 Monorepo + `apps/` 布局 | ✅ Accepted |
| [ADR-002](ADR-002-dataset-yaml-portable-path.md) | 数据集 yaml 的 `path` 写相对仓库根，运行期再解析绝对路径 | ✅ Accepted |
| [ADR-003](ADR-003-runtime-config-precedence.md) | 运行配置三层与优先级（CLI > YAML > DEFAULT），配置里不写机器相关值 | ✅ Accepted |
| [ADR-004](ADR-004-inference-output-and-viz.md) | 推理输出的落盘参数可配置；美化绘制当前为桩，默认路径退到原生绘制 | ✅ Accepted |