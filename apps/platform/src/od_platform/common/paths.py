#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : paths.py
# @Author    : ODPlatform Team
# @Project   : ODPlatform
# @Function  : 集中定义项目路径常量
#              - 使用 marker file 模式定位 ROOT_DIR(workspace 根)
#              - 引入 APP_DIR 概念,把【共享资产】和【端私有资产】分开
from pathlib import Path
from typing import List, Tuple

# ============================================================
# Workspace 根目录定位 (marker file 模式)
# ============================================================
WORKSPACE_MARKER: str = ".odp-workspace"

def _find_workspace_root(
    start: Path,
    markers: Tuple[str, ...] = (WORKSPACE_MARKER,),
) -> Path:
    """
    从 start 开始沿父目录向上查找,返回包含任一 marker 文件的目录。
    Args:
        start: 起始路径(通常是 Path(__file__))
        markers: 一组 marker 文件名,任一存在即视为找到
    Returns:
        Path: workspace 根目录
    Raises:
        FileNotFoundError: 一直爬到文件系统根仍没找到
    """
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for parent in [current, *current.parents]:
        for marker in markers:
            if (parent / marker).exists():
                return parent
    raise FileNotFoundError(
        f"找不到 workspace marker 文件 ({markers})。"
        f"请确认仓库根存在 {WORKSPACE_MARKER} 文件。"
    )

# 计算 ROOT_DIR(模块加载时执行一次)
ROOT_DIR: Path = _find_workspace_root(Path(__file__))

# ============================================================
# 端根目录 APP_DIR (platform 这一个端的根)
# ============================================================
APP_DIR: Path = ROOT_DIR / "apps" / "platform"

# ============================================================
# 【共享资产】(在 ROOT_DIR 下,所有端可访问)
# ============================================================
DATA_DIR: Path = ROOT_DIR / "data"
MODELS_DIR: Path = ROOT_DIR / "models"
RUNS_DIR: Path = ROOT_DIR / "runs"

# 模型子目录
PRETRAINED_MODELS_DIR: Path = MODELS_DIR / "pretrained"  # 下载的预训练权重(输入)
TRAINED_MODELS_DIR: Path = MODELS_DIR / "trained"        # 训练产出、长期保留的权重(产物)

# 数据集子目录
# 设计依据见 D0 决定 5:原始(只读)与派生(可再生)分家。
# train/val/test 划分属于派生产物,会在数据处理那天用固定种子一次性
# 冻结进 processed/ 内;init 阶段只建 raw / processed 两个空目录。
RAW_DATA_DIR: Path = DATA_DIR / "raw"              # 原始数据:只读,绝不就地改
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"  # 派生数据集:含冻结后的 train/val/test

# ============================================================
# 【端私有资产】(在 APP_DIR 下,只属于 platform 这个端)
# ============================================================
CONFIGS_DIR: Path = APP_DIR / "configs"
LOGGING_DIR: Path = APP_DIR / "logging"
UNIT_TEST_DIR: Path = APP_DIR / "tests"

# ============================================================
# 【顶层文档目录】(共享给所有人)
# ============================================================
DOCS_DIR: Path = ROOT_DIR / "docs"

# ============================================================
# 【工程基础设施目录】(共享)
# ============================================================
SCRIPTS_DIR: Path = ROOT_DIR / "scripts"

# ============================================================
# 【工具元数据目录】(audit trail / 工具自身日志)
# ============================================================
# 业务工具(train/val/inference)的日志 → LOGGING_DIR,会被 reset 清理
# 元工具(reset/doctor/upgrade)的日志 → META_LOGGING_DIR,是对清理动作本身的
# 审计追踪——如果存 LOGGING_DIR 里会被它们自己清掉,违背审计本意。
META_DIR: Path = ROOT_DIR / ".odp-meta"
META_LOGGING_DIR: Path = META_DIR / "logs"

# ============================================================
# 对外暴露的"要初始化的目录列表"
# ============================================================
def get_dirs_to_initialize() -> List[Path]:
    """
    返回项目启动时需要确保存在的所有目录列表。
    这是 init_project.py 的【唯一数据源】——
    paths.py 决定要哪些目录,init_project 只负责创建。
    Returns:
        所有需要初始化的目录路径列表
    """
    return [
        # 共享资产
        DATA_DIR,
        RUNS_DIR,
        MODELS_DIR,
        PRETRAINED_MODELS_DIR,
        TRAINED_MODELS_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        # 端私有资产
        CONFIGS_DIR,
        DATASET_CONFIGS_DIR,
        LOGGING_DIR,
        UNIT_TEST_DIR,
        # 工程基础设施
        SCRIPTS_DIR,
        DOCS_DIR,
        # 工具元数据(reset 等元工具的审计日志位置)
        META_LOGGING_DIR,
    ]

# ============================================================
# 【reset 可清理目录白名单】
# ============================================================
def get_dirs_to_reset() -> List[Path]:
    """
    返回 reset_project 可以安全清理的目录列表。

    和 get_dirs_to_initialize 的关键差异:
        - init:  "启动时要确保存在的目录"——可能含 git 追踪目录(如 docs/)
        - reset: "可以被清理的运行时产物目录"——【绝对不含】git 追踪目录

    安全原则:
        ✅ 只列【.gitignore 忽略的】运行时目录
        ❌ 任何 git 追踪的目录(代码/文档/测试/配置)绝不列入
        ❌ 用户的【原始数据】(RAW_DATA_DIR)绝不列入
    """
    return [
        # 划分后的派生数据集(含 train/val/test,由数据流水线产出,不进 git)
        PROCESSED_DATA_DIR,
        # 训练运行产物(不进 git)
        RUNS_DIR,
        # 训练产出的权重(可再生产物——重训能拿回来,所以可清)
        TRAINED_MODELS_DIR,
        # 端私有运行时日志(不进 git)
        LOGGING_DIR,
        # 注意:以下目录【不在】 reset 范围内——
        #   RAW_DATA_DIR           : 用户辛苦准备的原始数据,reset 绝不碰
        #   PRETRAINED_MODELS_DIR  : 预训练权重(下载来的输入,很大,重下载耗时)
        #   CONFIGS_DIR / UNIT_TEST_DIR / SCRIPTS_DIR / DOCS_DIR : 都进 git
    ]

# ============================================================
# 【绝对保护目录】reset 工具永远不能删除这些
# ============================================================
# 这是【最后一道安全网】——即使 get_dirs_to_reset() 写错了,
# reset 也会拒绝删除这些目录。
PROTECTED_DIRS: tuple = (
    ROOT_DIR,                    # workspace 根 - 永远不能删
    ROOT_DIR / "apps",
    ROOT_DIR / "packages",
    APP_DIR,                     # platform 端根
    APP_DIR / "src",             # 源码目录
    SCRIPTS_DIR,                 # 进 git 的脚本
    DOCS_DIR,                    # 进 git 的文档
    UNIT_TEST_DIR,               # 进 git 的测试
    CONFIGS_DIR,                 # 进 git 的配置
    ROOT_DIR / ".git",           # git 元数据 - 删了仓库就废了
    ROOT_DIR / ".odp-workspace", # marker file
    META_DIR,                    # 工具元数据根
    META_LOGGING_DIR,            # 工具自身日志 - 审计追踪不可被破坏
)


def is_protected(path: Path) -> bool:
    """
    路径是否受保护(任一满足即受保护):
        1. 路径本身在 PROTECTED_DIRS 中
        2. 路径是某个 PROTECTED_DIRS 成员的【祖先目录】
           (例如 ROOT_DIR 是 SCRIPTS_DIR 的祖先——删 ROOT_DIR
            就连带删了 SCRIPTS_DIR,所以删祖先也得拦)
    """
    path = path.resolve(strict=False)
    for protected in PROTECTED_DIRS:
        protected_resolved = protected.resolve(strict=False)
        if path == protected_resolved:
            return True
        # Python 3.9+ 的 is_relative_to,比 try/except ValueError 可读
        if protected_resolved.is_relative_to(path):
            return True
    return False

# ============================================================
# D3 增量: 数据集配置 & 运行输出目录
# ============================================================
# 数据集 yaml 的家: 描述【数据】本身(train/val/test 路径 + 类别).
# 跟 D5 的 RUNTIME_CONFIGS_DIR(描述【跑法】)语义完全不同,物理分目录.
DATASET_CONFIGS_DIR: Path = CONFIGS_DIR / "datasets"


def raw_dataset_root(name: str) -> Path:
    return RAW_DATA_DIR / name


def dataset_processed_dir(name: str) -> Path:
    return PROCESSED_DATA_DIR / name


def dataset_yaml_path(name: str) -> Path:
    return DATASET_CONFIGS_DIR / f"{name}.yaml"


# D4 增量:质检报告的家(按需创建,不进 init 清单)
VALIDATION_RUNS_DIR: Path = RUNS_DIR / "data_validation"


def validation_run_dir(run_id: str) -> Path:
    return VALIDATION_RUNS_DIR / run_id


def get_run_output_dir(task: str, name: str, run_id: str | None = None,
                       project: Path | None = None) -> Path:
    """拼装一次运行的输出目录.

    Args:
        task:   任务类型(detect / segment...),默认 runs/<task>/
        name:   运行名(如 exp_baseline)
        run_id: 6 位短 id,None 时由调用方/RunContext 自动生成
        project: 自定义项目根,None 时用 RUNS_DIR / task

    Returns:
        拼装好的 Path(不自动 mkdir,由 RunContext 决定)
    """
    if project is None:
        project = RUNS_DIR / task
    if run_id:
        return project / f"{name}_{run_id}"
    return project / name


if __name__ == "__main__":
    print(f"ROOT_DIR (workspace) = {ROOT_DIR}")
    print(f"APP_DIR  (platform)  = {APP_DIR}")
    print(f"\n共享资产:")
    print(f"  DATA_DIR    = {DATA_DIR.relative_to(ROOT_DIR)}")
    print(f"  MODELS_DIR  = {MODELS_DIR.relative_to(ROOT_DIR)}")
    print(f"  RUNS_DIR    = {RUNS_DIR.relative_to(ROOT_DIR)}")
    print(f"\n端私有资产:")
    print(f"  CONFIGS_DIR = {CONFIGS_DIR.relative_to(ROOT_DIR)}")
    print(f"  LOGGING_DIR = {LOGGING_DIR.relative_to(ROOT_DIR)}")
    print(f"\n要初始化的目录共 {len(get_dirs_to_initialize())} 个:")
    for d in get_dirs_to_initialize():
        print(f"  - {d.relative_to(ROOT_DIR)}")
