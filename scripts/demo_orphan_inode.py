#!/usr/bin/env python
"""
自指 bug 演示(Linux/macOS):孤儿 inode 静默丢失。

跑这个脚本,然后【在另一个终端】观察 fd 状态变化。
你会看到日志文件被删除后,fd 仍然有效——但写入的内容飞进黑洞。

仅 Linux/macOS,Windows 会在 step 2 的 rmtree 处提前 WinError 32。
"""
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGGING_DIR = REPO_ROOT / "apps" / "platform" / "logging"


def main() -> None:
    LOGGING_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGGING_DIR / "demo_orphan.log"

    # Step 1: 打开日志文件,持有 fd
    fd = open(log_path, "w")
    fd.write("step 1: log opened OK\n")
    fd.flush()

    pid = os.getpid()
    fd_no = fd.fileno()

    print(f"\n{'=' * 60}")
    print(f"PID = {pid}")
    print(f"日志文件: {log_path}")
    print(f"FD: {fd_no}")
    print(f"{'=' * 60}")
    print(f"\n👉 现在【另开一个终端】,跑下面命令(任选一条):")
    print(f"   # Linux(推荐 — /proc 直接看 fd 软链)")
    print(f"   ls -l /proc/{pid}/fd/{fd_no}")
    print(f"   # 跨平台(Linux/macOS 都行 — lsof 通用)")
    print(f"   lsof -p {pid} | grep demo_orphan")
    input("\n按回车继续(进入 step 2: 开始 rmtree)...")

    # Step 2: rmtree LOGGING_DIR(自指 bug 触发点)
    print(f"\n>>> 即将 shutil.rmtree({LOGGING_DIR})...")
    shutil.rmtree(LOGGING_DIR)
    print(f">>> rmtree 完成。Linux/macOS 上没有报错!这就是 bug 的隐蔽之处。")

    # Step 3: 在另一个终端验证孤儿 inode
    print(f"\n👉 现在【在另一个终端】再跑一次相同命令:")
    print(f"   # Linux:  ls -l /proc/{pid}/fd/{fd_no}")
    print(f"   # macOS:  lsof -p {pid} | grep demo_orphan")
    print(f"你应该看到软链路径后面多了 '(deleted)' 标记。")
    input("\n看完按回车继续(进入 step 4: 写入孤儿 fd)...")

    # Step 4: 后续写入静默丢失
    fd.write("step 2: 这条日志看起来写成功了,但是孤儿 inode\n")
    fd.flush()
    print(f"\n>>> fd.write() 没报错(write() syscall 在孤儿 fd 上完全合法)")
    print(f">>> 但是 cat {log_path}  ← 文件不存在")
    print(f">>> 进程退出后,孤儿 inode 被释放,数据【凭空消失】")
    print(f"\nWindows 在 step 2 的 rmtree 那一刻就 WinError 32——直接砸你脸上")
    print(f"Linux/macOS 不报,然后日志凭空消失——更要命")

    fd.close()


if __name__ == "__main__":
    main()
