"""启动新增门户，原manage.py和原配置不变。"""
import os
from pathlib import Path
import subprocess
import sys

from dotenv import load_dotenv


def main():
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / "football_portal/.env", override=False)
    os.environ["PYTHONUTF8"] = "1"
    subprocess.run([sys.executable, str(root / "manage_football.py"), "migrate", "--noinput"], cwd=root, check=True)
    host = os.environ.get("FOOTBALL_HOST", "127.0.0.1")
    port = int(os.environ.get("FOOTBALL_PORT", "8502"))
    if not 1 <= port <= 65535:
        raise ValueError("端口范围无效")
    print(f"Football portal: http://{host}:{port}/football/", flush=True)
    return subprocess.call([sys.executable, str(root / "manage_football.py"), "runserver", f"{host}:{port}", "--noreload"], cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
