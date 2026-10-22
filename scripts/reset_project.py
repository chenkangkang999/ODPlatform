# scripts/reset_project.py
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "apps" / "platform" / "src"))

from od_platform.cli.reset_project import main      # ← 调 main,不是 reset_project

if __name__ == "__main__":
    sys.exit(main())
