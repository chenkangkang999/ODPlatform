from __future__ import annotations

import os
from pathlib import Path
import secrets
import sys

from dotenv import load_dotenv

PORTAL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PORTAL_DIR.parent
load_dotenv(PORTAL_DIR / ".env", override=False)


def _path(name: str, default: Path | str = "") -> Path:
    path = Path(os.environ.get(name) or default)
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def _positive(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name}必须大于0")
    return value


FOOTBALL_DATA_DIR = _path("FOOTBALL_DATA_DIR", PORTAL_DIR / ".local")
FOOTBALL_DATA_DIR.mkdir(parents=True, exist_ok=True)
FOOTBALL_EVENT_DATA_DIR = _path("FOOTBALL_EVENT_DATA_DIR", FOOTBALL_DATA_DIR / "datasets/football_events")
FOOTBALL_CAROUSEL_SECONDS = _positive("FOOTBALL_CAROUSEL_SECONDS", 9)
FOOTBALL_MONITOR_REFRESH_SECONDS = _positive("FOOTBALL_MONITOR_REFRESH_SECONDS", 10)
secret_path = FOOTBALL_DATA_DIR / "django-secret.txt"
if not os.environ.get("DJANGO_SECRET_KEY"):
    if not secret_path.exists():
        secret_path.write_text(secrets.token_urlsafe(64), encoding="utf-8")
    os.environ["DJANGO_SECRET_KEY"] = secret_path.read_text(encoding="utf-8").strip()

from defect_manage.settings import *  # noqa: F403,E402

INSTALLED_APPS = [*INSTALLED_APPS, "football_portal.apps.FootballPortalConfig"]  # noqa: F405
ROOT_URLCONF = "football_portal.project_urls"
WSGI_APPLICATION = "football_portal.wsgi.application"
DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": FOOTBALL_DATA_DIR / "portal.sqlite3",
                          "OPTIONS": {"timeout": 30}}}
DEBUG = os.environ.get("FOOTBALL_DEBUG", "true").lower() in {"1", "true", "yes"}
ALLOWED_HOSTS = [s.strip() for s in os.environ.get("FOOTBALL_ALLOWED_HOSTS", "127.0.0.1,localhost,testserver").split(",") if s.strip()]
LANGUAGE_CODE = "zh-hans"
TIME_ZONE = "Asia/Shanghai"
STATICFILES_DIRS = [p for p in STATICFILES_DIRS if Path(p).exists()]  # noqa: F405
FILE_UPLOAD_TEMP_DIR = str(FOOTBALL_DATA_DIR / "tmp")
Path(FILE_UPLOAD_TEMP_DIR).mkdir(exist_ok=True)
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"
CSRF_COOKIE_SAMESITE = "Lax"
DATA_UPLOAD_MAX_MEMORY_SIZE = 8 * 1024 * 1024
FOOTBALL_MAX_IMAGE_MB = _positive("FOOTBALL_MAX_IMAGE_MB", 15)
FOOTBALL_MAX_VIDEO_MB = _positive("FOOTBALL_MAX_VIDEO_MB", 200)
FOOTBALL_MAX_IMAGE_BYTES = FOOTBALL_MAX_IMAGE_MB * 1024 * 1024
FOOTBALL_MAX_VIDEO_BYTES = FOOTBALL_MAX_VIDEO_MB * 1024 * 1024
FOOTBALL_MAX_QUEUE = _positive("FOOTBALL_MAX_QUEUE", 6)
FOOTBALL_MAX_FRAMES = _positive("FOOTBALL_MAX_FRAMES", 1500)
FOOTBALL_JOB_TIMEOUT = _positive("FOOTBALL_JOB_TIMEOUT", 900)
FOOTBALL_DEVICE = os.environ.get("FOOTBALL_DEVICE", "cpu")
FOOTBALL_INFERENCE_PYTHON = str(_path("FOOTBALL_INFERENCE_PYTHON", sys.executable))
FOOTBALL_PLATFORM_ROOT = str(_path("FOOTBALL_PLATFORM_ROOT", PROJECT_DIR.parent / "ODPlatform"))
FOOTBALL_MODEL_PATH = str(_path("FOOTBALL_MODEL_PATH")) if os.environ.get("FOOTBALL_MODEL_PATH") else ""
FOOTBALL_MODEL_DIR = str(_path("FOOTBALL_MODEL_DIR")) if os.environ.get("FOOTBALL_MODEL_DIR") else ""
FOOTBALL_MODEL_METADATA = str(_path("FOOTBALL_MODEL_METADATA")) if os.environ.get("FOOTBALL_MODEL_METADATA") else ""
FOOTBALL_MODEL_LABEL = os.environ.get("FOOTBALL_MODEL_LABEL", "足球检测模型")
FOOTBALL_DEMO_IMAGE_PATH = str(_path("FOOTBALL_DEMO_IMAGE_PATH")) if os.environ.get("FOOTBALL_DEMO_IMAGE_PATH") else ""
FOOTBALL_RUN_JOBS_INLINE = False
FOOTBALL_TEST_WORKER = None
