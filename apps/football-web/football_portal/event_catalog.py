"""One event definition shared by API, review validation and readiness views."""
import json
from pathlib import Path


def taxonomy():
    return json.loads((Path(__file__).resolve().parent / "data/event_taxonomy.json").read_text(encoding="utf-8"))


def event_types():
    # Preserve the older combined-card records while new cases use specific types.
    return {**{item["code"]: item["name"] for item in taxonomy()["events"]}, "card": "红黄牌 / 受牌（历史分类）"}
