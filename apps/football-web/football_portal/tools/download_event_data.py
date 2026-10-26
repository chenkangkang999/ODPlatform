"""Download a small, attributable football research corpus without credentials.

Only public, selected files are downloaded. This does not request gated access,
accept an NDA, download a complete dataset, or train an event model.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / ".local/datasets/football_events"
DEFAULT_PLAN = Path(__file__).resolve().parents[1] / "data/event_download_plan.json"
SOURCE_LICENSES = {}


def request(url: str):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "FootballResearchDownloader/1.0"}), timeout=90)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def fetch(root: Path, path: str, url: str, kind: str, source_id: str, coverage=None, **extra) -> dict:
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Download path leaves the selected dataset directory")
    if urllib.parse.urlparse(url).scheme != "https":
        raise ValueError("Download sources must use HTTPS")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file() or target.stat().st_size == 0:
        last_error = None
        for attempt in range(3):
            try:
                with request(url) as response, target.with_suffix(target.suffix + ".part").open("wb") as out:
                    expected_length = response.headers.get("Content-Length")
                    copied = 0
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        copied += len(chunk)
                        if copied > 150_000_000:
                            raise ValueError("Selected individual file exceeds 150MB guard")
                        out.write(chunk)
                    if expected_length and copied != int(expected_length):
                        raise ValueError("Incomplete download")
                target.with_suffix(target.suffix + ".part").replace(target)
                break
            except (urllib.error.URLError, OSError, ValueError) as exc:
                last_error = exc
                if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403):
                    break
                time.sleep(attempt + 1)
        if not target.is_file():
            raise RuntimeError(f"Download failed: {path}: {type(last_error).__name__}") from last_error
    return {"path": path, "url": url, "bytes": target.stat().st_size, "sha256": sha256(target),
            "kind": kind, "source_id": source_id, "coverage_event_ids": sorted(set(coverage or [])),
            "license": SOURCE_LICENSES[source_id],
            **extra}


def statsbomb_coverage(events: list[dict]) -> tuple[list[int], dict]:
    """Map actual annotation values; no claim that JSON is visual training data."""
    counts = Counter()
    values = Counter()
    for event in events:
        name = event.get("type", {}).get("name", "")
        values[name] += 1
        ids = []
        if name == "Offside" or event.get("pass", {}).get("outcome", {}).get("name") == "Pass Offside":
            ids.append(1)
        if name == "Shot":
            ids.append(14)
            shot = event.get("shot", {})
            if shot.get("outcome", {}).get("name") == "Goal":
                ids.append(2)
            if shot.get("outcome", {}).get("name") == "Blocked":
                ids.append(23)
            if shot.get("body_part", {}).get("name") == "Head":
                ids.append(19)
            if shot.get("type", {}).get("name") == "Penalty":
                ids.append(8)
        if name in {"Foul Committed", "Foul Won"}:
            ids.append(3)
            if event.get("location"):
                ids.append(13)
        for obj in (event.get("foul_committed", {}), event.get("bad_behaviour", {})):
            if obj.get("type", {}).get("name") == "Handball":
                ids.append(4)
            if obj.get("penalty"):
                ids.append(8)
            card = obj.get("card", {}).get("name")
            if card == "Yellow Card":
                ids.append(5)
            if card == "Red Card":
                ids.append(6)
            if card == "Second Yellow":
                ids.append(7)
        if name == "Pass":
            ids.append(15)
            subtype = event.get("pass", {}).get("type", {}).get("name")
            ids.extend({"Free Kick": [9], "Corner": [10], "Throw-in": [11], "Goal Kick": [12], "Kick Off": [24]}.get(subtype, []))
            if event.get("pass", {}).get("body_part", {}).get("name") == "Head":
                ids.append(19)
        if name == "Dribble":
            ids.append(16)
        if name == "Duel" and event.get("duel", {}).get("type", {}).get("name") == "Tackle":
            ids.append(17)
        if name == "Goal Keeper" and event.get("goalkeeper", {}).get("type", {}).get("name") in {"Shot Saved", "Penalty Saved", "Shot Saved Off T", "Penalty Saved To Post"}:
            ids.append(18)
        ids.extend({"Substitution": [25], "Injury Stoppage": [26], "Half End": [28], "Half Start": [30], "Starting XI": [33], "Tactical Shift": [33]}.get(name, []))
        if event.get("play_pattern", {}).get("name") == "From Counter":
            ids.append(37)
        for item in set(ids):
            counts[item] += 1
    return sorted(counts), {"event_type_counts": dict(values), "coverage_label_counts": {str(k): v for k, v in sorted(counts.items())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "--root", dest="root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--groups", nargs="+", choices=("videos", "images", "events"), default=["videos", "images", "events"])
    parser.add_argument("--clips-per-category", type=int, choices=range(1, 12))
    parser.add_argument("--event-match-limit", type=int, help="Optional cap for each configured event competition")
    parser.add_argument("--skip-videos", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    sources = plan["sources"]
    SOURCE_LICENSES.update({key: value["license"] for key, value in sources.items()})

    def remote(source, path):
        return sources[source]["download_base"] + urllib.parse.quote(path)

    files = []
    failures = []
    for item in plan["bootstrap_files"]:
        if item.get("group") and item["group"] not in args.groups:
            continue
        source = item["source_id"]
        extra = {key: value for key, value in item.items() if key not in {"path", "remote_path", "kind", "source_id", "group", "coverage_event_ids"}}
        files.append(fetch(root, item["path"], remote(source, item["remote_path"]), item["kind"], source, item.get("coverage_event_ids", []), **extra))
    rows = list(csv.DictReader(io.StringIO((root / plan["videos"]["index_path"]).read_text(encoding="utf-8"))))
    selected = []
    for category in plan["videos"]["categories"] if "videos" in args.groups else []:
        candidates = sorted((r for r in rows if r["event_category"] == category), key=lambda r: (r["game_date"], r["asset_id"]), reverse=True)
        chosen = []
        for subtype in plan["videos"]["preferred_subtypes"]:
            match = next((r for r in candidates if r.get("event_subtype") == subtype and r not in chosen), None)
            if match:
                chosen.append(match)
        chosen.extend(r for r in candidates if r not in chosen)
        selected.extend(chosen[:args.clips_per_category or plan["videos"]["clips_per_category"]])
    tasks = []
    for row in selected:
        coverage = {"goal": [2], "yellow_card": [5], "red_card": [6]}[row["event_category"]].copy()
        if row.get("event_subtype") == "Header":
            coverage.append(19)
        if row.get("event_subtype") == "Free kick shot":
            coverage.append(9)
        if row.get("event_subtype") == "Second Yellow Card":
            coverage.append(7)
        for key, kind in (("json_file", "annotations"), ("mp4_file", "video")):
            if kind == "video" and args.skip_videos:
                continue
            path = row[key]
            tasks.append(("infactory/data/" + path, remote(plan["videos"]["source_id"], "data/" + path), kind, plan["videos"]["source_id"], coverage,
                          {"annotation_origin": "provider multimodal auto-annotation; human review required", "match_date": row["game_date"],
                           "home_team": row["home_team"], "away_team": row["away_team"], "event_category": row["event_category"],
                           "event_subtype": row.get("event_subtype", ""), "source_duration_seconds": float(row["duration_seconds"])}))
    for competition in plan["events"]["competitions"] if "events" in args.groups else []:
        comp, season = competition["competition_id"], competition["season_id"]
        path = f"statsbomb/matches/{comp}/{season}.json"
        files.append(fetch(root, path, remote(plan["events"]["source_id"], f"data/matches/{comp}/{season}.json"), "metadata", plan["events"]["source_id"]))
        matches = json.loads((root / path).read_text(encoding="utf-8"))
        matches = sorted(matches, key=lambda r: (r["match_date"], r["match_id"]), reverse=True)
        limit = args.event_match_limit if args.event_match_limit is not None else competition["limit"]
        chosen = matches if limit is None else matches[:max(0, limit)]
        for match in chosen:
            match_id = match["match_id"]
            for folder in ("events", "lineups"):
                rel = f"data/{folder}/{match_id}.json"
                tasks.append(("statsbomb/" + rel[5:], remote(plan["events"]["source_id"], rel), "annotations" if folder == "events" else "metadata", plan["events"]["source_id"], [],
                              {"match_date": match["match_date"], "match_id": match_id,
                               "home_team": match["home_team"]["home_team_name"], "away_team": match["away_team"]["away_team_name"]}))
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch, root, path, remote, kind, source, coverage, **extra): path for path, remote, kind, source, coverage, extra in tasks}
        for future in as_completed(futures):
            path = futures[future]
            try:
                entry = future.result()
                if entry["source_id"] == "statsbomb_open_data" and entry["kind"] == "annotations":
                    coverage, detail = statsbomb_coverage(json.loads((root / entry["path"]).read_text(encoding="utf-8")))
                    entry["coverage_event_ids"] = coverage
                    entry.update(detail)
                files.append(entry)
                print(json.dumps({"downloaded": path, "bytes": entry["bytes"], "kind": entry["kind"]}), flush=True)
            except Exception as exc:
                failures.append({"path": path, "error": str(exc)})
                print(json.dumps({"failed": path, "error": str(exc)}), flush=True)
    manifest = {"schema": "football.local-corpus.v1", "created_at": datetime.now(timezone.utc).isoformat(),
                "purpose": "First research subset for event UI, statistics and future model preparation; not a complete 42-event training corpus.",
                "sources": [{k: v for k, v in value.items() if k != "download_base"} for value in sources.values()],
                "download_groups": args.groups,
                "files": sorted(files, key=lambda r: r["path"]), "blocked_sources": plan["blocked_sources"], "failed_downloads": failures,
                "counts": dict(Counter(item["kind"] for item in files)), "total_bytes": sum(item["bytes"] for item in files),
                "complete_42_event_training_dataset": False, "model_trained": False, "nda_accepted": False, "paid": False,
                "coverage_note": "coverage_event_ids denotes source-labeled examples/fields, not model capability or full subcase coverage. Goal means source goal label, not valid/invalid judgment; cards do not identify recipient; free kick may not distinguish direct/indirect; formation is event-provider tactical metadata."}
    # Keep already verified files/previews when rerunning a selected group.
    previous = root / "manifest.json"
    if previous.is_file():
        old = json.loads(previous.read_text(encoding="utf-8"))
        indexed = {row["path"]: row for row in manifest["files"]}
        for row in old.get("files", []):
            if row["path"] not in indexed and (root / row["path"]).is_file():
                indexed[row["path"]] = row
        manifest["files"] = sorted(indexed.values(), key=lambda row: row["path"])
        manifest["counts"] = dict(Counter(row["kind"] for row in manifest["files"]))
        manifest["total_bytes"] = sum(row["bytes"] for row in manifest["files"])
    temporary = root / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(root / "manifest.json")
    print(json.dumps({"manifest": str(root / "manifest.json"), "counts": manifest["counts"], "total_bytes": manifest["total_bytes"], "failures": len(failures)}), flush=True)


if __name__ == "__main__":
    main()
