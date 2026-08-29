#!/usr/bin/env python3
"""Compute the current prediction and replace the remote live-data branch."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from realtime_inference import predict_now


SCHEMA_VERSION = 1


def prediction_document(model: Path) -> dict[str, Any]:
    """Return the public document, including a publishable failure state."""
    published_at = datetime.now(timezone.utc).isoformat()
    try:
        prediction = predict_now(model)
    except Exception as error:
        message = str(error)
        status = (
            "outside_prediction_window"
            if "outside the model's 06:30-19:59 window" in message
            else "unavailable"
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "published_at": published_at,
            "status": status,
            "message": message,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "published_at": published_at,
        **prediction,
    }


def git(*args: str, cwd: Path | None = None, capture: bool = False) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else ""


def update_history(checkout: Path, document: dict[str, Any]) -> None:
    prediction_time = document.get("prediction_time")
    if not prediction_time:
        return
    date_text = str(prediction_time)[:10]
    days_dir = checkout / "days"
    days_dir.mkdir(exist_ok=True)
    path = days_dir / f"{date_text}.json"
    if path.exists():
        day = json.loads(path.read_text())
    else:
        day = {
            "schema_version": SCHEMA_VERSION,
            "date": date_text,
            "source": "live_published",
            "source_note": "Published contemporaneously by the live inference computer.",
            "label": None,
            "event_onset_minutes": None,
            "threshold": None,
            "points": [],
        }
    points = {
        point["prediction_time"]: point
        for point in day["points"]
        if point.get("prediction_time")
    }
    points[str(prediction_time)] = document
    day["points"] = [points[key] for key in sorted(points)]
    if day["source"] != "retrospective_reconstruction":
        day["source"] = "live_published"
    path.write_text(json.dumps(day, indent=2, sort_keys=True) + "\n")

    dates = []
    for daily_path in sorted(days_dir.glob("*.json")):
        item = json.loads(daily_path.read_text())
        dates.append(
            {
                "date": item["date"],
                "source": item["source"],
                "label": item.get("label"),
                "points": len(item["points"]),
            }
        )
    (checkout / "dates.json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "dates": dates}, indent=2)
        + "\n"
    )


def publish(
    document: dict[str, Any], branch: str, remote: str, seed_history: Path | None
) -> None:
    """Amend the data-only branch root commit, preserving its daily archive."""
    remote_url = git("remote", "get-url", remote, capture=True)
    with tempfile.TemporaryDirectory(prefix="pioupiou-live-") as directory:
        checkout = Path(directory) / "checkout"
        git(
            "clone",
            "--quiet",
            "--depth",
            "1",
            "--branch",
            branch,
            remote_url,
            str(checkout),
        )
        if seed_history is not None:
            shutil.copytree(seed_history, checkout, dirs_exist_ok=True)
        (checkout / "current_prediction.json").write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n"
        )
        update_history(checkout, document)
        git("add", "-A", cwd=checkout)
        git("commit", "--quiet", "--amend", "--no-edit", cwd=checkout)
        git("push", "--force", remote_url, f"HEAD:refs/heads/{branch}", cwd=checkout)


def run_once(
    model: Path,
    branch: str,
    remote: str,
    dry_run: bool,
    seed_history: Path | None,
) -> None:
    document = prediction_document(model)
    if dry_run:
        print(json.dumps(document, indent=2, sort_keys=True))
        return
    publish(document, branch, remote, seed_history)
    print(f"Published {document['status']} prediction at {document['published_at']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("artifacts/traverse_model.joblib")
    )
    parser.add_argument("--branch", default="live-data")
    parser.add_argument("--remote", default="origin")
    parser.add_argument(
        "--watch", action="store_true", help="Publish every --interval seconds"
    )
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument(
        "--seed-history",
        type=Path,
        help="Merge a generated history directory before publishing",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print JSON without using Git"
    )
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")

    while True:
        try:
            run_once(
                args.model,
                args.branch,
                args.remote,
                args.dry_run,
                args.seed_history,
            )
        except subprocess.CalledProcessError as error:
            print(f"Publish failed: {error}")
        if not args.watch:
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
