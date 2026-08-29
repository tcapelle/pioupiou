#!/usr/bin/env python3
"""Compute the current prediction and replace the remote live-data branch."""

from __future__ import annotations

import argparse
import json
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


def publish(document: dict[str, Any], branch: str, remote: str) -> None:
    """Force-replace one data-only branch with a single root commit."""
    remote_url = git("remote", "get-url", remote, capture=True)
    with tempfile.TemporaryDirectory(prefix="pioupiou-live-") as directory:
        checkout = Path(directory)
        git("init", "--quiet", cwd=checkout)
        git("remote", "add", remote, remote_url, cwd=checkout)
        (checkout / "current_prediction.json").write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n"
        )
        git("add", "current_prediction.json", cwd=checkout)
        git("commit", "--quiet", "-m", "Update live prediction", cwd=checkout)
        git("push", "--force", remote, f"HEAD:refs/heads/{branch}", cwd=checkout)


def run_once(model: Path, branch: str, remote: str, dry_run: bool) -> None:
    document = prediction_document(model)
    if dry_run:
        print(json.dumps(document, indent=2, sort_keys=True))
        return
    publish(document, branch, remote)
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
        "--dry-run", action="store_true", help="Print JSON without using Git"
    )
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")

    while True:
        try:
            run_once(args.model, args.branch, args.remote, args.dry_run)
        except subprocess.CalledProcessError as error:
            print(f"Publish failed: {error}")
        if not args.watch:
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
