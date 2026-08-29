#!/usr/bin/env python3
"""Build static daily web-history files from retrospective predictions."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from pioupiou.data.daily import LabelConfig, group_piou_by_local_day, iter_unique_piou
from realtime_inference import current_wind


SCHEMA_VERSION = 1


def finite_float(value: Any) -> float | None:
    parsed = float(value)
    return parsed if pd.notna(parsed) else None


def build_day(
    date_text: str,
    rows: pd.DataFrame,
    observations,
    local_timezone: ZoneInfo,
) -> dict[str, Any]:
    points = []
    for row in rows.sort_values("issue_minutes").to_dict("records"):
        issue_minutes = int(row["issue_minutes"])
        cutoff = datetime.combine(
            datetime.fromisoformat(date_text).date(),
            time(issue_minutes // 60, issue_minutes % 60),
            local_timezone,
        )
        point = {
            "prediction_time": cutoff.isoformat(),
            "status": "retrospective_reconstruction",
            "traverse_probability": float(row["traverse_probability"]),
            "predict_traverse": bool(int(row["predict_traverse"])),
            "onset_evidence": float(row["onset_evidence"]),
            "onset_within_probabilities": {
                f"{horizon}m": float(
                    row[f"probability_onset_within_{horizon}m"]
                )
                for horizon in (60, 120, 180)
            },
            "current_wind": current_wind(observations, cutoff),
        }
        points.append(point)
    first = rows.iloc[0]
    return {
        "schema_version": SCHEMA_VERSION,
        "date": date_text,
        "source": "retrospective_reconstruction",
        "source_note": (
            "Reconstructed after the date from archived observations using the "
            "frozen deployment model; not a contemporaneously published forecast."
        ),
        "label": bool(int(first["label"])),
        "event_onset_minutes": finite_float(first["event_onset_minutes"]),
        "threshold": float(first["threshold"]),
        "points": points,
    }


def write_history(
    predictions_path: Path, input_dir: Path, output_dir: Path
) -> dict[str, Any]:
    predictions = pd.read_csv(predictions_path)
    required = {
        "date",
        "issue_minutes",
        "label",
        "event_onset_minutes",
        "traverse_probability",
        "onset_evidence",
        "predict_traverse",
        "threshold",
        *(f"probability_onset_within_{horizon}m" for horizon in (60, 120, 180)),
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Prediction history is missing columns: {missing}")

    config = LabelConfig()
    local_timezone = ZoneInfo(config.timezone_name)
    requested_dates = set(predictions["date"].astype(str))
    iterator, _ = iter_unique_piou(input_dir, local_timezone)
    observations_by_day = {
        day.isoformat(): observations
        for day, observations in group_piou_by_local_day(iterator)
        if day.isoformat() in requested_dates
    }
    missing_observations = sorted(requested_dates.difference(observations_by_day))
    if missing_observations:
        raise ValueError(f"No PiouPiou observations for dates: {missing_observations}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    days_dir = output_dir / "days"
    days_dir.mkdir(parents=True)
    index = []
    for date_text, rows in predictions.groupby("date", sort=True):
        date_text = str(date_text)
        document = build_day(
            date_text, rows, observations_by_day[date_text], local_timezone
        )
        (days_dir / f"{date_text}.json").write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n"
        )
        index.append(
            {
                "date": date_text,
                "source": document["source"],
                "label": document["label"],
                "points": len(document["points"]),
            }
        )
    dates = {"schema_version": SCHEMA_VERSION, "dates": index}
    (output_dir / "dates.json").write_text(
        json.dumps(dates, indent=2, sort_keys=True) + "\n"
    )
    return {"days": len(index), "points": len(predictions), "output": str(output_dir)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/traverse_predictions_2026.csv"),
    )
    parser.add_argument("--input-dir", type=Path, default=Path("pioudata"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/web_history")
    )
    args = parser.parse_args()
    print(
        json.dumps(
            write_history(args.predictions, args.input_dir, args.output_dir),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
