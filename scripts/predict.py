#!/usr/bin/env python3
"""Score one prepared or historical row with a trained Traverse model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from pioupiou.inference.model import (
    load_artifact_with_sha256,
    onset_evidence,
    predict_loaded,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("artifacts/traverse_model.joblib")
    )
    parser.add_argument(
        "--time",
        help="HH:MM issue time for a same-day timestep dataset",
    )
    parser.add_argument(
        "--model-sha256",
        help="Trusted out-of-band SHA-256 to verify before joblib deserialization",
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("artifacts/traverse_timestep.csv")
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--date", help="Select a prepared historical day from --dataset")
    source.add_argument("--features", type=Path, help="CSV containing one prepared row")
    args = parser.parse_args()

    if args.features:
        row = pd.read_csv(args.features)
        if len(row) != 1:
            raise SystemExit(f"Expected exactly one row in {args.features}, found {len(row)}")
        prediction_date = str(row.iloc[0].get("date", "unknown"))
        try:
            expected_model_sha256 = args.model_sha256
            if expected_model_sha256 is None and "meta_model_sha256" in row:
                expected_model_sha256 = str(row.iloc[0]["meta_model_sha256"])
            loaded_artifact = load_artifact_with_sha256(
                args.model, expected_model_sha256
            )
            if "issue_minutes" not in row:
                raise ValueError("same-day features require issue_minutes")
        except (ValueError, KeyError) as error:
            print(json.dumps({"date": prediction_date, "status": str(error)}, indent=2))
            return 2
    else:
        try:
            loaded_artifact = load_artifact_with_sha256(
                args.model, args.model_sha256
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        model_payload = loaded_artifact[0]
        expected_dataset_hash = model_payload.get("provenance", {}).get("dataset_sha256")
        if not expected_dataset_hash or sha256_file(args.dataset) != expected_dataset_hash:
            raise SystemExit("Historical dataset SHA-256 does not match the model artifact")
        frame = pd.read_csv(args.dataset)
        row = frame[frame["date"].astype(str) == args.date]
        if "issue_minutes" in row.columns:
            if args.time is None:
                raise SystemExit("--time is required for a same-day timestep dataset")
            try:
                parsed_time = datetime.strptime(args.time, "%H:%M")
            except ValueError as error:
                raise SystemExit("--time must use HH:MM") from error
            requested_minutes = parsed_time.hour * 60 + parsed_time.minute
            row = row[
                pd.to_numeric(row["issue_minutes"], errors="coerce")
                == requested_minutes
            ]
        if len(row) != 1:
            raise SystemExit(f"Expected exactly one prepared row for {args.date}, found {len(row)}")
        prediction_date = args.date
    try:
        probability, predicted, contributions = predict_loaded(
            loaded_artifact[0], loaded_artifact[1], row
        )
    except ValueError as error:
        print(json.dumps({"date": prediction_date, "status": str(error)}, indent=2))
        return 2
    print(
        json.dumps(
            {
                "date": prediction_date,
                "issue_time": (
                    f"{int(row.iloc[0]['issue_minutes']) // 60:02d}:"
                    f"{int(row.iloc[0]['issue_minutes']) % 60:02d}"
                    if "issue_minutes" in row.columns
                    else None
                ),
                "status": "ok",
                "traverse_probability": float(probability[0]),
                "model_kind": loaded_artifact[0].get("model_kind", "advance_onset"),
                "predict_traverse": None if predicted[0] is None else bool(predicted[0]),
                "onset_evidence": float(
                    onset_evidence(
                        row,
                        float(
                            loaded_artifact[0]
                            .get("label", {})
                            .get("speed_threshold_kmh", 18.52)
                        ),
                    )[0]
                ),
                "model_contributions": [
                    {"feature": name, "contribution": value}
                    for name, value in contributions[0]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
