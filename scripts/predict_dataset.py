#!/usr/bin/env python3
"""Score a held-out year without changing the trained model or threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pioupiou.inference.model import (
    anticipation_metrics,
    anticipation_weights,
    classification_metrics,
    load_artifact_with_sha256,
    load_dataset,
    onset_evidence,
    predict_probabilities,
    save_json,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("artifacts/traverse_model.joblib")
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("artifacts/traverse_timestep.csv")
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/traverse_predictions.csv")
    )
    args = parser.parse_args()

    payload, pipeline, model_sha256 = load_artifact_with_sha256(args.model)
    frame = load_dataset(args.dataset)
    held_out = frame[frame["year"] == args.year].copy()
    if held_out.empty:
        raise SystemExit(f"Dataset contains no labeled rows for {args.year}")
    trained_years = set(payload["split"]["train_years"])
    validation_years = set(payload["split"]["validation_years"])
    if args.year in trained_years | validation_years:
        raise SystemExit(f"Refusing to describe {args.year} as held out")

    feature_names = list(payload["feature_names"])
    required_columns = {
        *feature_names,
        "meta_anticipation_weight",
        "meta_minutes_before_onset",
    }
    missing = sorted(required_columns.difference(held_out.columns))
    if missing:
        raise SystemExit(f"Dataset is missing required model columns: {missing}")
    probability = predict_probabilities(pipeline, held_out, feature_names)
    speed_threshold = float(
        payload.get("label", {}).get("speed_threshold_kmh", 18.52)
    )
    evidence = onset_evidence(held_out, speed_threshold)
    threshold = float(payload["model"]["threshold"])
    predicted = probability >= threshold
    event_onset_minutes = (
        held_out["issue_minutes"] + held_out["meta_minutes_before_onset"]
    ).where(held_out["label"] == 1)
    output = pd.DataFrame(
        {
            "date": held_out["date"].dt.strftime("%Y-%m-%d"),
            "year": held_out["year"].to_numpy(),
            "issue_minutes": held_out["issue_minutes"].to_numpy(),
            "label": held_out["label"].to_numpy(),
            "event_onset_minutes": event_onset_minutes.to_numpy(),
            "traverse_probability": probability,
            "onset_evidence": evidence,
            "predict_traverse": predicted.astype(int),
            "threshold": threshold,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    metrics = {
        **classification_metrics(
            held_out["label"].to_numpy(dtype=int),
            probability,
            threshold,
            anticipation_weights(held_out),
        ),
        **anticipation_metrics(held_out, probability, threshold),
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    save_json(
        {
            "year": args.year,
            "rows": len(output),
            "days": int(output["date"].nunique()),
            "model_sha256": model_sha256,
            "dataset": str(args.dataset),
            "dataset_sha256": sha256_file(args.dataset),
            "output_sha256": sha256_file(args.output),
            "threshold": threshold,
            "metrics": metrics,
        },
        metadata_path,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "metadata": str(metadata_path),
                "rows": len(output),
                "days": int(output["date"].nunique()),
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
