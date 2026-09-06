#!/usr/bin/env python3
"""Evaluate the fitted Traverse model on the 2026 event season."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pioupiou.inference.model import (
    anticipation_metrics,
    anticipation_weights,
    classification_metrics,
    load_bundle,
    load_dataset,
    onset_horizon_labels,
    predict_bundle_onset_probabilities,
    predict_probabilities,
    save_json,
    sha256_file,
)


EVALUATION_YEAR = 2026


def event_scores(
    frame: pd.DataFrame, probabilities: np.ndarray, threshold: float
) -> list[dict[str, Any]]:
    """Return one interpretable score record for each positive event day."""
    scored = frame.loc[:, ["date", "issue_minutes", "label", "meta_minutes_before_onset"]].copy()
    scored["probability"] = np.asarray(probabilities, dtype=float)
    scored["alert"] = scored["probability"] >= threshold
    events: list[dict[str, Any]] = []
    for event_date, day in scored[scored["label"] == 1].groupby("date", sort=True):
        onset_minutes = day["issue_minutes"] + day["meta_minutes_before_onset"]
        eligible = day[day["meta_minutes_before_onset"] >= 180]
        alerts = day[day["alert"]]
        events.append(
            {
                "date": pd.Timestamp(event_date).strftime("%Y-%m-%d"),
                "onset_minutes": int(round(float(onset_minutes.median()))),
                "max_probability_at_least_3h": (
                    float(eligible["probability"].max()) if len(eligible) else None
                ),
                "alerted_at_least_3h": bool(eligible["alert"].any()),
                "maximum_warning_minutes": (
                    float(alerts["meta_minutes_before_onset"].max())
                    if len(alerts)
                    else None
                ),
            }
        )
    return events


def event_day_confusion(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    events: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Count operational event-day outcomes at three hours and at any lead."""
    positive_days = len(events)
    negative = frame[frame["label"] == 0].copy()
    negative["alert"] = np.asarray(probabilities, dtype=float)[
        frame["label"].to_numpy(dtype=int) == 0
    ] >= threshold
    false_positive_days = int(
        negative.loc[negative["alert"], "date"].nunique()
    )
    negative_days = int(negative["date"].nunique())

    def counts(detected: int) -> dict[str, int]:
        return {
            "detected_events": detected,
            "false_negative_events": positive_days - detected,
            "false_positive_days": false_positive_days,
            "true_negative_days": negative_days - false_positive_days,
            "positive_event_days": positive_days,
            "negative_days": negative_days,
        }

    return {
        "at_least_3h": counts(
            sum(bool(event["alerted_at_least_3h"]) for event in events)
        ),
        "any_lead": counts(
            sum(event["maximum_warning_minutes"] is not None for event in events)
        ),
    }


def evaluate_2026(bundle: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    """Score 2026 without fitting or selecting any model parameter."""
    metadata = bundle["metadata"]
    if metadata.get("model_kind") == "remaining_wind":
        raise ValueError("This evaluator requires the legacy onset model and labels. Use scripts.remaining_wind for remaining-window evaluation.")
    fitted_years = {int(year) for year in metadata["split"]["train_years"]}
    if EVALUATION_YEAR in fitted_years:
        raise ValueError(f"Refusing to evaluate fitted training year {EVALUATION_YEAR}")

    evaluation = frame[frame["year"] == EVALUATION_YEAR].copy()
    if evaluation.empty:
        raise ValueError(f"Dataset contains no labeled rows for {EVALUATION_YEAR}")

    feature_names = list(metadata["feature_names"])
    required = {
        *feature_names,
        "issue_minutes",
        "meta_anticipation_weight",
        "meta_minutes_before_onset",
    }
    missing = sorted(required.difference(evaluation.columns))
    if missing:
        raise ValueError(f"Dataset is missing required model columns: {missing}")

    threshold = float(metadata["model"]["threshold"])
    probabilities = predict_probabilities(
        bundle["pipeline"], evaluation, feature_names
    )
    primary_metrics = {
        **classification_metrics(
            evaluation["label"].to_numpy(dtype=int),
            probabilities,
            threshold,
            anticipation_weights(evaluation),
        ),
        **anticipation_metrics(evaluation, probabilities, threshold),
    }

    onset_metrics: dict[str, dict[str, float]] = {}
    onset_probabilities = predict_bundle_onset_probabilities(bundle, evaluation)
    onset_thresholds = (metadata.get("onset_model") or {}).get("thresholds", {})
    for horizon, horizon_probabilities in onset_probabilities.items():
        onset_metrics[f"{horizon}m"] = classification_metrics(
            onset_horizon_labels(evaluation, horizon),
            horizon_probabilities,
            float(onset_thresholds[str(horizon)]),
        )

    events = event_scores(evaluation, probabilities, threshold)
    evaluation_dates = pd.to_datetime(evaluation["date"], errors="raise")
    return {
        "year": EVALUATION_YEAR,
        "validation_status": "evolving",
        "date_range": [
            evaluation_dates.min().strftime("%Y-%m-%d"),
            evaluation_dates.max().strftime("%Y-%m-%d"),
        ],
        "through": evaluation_dates.max().strftime("%Y-%m-%d"),
        "rows": len(evaluation),
        "days": int(evaluation["date"].nunique()),
        "threshold": threshold,
        "metrics": primary_metrics,
        "event_day_confusion": event_day_confusion(
            evaluation, probabilities, threshold, events
        ),
        "onset_horizon_metrics": onset_metrics,
        "events": events,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("artifacts/traverse_model.joblib")
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("artifacts/traverse_timestep.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/eval_2026.json")
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    bundle, model_sha256 = load_bundle(args.model)
    frame = load_dataset(args.dataset)
    report = {
        **evaluate_2026(bundle, frame),
        "model": str(args.model),
        "model_sha256": model_sha256,
        "dataset": str(args.dataset),
        "dataset_sha256": sha256_file(args.dataset),
    }
    save_json(report, args.output)
    print(json.dumps({**report, "output": str(args.output)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
