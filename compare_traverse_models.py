#!/usr/bin/env python3
"""Reproduce the paired-day uncertainty interval for the AP improvement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from traverse_model import (
    average_precision,
    load_artifact,
    load_daily_dataset,
    predict_probabilities,
    save_json,
    sha256_file,
)


def load_validated_inputs(dataset: Path, baseline: Path, variant: Path):
    frame = load_daily_dataset(dataset)
    baseline_values = load_artifact(baseline)
    variant_values = load_artifact(variant)
    baseline_payload = baseline_values[0]
    variant_payload = variant_values[0]
    if baseline_payload.get("role") != "baseline" or variant_payload.get("role") != "variant":
        raise ValueError("Expected baseline and variant model artifacts")
    if baseline_payload["split"] != variant_payload["split"]:
        raise ValueError("Baseline and variant splits do not match")
    actual_dataset_hash = sha256_file(dataset)
    baseline_hash = baseline_payload.get("provenance", {}).get("dataset_sha256")
    variant_hash = variant_payload.get("provenance", {}).get("dataset_sha256")
    if not baseline_hash or baseline_hash != variant_hash:
        raise ValueError("Baseline and variant dataset hashes do not match")
    if actual_dataset_hash != baseline_hash:
        raise ValueError("Supplied dataset SHA-256 does not match the model artifacts")
    test_years = [int(value) for value in variant_payload["split"]["test_years"]]
    test = frame[frame["year"].isin(test_years)].copy()
    actual_years = sorted(int(value) for value in test["year"].unique())
    if actual_years != sorted(test_years):
        raise ValueError("Supplied dataset does not contain every declared test year")
    if len(test) != int(variant_payload["split"]["test_rows"]):
        raise ValueError("Supplied dataset test row count does not match the model artifacts")
    return frame, test, baseline_values, variant_values, actual_dataset_hash


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/traverse_daily.csv"))
    parser.add_argument(
        "--baseline", type=Path, default=Path("artifacts/traverse_model_baseline.joblib")
    )
    parser.add_argument(
        "--variant", type=Path, default=Path("artifacts/traverse_model_variant.joblib")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/traverse_comparison.json")
    )
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    if args.replicates < 100:
        raise SystemExit("Use at least 100 bootstrap replicates")

    try:
        _, test, baseline_values, variant_values, dataset_hash = load_validated_inputs(
            args.dataset, args.baseline, args.variant
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    baseline_payload, baseline_pipeline = baseline_values
    variant_payload, variant_pipeline = variant_values
    test_years = variant_payload["split"]["test_years"]
    target = test["label"].to_numpy(dtype=int)
    baseline_probability = predict_probabilities(
        baseline_pipeline, test, baseline_payload["feature_names"]
    )
    variant_probability = predict_probabilities(
        variant_pipeline, test, variant_payload["feature_names"]
    )
    point_difference = average_precision(target, variant_probability) - average_precision(
        target, baseline_probability
    )

    generator = np.random.default_rng(args.seed)
    differences: list[float] = []
    for _ in range(args.replicates):
        positions = generator.integers(0, len(test), size=len(test))
        sampled_target = target[positions]
        if sampled_target.sum() == 0:
            continue
        differences.append(
            average_precision(sampled_target, variant_probability[positions])
            - average_precision(sampled_target, baseline_probability[positions])
        )
    interval = np.quantile(np.asarray(differences), [0.025, 0.5, 0.975])
    result = {
        "metric": "average_precision",
        "comparison": "variant_minus_baseline",
        "test_years": test_years,
        "test_rows": len(test),
        "point_difference": point_difference,
        "bootstrap": {
            "method": "paired iid resampling of held-out local days with replacement",
            "seed": args.seed,
            "requested_replicates": args.replicates,
            "completed_replicates": len(differences),
            "quantiles": {
                "2.5%": float(interval[0]),
                "50%": float(interval[1]),
                "97.5%": float(interval[2]),
            },
        },
        "dataset_sha256": dataset_hash,
        "model_artifact_sha256": {
            "baseline": sha256_file(args.baseline),
            "variant": sha256_file(args.variant),
        },
    }
    save_json(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
