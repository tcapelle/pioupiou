#!/usr/bin/env python3
"""Compare two Traverse models with a paired-day AP bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pioupiou.inference.model import (
    FEATURE_PREFIXES,
    average_precision,
    load_artifact,
    load_daily_dataset,
    predict_probabilities,
    save_json,
    sha256_file,
)


def sampled_date_cluster_positions(
    dates: np.ndarray, sampled_clusters: np.ndarray
) -> np.ndarray:
    """Expand sampled unique-date indexes to all row positions for each date."""
    values = np.asarray(dates)
    unique_dates = np.unique(values)
    return np.concatenate(
        [np.flatnonzero(values == unique_dates[index]) for index in sampled_clusters]
    )


def load_validated_inputs(dataset: Path, reference: Path, candidate: Path):
    frame = load_daily_dataset(dataset)
    reference_values = load_artifact(reference)
    candidate_values = load_artifact(candidate)
    reference_payload = reference_values[0]
    candidate_payload = candidate_values[0]
    reference_role = reference_payload.get("role")
    candidate_role = candidate_payload.get("role")
    if reference_role not in FEATURE_PREFIXES or candidate_role not in FEATURE_PREFIXES:
        raise ValueError("Expected known Traverse model roles")
    if reference_role == candidate_role:
        raise ValueError("Reference and candidate model roles must differ")
    if reference_payload["split"] != candidate_payload["split"]:
        raise ValueError("Reference and candidate splits do not match")
    reference_sources = reference_payload.get("provenance", {}).get("source_sha256")
    candidate_sources = candidate_payload.get("provenance", {}).get("source_sha256")
    if not reference_sources or reference_sources != candidate_sources:
        raise ValueError("Reference and candidate source revisions do not match")
    actual_dataset_hash = sha256_file(dataset)
    reference_hash = reference_payload.get("provenance", {}).get("dataset_sha256")
    candidate_hash = candidate_payload.get("provenance", {}).get("dataset_sha256")
    if not reference_hash or reference_hash != candidate_hash:
        raise ValueError("Reference and candidate dataset hashes do not match")
    if actual_dataset_hash != reference_hash:
        raise ValueError("Supplied dataset SHA-256 does not match the model artifacts")
    test_years = [int(value) for value in candidate_payload["split"]["test_years"]]
    test = frame[frame["year"].isin(test_years)].copy()
    actual_years = sorted(int(value) for value in test["year"].unique())
    if actual_years != sorted(test_years):
        raise ValueError("Supplied dataset does not contain every declared test year")
    if len(test) != int(candidate_payload["split"]["test_rows"]):
        raise ValueError("Supplied dataset test row count does not match the model artifacts")
    return frame, test, reference_values, candidate_values, actual_dataset_hash


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/traverse_daily.csv"))
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("artifacts/traverse_model_variant.joblib"),
        help="Reference model",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path("artifacts/traverse_model_spatial.joblib"),
        help="Candidate model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/traverse_comparison_spatial.json"),
    )
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    if args.replicates < 100:
        raise SystemExit("Use at least 100 bootstrap replicates")

    try:
        _, test, reference_values, candidate_values, dataset_hash = load_validated_inputs(
            args.dataset, args.reference, args.candidate
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    reference_payload, reference_pipeline = reference_values
    candidate_payload, candidate_pipeline = candidate_values
    reference_role = str(reference_payload["role"])
    candidate_role = str(candidate_payload["role"])
    test_years = candidate_payload["split"]["test_years"]
    target = test["label"].to_numpy(dtype=int)
    reference_probability = predict_probabilities(
        reference_pipeline, test, reference_payload["feature_names"]
    )
    candidate_probability = predict_probabilities(
        candidate_pipeline, test, candidate_payload["feature_names"]
    )
    point_difference = average_precision(target, candidate_probability) - average_precision(
        target, reference_probability
    )

    generator = np.random.default_rng(args.seed)
    test_dates = test["date"].to_numpy()
    date_count = int(test["date"].nunique())
    differences: list[float] = []
    for _ in range(args.replicates):
        sampled_clusters = generator.integers(0, date_count, size=date_count)
        positions = sampled_date_cluster_positions(test_dates, sampled_clusters)
        sampled_target = target[positions]
        if sampled_target.sum() == 0:
            continue
        differences.append(
            average_precision(sampled_target, candidate_probability[positions])
            - average_precision(sampled_target, reference_probability[positions])
        )
    interval = np.quantile(np.asarray(differences), [0.025, 0.5, 0.975])
    result = {
        "metric": "average_precision",
        "comparison": f"{candidate_role}_minus_{reference_role}",
        "test_years": test_years,
        "test_rows": len(test),
        "test_dates": date_count,
        "point_difference": point_difference,
        "bootstrap": {
            "method": (
                "paired cluster resampling of held-out local dates with replacement; "
                "all issue-time rows for a sampled date stay together"
            ),
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
            reference_role: sha256_file(args.reference),
            candidate_role: sha256_file(args.candidate),
        },
    }
    save_json(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
