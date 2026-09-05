#!/usr/bin/env python3
"""Compare the current weather contract with all available auxiliary fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from pioupiou.inference.model import (
    anticipation_metrics,
    anticipation_weights,
    build_pipeline,
    classification_metrics,
    day_alert_scores,
    day_normalized_anticipation_weights,
    expanding_year_l2_search,
    expanding_year_probabilities,
    feature_names,
    fit_pipeline,
    load_dataset,
    numeric_feature_frame,
    predict_probabilities,
    rolling_validation_years,
    select_anticipation_threshold,
    sha256_file,
    split_dataset,
)


AUXILIARY_PREFIXES = (
    "mf_belley_",
    "mf_novalaise_",
    "mf_mont_du_chat_",
)
WIND_TOKENS = ("wind_speed_", "wind_direction_", "west_component_")
MOISTURE_RAIN_TOKENS = ("dewpoint_", "relative_humidity_", "precipitation_")


def added_auxiliary_features(names: list[str]) -> dict[str, list[str]]:
    """Group fields that were excluded by the temperature-only contract."""
    added = [
        name
        for name in names
        if name.startswith(AUXILIARY_PREFIXES)
        and "temperature_c_" not in name
        and not name.endswith("_observation_count_morning")
        and not name.endswith("_last_age_minutes")
    ]
    wind = [name for name in added if any(token in name for token in WIND_TOKENS)]
    moisture_rain = [
        name
        for name in added
        if any(token in name for token in MOISTURE_RAIN_TOKENS)
    ]
    ungrouped = sorted(set(added).difference(wind, moisture_rain))
    if ungrouped:
        raise ValueError(f"Unclassified auxiliary weather fields: {ungrouped}")
    return {
        "auxiliary_wind": sorted(wind),
        "auxiliary_moisture_and_rain": sorted(moisture_rain),
        "all_auxiliary_weather": sorted(added),
    }


def paired_ap_interval(
    labels: np.ndarray,
    baseline_scores: np.ndarray,
    candidate_scores: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> dict[str, float]:
    """Bootstrap paired whole-day AP differences."""
    labels = np.asarray(labels, dtype=int)
    baseline_scores = np.asarray(baseline_scores, dtype=float)
    candidate_scores = np.asarray(candidate_scores, dtype=float)
    point = float(
        average_precision_score(labels, candidate_scores)
        - average_precision_score(labels, baseline_scores)
    )
    differences: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, len(labels), len(labels))
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size < 2:
            continue
        differences.append(
            float(
                average_precision_score(sampled_labels, candidate_scores[indices])
                - average_precision_score(sampled_labels, baseline_scores[indices])
            )
        )
    values = np.asarray(differences, dtype=float)
    return {
        "difference": point,
        "paired_95_percent_low": float(np.quantile(values, 0.025)),
        "paired_95_percent_high": float(np.quantile(values, 0.975)),
        "bootstrap_probability_improvement": float(np.mean(values > 0)),
        "bootstrap_samples": float(len(values)),
    }


def fit_variant(
    train,
    test,
    names: list[str],
    l2_candidates: tuple[float, ...],
) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray]]]:
    best_l2, l2_scores = expanding_year_l2_search(train, names, l2_candidates)
    cv_frame, cv_probability = expanding_year_probabilities(train, names, best_l2)
    threshold = select_anticipation_threshold(cv_frame, cv_probability)
    pipeline = build_pipeline(best_l2, names)
    fit_pipeline(
        pipeline,
        numeric_feature_frame(train, names),
        train["label"].to_numpy(dtype=int),
        day_normalized_anticipation_weights(train),
    )
    test_probability = predict_probabilities(pipeline, test, names)
    slices = {
        "rolling_oof_2020_2025": (cv_frame, cv_probability),
        "partial_2026_test": (test, test_probability),
    }
    metrics: dict[str, Any] = {
        "feature_count": len(names),
        "features": names,
        "selected_l2": best_l2,
        "l2_mean_fold_3h_ap": l2_scores,
        "threshold": threshold,
    }
    scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for slice_name, (slice_frame, probabilities) in slices.items():
        labels, day_scores = day_alert_scores(slice_frame, probabilities)
        scores[slice_name] = (labels, day_scores)
        metrics[slice_name] = {
            **classification_metrics(
                slice_frame["label"].to_numpy(dtype=int),
                probabilities,
                threshold,
                anticipation_weights(slice_frame),
            ),
            **anticipation_metrics(slice_frame, probabilities, threshold),
        }
    metrics["rolling_oof_years"] = rolling_validation_years(train)
    return metrics, scores


def compact_slice(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "event_day_3h_average_precision": metrics[
            "event_day_3h_average_precision"
        ],
        "event_day_3h_roc_auc": metrics["event_day_3h_roc_auc"],
        "event_alert_rate_lead_3h": metrics["event_alert_rate_lead_3h"],
        "false_alert_day_rate": metrics["false_alert_day_rate"],
    }


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# All-variable weather-station ablation",
        "",
        "The comparison changes only auxiliary-station feature columns. All variants use ",
        "identical rows, chronological folds, estimator settings, and day/lead weighting. ",
        "Each variant selects L2 on rolling 2020–2025 predictions; alert thresholds are ",
        "also selected only from those predictions. Partial 2026 is evaluated afterward.",
        "",
        "## Results",
        "",
        (
            "| Variant | Features | OOF 3 h AP | OOF ΔAP (paired 95% interval) "
            "| 2026 3 h AP | 2026 ΔAP (paired 95% interval) |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    baseline = payload["variants"]["temperature_only_auxiliary"]
    for name, result in payload["variants"].items():
        oof_ap = result["rolling_oof_2020_2025"]["event_day_3h_average_precision"]
        test_ap = result["partial_2026_test"]["event_day_3h_average_precision"]
        if name == "temperature_only_auxiliary":
            oof_delta = "—"
            test_delta = "—"
        else:
            oof = result["paired_difference_vs_temperature_only"][
                "rolling_oof_2020_2025"
            ]
            test = result["paired_difference_vs_temperature_only"][
                "partial_2026_test"
            ]
            oof_delta = (
                f"{oof['difference']:+.4f} "
                f"[{oof['paired_95_percent_low']:+.4f}, {oof['paired_95_percent_high']:+.4f}]"
            )
            test_delta = (
                f"{test['difference']:+.4f} "
                f"[{test['paired_95_percent_low']:+.4f}, {test['paired_95_percent_high']:+.4f}]"
            )
        lines.append(
            f"| `{name}` | {result['feature_count']} | {oof_ap:.4f} | "
            f"{oof_delta} | {test_ap:.4f} | {test_delta} |"
        )
    full = payload["variants"]["all_auxiliary_weather"]
    oof_delta = full["paired_difference_vs_temperature_only"][
        "rolling_oof_2020_2025"
    ]
    test_delta = full["paired_difference_vs_temperature_only"][
        "partial_2026_test"
    ]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The primary comparison is `all_auxiliary_weather` versus "
                "`temperature_only_auxiliary`. Its rolling OOF AP difference is "
                f"{oof_delta['difference']:+.4f}, with paired 95% interval "
                f"[{oof_delta['paired_95_percent_low']:+.4f}, "
                f"{oof_delta['paired_95_percent_high']:+.4f}]."
            ),
            (
                "On partial 2026, the corresponding AP difference is "
                f"{test_delta['difference']:+.4f}, with paired 95% interval "
                f"[{test_delta['paired_95_percent_low']:+.4f}, "
                f"{test_delta['paired_95_percent_high']:+.4f}]."
            ),
            "",
            "An interval spanning zero means this dataset does not establish a reliable ",
            "improvement. The 2026 result is especially uncertain because the slice contains ",
            f"only {int(baseline['partial_2026_test']['positive_days'])} event days.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/traverse_timestep_all_stations.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/weather_station_ablation.json"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        raise SystemExit("--bootstrap-samples must be positive")

    frame = load_dataset(args.dataset)
    train, test = split_dataset(frame)
    all_names = feature_names(train)
    additions = added_auxiliary_features(all_names)
    all_added = set(additions["all_auxiliary_weather"])
    baseline = [name for name in all_names if name not in all_added]
    variants = {
        "temperature_only_auxiliary": baseline,
        "auxiliary_wind": baseline + additions["auxiliary_wind"],
        "auxiliary_moisture_and_rain": (
            baseline + additions["auxiliary_moisture_and_rain"]
        ),
        "all_auxiliary_weather": all_names,
    }

    results: dict[str, Any] = {}
    scores: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for name, names in variants.items():
        result, variant_scores = fit_variant(train, test, names, (1.0, 10.0))
        results[name] = result
        scores[name] = variant_scores

    rng = np.random.default_rng(20260901)
    baseline_scores = scores["temperature_only_auxiliary"]
    for name, result in results.items():
        if name == "temperature_only_auxiliary":
            continue
        result["paired_difference_vs_temperature_only"] = {}
        for slice_name in baseline_scores:
            labels, reference = baseline_scores[slice_name]
            candidate_labels, candidate = scores[name][slice_name]
            if not np.array_equal(labels, candidate_labels):
                raise ValueError(f"Misaligned day labels for {name} {slice_name}")
            result["paired_difference_vs_temperature_only"][slice_name] = (
                paired_ap_interval(
                    labels,
                    reference,
                    candidate,
                    rng,
                    args.bootstrap_samples,
                )
            )

    payload = {
        "dataset": str(args.dataset),
        "dataset_sha256": sha256_file(args.dataset),
        "rows": len(frame),
        "days": int(frame["date"].nunique()),
        "protocol": {
            "fit_years": list(range(2017, 2026)),
            "rolling_oof_years": list(range(2020, 2026)),
            "test_year": 2026,
            "primary_metric": "3h event-day average precision",
            "uncertainty": "paired bootstrap resampling whole days",
            "bootstrap_samples": args.bootstrap_samples,
        },
        "added_feature_groups": additions,
        "variants": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    report_path = args.output.with_suffix(".md")
    report_path.write_text(markdown_report(payload))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report": str(report_path),
                "variants": {
                    name: {
                        "feature_count": result["feature_count"],
                        "rolling_oof": compact_slice(
                            result["rolling_oof_2020_2025"]
                        ),
                        "partial_2026_test": compact_slice(
                            result["partial_2026_test"]
                        ),
                    }
                    for name, result in results.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
