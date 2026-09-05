"""Audit afternoon wind labels and forecasts at morning and afternoon checkpoints."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from pioupiou.data.daily import (
    LabelConfig,
    group_piou_by_local_day,
    held_minutes,
    iter_unique_piou,
    local_boundary,
    monthly_files,
    target_label,
)
from pioupiou.inference.model import day_normalized_anticipation_weights, load_dataset, sha256_file
from pioupiou.data.timestep import traverse_event_onset


def afternoon_summary(day, observations, config: LabelConfig) -> dict:
    """Sensitivity checks use identical dates, coverage and duration rules."""
    current = target_label(day, observations, config)
    any_direction = replace(config, heading_min_degrees=0.0, heading_max_degrees=360.0)
    result = {
        "date": day.isoformat(),
        "current_label": current["label"],
        "coverage_fraction": current["meta_target_coverage_fraction"],
        "west_10kt_total_minutes": current["meta_target_qualifying_minutes"],
        "west_10kt_longest_minutes": current["meta_target_longest_qualifying_run_minutes"],
    }
    for knots in (8, 10, 12):
        for direction, rule in (("west", config), ("any", any_direction)):
            label = target_label(day, observations, replace(rule, speed_threshold_kmh=knots * 1.852))
            result[f"{direction}_{knots}kt_label"] = label["label"]
            if knots == 10 and direction == "any":
                result["any_10kt_total_minutes"] = label["meta_target_qualifying_minutes"]
                result["any_10kt_longest_minutes"] = label["meta_target_longest_qualifying_run_minutes"]

    start = local_boundary(day, config.cutoff_hour, ZoneInfo(config.timezone_name))
    end = local_boundary(day, config.target_end_hour, ZoneInfo(config.timezone_name))
    afternoon = [item for item in observations if start <= item.timestamp_local < end]
    durations = np.array([
        held_minutes(afternoon, index, end, config.sample_hold_cap_minutes)
        for index in range(len(afternoon))
    ])
    speeds = np.array([item.wind_speed_avg_kmh / 1.852 for item in afternoon])
    result["afternoon_mean_knots"] = float(np.average(speeds, weights=durations))
    result["afternoon_peak_sample_average_knots"] = float(speeds.max())
    onset = traverse_event_onset(day, observations, config, current)
    result["current_onset_minutes"] = (
        onset.hour * 60 + onset.minute + onset.second / 60
        if onset is not None else float("nan")
    )
    for hour in (21, 22):
        extended = target_label(day, observations, replace(config, target_end_hour=hour))
        result[f"west_10kt_until_{hour}_label"] = (
            extended["label"] if extended is not None else float("nan")
        )
    for hour in (14, 16, 18):
        remaining = target_label(day, observations, replace(config, cutoff_hour=hour))
        result[f"west_10kt_after_{hour}_label"] = (
            remaining["label"] if remaining is not None else float("nan")
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/traverse_timestep.csv"))
    parser.add_argument("--input-dir", type=Path, default=Path("pioudata"))
    parser.add_argument("--predictions", type=Path, default=Path("artifacts/weather_dynamics/oof_predictions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/windy_label_audit"))
    args = parser.parse_args()
    metadata = json.loads(args.dataset.with_suffix(".metadata.json").read_text())
    if sha256_file(args.dataset) != metadata["output_sha256"]:
        raise ValueError("Dataset checksum does not match metadata")
    config = LabelConfig(**metadata["label_config"])
    frame = load_dataset(args.dataset)
    frame = frame[frame.year.between(2017, 2025)].reset_index(drop=True)
    labels = frame.groupby("date").label.first()
    dates = set(labels.index.date)
    iterator, _ = iter_unique_piou(args.input_dir, ZoneInfo(config.timezone_name))
    records = [
        afternoon_summary(day, observations, config)
        for day, observations in group_piou_by_local_day(iterator)
        if day in dates
    ]
    days = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    np.testing.assert_array_equal(pd.to_datetime(days.date).to_numpy(), labels.index.to_numpy())
    np.testing.assert_array_equal(days.current_label.to_numpy(), labels.to_numpy())
    negative_days = days[days.current_label.eq(0)]
    comparison = {}
    for direction in ("west", "any"):
        for knots in (8, 10, 12):
            name = f"{direction}_{knots}kt_label"
            comparison[name] = {
                "positive_days": int(days[name].sum()),
                "added_vs_current": int((days[name].eq(1) & days.current_label.eq(0)).sum()),
                "removed_vs_current": int((days[name].eq(0) & days.current_label.eq(1)).sum()),
            }
    frame["fit_weight"] = day_normalized_anticipation_weights(frame)
    issue_time_weights = {}
    for minute in (390, 480, 540, 600, 720, 840, 960, 1080):
        subset = frame[frame.issue_minutes.eq(minute)]
        issue_time_weights[f"{minute // 60:02d}:{minute % 60:02d}"] = {
            "event_fraction": float(subset.label.mean()),
            "fit_weighted_event_fraction": float(np.average(subset.label, weights=subset.fit_weight)),
        }
    predictions = pd.read_csv(args.predictions)
    predictions = predictions[predictions.year.between(2020, 2025)].reset_index(drop=True)
    predictions["date"] = pd.to_datetime(predictions.date)
    identity = ["date", "year", "issue_minutes", "label"]
    expected = frame[frame.year.between(2020, 2025)][identity].reset_index(drop=True)
    pd.testing.assert_frame_equal(predictions[identity], expected)
    issue_time_metrics = {}
    for minute in (390, 480, 540, 600, 720, 840, 960, 1080):
        subset = predictions[predictions.issue_minutes.eq(minute)]
        issue_time_metrics[f"{minute // 60:02d}:{minute % 60:02d}"] = {
            "days": len(subset), "events": int(subset.label.sum()),
            "event_fraction": float(subset.label.mean()),
            "mean_prediction": float(subset.baseline.mean()),
            "ap": float(average_precision_score(subset.label, subset.baseline)),
            "brier": float(brier_score_loss(subset.label, subset.baseline)),
        }
    window_comparison = {}
    for column in [name for name in days if "until_" in name or "after_" in name]:
        eligible = days[days[column].notna()]
        window_comparison[column] = {
            "eligible_days": len(eligible), "unknown_days": int(days[column].isna().sum()),
            "positive_days": int(eligible[column].sum()),
            "added_vs_current_on_common_dates": int((eligible[column].eq(1) & eligible.current_label.eq(0)).sum()),
            "removed_vs_current_on_common_dates": int((eligible[column].eq(0) & eligible.current_label.eq(1)).sum()),
        }
    payload = {
        "dataset_sha256": sha256_file(args.dataset),
        "predictions_sha256": sha256_file(args.predictions),
        "script_sha256": sha256_file(__file__),
        "wind_source_sha256": {str(path): sha256_file(path) for path in monthly_files(args.input_dir)},
        "days": len(days), "current_positive_days": int(days.current_label.sum()),
        "recomputed_labels_match": True,
        "comparison": comparison,
        "current_negatives_with_30min_total_westerly_10kt": int(negative_days.west_10kt_total_minutes.ge(30).sum()),
        "current_negatives_with_60min_total_westerly_10kt": int(negative_days.west_10kt_total_minutes.ge(60).sum()),
        "current_negatives_with_westerly_run_20_to_30min": int(negative_days.west_10kt_longest_minutes.ge(20).sum()),
        "current_negatives_below_90pct_coverage": int(negative_days.coverage_fraction.lt(.9).sum()),
        "current_event_onsets": {
            "events": int(days.current_label.sum()),
            "at_or_after_14h": int(days.current_onset_minutes.ge(14 * 60).sum()),
            "at_or_after_16h": int(days.current_onset_minutes.ge(16 * 60).sum()),
            "at_or_after_18h": int(days.current_onset_minutes.ge(18 * 60).sum()),
        },
        "window_comparison": window_comparison,
        "issue_time_training_weights": issue_time_weights,
        "issue_time_oof_metrics_current_pre_onset_labels": issue_time_metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    days.to_csv(args.output_dir / "daily_labels.csv", index=False)
    (args.output_dir / "audit.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    display = {key: value for key, value in payload.items() if "sha256" not in key}
    print(json.dumps(display, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
