"""Build and evaluate a research model for useful wind still ahead today."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from threadpoolctl import threadpool_limits

from pioupiou.data.daily import (
    LabelConfig, calendar_features, group_piou_by_local_day, held_minutes,
    is_qualifying, iter_unique_piou, local_boundary, monthly_files, piou_features,
    qualifying_wind_summary, validate_label_config,
)
from pioupiou.data.timestep import (
    build_weather_timeline, cutoff_for_minutes, issue_time_features, weather_feed_ages,
)
from pioupiou.inference.model import (
    build_pipeline, day_equal_weights, day_normalized_anticipation_weights,
    feature_names, fit_pipeline, load_dataset, numeric_feature_frame,
    predict_probabilities, save_model_bundle, sha256_file,
)
from scripts.weather_dynamics_ablation import dynamic_weather_features, load_weather
from scripts.weather_station_ablation import added_auxiliary_features


CHECKPOINTS = (480, 600, 720, 840, 960, 1080)
GRID = tuple(range(390, 1201, 30))


def remaining_label(day, observations, cutoff: datetime, config: LabelConfig) -> dict | None:
    """Label only observed wind inside the remaining afternoon window."""
    local = ZoneInfo(config.timezone_name)
    start = max(cutoff, local_boundary(day, config.cutoff_hour, local))
    end = local_boundary(day, config.target_end_hour, local)
    minutes = (end - start).total_seconds() / 60
    if minutes < config.minimum_sustained_minutes:
        return None
    future = [item for item in observations if start <= item.timestamp_local < end]
    covered = sum(
        held_minutes(future, index, end, config.sample_hold_cap_minutes)
        for index in range(len(future))
    )
    if covered / minutes < config.minimum_target_coverage:
        return None
    summary = qualifying_wind_summary(future, end, config)
    before = [item for item in observations if item.timestamp_local < cutoff]
    return {
        "label": int(summary["event_onset"] is not None),
        "meta_future_coverage_fraction": covered / minutes,
        "meta_future_window_minutes": minutes,
        "meta_current_qualifies": int(bool(before) and is_qualifying(before[-1], config)),
    }


def build_features(args) -> None:
    old = load_dataset(args.reference)
    old = old[old.year.between(2017, 2025)]
    dates = set(old.date.dt.date)
    # Feature names retain their original westerly meaning whatever target is chosen.
    config = LabelConfig(piou_morning_start_hour=6, target_end_hour=21)
    local = ZoneInfo(config.timezone_name)
    print("Building uncensored historical weather features from local caches...", flush=True)
    weather = build_weather_timeline(args.cache_dir, config, 2017, 2025, GRID, False, True)
    raw_weather, weather_hashes = load_weather(old, args.cache_dir, config)
    iterator, _ = iter_unique_piou(args.input_dir, local)
    rows = []
    for day, observations in group_piou_by_local_day(iterator):
        if day not in dates:
            continue
        for minute in GRID:
            cutoff = cutoff_for_minutes(day, minute, local)
            wind = piou_features(day, observations, config, cutoff_local=cutoff)
            station_weather = weather.get((day, minute))
            if wind is None or station_weather is None:
                continue
            ages = list(weather_feed_ages(station_weather).values())
            if any(not np.isfinite(age) or age > config.maximum_weather_feature_age_minutes for age in ages):
                continue
            rows.append({
                **calendar_features(day), "issue_minutes": minute,
                **issue_time_features(minute), **wind, **station_weather,
                **dynamic_weather_features(raw_weather, cutoff, config),
            })
    frame = pd.DataFrame(rows).sort_values(["date", "issue_minutes"]).reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "features.csv"
    frame.to_csv(path, index=False)
    metadata = {
        "rows": len(frame), "days": int(frame.date.nunique()),
        "feature_config": asdict(config), "feature_sha256": sha256_file(path),
        "reference_sha256": sha256_file(args.reference),
        "weather_source_sha256": weather_hashes,
        "wind_source_sha256": {str(path): sha256_file(path) for path in monthly_files(args.input_dir)},
        "script_sha256": sha256_file(__file__), "years": list(range(2017, 2026)),
        "prediction_minutes": list(GRID), "post_onset_rows_retained": True,
    }
    (args.output_dir / "features.metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(frame):,} rows across {frame.date.nunique()} dates to {path}", flush=True)


def label_features(frame: pd.DataFrame, input_dir: Path, config: LabelConfig) -> pd.DataFrame:
    dates = set(frame.date.dt.date)
    local = ZoneInfo(config.timezone_name)
    iterator, _ = iter_unique_piou(input_dir, local)
    labels = []
    for day, observations in group_piou_by_local_day(iterator):
        if day not in dates:
            continue
        for minute in GRID:
            target = remaining_label(day, observations, cutoff_for_minutes(day, minute, local), config)
            if target is not None:
                labels.append({"date": pd.Timestamp(day), "issue_minutes": minute, **target})
    return frame.merge(pd.DataFrame(labels), on=["date", "issue_minutes"], validate="one_to_one")


def probability_metrics(y, probability) -> dict:
    y, probability = np.asarray(y), np.asarray(probability)
    if len(y) == 0:
        return {"rows": 0}
    return {
        "rows": len(y), "events": int(y.sum()), "event_rate": float(y.mean()),
        "mean_probability": float(probability.mean()),
        "ap": float(average_precision_score(y, probability)) if y.sum() else None,
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
    }


def checkpoint_metrics(frame: pd.DataFrame, probability: np.ndarray) -> dict:
    output = {}
    for minute in CHECKPOINTS:
        mask = frame.issue_minutes.eq(minute).to_numpy()
        not_windy = mask & frame.meta_current_qualifies.eq(0).to_numpy()
        output[f"{minute // 60:02d}:00"] = {
            **probability_metrics(frame.loc[mask, "label"], probability[mask]),
            "not_currently_windy": probability_metrics(frame.loc[not_windy, "label"], probability[not_windy]),
        }
    output["mean_checkpoint_ap"] = float(np.mean([v["ap"] for v in output.values()]))
    return output


def paired_checkpoint_interval(frame, reference, candidate, samples=1000) -> dict:
    """Resample whole dates; average AP differences equally across checkpoints."""
    codes, dates = pd.factorize(frame.date)
    rng = np.random.default_rng(20260905)
    masks = [frame.issue_minutes.eq(minute).to_numpy() for minute in CHECKPOINTS]
    y = frame.label.to_numpy()
    differences = []
    for _ in range(samples):
        weight = np.bincount(rng.integers(0, len(dates), len(dates)), minlength=len(dates))[codes]
        values = []
        for mask in masks:
            if not np.sum(weight[mask] * y[mask]):
                break
            values.append(
                average_precision_score(y[mask], candidate[mask], sample_weight=weight[mask])
                - average_precision_score(y[mask], reference[mask], sample_weight=weight[mask])
            )
        if len(values) == len(masks):
            differences.append(np.mean(values))
    return {
        "low": float(np.quantile(differences, .025)),
        "high": float(np.quantile(differences, .975)), "samples": len(differences),
    }


def train(args) -> None:
    path = args.output_dir / "features.csv"
    feature_metadata = json.loads((args.output_dir / "features.metadata.json").read_text())
    if sha256_file(path) != feature_metadata["feature_sha256"]:
        raise ValueError("Feature checksum mismatch")
    config = LabelConfig(
        piou_morning_start_hour=6, speed_threshold_kmh=args.knots * 1.852,
        heading_min_degrees=0.0 if args.direction == "any" else 225.0,
        heading_max_degrees=360.0 if args.direction == "any" else 315.0,
        target_end_hour=args.end_hour, minimum_sustained_minutes=args.duration,
    )
    validate_label_config(config)
    frame = pd.read_csv(path, parse_dates=["date"])
    print("Constructing remaining-window labels...", flush=True)
    frame = label_features(frame, args.input_dir, config).sort_values(["date", "issue_minutes"]).reset_index(drop=True)
    frame.to_csv(args.output_dir / "dataset.csv", index=False)
    old = load_dataset(args.reference)
    old = old[old.year.between(2017, 2025)]
    base_names = feature_names(old)
    excluded = set(added_auxiliary_features(base_names)["all_auxiliary_weather"])
    base_names = [name for name in base_names if name not in excluded]
    dynamics = sorted(name for name in frame if name.startswith("mf_dyn_"))
    variants = {"remaining_wind": base_names, "remaining_wind_dynamics": base_names + dynamics}
    outputs = []
    with threadpool_limits(limits=1):
        for year in range(2020, 2026):
            print(f"Training on earlier years; predicting {year}...", flush=True)
            fit = frame[frame.year.lt(year)]
            valid = frame[frame.year.eq(year)].copy()
            prior = fit.groupby("issue_minutes").label.mean()
            valid["historical_rate"] = valid.issue_minutes.map(prior)
            for name, names in variants.items():
                pipeline = build_pipeline(10.0, names)
                fit_pipeline(pipeline, numeric_feature_frame(fit, names), fit.label.to_numpy(), day_equal_weights(fit))
                valid[name] = predict_probabilities(pipeline, valid, names)
            legacy_fit = old[old.year.lt(year)]
            legacy = build_pipeline(10.0, base_names)
            fit_pipeline(
                legacy, numeric_feature_frame(legacy_fit, base_names), legacy_fit.label.to_numpy(),
                day_normalized_anticipation_weights(legacy_fit),
            )
            valid["legacy_objective"] = predict_probabilities(legacy, valid, base_names)
            outputs.append(valid)
    oof = pd.concat(outputs, ignore_index=True)
    names = ["historical_rate", "legacy_objective", *variants]
    metrics = {name: checkpoint_metrics(oof, oof[name].to_numpy()) for name in names}
    print("Computing paired whole-day uncertainty...", flush=True)
    intervals = {
        name: paired_checkpoint_interval(oof, oof.legacy_objective.to_numpy(), oof[name].to_numpy())
        for name in variants
    }
    dynamics_interval = paired_checkpoint_interval(
        oof, oof.remaining_wind.to_numpy(), oof.remaining_wind_dynamics.to_numpy(),
    )
    selected = "remaining_wind_dynamics" if dynamics_interval["low"] > 0 else "remaining_wind"
    selected_names = variants[selected]
    with threadpool_limits(limits=1):
        pipeline = build_pipeline(10.0, selected_names)
        fit_pipeline(pipeline, numeric_feature_frame(frame, selected_names), frame.label.to_numpy(), day_equal_weights(frame))
    metadata = {
        "model_kind": "remaining_wind_research", "feature_names": selected_names,
        "target_config": asdict(config), "selected_variant": selected,
        "fit_years": list(range(2017, 2026)), "oof_years": list(range(2020, 2026)),
        "l2": 10.0, "fit_weighting": "equal_per_day_uniform_within_day",
        "feature_provenance": feature_metadata, "script_sha256": sha256_file(__file__),
        "dataset_sha256": sha256_file(args.output_dir / "dataset.csv"),
        "post_onset_rows_retained": True, "test_2026_evaluated": False,
        "sklearn_version": sklearn.__version__, "uv_lock_sha256": sha256_file("uv.lock"),
    }
    save_model_bundle({"pipeline": pipeline, "metadata": metadata}, args.output_dir / "model.joblib")
    identity = ["date", "year", "issue_minutes", "label", "meta_current_qualifies"]
    oof[identity + names].to_csv(args.output_dir / "oof_predictions.csv", index=False)
    report = {
        "rows": len(frame), "days": int(frame.date.nunique()), "oof_rows": len(oof),
        "oof_days": int(oof.date.nunique()), "model": metadata,
        "metrics": metrics, "mean_checkpoint_ap_intervals_vs_legacy": intervals,
        "dynamics_vs_base_interval": dynamics_interval,
        "by_year": {
            str(year): {name: checkpoint_metrics(group, group[name].to_numpy()) for name in names}
            for year, group in oof.groupby("year")
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Remaining-wind research model", "", "| Issue time | Historical rate AP | Legacy AP | New target AP | New target + dynamics AP |", "|---|---:|---:|---:|---:|"]
    for minute in CHECKPOINTS:
        clock = f"{minute // 60:02d}:00"
        lines.append(f"| {clock} | " + " | ".join(f"{metrics[name][clock]['ap']:.4f}" for name in names) + " |")
    lines.extend(["", f"Selected candidate: {selected}.", "", "All models are scored against the same remaining-window labels on the same dates.", "Legacy scores are transferred from the old target, including rows outside its pre-onset", "training population. These are historical development folds, not a fresh test.", "", "See report.json for calibration, not-currently-windy subsets, per-year results and paired intervals."])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"selected": selected, "mean_ap": {name: metrics[name]["mean_checkpoint_ap"] for name in names}, "intervals_vs_legacy": intervals}, indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("features", "train", "predict"))
    parser.add_argument("--reference", type=Path, default=Path("artifacts/traverse_timestep.csv"))
    parser.add_argument("--input-dir", type=Path, default=Path("pioudata"))
    parser.add_argument("--cache-dir", type=Path, default=Path("pioudata/.weather_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/remaining_wind"))
    parser.add_argument("--knots", type=float)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--direction", choices=("any", "west"))
    parser.add_argument("--end-hour", type=int, choices=(20, 21, 22))
    parser.add_argument("--date")
    parser.add_argument("--time")
    args = parser.parse_args()
    if args.command == "features":
        build_features(args)
    elif args.command == "train":
        if args.knots is None or args.direction is None or args.end_hour is None:
            parser.error("train requires --knots, --direction and --end-hour")
        train(args)
    else:
        if args.date is None or args.time is None:
            parser.error("predict requires --date and --time")
        bundle = joblib.load(args.output_dir / "model.joblib")
        if bundle["metadata"]["model_kind"] != "remaining_wind_research":
            parser.error("Expected a remaining-wind research bundle")
        frame = pd.read_csv(args.output_dir / "features.csv")
        hour, minute = map(int, args.time.split(":"))
        config = bundle["metadata"]["target_config"]
        if config["target_end_hour"] * 60 - max(720, hour * 60 + minute) < config["minimum_sustained_minutes"]:
            parser.error("Too little time remains for a complete useful-wind spell")
        row = frame[frame.date.eq(args.date) & frame.issue_minutes.eq(hour * 60 + minute)]
        if len(row) != 1:
            parser.error("Requested feature row is unavailable")
        probability = predict_probabilities(bundle["pipeline"], row, bundle["metadata"]["feature_names"])[0]
        print(json.dumps({"date": args.date, "time": args.time, "remaining_wind_probability": float(probability), "evaluation": "fitted historical model; use oof_predictions.csv for evaluation"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
