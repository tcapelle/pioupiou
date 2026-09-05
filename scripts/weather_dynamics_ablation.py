"""A fixed-model, historical-only ablation of recent weather changes."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_curve
from threadpoolctl import threadpool_limits

from pioupiou.data.daily import (
    WEATHER_STATIONS,
    LabelConfig,
    cache_weather_resource,
    discover_weather_resources,
    iter_cached_weather,
)
from pioupiou.inference.model import (
    day_alert_scores,
    expanding_year_probabilities,
    feature_names,
    load_dataset,
    sha256_file,
)
from scripts.weather_station_ablation import added_auxiliary_features, paired_ap_interval


THERMAL_PAIRS = (("belley", "airport"), ("novalaise", "airport"), ("belley", "mont_du_chat"))
WIND_STATIONS = ("airport", "belley", "mont_du_chat")


def dynamic_weather_features(observations: dict, cutoff: datetime, config: LabelConfig) -> dict[str, float]:
    """Use exact hourly slots strictly before cutoff; gaps are not compressed."""
    local = cutoff.astimezone(ZoneInfo(config.timezone_name))
    latest = (local - timedelta(microseconds=1)).replace(minute=0, second=0, microsecond=0)

    def reading(station: str, field: str, lag: int = 0) -> float:
        timestamp = latest - timedelta(hours=lag)
        if timestamp.date() != local.date() or timestamp.hour < config.weather_morning_start_hour:
            return float("nan")
        return float(observations.get((station, timestamp), {}).get(field, float("nan")))

    def wind(station: str, component: str, lag: int = 0) -> float:
        speed = reading(station, "FF", lag)
        direction = math.radians(reading(station, "DD", lag))
        # Meteorological direction is where air comes FROM; u/v point toward.
        return -speed * (math.sin(direction) if component == "u" else math.cos(direction))

    output = {}
    for hours in (1, 3):
        for station in WEATHER_STATIONS:
            output[f"mf_dyn_thermal_{station.slug}_temperature_delta_{hours}h_c"] = (
                reading(station.slug, "T") - reading(station.slug, "T", hours)
            )
        for left, right in THERMAL_PAIRS:
            output[f"mf_dyn_thermal_{left}_minus_{right}_delta_{hours}h_c"] = (
                (reading(left, "T") - reading(right, "T"))
                - (reading(left, "T", hours) - reading(right, "T", hours))
            )
        output[f"mf_dyn_thermal_airport_radiation_delta_{hours}h_w_m2"] = (
            reading("airport", "GLO") - reading("airport", "GLO", hours)
        ) * (10000.0 / 3600.0)
    output["mf_dyn_thermal_airport_radiation_latest_w_m2"] = reading("airport", "GLO") * (10000.0 / 3600.0)

    for component in ("u", "v"):
        for station in WIND_STATIONS:
            output[f"mf_dyn_wind_snapshot_{station}_{component}_ms"] = wind(station, component)
            for hours in (1, 3):
                output[f"mf_dyn_wind_change_{station}_{component}_delta_{hours}h_ms"] = (
                    wind(station, component) - wind(station, component, hours)
                )
        for station in ("belley", "mont_du_chat"):
            output[f"mf_dyn_wind_snapshot_{station}_minus_airport_{component}_ms"] = (
                wind(station, component) - wind("airport", component)
            )
    return output


def load_weather(frame: pd.DataFrame, cache_dir: Path, config: LabelConfig) -> tuple[dict, dict]:
    """Read and verify existing caches through the standard quality/location filter."""
    resources = discover_weather_resources(
        cache_dir, int(frame.year.min()), int(frame.year.max()), offline=True,
        departments=tuple(sorted({station.department for station in WEATHER_STATIONS})),
    )
    paths = {station.slug: [] for station in WEATHER_STATIONS}
    source_hashes = {}
    for resource in resources:
        stations = [station for station in WEATHER_STATIONS if station.department == resource.department]
        path, counts = cache_weather_resource(
            cache_dir, resource, [station.station_id for station in stations], refresh=False, offline=True,
        )
        source_hashes[str(path)] = sha256_file(path)
        for station in stations:
            if counts[station.station_id]:
                paths[station.slug].append(path)
    dates = set(frame.date.dt.date)
    observations = {}
    for station in WEATHER_STATIONS:
        for observation in iter_cached_weather(paths[station.slug], ZoneInfo(config.timezone_name), station):
            timestamp = observation["timestamp_local"]
            if timestamp.date() in dates and config.weather_morning_start_hour <= timestamp.hour < config.target_end_hour:
                observations.setdefault((station.slug, timestamp), observation)
    return observations, source_hashes


def coverage_at_false_alert_ceiling(labels: np.ndarray, scores: np.ndarray, ceiling: float = 0.2) -> dict:
    """Descriptive ROC point; the threshold is selected on this same scored slice."""
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    eligible = np.flatnonzero(fpr <= ceiling)
    best = max(eligible, key=lambda index: (tpr[index], -fpr[index]))
    return {
        "event_coverage": float(tpr[best]),
        "false_alert_day_rate": float(fpr[best]),
        "threshold": float(thresholds[best]) if np.isfinite(thresholds[best]) else None,
    }


def evaluate(frame: pd.DataFrame, probability: np.ndarray) -> tuple[dict, tuple[np.ndarray, np.ndarray]]:
    labels, scores = day_alert_scores(frame, probability)
    by_year = {}
    for year in sorted(frame.year.unique()):
        mask = frame.year.eq(year).to_numpy()
        year_labels, year_scores = day_alert_scores(frame.loc[mask], probability[mask])
        by_year[str(year)] = float(average_precision_score(year_labels, year_scores))
    fixed_times = {}
    for hour in (9, 12, 15):
        mask = frame.issue_minutes.eq(hour * 60).to_numpy()
        subset = frame.loc[mask]
        fixed_times[f"{hour:02d}:00"] = {
            "days": len(subset),
            "events": int(subset.label.sum()),
            "event_rate": float(subset.label.mean()),
            "remaining_event_ap": float(average_precision_score(subset.label, probability[mask])),
        }
    return {
        "event_day_3h_ap": float(average_precision_score(labels, scores)),
        "mean_year_3h_ap": float(np.mean(list(by_year.values()))),
        "by_year_3h_ap": by_year,
        "at_20pct_false_alert_ceiling": coverage_at_false_alert_ceiling(labels, scores),
        "fixed_issue_times": fixed_times,
    }, (labels, scores)


def markdown_report(payload: dict) -> str:
    lines = [
        "# Recent weather changes: historical ablation", "",
        "Fixed L2=10, seven leaves, 200 iterations; identical rows and day/lead weights.",
        "Expanding-year predictions for 2020–2025. No 2026 evaluation or deployment fit.", "",
        "| Variant | Features | Pooled 3 h AP | ΔAP, paired 95% interval | Coverage | False alerts |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for name, result in payload["variants"].items():
        interval = result.get("paired_ap_vs_baseline")
        difference = "—" if interval is None else (
            f"{interval['difference']:+.4f} [{interval['paired_95_percent_low']:+.4f}, "
            f"{interval['paired_95_percent_high']:+.4f}]"
        )
        operating = result["at_20pct_false_alert_ceiling"]
        lines.append(
            f"| {name} | {result['feature_count']} | {result['event_day_3h_ap']:.4f} | {difference} "
            f"| {operating['event_coverage']:.1%} | {operating['false_alert_day_rate']:.1%} |"
        )
    lines.extend([
        "", "Coverage uses a common 20% false-alert ceiling. Thresholds are selected on these",
        "same historical predictions, so these are descriptive ROC comparisons, not forward",
        "estimates of an operational alert policy. Intervals resample whole days and condition",
        "on the fitted folds; they do not include model-selection or serial weather uncertainty.", "",
        "## AP by validation year", "", "| Variant | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | Mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, result in payload["variants"].items():
        values = [*result["by_year_3h_ap"].values(), result["mean_year_3h_ap"]]
        lines.append(f"| {name} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    lines.extend([
        "", "## Fixed issue-time AP", "",
        "These predict a remaining same-day event among pre-onset rows; they do not require",
        "three hours of warning. Post-onset events are absent from the dataset.", "",
        "| Variant | 09:00 | 12:00 | 15:00 |", "|---|---:|---:|---:|",
    ])
    for name, result in payload["variants"].items():
        values = [item["remaining_event_ap"] for item in result["fixed_issue_times"].values()]
        lines.append(f"| {name} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/traverse_timestep.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("pioudata/.weather_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/weather_dynamics"))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    frame = load_dataset(args.dataset)
    frame = frame[frame.year.between(2017, 2025)].reset_index(drop=True)
    metadata = json.loads(args.dataset.with_suffix(".metadata.json").read_text())
    if sha256_file(args.dataset) != metadata["output_sha256"]:
        raise ValueError("Baseline dataset checksum does not match its metadata")
    config = LabelConfig(**metadata["label_config"])
    all_names = feature_names(frame)
    excluded = set(added_auxiliary_features(all_names)["all_auxiliary_weather"])
    baseline = [name for name in all_names if name not in excluded]
    if len(baseline) != 94:
        raise ValueError(f"Expected the 94-feature baseline; found {len(baseline)} features")
    print("Reading verified local weather caches...", flush=True)
    observations, source_hashes = load_weather(frame, args.cache_dir, config)
    timezone_local = ZoneInfo(config.timezone_name)
    additions = pd.DataFrame([
        dynamic_weather_features(
            observations,
            row.date.to_pydatetime().replace(tzinfo=timezone_local) + timedelta(minutes=int(row.issue_minutes)),
            config,
        )
        for row in frame.itertuples(index=False)
    ])
    frame = pd.concat([frame, additions], axis=1)
    thermal = [name for name in additions if name.startswith("mf_dyn_thermal_")]
    snapshots = [name for name in additions if name.startswith("mf_dyn_wind_snapshot_")]
    wind_changes = [name for name in additions if name.startswith("mf_dyn_wind_change_")]
    variants = {
        "baseline": baseline,
        "thermal_changes": baseline + thermal,
        "wind_snapshots": baseline + snapshots,
        "wind_changes": baseline + snapshots + wind_changes,
        "thermal_and_wind_changes": baseline + thermal + snapshots + wind_changes,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.output_dir / "features.csv"
    frame.to_csv(feature_path, index=False)
    results, scores = {}, {}
    predictions = None
    # Small shallow fits are faster without oversubscribing the local machine.
    with threadpool_limits(limits=1):
        for name, names in variants.items():
            print(f"Fitting {name}: {len(names)} features, six chronological folds...", flush=True)
            cv_frame, probability = expanding_year_probabilities(frame, names, l2=10.0)
            result, scores[name] = evaluate(cv_frame, probability)
            result.update(feature_count=len(names), features=names)
            results[name] = result
            identity = cv_frame[["date", "year", "issue_minutes", "label", "meta_minutes_before_onset"]].reset_index(drop=True)
            if predictions is None:
                predictions = identity.copy()
            else:
                pd.testing.assert_frame_equal(predictions[identity.columns], identity)
            predictions[name] = probability
            print(f"  pooled 3 h AP={result['event_day_3h_ap']:.4f}", flush=True)
    labels, reference = scores["baseline"]
    for name in results:
        if name == "baseline":
            continue
        candidate_labels, candidate = scores[name]
        np.testing.assert_array_equal(candidate_labels, labels)
        results[name]["paired_ap_vs_baseline"] = paired_ap_interval(
            labels, reference, candidate, np.random.default_rng(20260906), args.bootstrap_samples,
        )
    predictions.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    payload = {
        "dataset": str(args.dataset), "dataset_sha256": sha256_file(args.dataset),
        "source_cache_sha256": source_hashes, "features_sha256": sha256_file(feature_path),
        "script_sha256": sha256_file(__file__), "sklearn_version": sklearn.__version__,
        "rows": len(frame), "days": int(frame.date.nunique()),
        "events": int(frame.groupby("date").label.first().sum()),
        "oof_days": int(predictions.date.nunique()), "oof_events": int(labels.sum()),
        "protocol": {"l2": 10.0, "validation_years": list(range(2020, 2026)), "test_2026_evaluated": False},
        "added_feature_nonmissing_fraction": additions.notna().mean().to_dict(),
        "variants": results,
    }
    (args.output_dir / "report.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "report.md").write_text(markdown_report(payload))
    print(f"Report: {args.output_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
