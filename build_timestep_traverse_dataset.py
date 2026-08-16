#!/usr/bin/env python3
"""Build same-day Traverse predictions on a dense local-time grid."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import numpy as np

from build_traverse_dataset import (
    DATA_GOUV_API,
    DATA_GOUV_DATASET_ID,
    METEO_FRANCE_SOURCE_SCHEMA,
    PIOU_ARCHIVE_API,
    PIOU_LATITUDE,
    PIOU_LONGITUDE,
    PIOU_MAX_LOCATION_DISTANCE_KM,
    PIOU_SOURCE_SCHEMA,
    PIOU_STATION_ID,
    PRIMARY_WEATHER_STATION,
    WEATHER_MAX_LOCATION_DISTANCE_KM,
    WEATHER_RAW_FIELDS,
    WEATHER_STATIONS,
    LabelConfig,
    PiouObservation,
    WeatherResource,
    cache_weather_resource,
    calendar_features,
    discover_weather_resources,
    distance_km,
    elapsed_minutes,
    group_piou_by_local_day,
    held_minutes,
    is_qualifying,
    iter_cached_weather,
    iter_unique_piou,
    local_boundary,
    piou_features,
    sha256_file,
    target_label,
    validate_label_config,
    weather_features,
    write_dataset,
)


TIMESTEP_DATASET_SCHEMA = "traverse-same-day-timestep-v1"
DEFAULT_START_MINUTES = 6 * 60 + 30
DEFAULT_END_MINUTES = 20 * 60
DEFAULT_STEP_MINUTES = 30


def parse_clock(value: str) -> int:
    try:
        parsed = datetime.strptime(value, "%H:%M")
    except ValueError as error:
        raise argparse.ArgumentTypeError("Times must use HH:MM") from error
    return parsed.hour * 60 + parsed.minute


def format_clock(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def prediction_grid(start: int, end: int, step: int) -> tuple[int, ...]:
    if not 0 <= start < end <= 24 * 60:
        raise ValueError("Require 00:00 <= prediction start < end <= 24:00")
    if step <= 0 or step > 24 * 60:
        raise ValueError("prediction step must be between 1 and 1440 minutes")
    return tuple(range(start, end, step))


def cutoff_for_minutes(
    local_day: date, issue_minutes: int, timezone_local: ZoneInfo
) -> datetime:
    hour, minute = divmod(issue_minutes, 60)
    if hour == 24:
        return datetime.combine(local_day + timedelta(days=1), time(0), timezone_local)
    return datetime.combine(local_day, time(hour, minute), timezone_local)


def issue_time_features(issue_minutes: int) -> dict[str, float]:
    angle = 2.0 * math.pi * issue_minutes / (24.0 * 60.0)
    return {
        "cal_issue_time_sin": math.sin(angle),
        "cal_issue_time_cos": math.cos(angle),
    }


def historical_traverse_features(
    local_day: date, previous_labels: dict[date, int]
) -> dict[str, float]:
    """Summarize Traverse outcomes from dates before ``local_day``."""
    output: dict[str, float] = {}
    yesterday = previous_labels.get(local_day - timedelta(days=1))
    output["lag_traverse_1d"] = (
        float(yesterday) if yesterday is not None else float("nan")
    )
    for days in (3, 7, 14):
        values = [
            previous_labels[candidate]
            for offset in range(1, days + 1)
            if (candidate := local_day - timedelta(days=offset)) in previous_labels
        ]
        output[f"lag_traverse_rate_{days}d"] = (
            float(np.mean(values)) if values else float("nan")
        )
        output[f"lag_known_days_{days}d"] = float(len(values))

    positive_dates = [day for day, label in previous_labels.items() if label == 1]
    output["lag_days_since_traverse"] = (
        float((local_day - max(positive_dates)).days)
        if positive_dates
        else float("nan")
    )
    day_of_year = local_day.timetuple().tm_yday
    similar = [
        label
        for day, label in previous_labels.items()
        if day.year < local_day.year
        and min(
            abs(day.timetuple().tm_yday - day_of_year),
            366 - abs(day.timetuple().tm_yday - day_of_year),
        )
        <= 14
    ]
    output["lag_similar_doy_traverse_rate"] = (
        float(np.mean(similar)) if similar else float("nan")
    )
    output["lag_similar_doy_known_days"] = float(len(similar))
    return output


def traverse_progress_features(
    local_day: date,
    observations: Sequence[PiouObservation],
    config: LabelConfig,
    cutoff: datetime,
) -> dict[str, float]:
    """Describe qualifying observations already seen, strictly before cutoff."""
    timezone_local = ZoneInfo(config.timezone_name)
    target_start = local_boundary(local_day, config.cutoff_hour, timezone_local)
    target_end = local_boundary(local_day, config.target_end_hour, timezone_local)
    effective_end = min(cutoff, target_end)
    if effective_end <= target_start:
        return {
            "piou_traverse_observations_so_far": 0.0,
            "piou_traverse_qualifying_minutes_so_far": 0.0,
            "piou_traverse_longest_run_so_far": 0.0,
            "piou_traverse_observed_so_far": 0.0,
        }
    start_utc = target_start.astimezone(timezone.utc)
    end_utc = effective_end.astimezone(timezone.utc)
    observed = [
        item
        for item in observations
        if start_utc <= item.timestamp_utc < end_utc
    ]
    qualifying_minutes = 0.0
    longest_run = 0
    current_run = 0
    previous_qualifying_time: datetime | None = None
    for index, observation in enumerate(observed):
        if is_qualifying(observation, config):
            gap = (
                elapsed_minutes(
                    observation.timestamp_local, previous_qualifying_time
                )
                if previous_qualifying_time is not None
                else math.inf
            )
            current_run = (
                current_run + 1
                if gap <= config.maximum_consecutive_gap_minutes
                else 1
            )
            longest_run = max(longest_run, current_run)
            previous_qualifying_time = observation.timestamp_local
            qualifying_minutes += held_minutes(
                observed, index, effective_end, config.sample_hold_cap_minutes
            )
        else:
            current_run = 0
            previous_qualifying_time = None
    event_seen = (
        qualifying_minutes >= config.minimum_cumulative_minutes
        and longest_run >= config.minimum_consecutive_samples
    )
    return {
        "piou_traverse_observations_so_far": float(len(observed)),
        "piou_traverse_qualifying_minutes_so_far": qualifying_minutes,
        "piou_traverse_longest_run_so_far": float(longest_run),
        "piou_traverse_observed_so_far": float(event_seen),
    }


def build_timestep_piou_rows(
    input_dir: Path,
    config: LabelConfig,
    issue_minutes_grid: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    timezone_local = ZoneInfo(config.timezone_name)
    iterator, counters = iter_unique_piou(input_dir, timezone_local)
    counters.update(
        {
            "candidate_days": 0,
            "unknown_label_days": 0,
            "candidate_rows": 0,
            "stale_or_missing_feature_rows": 0,
            "usable_rows": 0,
            "positive_rows": 0,
        }
    )
    rows: list[dict[str, Any]] = []
    previous_labels: dict[date, int] = {}
    for local_day, observations in group_piou_by_local_day(iterator):
        counters["candidate_days"] += 1
        daily_target = target_label(local_day, observations, config)
        if daily_target is None:
            counters["unknown_label_days"] += 1
            continue
        history = historical_traverse_features(local_day, previous_labels)
        for issue_minutes in issue_minutes_grid:
            counters["candidate_rows"] += 1
            cutoff = cutoff_for_minutes(local_day, issue_minutes, timezone_local)
            features = piou_features(
                local_day, observations, config, cutoff_local=cutoff
            )
            if features is None:
                counters["stale_or_missing_feature_rows"] += 1
                continue
            row = {
                **calendar_features(local_day),
                "issue_minutes": issue_minutes,
                **issue_time_features(issue_minutes),
                **history,
                **features,
                **traverse_progress_features(
                    local_day, observations, config, cutoff
                ),
                **daily_target,
            }
            rows.append(row)
            counters["usable_rows"] += 1
            counters["positive_rows"] += int(row["label"])
        previous_labels[local_day] = int(daily_target["label"])
    return rows, counters


def build_primary_weather_timeline(
    cache_dir: Path,
    config: LabelConfig,
    start_year: int,
    end_year: int,
    issue_minutes_grid: tuple[int, ...],
    refresh: bool,
    offline: bool,
) -> tuple[
    dict[tuple[date, int], dict[str, float]],
    list[WeatherResource],
    dict[str, dict[str, int]],
    dict[str, str],
]:
    resources = discover_weather_resources(
        cache_dir,
        start_year,
        end_year,
        offline,
        departments=(PRIMARY_WEATHER_STATION.department,),
    )
    paths: list[Path] = []
    resource_rows: dict[str, dict[str, int]] = {}
    cache_sha256: dict[str, str] = {}
    department_stations = [
        station
        for station in WEATHER_STATIONS
        if station.department == PRIMARY_WEATHER_STATION.department
    ]
    for resource in resources:
        path, counts = cache_weather_resource(
            cache_dir,
            resource,
            [station.station_id for station in department_stations],
            refresh,
            offline,
        )
        resource_rows[resource.resource_id] = counts
        cache_sha256[resource.resource_id] = sha256_file(path)
        if counts.get(PRIMARY_WEATHER_STATION.station_id, 0) > 0:
            paths.append(path)

    timezone_local = ZoneInfo(config.timezone_name)
    grouped: dict[date, list[dict[str, Any]]] = defaultdict(list)
    seen: dict[datetime, dict[str, Any]] = {}
    maximum_minutes = max(issue_minutes_grid)
    for observation in iter_cached_weather(
        paths, timezone_local, PRIMARY_WEATHER_STATION
    ):
        local_timestamp = observation["timestamp_local"]
        if not start_year <= local_timestamp.year <= end_year:
            continue
        issue_end = cutoff_for_minutes(
            local_timestamp.date(), maximum_minutes, timezone_local
        )
        weather_start = local_boundary(
            local_timestamp.date(), config.weather_morning_start_hour, timezone_local
        )
        if not weather_start <= local_timestamp < issue_end:
            continue
        timestamp_utc = observation["timestamp_utc"]
        previous = seen.get(timestamp_utc)
        if previous is not None:
            conflicts = [
                field
                for field in WEATHER_RAW_FIELDS
                if not (
                    np.isnan(previous[field]) and np.isnan(observation[field])
                )
                and previous[field] != observation[field]
            ]
            if conflicts:
                raise ValueError(
                    f"Conflicting Météo-France rows at {timestamp_utc.isoformat()}: "
                    f"{conflicts}"
                )
            continue
        seen[timestamp_utc] = observation
        grouped[local_timestamp.date()].append(observation)

    timeline: dict[tuple[date, int], dict[str, float]] = {}
    for local_day, values in grouped.items():
        ordered = sorted(values, key=lambda item: item["timestamp_local"])
        for issue_minutes in issue_minutes_grid:
            cutoff = cutoff_for_minutes(local_day, issue_minutes, timezone_local)
            available = [
                item for item in ordered if item["timestamp_local"] < cutoff
            ]
            if available:
                timeline[(local_day, issue_minutes)] = weather_features(
                    available,
                    cutoff=cutoff,
                    maximum_age_minutes=config.maximum_weather_feature_age_minutes,
                )
    return timeline, resources, resource_rows, cache_sha256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("pioudata"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/traverse_timestep.csv")
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("artifacts/traverse_timestep.metadata.json"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("pioudata/.weather_cache"))
    parser.add_argument("--prediction-start", type=parse_clock, default=DEFAULT_START_MINUTES)
    parser.add_argument("--prediction-end", type=parse_clock, default=DEFAULT_END_MINUTES)
    parser.add_argument("--prediction-step-minutes", type=int, default=DEFAULT_STEP_MINUTES)
    parser.add_argument("--timezone", default="Europe/Paris")
    parser.add_argument("--target-start-hour", type=int, default=12)
    parser.add_argument("--target-end-hour", type=int, default=20)
    parser.add_argument("--speed-threshold-kmh", type=float, default=18.52)
    parser.add_argument("--heading-min-degrees", type=float, default=225.0)
    parser.add_argument("--heading-max-degrees", type=float, default=315.0)
    parser.add_argument("--minimum-cumulative-minutes", type=float, default=30.0)
    parser.add_argument("--minimum-consecutive-samples", type=int, default=3)
    parser.add_argument("--maximum-consecutive-gap-minutes", type=float, default=10.0)
    parser.add_argument("--sample-hold-cap-minutes", type=float, default=5.0)
    parser.add_argument("--minimum-target-coverage", type=float, default=0.75)
    parser.add_argument("--piou-feature-start-hour", type=int, default=6)
    parser.add_argument("--maximum-feature-age-minutes", type=float, default=30.0)
    parser.add_argument("--weather-feature-start-hour", type=int, default=6)
    parser.add_argument("--maximum-weather-feature-age-minutes", type=float, default=90.0)
    parser.add_argument("--refresh-weather", action="store_true")
    parser.add_argument("--offline", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.refresh_weather and args.offline:
        raise SystemExit("--refresh-weather and --offline are mutually exclusive")
    try:
        issue_minutes_grid = prediction_grid(
            args.prediction_start,
            args.prediction_end,
            args.prediction_step_minutes,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    config = LabelConfig(
        timezone_name=args.timezone,
        cutoff_hour=args.target_start_hour,
        target_end_hour=args.target_end_hour,
        speed_threshold_kmh=args.speed_threshold_kmh,
        heading_min_degrees=args.heading_min_degrees,
        heading_max_degrees=args.heading_max_degrees,
        minimum_cumulative_minutes=args.minimum_cumulative_minutes,
        minimum_consecutive_samples=args.minimum_consecutive_samples,
        maximum_consecutive_gap_minutes=args.maximum_consecutive_gap_minutes,
        sample_hold_cap_minutes=args.sample_hold_cap_minutes,
        minimum_target_coverage=args.minimum_target_coverage,
        piou_morning_start_hour=args.piou_feature_start_hour,
        maximum_feature_age_minutes=args.maximum_feature_age_minutes,
        weather_morning_start_hour=args.weather_feature_start_hour,
        maximum_weather_feature_age_minutes=args.maximum_weather_feature_age_minutes,
    )
    try:
        validate_label_config(config)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    piou_rows, counters = build_timestep_piou_rows(
        args.input_dir, config, issue_minutes_grid
    )
    if not piou_rows:
        raise SystemExit("No usable PiouPiou timestep rows were produced")
    start_year = min(int(row["year"]) for row in piou_rows)
    end_year = max(int(row["year"]) for row in piou_rows)
    weather, resources, cache_counts, cache_sha256 = build_primary_weather_timeline(
        args.cache_dir,
        config,
        start_year,
        end_year,
        issue_minutes_grid,
        args.refresh_weather,
        args.offline,
    )

    joined: list[dict[str, Any]] = []
    missing_weather_rows = 0
    stale_weather_rows = 0
    airport_distance_km = round(
        distance_km(
            PIOU_LATITUDE,
            PIOU_LONGITUDE,
            PRIMARY_WEATHER_STATION.latitude,
            PRIMARY_WEATHER_STATION.longitude,
        ),
        2,
    )
    for row in piou_rows:
        key = (date.fromisoformat(str(row["date"])), int(row["issue_minutes"]))
        airport = weather.get(key)
        if airport is None:
            missing_weather_rows += 1
            continue
        age = float(airport.get("mf_last_age_minutes", float("nan")))
        if not np.isfinite(age) or age > config.maximum_weather_feature_age_minutes:
            stale_weather_rows += 1
            continue
        joined.append(
            {
                **row,
                **airport,
                "meta_weather_station_id": PRIMARY_WEATHER_STATION.station_id,
                "meta_weather_station_distance_km": airport_distance_km,
            }
        )
    if not joined:
        raise SystemExit("No timestep rows have fresh Météo-France features")
    joined.sort(key=lambda row: (str(row["date"]), int(row["issue_minutes"])))
    fields = write_dataset(joined, args.output)
    prediction_times = [format_clock(value) for value in issue_minutes_grid]
    metadata = {
        "dataset_schema": TIMESTEP_DATASET_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "row_count": len(joined),
        "columns": fields,
        "date_range": [joined[0]["date"], joined[-1]["date"]],
        "prediction_times": prediction_times,
        "prediction_step_minutes": args.prediction_step_minutes,
        "prediction_window_minutes": [args.prediction_start, args.prediction_end],
        "target_semantics": (
            "qualifying Traverse during the same local day's fixed "
            "[target_start_hour, target_end_hour) window"
        ),
        "history_policy": (
            "history uses only fully observed dates before the forecast date"
        ),
        "label_config": config.__dict__,
        "pioupiou": {
            "input_dir": str(args.input_dir),
            "archive_api": PIOU_ARCHIVE_API,
            "station_id": PIOU_STATION_ID,
            "source_schema": PIOU_SOURCE_SCHEMA,
            "expected_coordinates": [PIOU_LATITUDE, PIOU_LONGITUDE],
            "maximum_location_distance_km": PIOU_MAX_LOCATION_DISTANCE_KM,
            "counters": counters,
        },
        "meteofrance": {
            "dataset_id": DATA_GOUV_DATASET_ID,
            "dataset_api_url": DATA_GOUV_API,
            "station_id": PRIMARY_WEATHER_STATION.station_id,
            "station_name": PRIMARY_WEATHER_STATION.name,
            "source_schema": METEO_FRANCE_SOURCE_SCHEMA,
            "station_coordinates": [
                PRIMARY_WEATHER_STATION.latitude,
                PRIMARY_WEATHER_STATION.longitude,
            ],
            "maximum_location_distance_km": WEATHER_MAX_LOCATION_DISTANCE_KM,
            "resource_rows": cache_counts,
            "resource_cache_sha256": cache_sha256,
            "resources": [resource.__dict__ for resource in resources],
            "missing_weather_rows": missing_weather_rows,
            "stale_weather_rows": stale_weather_rows,
        },
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "metadata": str(args.metadata_output),
                "rows": len(joined),
                "prediction_times": len(prediction_times),
                "first_prediction_time": prediction_times[0],
                "last_prediction_time": prediction_times[-1],
                "positive_rate": sum(int(row["label"]) for row in joined) / len(joined),
                "missing_weather_rows": missing_weather_rows,
                "stale_weather_rows": stale_weather_rows,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
