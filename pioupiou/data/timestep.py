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

from pioupiou.data.daily import (
    PRIMARY_WEATHER_STATION,
    WEATHER_STATIONS,
    LabelConfig,
    PiouObservation,
    cache_weather_resource,
    calendar_features,
    discover_weather_resources,
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
    weather_features,
    write_dataset,
)


DEFAULT_START_MINUTES = 6 * 60 + 30
DEFAULT_END_MINUTES = 20 * 60
DEFAULT_STEP_MINUTES = 30
PREDICTION_MINUTES = tuple(
    range(DEFAULT_START_MINUTES, DEFAULT_END_MINUTES, DEFAULT_STEP_MINUTES)
)


def parse_clock(value: str) -> int:
    try:
        parsed = datetime.strptime(value, "%H:%M")
    except ValueError as error:
        raise argparse.ArgumentTypeError("Times must use HH:MM") from error
    return parsed.hour * 60 + parsed.minute


def cutoff_for_minutes(
    local_day: date, issue_minutes: int, timezone_local: ZoneInfo
) -> datetime:
    hour, minute = divmod(issue_minutes, 60)
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
) -> list[dict[str, Any]]:
    timezone_local = ZoneInfo(config.timezone_name)
    iterator, _ = iter_unique_piou(input_dir, timezone_local)
    rows: list[dict[str, Any]] = []
    previous_labels: dict[date, int] = {}
    for local_day, observations in group_piou_by_local_day(iterator):
        daily_target = target_label(local_day, observations, config)
        if daily_target is None:
            continue
        history = historical_traverse_features(local_day, previous_labels)
        for issue_minutes in issue_minutes_grid:
            cutoff = cutoff_for_minutes(local_day, issue_minutes, timezone_local)
            features = piou_features(
                local_day, observations, config, cutoff_local=cutoff
            )
            if features is None:
                continue
            rows.append(
                {
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
            )
        previous_labels[local_day] = int(daily_target["label"])
    return rows


def build_primary_weather_timeline(
    cache_dir: Path,
    config: LabelConfig,
    start_year: int,
    end_year: int,
    issue_minutes_grid: tuple[int, ...],
    refresh: bool,
    offline: bool,
) -> dict[tuple[date, int], dict[str, float]]:
    resources = discover_weather_resources(
        cache_dir,
        start_year,
        end_year,
        offline,
        departments=(PRIMARY_WEATHER_STATION.department,),
    )
    paths: list[Path] = []
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
        if counts.get(PRIMARY_WEATHER_STATION.station_id, 0) > 0:
            paths.append(path)

    timezone_local = ZoneInfo(config.timezone_name)
    grouped: dict[date, dict[datetime, dict[str, Any]]] = defaultdict(dict)
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
        grouped[local_timestamp.date()].setdefault(
            observation["timestamp_utc"], observation
        )

    timeline: dict[tuple[date, int], dict[str, float]] = {}
    for local_day, values in grouped.items():
        ordered = sorted(values.values(), key=lambda item: item["timestamp_local"])
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
    return timeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("pioudata"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/traverse_timestep.csv")
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("pioudata/.weather_cache"))
    parser.add_argument("--refresh-weather", action="store_true")
    parser.add_argument("--offline", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.refresh_weather and args.offline:
        raise SystemExit("--refresh-weather and --offline are mutually exclusive")
    config = LabelConfig(piou_morning_start_hour=6)
    piou_rows = build_timestep_piou_rows(
        args.input_dir, config, PREDICTION_MINUTES
    )
    if not piou_rows:
        raise SystemExit("No usable PiouPiou timestep rows were produced")
    start_year = min(int(row["year"]) for row in piou_rows)
    end_year = max(int(row["year"]) for row in piou_rows)
    weather = build_primary_weather_timeline(
        args.cache_dir,
        config,
        start_year,
        end_year,
        PREDICTION_MINUTES,
        args.refresh_weather,
        args.offline,
    )

    joined: list[dict[str, Any]] = []
    for row in piou_rows:
        key = (date.fromisoformat(str(row["date"])), int(row["issue_minutes"]))
        airport = weather.get(key)
        if airport is None:
            continue
        age = float(airport.get("mf_last_age_minutes", float("nan")))
        if not np.isfinite(age) or age > config.maximum_weather_feature_age_minutes:
            continue
        joined.append({**row, **airport})
    if not joined:
        raise SystemExit("No timestep rows have fresh Météo-France features")
    joined.sort(key=lambda row: (str(row["date"]), int(row["issue_minutes"])))
    write_dataset(joined, args.output)
    metadata = {
        "output_sha256": sha256_file(args.output),
        "row_count": len(joined),
        "date_range": [joined[0]["date"], joined[-1]["date"]],
        "label_config": config.__dict__,
    }
    metadata_output = args.output.with_suffix(".metadata.json")
    metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "metadata": str(metadata_output),
                "rows": len(joined),
                "positive_rate": sum(int(row["label"]) for row in joined) / len(joined),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
