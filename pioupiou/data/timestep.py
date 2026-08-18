"""Build same-day Traverse predictions on a dense local-time grid."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np

from pioupiou.data.daily import (
    BELLEY_WEATHER_STATION,
    MONT_DU_CHAT_WEATHER_STATION,
    NOVALAISE_WEATHER_STATION,
    PRIMARY_WEATHER_STATION,
    WEATHER_STATIONS,
    LabelConfig,
    PiouObservation,
    WeatherStation,
    cache_daily_weather_resource,
    cache_weather_resource,
    calendar_features,
    discover_daily_weather_resources,
    discover_weather_resources,
    elapsed_minutes,
    group_piou_by_local_day,
    held_minutes,
    is_qualifying,
    iter_cached_daily_max_temperature,
    iter_cached_weather,
    iter_unique_piou,
    local_boundary,
    monthly_files,
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


def traverse_progress_features(
    local_day: date,
    observations: Sequence[PiouObservation],
    config: LabelConfig,
    cutoff: datetime,
) -> dict[str, float]:
    """Describe candidate wind-event evidence seen strictly before cutoff."""
    timezone_local = ZoneInfo(config.timezone_name)
    target_start = local_boundary(local_day, config.cutoff_hour, timezone_local)
    target_end = local_boundary(local_day, config.target_end_hour, timezone_local)
    effective_end = min(cutoff, target_end)
    if effective_end <= target_start:
        return {
            "piou_wind_event_observations_so_far": 0.0,
            "piou_wind_event_qualifying_minutes_so_far": 0.0,
            "piou_wind_event_longest_run_so_far": 0.0,
            "piou_wind_event_observed_so_far": 0.0,
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
        "piou_wind_event_observations_so_far": float(len(observed)),
        "piou_wind_event_qualifying_minutes_so_far": qualifying_minutes,
        "piou_wind_event_longest_run_so_far": float(longest_run),
        "piou_wind_event_observed_so_far": float(event_seen),
    }


def build_timestep_piou_rows(
    input_dir: Path,
    config: LabelConfig,
    issue_minutes_grid: tuple[int, ...],
    daily_max_temperatures: Mapping[date, float],
) -> list[dict[str, Any]]:
    timezone_local = ZoneInfo(config.timezone_name)
    iterator, _ = iter_unique_piou(input_dir, timezone_local)
    rows: list[dict[str, Any]] = []
    for local_day, observations in group_piou_by_local_day(iterator):
        daily_target = target_label(
            local_day,
            observations,
            config,
            daily_max_temperatures.get(local_day, float("nan")),
        )
        if daily_target is None:
            continue
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
                    **features,
                    **traverse_progress_features(
                        local_day, observations, config, cutoff
                    ),
                    **daily_target,
                }
            )
    return rows


def weather_prefix(station: WeatherStation) -> str:
    return "mf" if station == PRIMARY_WEATHER_STATION else f"mf_{station.slug}"


def weather_feed_ages(features: Mapping[str, float]) -> dict[str, float]:
    return {
        station.slug: float(
            features.get(
                f"{weather_prefix(station)}_last_age_minutes", float("nan")
            )
        )
        for station in WEATHER_STATIONS
    }


def station_weather_features(
    station: WeatherStation,
    observations: Sequence[dict[str, Any]],
    cutoff: datetime,
    config: LabelConfig,
) -> dict[str, float]:
    prefix = weather_prefix(station)
    features = weather_features(
        observations,
        cutoff=cutoff,
        maximum_age_minutes=config.maximum_weather_feature_age_minutes,
        prefix=prefix,
        freshness_fields=(
            ("T",)
            if station != PRIMARY_WEATHER_STATION
            else ("T", "U", "FF", "DD")
        ),
    )
    if station == PRIMARY_WEATHER_STATION:
        return features
    keep = {
        f"{prefix}_observation_count_morning",
        f"{prefix}_core_observation_count_morning",
        f"{prefix}_last_age_minutes",
        f"{prefix}_temperature_c_latest",
        f"{prefix}_temperature_c_mean",
        f"{prefix}_temperature_c_delta_morning",
    }
    return {name: value for name, value in features.items() if name in keep}


def temperature_contrast_features(features: Mapping[str, float]) -> dict[str, float]:
    pairs = (
        (BELLEY_WEATHER_STATION, PRIMARY_WEATHER_STATION),
        (NOVALAISE_WEATHER_STATION, PRIMARY_WEATHER_STATION),
        (BELLEY_WEATHER_STATION, MONT_DU_CHAT_WEATHER_STATION),
    )
    output: dict[str, float] = {}
    for left, right in pairs:
        left_prefix = weather_prefix(left)
        right_prefix = weather_prefix(right)
        for suffix in ("latest", "mean", "delta_morning"):
            left_name = f"{left_prefix}_temperature_c_{suffix}"
            right_name = f"{right_prefix}_temperature_c_{suffix}"
            left_value = float(features.get(left_name, float("nan")))
            right_value = float(features.get(right_name, float("nan")))
            output[
                f"mf_contrast_{left.slug}_minus_{right.slug}_temperature_c_{suffix}"
            ] = (
                left_value - right_value
                if np.isfinite(left_value) and np.isfinite(right_value)
                else float("nan")
            )
    return output


def build_weather_timeline(
    cache_dir: Path,
    config: LabelConfig,
    start_year: int,
    end_year: int,
    issue_minutes_grid: tuple[int, ...],
    refresh: bool,
    offline: bool,
) -> dict[tuple[date, int], dict[str, float]]:
    departments = tuple(sorted({station.department for station in WEATHER_STATIONS}))
    resources = discover_weather_resources(
        cache_dir,
        start_year,
        end_year,
        offline,
        departments=departments,
    )
    paths_by_station: dict[str, list[Path]] = {
        station.station_id: [] for station in WEATHER_STATIONS
    }
    for resource in resources:
        stations = [
            station
            for station in WEATHER_STATIONS
            if station.department == resource.department
        ]
        path, counts = cache_weather_resource(
            cache_dir,
            resource,
            [station.station_id for station in stations],
            refresh,
            offline,
        )
        for station in stations:
            if counts.get(station.station_id, 0) > 0:
                paths_by_station[station.station_id].append(path)

    timezone_local = ZoneInfo(config.timezone_name)
    maximum_minutes = max(issue_minutes_grid)
    combined: dict[tuple[date, int], dict[str, float]] | None = None
    for station in WEATHER_STATIONS:
        grouped: dict[date, dict[datetime, dict[str, Any]]] = defaultdict(dict)
        for observation in iter_cached_weather(
            paths_by_station[station.station_id], timezone_local, station
        ):
            local_timestamp = observation["timestamp_local"]
            if not start_year <= local_timestamp.year <= end_year:
                continue
            issue_end = cutoff_for_minutes(
                local_timestamp.date(), maximum_minutes, timezone_local
            )
            weather_start = local_boundary(
                local_timestamp.date(),
                config.weather_morning_start_hour,
                timezone_local,
            )
            if weather_start <= local_timestamp < issue_end:
                grouped[local_timestamp.date()].setdefault(
                    observation["timestamp_utc"], observation
                )
        station_timeline: dict[tuple[date, int], dict[str, float]] = {}
        for local_day, values in grouped.items():
            ordered = sorted(
                values.values(), key=lambda item: item["timestamp_local"]
            )
            for issue_minutes in issue_minutes_grid:
                cutoff = cutoff_for_minutes(
                    local_day, issue_minutes, timezone_local
                )
                available = [
                    item for item in ordered if item["timestamp_local"] < cutoff
                ]
                if available:
                    station_timeline[(local_day, issue_minutes)] = (
                        station_weather_features(
                            station, available, cutoff, config
                        )
                    )
        combined = (
            station_timeline
            if combined is None
            else {
                key: {**features, **station_timeline[key]}
                for key, features in combined.items()
                if key in station_timeline
            }
        )

    assert combined is not None
    for features in combined.values():
        features.update(temperature_contrast_features(features))
    return combined


def build_daily_max_temperatures(
    cache_dir: Path,
    config: LabelConfig,
    start_year: int,
    end_year: int,
    refresh: bool,
    offline: bool,
) -> dict[date, float]:
    station = next(
        (
            item
            for item in WEATHER_STATIONS
            if item.station_id == config.hot_day_station_id
        ),
        None,
    )
    if station is None:
        raise ValueError(
            f"Unknown hot-day station {config.hot_day_station_id!r}"
        )
    resources = discover_daily_weather_resources(
        cache_dir,
        start_year,
        end_year,
        offline,
        departments=(station.department,),
    )
    paths = [
        cache_daily_weather_resource(
            cache_dir, resource, station, refresh, offline
        )
        for resource in resources
    ]
    return {
        local_day: temperature
        for local_day, temperature in iter_cached_daily_max_temperature(
            paths, station
        )
        if start_year <= local_day.year <= end_year
    }


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
    source_files = monthly_files(args.input_dir)
    if not source_files:
        raise SystemExit("No monthly PiouPiou files were found")
    source_years = [int(path.stem[:4]) for path in source_files]
    start_year = min(source_years)
    end_year = max(source_years)
    daily_max_temperatures = build_daily_max_temperatures(
        args.cache_dir,
        config,
        start_year,
        end_year,
        args.refresh_weather,
        args.offline,
    )
    piou_rows = build_timestep_piou_rows(
        args.input_dir,
        config,
        PREDICTION_MINUTES,
        daily_max_temperatures,
    )
    if not piou_rows:
        raise SystemExit("No usable PiouPiou timestep rows were produced")
    start_year = min(int(row["year"]) for row in piou_rows)
    end_year = max(int(row["year"]) for row in piou_rows)
    weather = build_weather_timeline(
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
        station_weather = weather.get(key)
        if station_weather is None:
            continue
        ages = weather_feed_ages(station_weather).values()
        if any(
            not np.isfinite(age)
            or age > config.maximum_weather_feature_age_minutes
            for age in ages
        ):
            continue
        joined.append({**row, **station_weather})
    if not joined:
        raise SystemExit("No timestep rows have fresh Météo-France features")
    joined.sort(key=lambda row: (str(row["date"]), int(row["issue_minutes"])))
    write_dataset(joined, args.output)
    metadata = {
        "output_sha256": sha256_file(args.output),
        "row_count": len(joined),
        "date_range": [joined[0]["date"], joined[-1]["date"]],
        "label_config": config.__dict__,
        "weather_stations": [station.__dict__ for station in WEATHER_STATIONS],
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
