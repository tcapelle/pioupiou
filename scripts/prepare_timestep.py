#!/usr/bin/env python3
"""Prepare one leakage-safe same-day Traverse row at an arbitrary local time."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from pioupiou.data.timestep import (
    DEFAULT_END_MINUTES,
    DEFAULT_START_MINUTES,
    build_weather_timeline,
    cutoff_for_minutes,
    issue_time_features,
    parse_clock,
    weather_feed_ages,
)
from pioupiou.data.daily import (
    calendar_features,
    deduplicate_piou_observations,
    is_traverse_season,
    label_config_from_payload,
    piou_features,
    read_month,
)
from pioupiou.inference.model import load_artifact_with_sha256


def write_feature_row(row: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    leading = [name for name in ("date", "year", "issue_minutes") if name in row]
    fields = leading + sorted(set(row).difference(leading))
    temporary = output.with_suffix(output.suffix + ".part")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                name: ""
                if isinstance(row.get(name), float) and not math.isfinite(row[name])
                else row.get(name, "")
                for name in fields
            }
        )
    temporary.replace(output)


def bind_to_model_schema(
    prepared: dict[str, object], feature_names: list[str]
) -> dict[str, object]:
    """Return identity fields and the exact ordered model feature schema."""
    missing = sorted({"date", "year", *feature_names}.difference(prepared))
    if missing:
        raise ValueError(f"Cannot construct trained model features: {missing}")
    return {
        "date": prepared["date"],
        "year": prepared["year"],
        **{name: prepared[name] for name in feature_names},
    }


def load_piou_day(input_dir: Path, local_day: date, timezone_name: str) -> list:
    path = input_dir / f"{local_day:%Y-%m}.csv"
    if not path.exists():
        raise FileNotFoundError(f"PiouPiou month file not found: {path}")
    timezone_local = ZoneInfo(timezone_name)
    return deduplicate_piou_observations(
        item
        for item in read_month(path, timezone_local)
        if item.timestamp_local.date() == local_day
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Local date in YYYY-MM-DD")
    parser.add_argument("--time", required=True, type=parse_clock, help="Local HH:MM")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/traverse_model.joblib"),
    )
    parser.add_argument("--piou-input-dir", type=Path, default=Path("pioudata"))
    parser.add_argument("--cache-dir", type=Path, default=Path("pioudata/.weather_cache"))
    parser.add_argument("--refresh-weather", action="store_true")
    parser.add_argument("--offline-weather", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/timestep_features.csv")
    )
    args = parser.parse_args()
    if args.refresh_weather and args.offline_weather:
        raise SystemExit("--refresh-weather and --offline-weather are mutually exclusive")

    try:
        payload, _, _ = load_artifact_with_sha256(args.model)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    try:
        config = label_config_from_payload(payload.get("label"))
    except ValueError as error:
        raise SystemExit(f"Model has an invalid label contract: {error}") from error
    local_day = date.fromisoformat(args.date)
    if not is_traverse_season(local_day, config):
        raise SystemExit(
            f"Requested date is outside the model's May-September season: {local_day}"
        )
    issue_minutes = int(args.time)
    if not DEFAULT_START_MINUTES <= issue_minutes < DEFAULT_END_MINUTES:
        raise SystemExit(
            f"Requested time is outside [{DEFAULT_START_MINUTES}, "
            f"{DEFAULT_END_MINUTES}) minutes"
        )
    timezone_local = ZoneInfo(config.timezone_name)
    cutoff = cutoff_for_minutes(local_day, issue_minutes, timezone_local)
    if cutoff.astimezone(timezone.utc) > datetime.now(timezone.utc):
        raise SystemExit("Requested feature cutoff has not occurred yet")

    observations = load_piou_day(
        args.piou_input_dir, local_day, config.timezone_name
    )
    if not observations:
        raise SystemExit(f"No PiouPiou observations found for {local_day}")
    piou = piou_features(
        local_day, observations, config, cutoff_local=cutoff
    )
    if piou is None:
        raise SystemExit("insufficient_data: missing or stale PiouPiou observations")
    weather = build_weather_timeline(
        args.cache_dir,
        config,
        local_day.year,
        local_day.year,
        (issue_minutes,),
        args.refresh_weather,
        args.offline_weather,
    )
    station_weather = weather.get((local_day, issue_minutes))
    if station_weather is None:
        raise SystemExit("insufficient_data: no Météo-France observations")
    weather_ages = weather_feed_ages(station_weather)
    for station, weather_age in weather_ages.items():
        if (
            not math.isfinite(weather_age)
            or weather_age > config.maximum_weather_feature_age_minutes
        ):
            raise SystemExit(
                f"insufficient_data: {station} Météo-France feed age is "
                f"{weather_age!r} minutes"
            )

    prepared = {
        **calendar_features(local_day),
        "issue_minutes": issue_minutes,
        **issue_time_features(issue_minutes),
        **piou,
        **station_weather,
    }
    try:
        row = bind_to_model_schema(prepared, list(payload["feature_names"]))
    except ValueError as error:
        raise SystemExit(str(error)) from error
    row["issue_minutes"] = issue_minutes
    write_feature_row(row, args.output)
    print(
        json.dumps(
            {
                "date": local_day.isoformat(),
                "time": f"{issue_minutes // 60:02d}:{issue_minutes % 60:02d}",
                "output": str(args.output),
                "piou_last_age_minutes": piou["piou_last_age_minutes"],
                "weather_last_age_minutes": weather_ages,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
