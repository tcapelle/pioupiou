#!/usr/bin/env python3
"""Prepare one leakage-safe same-day Traverse row at an arbitrary local time."""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from pioupiou.data.timestep import (
    DEFAULT_END_MINUTES,
    DEFAULT_START_MINUTES,
    build_primary_weather_timeline,
    cutoff_for_minutes,
    historical_traverse_features,
    issue_time_features,
    parse_clock,
    traverse_progress_features,
)
from pioupiou.data.daily import (
    calendar_features,
    group_piou_by_local_day,
    iter_unique_piou,
    label_config_from_payload,
    piou_features,
    target_label,
)
from pioupiou.inference.model import load_artifact_with_sha256
from scripts.prepare_noon import bind_to_model_schema, write_feature_row


def load_piou_history_and_day(
    input_dir: Path, local_day: date, config
) -> tuple[dict[date, int], list]:
    iterator, _ = iter_unique_piou(input_dir, ZoneInfo(config.timezone_name))
    previous_labels: dict[date, int] = {}
    requested_observations = []
    for candidate_day, observations in group_piou_by_local_day(iterator):
        if candidate_day < local_day:
            values = target_label(candidate_day, observations, config)
            if values is not None:
                previous_labels[candidate_day] = int(values["label"])
        elif candidate_day == local_day:
            requested_observations = observations
        elif candidate_day > local_day:
            break
    return previous_labels, requested_observations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Local date in YYYY-MM-DD")
    parser.add_argument("--time", required=True, type=parse_clock, help="Local HH:MM")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/traverse_model_same_day.joblib"),
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
    if payload.get("role") != "same_day":
        raise SystemExit("Arbitrary-time preparation requires a same_day model")
    try:
        config = label_config_from_payload(payload.get("label"))
    except ValueError as error:
        raise SystemExit(f"Model has an invalid label contract: {error}") from error
    local_day = date.fromisoformat(args.date)
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

    history_labels, observations = load_piou_history_and_day(
        args.piou_input_dir, local_day, config
    )
    if not observations:
        raise SystemExit(f"No PiouPiou observations found for {local_day}")
    piou = piou_features(
        local_day, observations, config, cutoff_local=cutoff
    )
    if piou is None:
        raise SystemExit("insufficient_data: missing or stale PiouPiou observations")
    weather = build_primary_weather_timeline(
        args.cache_dir,
        config,
        local_day.year,
        local_day.year,
        (issue_minutes,),
        args.refresh_weather,
        args.offline_weather,
    )
    airport = weather.get((local_day, issue_minutes))
    if airport is None:
        raise SystemExit("insufficient_data: no Météo-France observations")
    weather_age = float(airport.get("mf_last_age_minutes", float("nan")))
    if (
        not math.isfinite(weather_age)
        or weather_age > config.maximum_weather_feature_age_minutes
    ):
        raise SystemExit(
            f"insufficient_data: Météo-France feed age is {weather_age!r} minutes"
        )

    prepared = {
        **calendar_features(local_day),
        "issue_minutes": issue_minutes,
        **issue_time_features(issue_minutes),
        **historical_traverse_features(local_day, history_labels),
        **piou,
        **traverse_progress_features(local_day, observations, config, cutoff),
        **airport,
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
                "mf_last_age_minutes": weather_age,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
