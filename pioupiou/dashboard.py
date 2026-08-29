"""Prepare the historical Traverse data used by the local dashboard."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, time, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from pioupiou.data.daily import (
    PRIMARY_WEATHER_STATION,
    LabelConfig,
    group_piou_by_local_day,
    iter_cached_weather,
    iter_unique_piou,
    monthly_files,
    target_label,
)
from pioupiou.data.timestep import build_daily_max_temperatures


KMH_PER_KNOT = 1.852
DASHBOARD_START_MINUTES = 12 * 60
DASHBOARD_END_MINUTES = 21 * 60


def kmh_to_knots(value: float) -> float:
    return value / KMH_PER_KNOT


def serialize_wind_day(
    local_day, observations, label: dict[str, float | int]
) -> dict[str, Any]:
    return {
        "date": local_day.isoformat(),
        "year": local_day.year,
        "traverse": bool(label["label"]),
        "sustained_minutes": round(
            float(label["meta_target_longest_qualifying_run_minutes"]), 1
        ),
        "coverage": round(
            100.0 * float(label["meta_target_coverage_fraction"]), 1
        ),
        "peak_gust_knots": round(
            max(kmh_to_knots(item.wind_speed_max_kmh) for item in observations), 1
        ),
        "observations": [
            {
                "time": item.timestamp_local.isoformat(),
                "min": round(kmh_to_knots(item.wind_speed_min_kmh), 2),
                "avg": round(kmh_to_knots(item.wind_speed_avg_kmh), 2),
                "max": round(kmh_to_knots(item.wind_speed_max_kmh), 2),
                "direction": round(item.wind_heading_degrees, 1),
            }
            for item in observations
        ],
    }


def load_predictions(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    labels: dict[str, bool] = {}
    event_onsets: dict[str, float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            day = row["date"]
            labels[day] = bool(int(row["label"]))
            onset = row.get("event_onset_minutes", "")
            if onset:
                event_onsets[day] = round(float(onset), 3)
            point = {
                "issue_minutes": int(row["issue_minutes"]),
                "probability": round(float(row["traverse_probability"]), 6),
                "onset_evidence": round(float(row["onset_evidence"]), 6),
            }
            for horizon in (60, 120, 180):
                column = f"probability_onset_within_{horizon}m"
                if row.get(column):
                    point[horizon] = round(float(row[column]), 6)
            grouped[day].append(point)
    metadata_path = path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    return {
        "days": [
            {
                "date": day,
                "label": labels[day],
                "event_onset_minutes": event_onsets.get(day),
                "predictions": grouped[day],
            }
            for day in sorted(grouped)
        ],
        "metadata": metadata,
    }


def load_weather_series(
    cache_dir: Path, dates: set[str], local_timezone: ZoneInfo
) -> dict[str, list[dict[str, Any]]]:
    pattern = f"meteofrance_{PRIMARY_WEATHER_STATION.station_id}_*.csv.gz"
    paths = sorted(cache_dir.glob(pattern))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in iter_cached_weather(
        paths, local_timezone, PRIMARY_WEATHER_STATION
    ):
        timestamp = observation["timestamp_local"]
        day = timestamp.date().isoformat()
        minutes = timestamp.hour * 60 + timestamp.minute
        if (
            day not in dates
            or not DASHBOARD_START_MINUTES <= minutes < DASHBOARD_END_MINUTES
        ):
            continue
        temperature = float(observation["T"])
        cloud_cover = float(observation["N"])
        global_radiation = float(observation["GLO"])
        if not any(
            math.isfinite(value)
            for value in (temperature, cloud_cover, global_radiation)
        ):
            continue
        grouped[day].append(
            {
                "time": timestamp.isoformat(),
                "temperature_c": (
                    round(temperature, 1) if math.isfinite(temperature) else None
                ),
                "cloud_cover_oktas": (
                    round(cloud_cover, 1) if math.isfinite(cloud_cover) else None
                ),
                "global_radiation_wm2": (
                    round(global_radiation * 10000.0 / 3600.0, 1)
                    if math.isfinite(global_radiation)
                    else None
                ),
            }
        )
    return grouped


@lru_cache(maxsize=1)
def build_dashboard_data(
    input_dir: Path = Path("pioudata"),
    predictions_path: Path = Path("artifacts/traverse_predictions_2026.csv"),
    weather_cache_dir: Path = Path("pioudata/.weather_cache"),
) -> dict[str, Any]:
    """Return annual event counts and each event's target-window observations."""
    config = LabelConfig()
    local_timezone = ZoneInfo(config.timezone_name)
    predictions = load_predictions(predictions_path)
    prediction_dates = (
        {item["date"] for item in predictions["days"]} if predictions else set()
    )
    source_years = [int(path.stem[:4]) for path in monthly_files(input_dir)]
    if not source_years:
        raise ValueError(f"No monthly PiouPiou files found in {input_dir}")
    daily_max_temperatures = build_daily_max_temperatures(
        weather_cache_dir,
        config,
        min(source_years),
        max(source_years),
        refresh=False,
        offline=True,
    )
    iterator, counters = iter_unique_piou(input_dir, local_timezone)
    annual: dict[int, dict[str, int]] = defaultdict(
        lambda: {"events": 0, "observed_days": 0}
    )
    events: list[dict[str, Any]] = []
    prediction_wind: dict[str, dict[str, Any]] = {}
    first_observation = None
    latest_observation = None

    for local_day, observations in group_piou_by_local_day(iterator):
        first_observation = first_observation or observations[0].timestamp_local
        latest_observation = observations[-1].timestamp_local
        label = target_label(
            local_day,
            observations,
            config,
            daily_max_temperatures.get(local_day, float("nan")),
        )
        if label is None:
            continue
        annual[local_day.year]["observed_days"] += 1
        day_text = local_day.isoformat()
        if not label["label"] and day_text not in prediction_dates:
            continue
        display_start = datetime.combine(
            local_day, time(12), local_timezone
        ).astimezone(timezone.utc)
        display_end = datetime.combine(
            local_day, time(21), local_timezone
        ).astimezone(timezone.utc)
        displayed = [
            item
            for item in observations
            if display_start <= item.timestamp_utc < display_end
        ]
        wind_day = serialize_wind_day(local_day, displayed, label)
        if day_text in prediction_dates:
            prediction_wind[day_text] = wind_day
        if label["label"]:
            annual[local_day.year]["events"] += 1
            events.append(wind_day)

    if predictions:
        for item in predictions["days"]:
            item["wind"] = prediction_wind[item["date"]]

    wind_days = {item["date"]: item for item in events}
    wind_days.update(prediction_wind)
    weather = load_weather_series(
        weather_cache_dir, set(wind_days), local_timezone
    )
    for day, wind in wind_days.items():
        wind["weather"] = weather.get(day, [])

    years = [
        {"year": year, **annual[year]}
        for year in sorted(annual)
    ]
    return {
        "years": years,
        "events": events,
        "predictions": predictions,
        "metadata": {
            "first_observation": (
                first_observation.isoformat() if first_observation else None
            ),
            "latest_observation": (
                latest_observation.isoformat() if latest_observation else None
            ),
            "timezone": config.timezone_name,
            "window": f"{config.cutoff_hour:02d}:00–{config.target_end_hour:02d}:00",
            "display_window": "12:00–21:00",
            "speed_threshold_knots": round(
                kmh_to_knots(config.speed_threshold_kmh), 1
            ),
            "minimum_coverage_percent": round(
                config.minimum_target_coverage * 100
            ),
            "unique_observations": counters["unique_rows"],
            "weather_station": PRIMARY_WEATHER_STATION.name,
        },
    }
