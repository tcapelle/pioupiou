#!/usr/bin/env python3
"""Build one leakage-free noon-prediction row per local day.

PiouPiou wind observations provide both the morning predictors and the Traverse
label. Official hourly CHAMBERY-AIX observations are streamed from the public
Meteo-France climatology archives and cached after filtering to one station.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import re
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence
from zoneinfo import ZoneInfo

import numpy as np


DATA_GOUV_DATASET_ID = "6569b4473bedf2e7abad3b72"
DATA_GOUV_API = f"https://www.data.gouv.fr/api/1/datasets/{DATA_GOUV_DATASET_ID}/"
PIOU_ARCHIVE_API = "https://api.pioupiou.fr/v1/archive"
PIOU_SOURCE_SCHEMA = "pioupiou-archive-v1-kmh"
METEO_FRANCE_SOURCE_SCHEMA = "meteofrance-base-hourly-v1"
METEO_FRANCE_STATION_ID = "73329001"
METEO_FRANCE_STATION_NAME = "CHAMBERY-AIX"
MONTHLY_NAME = re.compile(r"^\d{4}-\d{2}\.csv$")
PIOU_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}T")
RESOURCE_PERIOD = re.compile(r"HOR_departement_73_periode_(\d{4})-(\d{4})")
USER_AGENT = "pioupiou-traverse-research/1.0"
QUALITY_ACCEPTED = {"0", "1", "9"}
WEATHER_RAW_FIELDS = (
    "T",
    "TD",
    "U",
    "RR1",
    "PMER",
    "PSTAT",
    "N",
    "VV",
    "GLO",
    "DIR",
    "DIF",
    "INS",
    "FF",
    "DD",
)
WEATHER_FRESHNESS_FIELDS = ("T", "U", "FF", "DD")
WEATHER_CACHE_FIELDS = (
    "NUM_POSTE",
    "NOM_USUEL",
    "LAT",
    "LON",
    "ALTI",
    "AAAAMMJJHH",
) + tuple(item for field in WEATHER_RAW_FIELDS for item in (field, f"Q{field}"))


@dataclass(frozen=True)
class LabelConfig:
    timezone_name: str = "Europe/Paris"
    cutoff_hour: int = 12
    target_end_hour: int = 20
    speed_threshold_kmh: float = 18.52
    heading_min_degrees: float = 225.0
    heading_max_degrees: float = 315.0
    minimum_cumulative_minutes: float = 30.0
    minimum_consecutive_samples: int = 3
    maximum_consecutive_gap_minutes: float = 10.0
    sample_hold_cap_minutes: float = 5.0
    minimum_target_coverage: float = 0.75
    piou_morning_start_hour: int = 8
    maximum_feature_age_minutes: float = 30.0
    weather_morning_start_hour: int = 6
    maximum_weather_feature_age_minutes: float = 90.0


@dataclass(frozen=True)
class PiouObservation:
    timestamp_utc: datetime
    timestamp_local: datetime
    wind_speed_min_kmh: float
    wind_speed_avg_kmh: float
    wind_speed_max_kmh: float
    wind_heading_degrees: float


@dataclass(frozen=True)
class WeatherResource:
    resource_id: str
    title: str
    url: str
    start_year: int
    end_year: int
    last_modified: str
    filesize: int | None


def open_url(url: str, timeout: int = 120):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=timeout)


def validate_label_config(config: LabelConfig) -> None:
    """Reject label/feature contracts that are ambiguous or internally invalid."""
    try:
        ZoneInfo(config.timezone_name)
    except Exception as error:
        raise ValueError(f"Invalid timezone_name: {config.timezone_name!r}") from error
    if not (
        0 <= config.weather_morning_start_hour < config.cutoff_hour
        and 0 <= config.piou_morning_start_hour < config.cutoff_hour
        and config.cutoff_hour < config.target_end_hour <= 24
    ):
        raise ValueError(
            "Require both morning starts < cutoff_hour < target_end_hour <= 24"
        )
    if not (
        config.speed_threshold_kmh > 0
        and 0 <= config.heading_min_degrees <= 360
        and 0 <= config.heading_max_degrees <= 360
        and config.minimum_cumulative_minutes > 0
        and config.minimum_consecutive_samples >= 1
        and config.maximum_consecutive_gap_minutes > 0
        and config.sample_hold_cap_minutes > 0
        and 0 < config.minimum_target_coverage <= 1
        and config.maximum_feature_age_minutes > 0
        and config.maximum_weather_feature_age_minutes > 0
    ):
        raise ValueError("Label thresholds, coverage, and feature ages must be valid")


def label_config_from_payload(payload: Any) -> LabelConfig:
    """Parse a complete, typed LabelConfig from dataset metadata."""
    if not isinstance(payload, dict):
        raise ValueError("Dataset metadata must contain a label_config object")
    expected = set(LabelConfig.__dataclass_fields__)
    supplied = set(payload)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise ValueError(f"Invalid label_config keys; missing={missing}, extra={extra}")
    integer_fields = {
        "cutoff_hour",
        "target_end_hour",
        "minimum_consecutive_samples",
        "piou_morning_start_hour",
        "weather_morning_start_hour",
    }
    for name, value in payload.items():
        if name == "timezone_name":
            if not isinstance(value, str) or not value:
                raise ValueError("label_config.timezone_name must be a non-empty string")
        elif name in integer_fields:
            if type(value) is not int:
                raise ValueError(f"label_config.{name} must be an integer")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"label_config.{name} must be a finite number")
    config = LabelConfig(**payload)
    validate_label_config(config)
    return config


def parse_piou_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def piou_observations_from_archive_payload(
    payload: dict[str, Any], local_timezone: ZoneInfo
) -> list[PiouObservation]:
    required = {
        "time",
        "wind_speed_min",
        "wind_speed_avg",
        "wind_speed_max",
        "wind_heading",
    }
    legend = payload.get("legend")
    if not isinstance(legend, list) or not required.issubset(legend):
        raise ValueError("PiouPiou archive response has an unexpected legend")
    indexes = {name: legend.index(name) for name in required}
    observations: list[PiouObservation] = []
    for values in payload.get("data", []):
        if any(values[indexes[name]] is None for name in required):
            continue
        timestamp_utc = parse_piou_timestamp(str(values[indexes["time"]]))
        observations.append(
            PiouObservation(
                timestamp_utc=timestamp_utc,
                timestamp_local=timestamp_utc.astimezone(local_timezone),
                wind_speed_min_kmh=float(values[indexes["wind_speed_min"]]),
                wind_speed_avg_kmh=float(values[indexes["wind_speed_avg"]]),
                wind_speed_max_kmh=float(values[indexes["wind_speed_max"]]),
                wind_heading_degrees=float(values[indexes["wind_heading"]]),
            )
        )
    observations.sort(key=lambda item: item.timestamp_utc)
    return observations


def fetch_piou_morning(
    local_day: date, config: LabelConfig, station_id: int = 456
) -> list[PiouObservation]:
    """Fetch the local morning window from the public PiouPiou archive API."""
    local_timezone = ZoneInfo(config.timezone_name)
    start = local_boundary(local_day, config.piou_morning_start_hour, local_timezone)
    stop = local_boundary(local_day, config.cutoff_hour, local_timezone)
    query = urllib.parse.urlencode(
        {
            "start": start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "stop": stop.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "format": "json",
        }
    )
    with open_url(f"{PIOU_ARCHIVE_API}/{station_id}?{query}", timeout=60) as response:
        payload = json.load(response)
    observations = deduplicate_piou_observations(
        piou_observations_from_archive_payload(payload, local_timezone)
    )
    start_utc = start.astimezone(timezone.utc)
    stop_utc = stop.astimezone(timezone.utc)
    return [item for item in observations if start_utc <= item.timestamp_utc < stop_utc]


def monthly_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if MONTHLY_NAME.match(path.name))


def read_month(path: Path, local_timezone: ZoneInfo) -> list[PiouObservation]:
    observations: list[PiouObservation] = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 8 or not PIOU_TIME.match(row[0]):
                continue
            timestamp_utc = parse_piou_timestamp(row[0])
            observations.append(
                PiouObservation(
                    timestamp_utc=timestamp_utc,
                    timestamp_local=timestamp_utc.astimezone(local_timezone),
                    wind_speed_min_kmh=float(row[3]),
                    wind_speed_avg_kmh=float(row[4]),
                    wind_speed_max_kmh=float(row[5]),
                    wind_heading_degrees=float(row[6]),
                )
            )
    observations.sort(key=lambda item: item.timestamp_utc)
    return observations


def deduplicate_piou_observations(
    observations: Iterable[PiouObservation],
) -> list[PiouObservation]:
    unique: list[PiouObservation] = []
    for observation in sorted(observations, key=lambda item: item.timestamp_utc):
        if unique and observation.timestamp_utc == unique[-1].timestamp_utc:
            if observation != unique[-1]:
                raise ValueError(
                    f"Conflicting PiouPiou rows at {observation.timestamp_utc.isoformat()}"
                )
            continue
        unique.append(observation)
    return unique


def iter_unique_piou(
    input_dir: Path, local_timezone: ZoneInfo
) -> tuple[Iterator[PiouObservation], dict[str, int]]:
    counters = {"source_rows": 0, "unique_rows": 0, "duplicate_timestamps": 0}

    def generate() -> Iterator[PiouObservation]:
        previous: PiouObservation | None = None
        for path in monthly_files(input_dir):
            for observation in read_month(path, local_timezone):
                counters["source_rows"] += 1
                if previous is not None and observation.timestamp_utc == previous.timestamp_utc:
                    counters["duplicate_timestamps"] += 1
                    if observation != previous:
                        raise ValueError(
                            f"Conflicting PiouPiou rows at "
                            f"{observation.timestamp_utc.isoformat()}"
                        )
                    continue
                if previous is not None and observation.timestamp_utc < previous.timestamp_utc:
                    raise ValueError(f"Monthly source order overlaps unexpectedly at {path}")
                previous = observation
                counters["unique_rows"] += 1
                yield observation

    return generate(), counters


def group_piou_by_local_day(
    observations: Iterable[PiouObservation],
) -> Iterator[tuple[date, list[PiouObservation]]]:
    current_day: date | None = None
    bucket: list[PiouObservation] = []
    for observation in observations:
        observation_day = observation.timestamp_local.date()
        if current_day is not None and observation_day != current_day:
            yield current_day, bucket
            bucket = []
        current_day = observation_day
        bucket.append(observation)
    if current_day is not None:
        yield current_day, bucket


def is_westerly(heading: float, config: LabelConfig) -> bool:
    if config.heading_min_degrees <= config.heading_max_degrees:
        return config.heading_min_degrees <= heading <= config.heading_max_degrees
    return heading >= config.heading_min_degrees or heading <= config.heading_max_degrees


def is_qualifying(observation: PiouObservation, config: LabelConfig) -> bool:
    return (
        observation.wind_speed_avg_kmh >= config.speed_threshold_kmh
        and is_westerly(observation.wind_heading_degrees, config)
    )


def held_minutes(
    observations: Sequence[PiouObservation], index: int, window_end: datetime, cap_minutes: float
) -> float:
    current = observations[index].timestamp_local
    if index + 1 < len(observations):
        following = observations[index + 1].timestamp_local
    else:
        following = current + timedelta(minutes=cap_minutes)
    return max(
        0.0,
        min(
            elapsed_minutes(following, current),
            cap_minutes,
            elapsed_minutes(window_end, current),
        ),
    )


def local_boundary(local_day: date, hour: int, timezone_local: ZoneInfo) -> datetime:
    if hour == 24:
        return datetime.combine(local_day + timedelta(days=1), time(0), timezone_local)
    return datetime.combine(local_day, time(hour), timezone_local)


def elapsed_minutes(later: datetime, earlier: datetime) -> float:
    return (
        later.astimezone(timezone.utc) - earlier.astimezone(timezone.utc)
    ).total_seconds() / 60.0


def target_label(
    local_day: date, observations: Sequence[PiouObservation], config: LabelConfig
) -> dict[str, float | int] | None:
    timezone_local = ZoneInfo(config.timezone_name)
    start = local_boundary(local_day, config.cutoff_hour, timezone_local)
    end = local_boundary(local_day, config.target_end_hour, timezone_local)
    start_utc = start.astimezone(timezone.utc)
    end_utc = end.astimezone(timezone.utc)
    target = [
        item
        for item in observations
        if start_utc <= item.timestamp_utc < end_utc
    ]
    if not target:
        return None
    coverage_minutes = sum(
        held_minutes(target, index, end, config.sample_hold_cap_minutes)
        for index in range(len(target))
    )
    window_minutes = elapsed_minutes(end, start)
    coverage_fraction = coverage_minutes / window_minutes
    if coverage_fraction < config.minimum_target_coverage:
        return None

    qualifying_minutes = 0.0
    longest_run = 0
    current_run = 0
    previous_qualifying_time: datetime | None = None
    for index, observation in enumerate(target):
        if is_qualifying(observation, config):
            gap = (
                elapsed_minutes(observation.timestamp_local, previous_qualifying_time)
                if previous_qualifying_time is not None
                else math.inf
            )
            current_run = current_run + 1 if gap <= config.maximum_consecutive_gap_minutes else 1
            longest_run = max(longest_run, current_run)
            previous_qualifying_time = observation.timestamp_local
            qualifying_minutes += held_minutes(
                target, index, end, config.sample_hold_cap_minutes
            )
        else:
            current_run = 0
            previous_qualifying_time = None
    positive = (
        qualifying_minutes >= config.minimum_cumulative_minutes
        and longest_run >= config.minimum_consecutive_samples
    )
    return {
        "label": int(positive),
        "meta_target_observations": len(target),
        "meta_target_coverage_fraction": coverage_fraction,
        "meta_target_qualifying_minutes": qualifying_minutes,
        "meta_target_longest_qualifying_run": longest_run,
    }


def safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def safe_max(values: Sequence[float]) -> float:
    return float(np.max(values)) if values else float("nan")


def safe_sum(values: Sequence[float]) -> float:
    return float(np.sum(values)) if values else float("nan")


def linear_slope_per_hour(times: Sequence[datetime], values: Sequence[float]) -> float:
    if len(values) < 2:
        return float("nan")
    origin = times[0]
    hours = np.asarray([elapsed_minutes(item, origin) / 60.0 for item in times])
    if np.ptp(hours) < 1e-9:
        return 0.0
    return float(np.polyfit(hours, np.asarray(values, dtype=float), 1)[0])


def subset_before_cutoff(
    observations: Sequence[PiouObservation], cutoff: datetime, minutes: float
) -> list[PiouObservation]:
    cutoff_utc = cutoff.astimezone(timezone.utc)
    start_utc = cutoff_utc - timedelta(minutes=minutes)
    return [item for item in observations if start_utc <= item.timestamp_utc < cutoff_utc]


def piou_features(
    local_day: date, observations: Sequence[PiouObservation], config: LabelConfig
) -> dict[str, float] | None:
    timezone_local = ZoneInfo(config.timezone_name)
    cutoff = local_boundary(local_day, config.cutoff_hour, timezone_local)
    morning_start = local_boundary(
        local_day, config.piou_morning_start_hour, timezone_local
    )
    cutoff_utc = cutoff.astimezone(timezone.utc)
    morning_start_utc = morning_start.astimezone(timezone.utc)
    morning = [
        item
        for item in observations
        if morning_start_utc <= item.timestamp_utc < cutoff_utc
    ]
    if not morning:
        return None
    last_age = elapsed_minutes(cutoff, morning[-1].timestamp_local)
    if last_age > config.maximum_feature_age_minutes:
        return None
    output: dict[str, float] = {
        "piou_observation_count_morning": float(len(morning)),
        "piou_last_age_minutes": last_age,
        "piou_last_wind_avg_kmh": morning[-1].wind_speed_avg_kmh,
        "piou_last_wind_max_kmh": morning[-1].wind_speed_max_kmh,
        "piou_last_heading_sin": math.sin(math.radians(morning[-1].wind_heading_degrees)),
        "piou_last_heading_cos": math.cos(math.radians(morning[-1].wind_heading_degrees)),
    }
    for minutes, label in ((30, "30m"), (60, "1h"), (180, "3h")):
        recent = subset_before_cutoff(morning, cutoff, minutes)
        speeds = [item.wind_speed_avg_kmh for item in recent]
        maximums = [item.wind_speed_max_kmh for item in recent]
        headings_radians = [math.radians(item.wind_heading_degrees) for item in recent]
        output[f"piou_mean_wind_avg_{label}_kmh"] = safe_mean(speeds)
        output[f"piou_max_wind_avg_{label}_kmh"] = safe_max(speeds)
        output[f"piou_max_wind_gust_{label}_kmh"] = safe_max(maximums)
        output[f"piou_west_fraction_{label}"] = safe_mean(
            [float(is_westerly(item.wind_heading_degrees, config)) for item in recent]
        )
        output[f"piou_heading_sin_mean_{label}"] = safe_mean(
            [math.sin(value) for value in headings_radians]
        )
        output[f"piou_heading_cos_mean_{label}"] = safe_mean(
            [math.cos(value) for value in headings_radians]
        )
        output[f"piou_west_component_mean_{label}_kmh"] = safe_mean(
            [
                item.wind_speed_avg_kmh
                * math.cos(math.radians(item.wind_heading_degrees - 270.0))
                for item in recent
            ]
        )
    recent_three_hours = subset_before_cutoff(morning, cutoff, 180)
    output["piou_wind_avg_trend_3h_kmh_per_hour"] = linear_slope_per_hour(
        [item.timestamp_local for item in recent_three_hours],
        [item.wind_speed_avg_kmh for item in recent_three_hours],
    )
    mean_speed = output["piou_mean_wind_avg_3h_kmh"]
    output["piou_gust_factor_3h"] = (
        output["piou_max_wind_gust_3h_kmh"] / mean_speed
        if np.isfinite(mean_speed) and mean_speed > 0
        else float("nan")
    )
    return output


def calendar_features(local_day: date) -> dict[str, float | int | str]:
    day_of_year = local_day.timetuple().tm_yday
    days_in_year = date(local_day.year, 12, 31).timetuple().tm_yday
    return {
        "date": local_day.isoformat(),
        "year": local_day.year,
        "cal_doy_sin": math.sin(2.0 * math.pi * day_of_year / days_in_year),
        "cal_doy_cos": math.cos(2.0 * math.pi * day_of_year / days_in_year),
    }


def build_piou_days(input_dir: Path, config: LabelConfig) -> tuple[dict[date, dict[str, Any]], dict[str, int]]:
    timezone_local = ZoneInfo(config.timezone_name)
    iterator, counters = iter_unique_piou(input_dir, timezone_local)
    rows: dict[date, dict[str, Any]] = {}
    counters.update(
        {
            "candidate_days": 0,
            "insufficient_target_coverage_days": 0,
            "stale_or_missing_noon_feature_days": 0,
            "usable_days": 0,
            "positive_days": 0,
        }
    )
    for local_day, observations in group_piou_by_local_day(iterator):
        counters["candidate_days"] += 1
        label_values = target_label(local_day, observations, config)
        if label_values is None:
            counters["insufficient_target_coverage_days"] += 1
            continue
        features = piou_features(local_day, observations, config)
        if features is None:
            counters["stale_or_missing_noon_feature_days"] += 1
            continue
        row: dict[str, Any] = {
            **calendar_features(local_day),
            "label": label_values["label"],
            **label_values,
            **features,
        }
        rows[local_day] = row
        counters["usable_days"] += 1
        counters["positive_days"] += int(row["label"])
    return rows, counters


def resource_from_payload(payload: dict[str, Any]) -> WeatherResource | None:
    title = str(payload.get("title") or "")
    match = RESOURCE_PERIOD.search(title)
    if not match:
        return None
    return WeatherResource(
        resource_id=str(payload["id"]),
        title=title,
        url=str(payload["url"]),
        start_year=int(match.group(1)),
        end_year=int(match.group(2)),
        last_modified=str(payload.get("last_modified") or ""),
        filesize=int(payload["filesize"]) if payload.get("filesize") is not None else None,
    )


def discover_weather_resources(
    cache_dir: Path, start_year: int, end_year: int, offline: bool
) -> list[WeatherResource]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_cache = cache_dir / "meteofrance_hourly_resources.json"
    if offline:
        if not metadata_cache.exists():
            raise FileNotFoundError(f"Offline resource metadata not found: {metadata_cache}")
        payload = json.loads(metadata_cache.read_text())
    else:
        with open_url(DATA_GOUV_API, timeout=60) as response:
            payload = json.load(response)
        metadata_cache.write_text(json.dumps(payload, indent=2) + "\n")
    resources = [resource_from_payload(item) for item in payload["resources"]]
    selected = [
        item
        for item in resources
        if item is not None and item.start_year <= end_year and item.end_year >= start_year
    ]
    if not selected:
        raise ValueError(f"No department-73 hourly resources cover {start_year}-{end_year}")
    return sorted(selected, key=lambda item: item.start_year)


def filtered_resource_path(cache_dir: Path, resource: WeatherResource, station_id: str) -> Path:
    return cache_dir / f"meteofrance_{station_id}_{resource.resource_id}.csv.gz"


def cache_station_resource(
    cache_dir: Path,
    resource: WeatherResource,
    station_id: str,
    refresh: bool,
    offline: bool,
) -> tuple[Path, int]:
    target = filtered_resource_path(cache_dir, resource, station_id)
    sidecar = target.with_name(target.name + ".metadata.json")
    if target.exists() and not refresh:
        if not sidecar.exists():
            if offline:
                raise ValueError(
                    f"Unverified legacy weather cache {target}; refresh it online first"
                )
            cache_matches = False
        else:
            cached = json.loads(sidecar.read_text())
            cache_matches = (
                cached.get("resource_id") == resource.resource_id
                and cached.get("last_modified") == resource.last_modified
                and cached.get("url") == resource.url
                and cached.get("station_id") == station_id
                and cached.get("cache_sha256") == sha256_file(target)
            )
        if cache_matches:
            with gzip.open(target, "rt", newline="") as handle:
                row_count = sum(1 for _ in csv.DictReader(handle))
            if row_count == 0:
                raise ValueError(f"Station {station_id} was absent from cached {resource.title}")
            if int(cached.get("row_count", -1)) != row_count:
                raise ValueError(f"Weather cache row count changed for {target}")
            return target, row_count
        if offline:
            raise ValueError(f"Weather cache metadata or checksum mismatch for {target}")
    if offline:
        raise FileNotFoundError(f"Offline station cache not found: {target}")
    temporary = target.with_suffix(target.suffix + ".part")
    print(
        f"Downloading {resource.title} ({resource.filesize or 'unknown'} compressed bytes) ...",
        file=sys.stderr,
        flush=True,
    )
    row_count = 0
    try:
        with open_url(resource.url, timeout=300) as response:
            with gzip.GzipFile(fileobj=response, mode="rb") as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text_input:
                    reader = csv.DictReader(text_input, delimiter=";")
                    missing_columns = set(WEATHER_CACHE_FIELDS).difference(reader.fieldnames or [])
                    if missing_columns:
                        raise ValueError(
                            f"Meteo-France resource missing fields: {sorted(missing_columns)}"
                        )
                    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as output:
                        writer = csv.DictWriter(output, fieldnames=WEATHER_CACHE_FIELDS)
                        writer.writeheader()
                        for row in reader:
                            if row["NUM_POSTE"] == station_id:
                                writer.writerow({field: row.get(field, "") for field in WEATHER_CACHE_FIELDS})
                                row_count += 1
        temporary.replace(target)
        cache_sha256 = sha256_file(target)
        sidecar.write_text(
            json.dumps(
                {
                    "resource_id": resource.resource_id,
                    "last_modified": resource.last_modified,
                    "url": resource.url,
                    "station_id": station_id,
                    "row_count": row_count,
                    "cache_sha256": cache_sha256,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Cached {row_count:,} rows for station {station_id}.", file=sys.stderr, flush=True)
    if row_count == 0:
        raise ValueError(f"Station {station_id} was absent from {resource.title}")
    return target, row_count


def weather_value(row: dict[str, str], field: str) -> float:
    raw = row.get(field, "")
    quality = row.get(f"Q{field}", "")
    if raw == "" or quality not in QUALITY_ACCEPTED:
        return float("nan")
    return float(raw)


def iter_cached_weather(paths: Sequence[Path], timezone_local: ZoneInfo) -> Iterator[dict[str, Any]]:
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                timestamp_utc = datetime.strptime(row["AAAAMMJJHH"], "%Y%m%d%H").replace(
                    tzinfo=timezone.utc
                )
                values = {field: weather_value(row, field) for field in WEATHER_RAW_FIELDS}
                yield {
                    "timestamp_utc": timestamp_utc,
                    "timestamp_local": timestamp_utc.astimezone(timezone_local),
                    **values,
                }


def finite_values(observations: Sequence[dict[str, Any]], field: str) -> list[float]:
    return [float(item[field]) for item in observations if np.isfinite(item[field])]


def first_last_delta(observations: Sequence[dict[str, Any]], field: str) -> float:
    values = finite_values(observations, field)
    return values[-1] - values[0] if len(values) >= 2 else float("nan")


def latest(observations: Sequence[dict[str, Any]], field: str) -> float:
    values = finite_values(observations, field)
    return values[-1] if values else float("nan")


def latest_recent(
    observations: Sequence[dict[str, Any]],
    field: str,
    cutoff: datetime | None,
    maximum_age_minutes: float,
) -> float:
    for observation in reversed(observations):
        value = observation[field]
        timestamp = observation.get("timestamp_local")
        if not np.isfinite(value):
            continue
        if cutoff is None or not isinstance(timestamp, datetime):
            return float(value)
        age = elapsed_minutes(cutoff, timestamp)
        return float(value) if 0 <= age <= maximum_age_minutes else float("nan")
    return float("nan")


def weather_features(
    observations: Sequence[dict[str, Any]],
    cutoff: datetime | None = None,
    maximum_age_minutes: float = math.inf,
) -> dict[str, float]:
    output: dict[str, float] = {"mf_observation_count_morning": float(len(observations))}
    core_observations = [
        item
        for item in observations
        if isinstance(item.get("timestamp_local"), datetime)
        and all(np.isfinite(item[field]) for field in WEATHER_FRESHNESS_FIELDS)
    ]
    output["mf_core_observation_count_morning"] = float(len(core_observations))
    output["mf_last_age_minutes"] = (
        elapsed_minutes(cutoff, core_observations[-1]["timestamp_local"])
        if cutoff is not None and core_observations
        else float("nan")
    )
    for field, name in (
        ("T", "temperature_c"),
        ("TD", "dewpoint_c"),
        ("U", "relative_humidity_pct"),
        ("PMER", "pressure_msl_hpa"),
        ("PSTAT", "surface_pressure_hpa"),
        ("VV", "visibility_m"),
        ("FF", "wind_speed_10m_ms"),
    ):
        values = finite_values(observations, field)
        output[f"mf_{name}_latest"] = latest_recent(
            observations, field, cutoff, maximum_age_minutes
        )
        output[f"mf_{name}_mean"] = safe_mean(values)
        output[f"mf_{name}_delta_morning"] = first_last_delta(observations, field)
    temperature = output["mf_temperature_c_latest"]
    dewpoint = output["mf_dewpoint_c_latest"]
    output["mf_dewpoint_depression_c_latest"] = (
        temperature - dewpoint
        if np.isfinite(temperature) and np.isfinite(dewpoint)
        else float("nan")
    )
    output["mf_precipitation_morning_mm"] = safe_sum(
        finite_values(observations, "RR1")
    )
    output["mf_sunshine_morning_minutes"] = safe_sum(
        finite_values(observations, "INS")
    )
    for field, name in (("GLO", "global"), ("DIR", "direct"), ("DIF", "diffuse")):
        values = finite_values(observations, field)
        output[f"mf_{name}_radiation_morning_j_cm2"] = safe_sum(values)
        output[f"mf_{name}_radiation_mean_w_m2"] = safe_mean(
            [value * (10000.0 / 3600.0) for value in values]
        )
    clouds = finite_values(observations, "N")
    output["mf_cloud_cover_mean_oktas"] = safe_mean([value for value in clouds if value <= 8])
    output["mf_sky_obscured_fraction"] = safe_mean([float(value == 9) for value in clouds])
    direction_latest = latest_recent(observations, "DD", cutoff, maximum_age_minutes)
    output["mf_wind_direction_latest_sin"] = (
        math.sin(math.radians(direction_latest))
        if np.isfinite(direction_latest)
        else float("nan")
    )
    output["mf_wind_direction_latest_cos"] = (
        math.cos(math.radians(direction_latest))
        if np.isfinite(direction_latest)
        else float("nan")
    )
    paired = [
        (float(item["FF"]), float(item["DD"]))
        for item in observations
        if np.isfinite(item["FF"]) and np.isfinite(item["DD"])
    ]
    output["mf_west_component_mean_ms"] = safe_mean(
        [speed * math.cos(math.radians(direction - 270.0)) for speed, direction in paired]
    )
    return output


def build_weather_days(
    paths: Sequence[Path], config: LabelConfig, start_year: int, end_year: int
) -> dict[date, dict[str, float]]:
    timezone_local = ZoneInfo(config.timezone_name)
    grouped: dict[date, list[dict[str, Any]]] = defaultdict(list)
    seen: dict[datetime, dict[str, Any]] = {}
    for observation in iter_cached_weather(paths, timezone_local):
        local_timestamp = observation["timestamp_local"]
        if not start_year <= local_timestamp.year <= end_year:
            continue
        if config.weather_morning_start_hour <= local_timestamp.hour < config.cutoff_hour:
            timestamp_utc = observation["timestamp_utc"]
            previous = seen.get(timestamp_utc)
            if previous is not None:
                conflicts = [
                    field
                    for field in WEATHER_RAW_FIELDS
                    if not (
                        (np.isnan(previous[field]) and np.isnan(observation[field]))
                        or previous[field] == observation[field]
                    )
                ]
                if conflicts:
                    raise ValueError(
                        f"Conflicting Meteo-France rows at {timestamp_utc.isoformat()}: "
                        f"{conflicts}"
                    )
                continue
            seen[timestamp_utc] = observation
            grouped[local_timestamp.date()].append(observation)
    return {
        local_day: weather_features(
            sorted(values, key=lambda item: item["timestamp_local"]),
            cutoff=local_boundary(local_day, config.cutoff_hour, timezone_local),
            maximum_age_minutes=config.maximum_weather_feature_age_minutes,
        )
        for local_day, values in grouped.items()
    }


def write_dataset(rows: Sequence[dict[str, Any]], output: Path) -> list[str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    leading = ["date", "year", "label"]
    remaining = sorted(set().union(*(row.keys() for row in rows)).difference(leading))
    fields = leading + remaining
    temporary = output.with_suffix(output.suffix + ".part")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: "" if isinstance(row.get(field), float) and not np.isfinite(row[field]) else row.get(field, "")
                    for field in fields
                }
            )
    temporary.replace(output)
    return fields


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("pioudata"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/traverse_daily.csv"))
    parser.add_argument(
        "--metadata-output", type=Path, default=Path("artifacts/traverse_daily.metadata.json")
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("pioudata/.weather_cache"))
    parser.add_argument("--station-id", default=METEO_FRANCE_STATION_ID)
    parser.add_argument("--timezone", default="Europe/Paris")
    parser.add_argument("--cutoff-hour", type=int, default=12)
    parser.add_argument("--target-end-hour", type=int, default=20)
    parser.add_argument("--speed-threshold-kmh", type=float, default=18.52)
    parser.add_argument("--heading-min-degrees", type=float, default=225.0)
    parser.add_argument("--heading-max-degrees", type=float, default=315.0)
    parser.add_argument("--minimum-cumulative-minutes", type=float, default=30.0)
    parser.add_argument("--minimum-consecutive-samples", type=int, default=3)
    parser.add_argument("--maximum-consecutive-gap-minutes", type=float, default=10.0)
    parser.add_argument("--sample-hold-cap-minutes", type=float, default=5.0)
    parser.add_argument("--minimum-target-coverage", type=float, default=0.75)
    parser.add_argument("--piou-morning-start-hour", type=int, default=8)
    parser.add_argument("--maximum-feature-age-minutes", type=float, default=30.0)
    parser.add_argument("--weather-morning-start-hour", type=int, default=6)
    parser.add_argument("--maximum-weather-feature-age-minutes", type=float, default=90.0)
    parser.add_argument("--refresh-weather", action="store_true")
    parser.add_argument("--offline", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = LabelConfig(
        timezone_name=args.timezone,
        cutoff_hour=args.cutoff_hour,
        target_end_hour=args.target_end_hour,
        speed_threshold_kmh=args.speed_threshold_kmh,
        heading_min_degrees=args.heading_min_degrees,
        heading_max_degrees=args.heading_max_degrees,
        minimum_cumulative_minutes=args.minimum_cumulative_minutes,
        minimum_consecutive_samples=args.minimum_consecutive_samples,
        maximum_consecutive_gap_minutes=args.maximum_consecutive_gap_minutes,
        sample_hold_cap_minutes=args.sample_hold_cap_minutes,
        minimum_target_coverage=args.minimum_target_coverage,
        piou_morning_start_hour=args.piou_morning_start_hour,
        maximum_feature_age_minutes=args.maximum_feature_age_minutes,
        weather_morning_start_hour=args.weather_morning_start_hour,
        maximum_weather_feature_age_minutes=args.maximum_weather_feature_age_minutes,
    )
    try:
        validate_label_config(config)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    piou_days, counters = build_piou_days(args.input_dir, config)
    if not piou_days:
        raise SystemExit("No usable PiouPiou days were produced")
    start_year = min(item.year for item in piou_days)
    end_year = max(item.year for item in piou_days)
    resources = discover_weather_resources(args.cache_dir, start_year, end_year, args.offline)
    cache_paths: list[Path] = []
    cache_counts: dict[str, int] = {}
    cache_sha256: dict[str, str] = {}
    for resource in resources:
        path, count = cache_station_resource(
            args.cache_dir,
            resource,
            args.station_id,
            args.refresh_weather,
            args.offline,
        )
        cache_paths.append(path)
        cache_counts[resource.resource_id] = count
        cache_sha256[resource.resource_id] = sha256_file(path)
    weather_days = build_weather_days(cache_paths, config, start_year, end_year)
    rows: list[dict[str, Any]] = []
    days_without_weather = 0
    days_with_stale_weather = 0
    for local_day, row in sorted(piou_days.items()):
        weather = weather_days.get(local_day)
        if weather is None:
            days_without_weather += 1
            continue
        weather_age = weather.get("mf_last_age_minutes", float("nan"))
        if not np.isfinite(weather_age) or (
            weather_age > config.maximum_weather_feature_age_minutes
        ):
            days_with_stale_weather += 1
            continue
        rows.append(
            {
                **row,
                **weather,
                "meta_weather_station_id": args.station_id,
                "meta_weather_station_distance_km": 6.73,
            }
        )
    fields = write_dataset(rows, args.output)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "row_count": len(rows),
        "columns": fields,
        "date_range": [rows[0]["date"], rows[-1]["date"]],
        "label_config": asdict(config),
        "pioupiou": {
            "input_dir": str(args.input_dir),
            "station_id": 456,
            "source_schema": PIOU_SOURCE_SCHEMA,
            "license_url": "http://developers.pioupiou.fr/data-licensing",
            "attribution": "(c) contributors of the OpenWindMap wind network",
            "counters": counters,
        },
        "meteofrance": {
            "dataset_id": DATA_GOUV_DATASET_ID,
            "dataset_api_url": DATA_GOUV_API,
            "station_id": args.station_id,
            "station_name": METEO_FRANCE_STATION_NAME,
            "source_schema": METEO_FRANCE_SOURCE_SCHEMA,
            "station_coordinates": [45.641333, 5.877833],
            "station_elevation_m": 235,
            "license": "Licence Ouverte 2.0",
            "resource_rows": cache_counts,
            "resource_cache_sha256": cache_sha256,
            "resources": [asdict(item) for item in resources],
            "quality_policy": "empty values and quality code 2 are missing; codes 0, 1, and 9 are accepted",
            "days_without_weather": days_without_weather,
            "days_with_stale_weather": days_with_stale_weather,
            "feature_window": f"[{config.weather_morning_start_hour:02d}:00,{config.cutoff_hour:02d}:00) Europe/Paris",
        },
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "metadata": str(args.metadata_output),
                "rows": len(rows),
                "positives": counters["positive_days"],
                "positive_rate": counters["positive_days"] / len(rows),
                "days_without_weather": days_without_weather,
                "days_with_stale_weather": days_with_stale_weather,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
