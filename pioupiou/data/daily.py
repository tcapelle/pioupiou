"""Observation parsing and feature engineering for the same-day model."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import re
import sys
import urllib.request
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

DATA_GOUV_DATASET_ID = "6569b4473bedf2e7abad3b72"
DATA_GOUV_API = f"https://www.data.gouv.fr/api/1/datasets/{DATA_GOUV_DATASET_ID}/"
PIOU_LATITUDE = 45.701731
PIOU_LONGITUDE = 5.883505
PIOU_MAX_LOCATION_DISTANCE_KM = 1.0
WEATHER_MAX_LOCATION_DISTANCE_KM = 1.0
MONTHLY_NAME = re.compile(r"^\d{4}-\d{2}\.csv$")
PIOU_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}T")
RESOURCE_PERIOD = re.compile(
    r"HOR_departement_(?P<department>\d{2})_periode_"
    r"(?P<start_year>\d{4})-(?P<end_year>\d{4})"
)
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
class WeatherStation:
    slug: str
    station_id: str
    department: str
    name: str
    latitude: float
    longitude: float
    elevation_m: int


PRIMARY_WEATHER_STATION = WeatherStation(
    "airport",
    "73329001",
    "73",
    "CHAMBERY-AIX",
    45.641333,
    5.877833,
    235,
)


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
    department: str


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
        elif (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"label_config.{name} must be a finite number")
    config = LabelConfig(**payload)
    validate_label_config(config)
    return config


def parse_piou_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def distance_km(
    latitude: float,
    longitude: float,
    reference_latitude: float,
    reference_longitude: float,
) -> float:
    """Great-circle distance between two WGS84 points."""
    radius_km = 6371.0
    latitude_radians = math.radians(latitude)
    reference_latitude_radians = math.radians(reference_latitude)
    latitude_delta = latitude_radians - reference_latitude_radians
    longitude_delta = math.radians(longitude - reference_longitude)
    haversine = math.sin(latitude_delta / 2.0) ** 2 + (
        math.cos(reference_latitude_radians)
        * math.cos(latitude_radians)
        * math.sin(longitude_delta / 2.0) ** 2
    )
    return 2.0 * radius_km * math.asin(math.sqrt(haversine))


def valid_piou_location(latitude: object, longitude: object) -> bool:
    """Accept observations only while PP456 is physically at the lake site."""
    try:
        latitude_value = float(latitude)
        longitude_value = float(longitude)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(latitude_value) or not math.isfinite(longitude_value):
        return False
    return (
        distance_km(
            latitude_value,
            longitude_value,
            PIOU_LATITUDE,
            PIOU_LONGITUDE,
        )
        <= PIOU_MAX_LOCATION_DISTANCE_KM
    )


def valid_weather_station_location(
    latitude: object, longitude: object, station: WeatherStation
) -> bool:
    """Accept an official observation only at its configured station site."""
    try:
        latitude_value = float(latitude)
        longitude_value = float(longitude)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(latitude_value) or not math.isfinite(longitude_value):
        return False
    return (
        distance_km(
            latitude_value,
            longitude_value,
            station.latitude,
            station.longitude,
        )
        <= WEATHER_MAX_LOCATION_DISTANCE_KM
    )


def piou_observations_from_archive_payload(
    payload: dict[str, Any], local_timezone: ZoneInfo
) -> list[PiouObservation]:
    required = {
        "time",
        "latitude",
        "longitude",
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
        if not valid_piou_location(
            values[indexes["latitude"]], values[indexes["longitude"]]
        ):
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


def monthly_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if MONTHLY_NAME.match(path.name))


def read_month(
    path: Path,
    local_timezone: ZoneInfo,
    counters: dict[str, int] | None = None,
) -> list[PiouObservation]:
    observations: list[PiouObservation] = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 8 or not PIOU_TIME.match(row[0]):
                continue
            if counters is not None:
                counters["source_rows"] += 1
            if not valid_piou_location(row[1], row[2]):
                if counters is not None:
                    counters["invalid_location_rows"] += 1
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
    counters = {
        "source_rows": 0,
        "invalid_location_rows": 0,
        "unique_rows": 0,
        "duplicate_timestamps": 0,
    }

    def generate() -> Iterator[PiouObservation]:
        previous: PiouObservation | None = None
        for path in monthly_files(input_dir):
            for observation in read_month(path, local_timezone, counters):
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
    local_day: date,
    observations: Sequence[PiouObservation],
    config: LabelConfig,
    cutoff_local: datetime | None = None,
) -> dict[str, float] | None:
    timezone_local = ZoneInfo(config.timezone_name)
    cutoff = cutoff_local or local_boundary(
        local_day, config.cutoff_hour, timezone_local
    )
    if cutoff.tzinfo is None or cutoff.astimezone(timezone_local).date() != local_day:
        raise ValueError("PiouPiou feature cutoff must be timezone-aware and on local_day")
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


def resource_from_payload(payload: dict[str, Any]) -> WeatherResource | None:
    title = str(payload.get("title") or "")
    match = RESOURCE_PERIOD.search(title)
    if not match:
        return None
    return WeatherResource(
        resource_id=str(payload["id"]),
        title=title,
        url=str(payload["url"]),
        start_year=int(match.group("start_year")),
        end_year=int(match.group("end_year")),
        last_modified=str(payload.get("last_modified") or ""),
        filesize=int(payload["filesize"]) if payload.get("filesize") is not None else None,
        department=match.group("department"),
    )


def discover_weather_resources(
    cache_dir: Path,
    start_year: int,
    end_year: int,
    offline: bool,
    departments: Sequence[str] = ("73",),
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
        if item is not None
        and item.department in departments
        and item.start_year <= end_year
        and item.end_year >= start_year
    ]
    if not selected:
        raise ValueError(
            f"No hourly resources for departments {sorted(departments)} "
            f"cover {start_year}-{end_year}"
        )
    return sorted(selected, key=lambda item: (item.department, item.start_year))


def filtered_resources_path(
    cache_dir: Path, resource: WeatherResource, station_ids: Sequence[str]
) -> Path:
    station_ids = tuple(sorted(set(station_ids)))
    if len(station_ids) == 1:
        return cache_dir / f"meteofrance_{station_ids[0]}_{resource.resource_id}.csv.gz"
    station_token = "-".join(station_ids)
    return cache_dir / f"meteofrance_{station_token}_{resource.resource_id}.csv.gz"


def cache_weather_resource(
    cache_dir: Path,
    resource: WeatherResource,
    station_ids: Sequence[str],
    refresh: bool,
    offline: bool,
) -> tuple[Path, dict[str, int]]:
    """Download one department archive once and retain the selected stations."""
    station_ids = tuple(sorted(set(station_ids)))
    if not station_ids:
        raise ValueError("At least one weather station ID is required")
    target = filtered_resources_path(cache_dir, resource, station_ids)
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
                and cached.get("station_ids") == list(station_ids)
                and cached.get("cache_sha256") == sha256_file(target)
            )
        if cache_matches:
            counts = {station_id: 0 for station_id in station_ids}
            with gzip.open(target, "rt", newline="") as handle:
                for row in csv.DictReader(handle):
                    station_id = row["NUM_POSTE"]
                    if station_id in counts:
                        counts[station_id] += 1
            row_count = sum(counts.values())
            if int(cached.get("row_count", -1)) != row_count:
                raise ValueError(f"Weather cache row count changed for {target}")
            if cached.get("resource_rows") != counts:
                raise ValueError(f"Weather cache station counts changed for {target}")
            return target, counts
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
    counts = {station_id: 0 for station_id in station_ids}
    selected_ids = set(station_ids)
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
                            station_id = row["NUM_POSTE"]
                            if station_id in selected_ids:
                                writer.writerow(
                                    {
                                        field: row.get(field, "")
                                        for field in WEATHER_CACHE_FIELDS
                                    }
                                )
                                counts[station_id] += 1
        temporary.replace(target)
        row_count = sum(counts.values())
        cache_sha256 = sha256_file(target)
        sidecar.write_text(
            json.dumps(
                {
                    "resource_id": resource.resource_id,
                    "last_modified": resource.last_modified,
                    "url": resource.url,
                    "station_ids": list(station_ids),
                    "row_count": row_count,
                    "resource_rows": counts,
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
    print(
        f"Cached {sum(counts.values()):,} rows for {len(station_ids)} station(s).",
        file=sys.stderr,
        flush=True,
    )
    return target, counts


def weather_value(row: dict[str, str], field: str) -> float:
    raw = row.get(field, "")
    quality = row.get(f"Q{field}", "")
    if raw == "" or quality not in QUALITY_ACCEPTED:
        return float("nan")
    return float(raw)


def iter_cached_weather(
    paths: Sequence[Path],
    timezone_local: ZoneInfo,
    station: WeatherStation | None = None,
) -> Iterator[dict[str, Any]]:
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if station is not None:
                    if row["NUM_POSTE"] != station.station_id:
                        continue
                    if not valid_weather_station_location(
                        row["LAT"], row["LON"], station
                    ):
                        continue
                timestamp_utc = datetime.strptime(row["AAAAMMJJHH"], "%Y%m%d%H").replace(
                    tzinfo=timezone.utc
                )
                values = {field: weather_value(row, field) for field in WEATHER_RAW_FIELDS}
                yield {
                    "station_id": row["NUM_POSTE"],
                    "timestamp_utc": timestamp_utc,
                    "timestamp_local": timestamp_utc.astimezone(timezone_local),
                    **values,
                }


def finite_values(observations: Sequence[dict[str, Any]], field: str) -> list[float]:
    return [float(item[field]) for item in observations if np.isfinite(item[field])]


def first_last_delta(observations: Sequence[dict[str, Any]], field: str) -> float:
    values = finite_values(observations, field)
    return values[-1] - values[0] if len(values) >= 2 else float("nan")


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


def write_dataset(rows: Sequence[dict[str, Any]], output: Path) -> list[str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    leading = [
        name
        for name in ("date", "year", "issue_minutes", "label")
        if any(name in row for row in rows)
    ]
    remaining = sorted(set().union(*(row.keys() for row in rows)).difference(leading))
    fields = leading + remaining
    temporary = output.with_suffix(output.suffix + ".part")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        ""
                        if isinstance(row.get(field), float)
                        and not np.isfinite(row[field])
                        else row.get(field, "")
                    )
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
