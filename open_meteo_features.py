"""Leakage-safe ECMWF feature enrichment inspired by Quartz Solar Forecast.

Quartz uses gridded numerical-weather-prediction (NWP) variables from
Open-Meteo.  This module adapts that idea to the Traverse decision time: only
model values whose valid timestamps are strictly before noon are aggregated.
No realised or reconstructed afternoon weather enters a training row.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Sequence
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np


OPEN_METEO_HISTORICAL_FORECAST_API = (
    "https://historical-forecast-api.open-meteo.com/v1/forecast"
)
OPEN_METEO_LIVE_FORECAST_API = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_MODEL = "ecmwf_ifs"
OPEN_METEO_SOURCE_SCHEMA = "open-meteo-ecmwf-ifs-pre-noon-v1"
OPEN_METEO_ARCHIVE_START_YEAR = 2017
OPEN_METEO_LATITUDE = 45.701731
OPEN_METEO_LONGITUDE = 5.883505
OPEN_METEO_USER_AGENT = "pioupiou-traverse-research/1.0"
OPEN_METEO_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "surface_pressure",
    "pressure_msl",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",
    "wind_direction_10m",
    "is_day",
    "direct_radiation",
    "diffuse_radiation",
)
EXPECTED_UNITS = {
    "temperature_2m": "°C",
    "relative_humidity_2m": "%",
    "dew_point_2m": "°C",
    "precipitation": "mm",
    "surface_pressure": "hPa",
    "pressure_msl": "hPa",
    "cloud_cover": "%",
    "cloud_cover_low": "%",
    "cloud_cover_mid": "%",
    "cloud_cover_high": "%",
    "wind_speed_10m": "km/h",
    "wind_direction_10m": "°",
    "is_day": "",
    "direct_radiation": "W/m²",
    "diffuse_radiation": "W/m²",
}


def _finite(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _mean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _sum(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.sum(finite)) if finite else float("nan")


def _delta(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return finite[-1] - finite[0] if len(finite) >= 2 else float("nan")


def _latest(values: Sequence[float]) -> float:
    return next(
        (value for value in reversed(values) if math.isfinite(value)),
        float("nan"),
    )


def open_meteo_url(
    start_date: date,
    end_date: date,
    timezone_name: str,
    *,
    live: bool = False,
) -> str:
    """Build the exact Open-Meteo request used by training and inference."""
    if end_date < start_date:
        raise ValueError("Open-Meteo end date must not precede start date")
    parameters = {
        "latitude": str(OPEN_METEO_LATITUDE),
        "longitude": str(OPEN_METEO_LONGITUDE),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(OPEN_METEO_VARIABLES),
        "models": OPEN_METEO_MODEL,
        "timezone": timezone_name,
        "wind_speed_unit": "kmh",
    }
    endpoint = (
        OPEN_METEO_LIVE_FORECAST_API
        if live
        else OPEN_METEO_HISTORICAL_FORECAST_API
    )
    return f"{endpoint}?{urllib.parse.urlencode(parameters)}"


def _cache_path(cache_dir: Path, start_date: date, end_date: date) -> Path:
    return (
        cache_dir
        / "open_meteo"
        / f"{OPEN_METEO_MODEL}_{start_date.isoformat()}_{end_date.isoformat()}.json"
    )


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_payload(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return _sha256_bytes(canonical)


def _download_json(url: str, timeout: int = 120) -> tuple[dict[str, Any], str]:
    request = urllib.request.Request(url, headers={"User-Agent": OPEN_METEO_USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("Open-Meteo returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("Open-Meteo response must be a JSON object")
    return payload, _sha256_bytes(raw)


def _write_cache(path: Path, envelope: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".part", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(raw)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_open_meteo_payload(payload: dict[str, Any], timezone_name: str) -> None:
    if payload.get("error"):
        raise ValueError(f"Open-Meteo error: {payload.get('reason', 'unknown error')}")
    if payload.get("timezone") != timezone_name:
        raise ValueError("Open-Meteo response timezone does not match the request")
    latitude = _finite(payload.get("latitude"))
    longitude = _finite(payload.get("longitude"))
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        raise ValueError("Open-Meteo response is missing its grid coordinates")
    latitude_km = (latitude - OPEN_METEO_LATITUDE) * 111.0
    longitude_km = (
        (longitude - OPEN_METEO_LONGITUDE)
        * 111.0
        * math.cos(math.radians(OPEN_METEO_LATITUDE))
    )
    if math.hypot(latitude_km, longitude_km) > 25.0:
        raise ValueError("Open-Meteo response grid point is too far from the lake")
    hourly = payload.get("hourly")
    units = payload.get("hourly_units")
    if not isinstance(hourly, dict) or not isinstance(units, dict):
        raise ValueError("Open-Meteo response is missing hourly data or units")
    times = hourly.get("time")
    if not isinstance(times, list):
        raise ValueError("Open-Meteo response is missing hourly timestamps")
    for variable in OPEN_METEO_VARIABLES:
        values = hourly.get(variable)
        if not isinstance(values, list) or len(values) != len(times):
            raise ValueError(f"Open-Meteo field {variable!r} is missing or misaligned")
        if units.get(variable) != EXPECTED_UNITS[variable]:
            raise ValueError(
                f"Unexpected Open-Meteo unit for {variable}: {units.get(variable)!r}"
            )


def load_open_meteo_payload(
    cache_dir: Path,
    start_date: date,
    end_date: date,
    timezone_name: str,
    *,
    refresh: bool = False,
    offline: bool = False,
    live: bool = False,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Load a verified response cache, or download one atomically."""
    path = _cache_path(cache_dir, start_date, end_date)
    expected_url = open_meteo_url(start_date, end_date, timezone_name, live=live)
    envelope: dict[str, Any] | None = None
    if path.exists() and not refresh:
        try:
            candidate = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid Open-Meteo cache JSON: {path}") from error
        if (
            isinstance(candidate, dict)
            and candidate.get("source_schema") == OPEN_METEO_SOURCE_SCHEMA
            and candidate.get("request_url") == expected_url
            and isinstance(candidate.get("payload"), dict)
            and candidate.get("payload_sha256")
            == _sha256_payload(candidate["payload"])
        ):
            envelope = candidate
        elif offline:
            raise ValueError(f"Open-Meteo cache contract mismatch: {path}")
    if envelope is None:
        if offline:
            raise FileNotFoundError(f"Open-Meteo cache not available offline: {path}")
        payload, response_sha256 = _download_json(expected_url)
        envelope = {
            "source_schema": OPEN_METEO_SOURCE_SCHEMA,
            "request_url": expected_url,
            "response_sha256": response_sha256,
            "payload_sha256": _sha256_payload(payload),
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        validate_open_meteo_payload(payload, timezone_name)
        _write_cache(path, envelope)
    payload = envelope["payload"]
    validate_open_meteo_payload(payload, timezone_name)
    return payload, {
        "cache_path": str(path),
        "cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "request_url": expected_url,
        "response_sha256": str(envelope["response_sha256"]),
        "payload_sha256": str(envelope["payload_sha256"]),
    }


def open_meteo_morning_features(
    observations: Sequence[dict[str, Any]], cutoff: datetime
) -> dict[str, float]:
    """Aggregate gridded model values with valid times strictly before cutoff."""
    ordered = sorted(observations, key=lambda item: item["timestamp_local"])
    if not ordered:
        return {}
    core = [
        item
        for item in ordered
        if all(
            math.isfinite(_finite(item.get(name)))
            for name in (
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "wind_direction_10m",
            )
        )
    ]
    output: dict[str, float] = {
        "nwp_observation_count_morning": float(len(ordered)),
        "nwp_core_observation_count_morning": float(len(core)),
        "nwp_last_age_minutes": (
            (
                cutoff.astimezone(timezone.utc)
                - core[-1]["timestamp_local"].astimezone(timezone.utc)
            ).total_seconds()
            / 60.0
            if core
            else float("nan")
        ),
    }
    summary_variables = {
        "temperature_2m": "temperature_c",
        "relative_humidity_2m": "relative_humidity_pct",
        "dew_point_2m": "dewpoint_c",
        "surface_pressure": "surface_pressure_hpa",
        "pressure_msl": "pressure_msl_hpa",
        "cloud_cover": "cloud_cover_pct",
        "cloud_cover_low": "cloud_cover_low_pct",
        "cloud_cover_mid": "cloud_cover_mid_pct",
        "cloud_cover_high": "cloud_cover_high_pct",
        "wind_speed_10m": "wind_speed_10m_kmh",
        "direct_radiation": "direct_radiation_w_m2",
        "diffuse_radiation": "diffuse_radiation_w_m2",
    }
    for source, name in summary_variables.items():
        values = [_finite(item.get(source)) for item in ordered]
        output[f"nwp_{name}_latest"] = _latest(values)
        output[f"nwp_{name}_mean"] = _mean(values)
        output[f"nwp_{name}_delta_morning"] = _delta(values)
    output["nwp_precipitation_morning_mm"] = _sum(
        [_finite(item.get("precipitation")) for item in ordered]
    )
    output["nwp_daylight_fraction_morning"] = _mean(
        [_finite(item.get("is_day")) for item in ordered]
    )
    directions = [_finite(item.get("wind_direction_10m")) for item in ordered]
    latest_direction = _latest(directions)
    output["nwp_wind_direction_latest_sin"] = (
        math.sin(math.radians(latest_direction))
        if math.isfinite(latest_direction)
        else float("nan")
    )
    output["nwp_wind_direction_latest_cos"] = (
        math.cos(math.radians(latest_direction))
        if math.isfinite(latest_direction)
        else float("nan")
    )
    paired = [
        (_finite(item.get("wind_speed_10m")), _finite(item.get("wind_direction_10m")))
        for item in ordered
    ]
    paired = [
        (speed, direction)
        for speed, direction in paired
        if math.isfinite(speed) and math.isfinite(direction)
    ]
    output["nwp_west_component_mean_kmh"] = _mean(
        [speed * math.cos(math.radians(direction - 270.0)) for speed, direction in paired]
    )
    output["nwp_south_component_mean_kmh"] = _mean(
        [speed * math.cos(math.radians(direction - 180.0)) for speed, direction in paired]
    )
    return output


def open_meteo_days_from_payload(
    payload: dict[str, Any],
    timezone_name: str,
    morning_start_hour: int,
    cutoff_hour: int,
) -> dict[date, dict[str, float]]:
    """Convert an API payload into one pre-noon feature block per local day."""
    validate_open_meteo_payload(payload, timezone_name)
    timezone_local = ZoneInfo(timezone_name)
    hourly = payload["hourly"]
    grouped: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for index, raw_timestamp in enumerate(hourly["time"]):
        timestamp = datetime.fromisoformat(str(raw_timestamp)).replace(
            tzinfo=timezone_local
        )
        if morning_start_hour <= timestamp.hour < cutoff_hour:
            grouped[timestamp.date()].append(
                {
                    "timestamp_local": timestamp,
                    **{
                        variable: hourly[variable][index]
                        for variable in OPEN_METEO_VARIABLES
                    },
                }
            )
    return {
        local_day: open_meteo_morning_features(
            observations,
            datetime.combine(local_day, time(cutoff_hour), timezone_local),
        )
        for local_day, observations in grouped.items()
    }


def load_open_meteo_days(
    cache_dir: Path,
    start_date: date,
    end_date: date,
    timezone_name: str,
    morning_start_hour: int,
    cutoff_hour: int,
    *,
    refresh: bool = False,
    offline: bool = False,
    live: bool = False,
) -> tuple[dict[date, dict[str, float]], dict[str, str]]:
    payload, provenance = load_open_meteo_payload(
        cache_dir,
        start_date,
        end_date,
        timezone_name,
        refresh=refresh,
        offline=offline,
        live=live,
    )
    return (
        open_meteo_days_from_payload(
            payload, timezone_name, morning_start_hour, cutoff_hour
        ),
        provenance,
    )


def load_open_meteo_years(
    cache_dir: Path,
    start_year: int,
    end_year: int,
    timezone_name: str,
    morning_start_hour: int,
    cutoff_hour: int,
    *,
    refresh: bool = False,
    offline: bool = False,
) -> tuple[dict[date, dict[str, float]], list[dict[str, str]]]:
    """Load historical ECMWF fields in bounded, reusable calendar-year chunks."""
    daily: dict[date, dict[str, float]] = {}
    provenance: list[dict[str, str]] = []
    today = datetime.now(ZoneInfo(timezone_name)).date()
    for year in range(max(start_year, OPEN_METEO_ARCHIVE_START_YEAR), end_year + 1):
        request_start = date(year, 1, 1)
        request_end = min(date(year, 12, 31), today)
        if request_end < request_start:
            continue
        year_days, year_provenance = load_open_meteo_days(
            cache_dir,
            request_start,
            request_end,
            timezone_name,
            morning_start_hour,
            cutoff_hour,
            refresh=refresh,
            offline=offline,
        )
        overlap = set(daily).intersection(year_days)
        if overlap:
            raise ValueError(f"Duplicate Open-Meteo days: {sorted(overlap)[:5]}")
        daily.update(year_days)
        provenance.append(year_provenance)
    return daily, provenance
