#!/usr/bin/env python3
"""Fetch current observations and score today's Traverse probability."""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from pioupiou.data.daily import (
    PRIMARY_WEATHER_STATION,
    WEATHER_STATIONS,
    LabelConfig,
    WeatherStation,
    WEATHER_RAW_FIELDS,
    calendar_features,
    is_traverse_season,
    label_config_from_payload,
    piou_features,
    piou_observations_from_archive_payload,
    valid_weather_station_location,
)
from pioupiou.data.timestep import (
    issue_time_features,
    station_weather_features,
    temperature_contrast_features,
)
from pioupiou.inference.model import (
    load_artifact_with_sha256,
    onset_evidence,
    predict_loaded,
)


PIOU_URL = "https://api.pioupiou.fr/v1/archive/2176?start=last-day&stop=now"
METEO_URL_TEMPLATE = (
    "https://public-api.meteofrance.fr/public/DPPaquetObs/v1/paquet/horaire"
    "?id-departement={department}&format=json"
)


def fetch_json(url: str, token: str | None = None) -> Any:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def finite(value: Any, conversion=lambda item: item) -> float:
    return float("nan") if value is None else float(conversion(float(value)))


def meteofrance_observations(
    payload: list[dict[str, Any]],
    cutoff: datetime,
    station: WeatherStation = PRIMARY_WEATHER_STATION,
) -> list[dict[str, Any]]:
    local_timezone = cutoff.tzinfo
    assert local_timezone is not None
    start = datetime.combine(cutoff.date(), time(6), local_timezone)
    observations: dict[datetime, dict[str, Any]] = {}
    for row in payload:
        if str(row.get("geo_id_insee")) != station.station_id:
            continue
        if not valid_weather_station_location(
            row.get("lat"), row.get("lon"), station
        ):
            continue
        timestamp_utc = datetime.fromisoformat(
            str(row["validity_time"]).replace("Z", "+00:00")
        )
        timestamp_local = timestamp_utc.astimezone(local_timezone)
        if not start <= timestamp_local < cutoff:
            continue
        values = {field: float("nan") for field in WEATHER_RAW_FIELDS}
        values.update(
            {
                "T": finite(row.get("t"), lambda value: value - 273.15),
                "TD": finite(row.get("td"), lambda value: value - 273.15),
                "U": finite(row.get("u")),
                "RR1": finite(row.get("rr1")),
                "PMER": finite(row.get("pmer"), lambda value: value / 100.0),
                "PSTAT": finite(row.get("pres"), lambda value: value / 100.0),
                "N": finite(row.get("n")),
                "VV": finite(row.get("vv")),
                "GLO": finite(row.get("ray_glo01"), lambda value: value / 10000.0),
                "INS": finite(row.get("insolh")),
                "FF": finite(row.get("ff")),
                "DD": finite(row.get("dd")),
            }
        )
        observations[timestamp_utc] = {
            "station_id": station.station_id,
            "timestamp_utc": timestamp_utc,
            "timestamp_local": timestamp_local,
            **values,
        }
    return [observations[key] for key in sorted(observations)]


def current_weather_features(
    payloads: dict[str, list[dict[str, Any]]],
    cutoff: datetime,
    config: LabelConfig,
) -> tuple[dict[str, float], dict[str, list[dict[str, Any]]]]:
    features: dict[str, float] = {}
    observations: dict[str, list[dict[str, Any]]] = {}
    for station in WEATHER_STATIONS:
        station_observations = meteofrance_observations(
            payloads[station.department], cutoff, station
        )
        if not station_observations:
            raise ValueError(
                f"insufficient_data: no Météo-France observations for {station.name}"
            )
        observations[station.slug] = station_observations
        features.update(
            station_weather_features(
                station, station_observations, cutoff, config
            )
        )
    features.update(temperature_contrast_features(features))
    return features, observations


def predict_now(
    model: Path = Path("artifacts/traverse_model.joblib"),
) -> dict[str, Any]:
    token = os.environ.get("METEOFRANCE_TOKEN")
    if not token:
        raise ValueError("METEOFRANCE_TOKEN is not set")

    payload, pipeline, model_sha256 = load_artifact_with_sha256(model)
    config = label_config_from_payload(payload["label"])
    local_timezone = ZoneInfo(config.timezone_name)
    cutoff = datetime.now(local_timezone).replace(second=0, microsecond=0)
    issue_minutes = cutoff.hour * 60 + cutoff.minute
    if not 6 * 60 + 30 <= issue_minutes < 20 * 60:
        raise ValueError("Current time is outside the model's 06:30-19:59 window")
    if not is_traverse_season(cutoff.date(), config):
        return {
            "prediction_time": cutoff.isoformat(),
            "model_sha256": model_sha256,
            "status": "outside_traverse_season",
            "traverse_probability": 0.0,
            "predict_traverse": False,
        }

    piou_payload = fetch_json(PIOU_URL)
    piou_observations = piou_observations_from_archive_payload(
        piou_payload, local_timezone
    )
    piou = piou_features(
        cutoff.date(), piou_observations, config, cutoff_local=cutoff
    )
    if piou is None:
        raise ValueError("insufficient_data: missing or stale Windbird observations")
    piou_start = datetime.combine(
        cutoff.date(), time(config.piou_morning_start_hour), local_timezone
    )
    piou_latest = max(
        item.timestamp_local
        for item in piou_observations
        if piou_start <= item.timestamp_local < cutoff
    )

    meteo_payloads = {
        department: fetch_json(
            METEO_URL_TEMPLATE.format(department=department), token
        )
        for department in sorted(
            {station.department for station in WEATHER_STATIONS}
        )
    }
    meteo, meteo_observations = current_weather_features(
        meteo_payloads, cutoff, config
    )

    prepared = {
        **calendar_features(cutoff.date()),
        **issue_time_features(issue_minutes),
        **piou,
        **meteo,
    }
    feature_names = list(payload["feature_names"])
    missing = sorted(set(feature_names).difference(prepared))
    if missing:
        raise ValueError(f"Cannot construct model features: {missing}")
    frame = pd.DataFrame([{name: prepared[name] for name in feature_names}])
    probability, predicted, _ = predict_loaded(payload, pipeline, frame)

    return {
        "prediction_time": cutoff.isoformat(),
        "model_sha256": model_sha256,
        "piou_observation_time": piou_latest.isoformat(),
        "piou_last_age_minutes": piou["piou_last_age_minutes"],
        "mf_observation_time": meteo_observations[PRIMARY_WEATHER_STATION.slug][-1][
            "timestamp_local"
        ].isoformat(),
        "mf_last_age_minutes": meteo["mf_last_age_minutes"],
        "weather_observation_times": {
            slug: values[-1]["timestamp_local"].isoformat()
            for slug, values in meteo_observations.items()
        },
        "traverse_probability": float(probability[0]),
        "predict_traverse": bool(predicted[0]),
        "onset_evidence": float(
            onset_evidence(frame, config.speed_threshold_kmh)[0]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/traverse_model.joblib"),
    )
    args = parser.parse_args()
    try:
        result = predict_now(args.model)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
