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
    WEATHER_RAW_FIELDS,
    calendar_features,
    label_config_from_payload,
    piou_features,
    piou_observations_from_archive_payload,
    valid_weather_station_location,
    weather_features,
)
from pioupiou.data.timestep import issue_time_features, traverse_progress_features
from pioupiou.inference.model import load_artifact_with_sha256, predict_loaded


PIOU_URL = "https://api.pioupiou.fr/v1/archive/2176?start=last-day&stop=now"
METEO_URL = (
    "https://public-api.meteofrance.fr/public/DPPaquetObs/v1/paquet/horaire"
    "?id-departement=73&format=json"
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
    payload: list[dict[str, Any]], cutoff: datetime
) -> list[dict[str, Any]]:
    local_timezone = cutoff.tzinfo
    assert local_timezone is not None
    start = datetime.combine(cutoff.date(), time(6), local_timezone)
    observations: dict[datetime, dict[str, Any]] = {}
    for row in payload:
        if str(row.get("geo_id_insee")) != PRIMARY_WEATHER_STATION.station_id:
            continue
        if not valid_weather_station_location(
            row.get("lat"), row.get("lon"), PRIMARY_WEATHER_STATION
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
            "station_id": PRIMARY_WEATHER_STATION.station_id,
            "timestamp_utc": timestamp_utc,
            "timestamp_local": timestamp_local,
            **values,
        }
    return [observations[key] for key in sorted(observations)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/traverse_model.joblib"),
    )
    args = parser.parse_args()

    token = os.environ.get("METEOFRANCE_TOKEN")
    if not token:
        raise SystemExit("METEOFRANCE_TOKEN is not set")

    payload, pipeline, model_sha256 = load_artifact_with_sha256(args.model)
    config = label_config_from_payload(payload["label"])
    local_timezone = ZoneInfo(config.timezone_name)
    cutoff = datetime.now(local_timezone).replace(second=0, microsecond=0)
    issue_minutes = cutoff.hour * 60 + cutoff.minute
    if not 6 * 60 + 30 <= issue_minutes < 20 * 60:
        raise SystemExit("Current time is outside the model's 06:30-19:59 window")

    piou_payload = fetch_json(PIOU_URL)
    piou_observations = piou_observations_from_archive_payload(
        piou_payload, local_timezone
    )
    piou = piou_features(
        cutoff.date(), piou_observations, config, cutoff_local=cutoff
    )
    if piou is None:
        raise SystemExit("insufficient_data: missing or stale Windbird observations")

    meteo_payload = fetch_json(METEO_URL, token)
    meteo_observations = meteofrance_observations(meteo_payload, cutoff)
    if not meteo_observations:
        raise SystemExit("insufficient_data: no Météo-France observations")
    meteo = weather_features(
        meteo_observations,
        cutoff=cutoff,
        maximum_age_minutes=config.maximum_weather_feature_age_minutes,
    )

    prepared = {
        **calendar_features(cutoff.date()),
        **issue_time_features(issue_minutes),
        **piou,
        **traverse_progress_features(
            cutoff.date(), piou_observations, config, cutoff
        ),
        **meteo,
    }
    feature_names = list(payload["feature_names"])
    missing = sorted(set(feature_names).difference(prepared))
    if missing:
        raise SystemExit(f"Cannot construct model features: {missing}")
    frame = pd.DataFrame([{name: prepared[name] for name in feature_names}])
    probability, predicted, _ = predict_loaded(payload, pipeline, frame)

    print(
        json.dumps(
            {
                "issue_time": cutoff.isoformat(),
                "model_sha256": model_sha256,
                "piou_last_age_minutes": piou["piou_last_age_minutes"],
                "mf_last_age_minutes": meteo["mf_last_age_minutes"],
                "traverse_probability": float(probability[0]),
                "predict_traverse": bool(predicted[0]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
