#!/usr/bin/env python3
"""Prepare one unlabeled, leakage-safe feature row as it would exist at noon."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from build_traverse_dataset import (
    METEO_FRANCE_SOURCE_SCHEMA,
    PIOU_SOURCE_SCHEMA,
    PRIMARY_WEATHER_STATION,
    WEATHER_STATIONS,
    calendar_features,
    deduplicate_piou_observations,
    fetch_piou_morning,
    label_config_from_payload,
    load_station_weather,
    local_boundary,
    piou_features,
    read_month,
    spatial_weather_features,
)
from traverse_model import load_artifact_with_sha256, sha256_json


def local_piou_observations(input_dir: Path, local_day: date, timezone_name: str):
    path = input_dir / f"{local_day:%Y-%m}.csv"
    if not path.exists():
        raise FileNotFoundError(f"PiouPiou month file not found: {path}")
    timezone_local = ZoneInfo(timezone_name)
    return deduplicate_piou_observations(
        item
        for item in read_month(path, timezone_local)
        if item.timestamp_local.date() == local_day
    )


def write_feature_row(row: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    leading = [name for name in ("date", "year") if name in row]
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
    """Return only identity fields and the exact ordered model feature schema."""
    missing = sorted({"date", "year", *feature_names}.difference(prepared))
    if missing:
        raise ValueError(f"Cannot construct trained model features: {missing}")
    return {
        "date": prepared["date"],
        "year": prepared["year"],
        **{name: prepared[name] for name in feature_names},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Local date, default today in the model timezone")
    parser.add_argument(
        "--model", type=Path, default=Path("artifacts/traverse_model_variant.joblib")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/noon_features.csv")
    )
    parser.add_argument("--piou-station-id", type=int)
    parser.add_argument(
        "--piou-input-dir",
        type=Path,
        help="Use a local YYYY-MM.csv archive instead of the live PiouPiou API",
    )
    parser.add_argument("--weather-station-id")
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("pioudata/.weather_cache")
    )
    parser.add_argument("--refresh-weather", action="store_true")
    parser.add_argument(
        "--offline-weather", action="store_true", help="Do not query data.gouv.fr"
    )
    args = parser.parse_args()
    if args.refresh_weather and args.offline_weather:
        raise SystemExit("--refresh-weather and --offline-weather are mutually exclusive")

    try:
        model_payload, _, model_sha256 = load_artifact_with_sha256(args.model)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    role = model_payload.get("role")
    if role not in {"variant", "spatial"}:
        raise SystemExit("Noon preparation requires a weather-based model")
    try:
        config = label_config_from_payload(model_payload.get("label"))
    except ValueError as error:
        raise SystemExit(f"Model has an invalid label contract: {error}") from error
    contract = model_payload.get("input_contract")
    required_contract = {
        "schema_version",
        "dataset_sha256",
        "label_config_sha256",
        "feature_schema_sha256",
        "piou_station_id",
        "piou_source_schema",
        "weather_station_id",
        "weather_source_schema",
    }
    if not isinstance(contract, dict) or not required_contract.issubset(contract):
        raise SystemExit("Model is missing its input contract")
    expected_schema_version = 3 if role == "spatial" else 2
    if contract["schema_version"] != expected_schema_version:
        raise SystemExit("Unsupported model input contract version")
    if contract["label_config_sha256"] != sha256_json(model_payload["label"]):
        raise SystemExit("Model label contract fingerprint does not match its payload")
    if contract["feature_schema_sha256"] != sha256_json(
        model_payload["feature_names"]
    ):
        raise SystemExit("Model feature schema fingerprint does not match its payload")
    if contract["piou_source_schema"] != PIOU_SOURCE_SCHEMA or (
        contract["weather_source_schema"] != METEO_FRANCE_SOURCE_SCHEMA
    ):
        raise SystemExit("Model source schema is not supported by this preparer")
    if role == "spatial":
        station_manifest = [asdict(station) for station in WEATHER_STATIONS]
        if contract.get("weather_station_manifest") != station_manifest or contract.get(
            "weather_station_manifest_sha256"
        ) != sha256_json(station_manifest):
            raise SystemExit("Model weather station manifest is not supported")
    piou_station_id = str(args.piou_station_id or contract["piou_station_id"])
    weather_station_id = str(
        args.weather_station_id or contract["weather_station_id"]
    )
    if piou_station_id != str(contract["piou_station_id"]):
        raise SystemExit("PiouPiou station does not match the trained model")
    if weather_station_id != str(contract["weather_station_id"]):
        raise SystemExit("Météo-France station does not match the trained model")
    timezone_local = ZoneInfo(config.timezone_name)
    local_day = date.fromisoformat(args.date) if args.date else datetime.now(timezone_local).date()
    cutoff = local_boundary(local_day, config.cutoff_hour, timezone_local)
    if cutoff.astimezone(timezone.utc) > datetime.now(timezone.utc):
        raise SystemExit(f"Cannot prepare {local_day}: its noon cutoff has not occurred yet")

    try:
        piou_observations = (
            local_piou_observations(args.piou_input_dir, local_day, config.timezone_name)
            if args.piou_input_dir
            else fetch_piou_morning(local_day, config, station_id=int(piou_station_id))
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    piou = piou_features(local_day, piou_observations, config)
    if piou is None:
        raise SystemExit("insufficient_data: missing or stale pre-noon PiouPiou observations")

    stations = WEATHER_STATIONS if role == "spatial" else (PRIMARY_WEATHER_STATION,)
    weather_days, _, _, _ = load_station_weather(
        args.cache_dir,
        config,
        local_day.year,
        local_day.year,
        args.refresh_weather,
        args.offline_weather,
        stations=stations,
    )
    airport = weather_days[PRIMARY_WEATHER_STATION.slug].get(local_day)
    if airport is None:
        raise SystemExit("insufficient_data: no pre-noon Météo-France observations")
    weather_age = airport.get("mf_last_age_minutes", float("nan"))
    if not math.isfinite(weather_age) or (
        weather_age > config.maximum_weather_feature_age_minutes
    ):
        raise SystemExit(
            f"insufficient_data: Météo-France feed age is {weather_age!r} minutes"
        )

    prepared = {
        **calendar_features(local_day),
        **piou,
        **airport,
    }
    if role == "spatial":
        prepared.update(
            spatial_weather_features(
                {
                    station.slug: weather_days[station.slug].get(local_day)
                    for station in WEATHER_STATIONS[1:]
                },
                config.maximum_weather_feature_age_minutes,
            )
        )
    try:
        row = bind_to_model_schema(prepared, list(model_payload["feature_names"]))
    except ValueError as error:
        raise SystemExit(str(error)) from error
    row.update(
        {
            "meta_feature_cutoff_local": cutoff.isoformat(),
            "meta_contract_schema_version": contract["schema_version"],
            "meta_dataset_sha256": contract["dataset_sha256"],
            "meta_feature_schema_sha256": contract["feature_schema_sha256"],
            "meta_feature_prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "meta_label_config_sha256": contract["label_config_sha256"],
            "meta_model_sha256": model_sha256,
            "meta_piou_source_schema": contract["piou_source_schema"],
            "meta_piou_station_id": piou_station_id,
            "meta_piou_source": (
                "local_archive" if args.piou_input_dir else "public_archive_api"
            ),
            "meta_weather_source_schema": contract["weather_source_schema"],
            "meta_weather_station_id": weather_station_id,
        }
    )
    if role == "spatial":
        row["meta_weather_station_manifest_sha256"] = contract[
            "weather_station_manifest_sha256"
        ]
    write_feature_row(row, args.output)
    print(
        json.dumps(
            {
                "date": local_day.isoformat(),
                "feature_columns": len(row),
                "mf_last_age_minutes": weather_age,
                "output": str(args.output),
                "piou_last_age_minutes": piou["piou_last_age_minutes"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
