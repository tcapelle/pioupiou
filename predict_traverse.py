#!/usr/bin/env python3
"""Score one prepared noon feature row with a trained Traverse model."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from build_traverse_dataset import (
    METEO_FRANCE_SOURCE_SCHEMA,
    PIOU_SOURCE_SCHEMA,
    label_config_from_payload,
    local_boundary,
)
from traverse_model import (
    load_artifact_with_sha256,
    predict_loaded,
    sha256_file,
    sha256_json,
)


def validate_prepared_contract(
    model_path: Path, pd_row: pd.DataFrame, loaded_artifact=None
) -> None:
    payload, _, model_sha256 = (
        loaded_artifact
        if loaded_artifact is not None
        else load_artifact_with_sha256(model_path)
    )
    row = pd_row
    role = payload.get("role")
    if role not in {"variant", "spatial"}:
        raise ValueError("invalid_model: prepared rows require a weather-based model")
    contract = payload.get("input_contract")
    expected_schema_version = 3 if role == "spatial" else 2
    if (
        not isinstance(contract, dict)
        or contract.get("schema_version") != expected_schema_version
    ):
        raise ValueError("invalid_model: missing or unsupported input contract")
    if (
        contract.get("piou_source_schema") != PIOU_SOURCE_SCHEMA
        or contract.get("weather_source_schema") != METEO_FRANCE_SOURCE_SCHEMA
    ):
        raise ValueError("invalid_model: unsupported source schema")
    required = {
        "date",
        "meta_contract_schema_version",
        "meta_dataset_sha256",
        "meta_feature_cutoff_local",
        "meta_feature_prepared_at_utc",
        "meta_feature_schema_sha256",
        "meta_label_config_sha256",
        "meta_model_sha256",
        "meta_piou_source_schema",
        "meta_piou_station_id",
        "meta_weather_source_schema",
        "meta_weather_station_id",
    }
    if role == "spatial":
        required.add("meta_weather_station_manifest_sha256")
    missing = sorted(required.difference(row.columns))
    if missing:
        raise ValueError(f"incompatible_features: missing contract columns {missing}")
    expected_features = set(payload.get("feature_names", []))
    supplied_features = {
        name
        for name in row.columns
        if name.startswith(("cal_", "piou_", "mf_", "mfs_"))
    }
    if supplied_features != expected_features:
        missing_features = sorted(expected_features - supplied_features)
        extra_features = sorted(supplied_features - expected_features)
        raise ValueError(
            "incompatible_features: modeled column mismatch; "
            f"missing={missing_features}, extra={extra_features}"
        )

    def value(name: str) -> str:
        return str(row.iloc[0][name])

    expected = {
        "meta_dataset_sha256": contract["dataset_sha256"],
        "meta_feature_schema_sha256": contract["feature_schema_sha256"],
        "meta_label_config_sha256": contract["label_config_sha256"],
        "meta_model_sha256": model_sha256,
        "meta_piou_source_schema": contract["piou_source_schema"],
        "meta_piou_station_id": str(contract["piou_station_id"]),
        "meta_weather_source_schema": contract["weather_source_schema"],
        "meta_weather_station_id": str(contract["weather_station_id"]),
    }
    if role == "spatial":
        station_manifest = contract.get("weather_station_manifest")
        station_manifest_sha256 = contract.get("weather_station_manifest_sha256")
        if station_manifest_sha256 != sha256_json(station_manifest):
            raise ValueError("invalid_model: weather station manifest fingerprint mismatch")
        expected["meta_weather_station_manifest_sha256"] = station_manifest_sha256
    mismatches = [
        name for name, expected_value in expected.items() if value(name) != expected_value
    ]
    if int(float(value("meta_contract_schema_version"))) != int(contract["schema_version"]):
        mismatches.append("meta_contract_schema_version")
    if contract["label_config_sha256"] != sha256_json(payload.get("label")):
        raise ValueError("invalid_model: label contract fingerprint mismatch")
    if contract["feature_schema_sha256"] != sha256_json(
        payload.get("feature_names")
    ):
        raise ValueError("invalid_model: feature schema fingerprint mismatch")
    if mismatches:
        raise ValueError(
            f"incompatible_features: contract mismatch in {sorted(set(mismatches))}"
        )

    config = label_config_from_payload(payload.get("label"))
    local_day = date.fromisoformat(value("date"))
    expected_cutoff = local_boundary(
        local_day, config.cutoff_hour, ZoneInfo(config.timezone_name)
    )
    try:
        supplied_cutoff = datetime.fromisoformat(value("meta_feature_cutoff_local"))
        prepared_at = datetime.fromisoformat(value("meta_feature_prepared_at_utc"))
    except ValueError as error:
        raise ValueError("incompatible_features: invalid contract timestamp") from error
    if supplied_cutoff.tzinfo is None or supplied_cutoff != expected_cutoff:
        raise ValueError("incompatible_features: cutoff does not match model contract")
    if supplied_cutoff.astimezone(timezone.utc) > datetime.now(timezone.utc):
        raise ValueError("incompatible_features: cutoff has not occurred yet")
    if prepared_at.tzinfo is None or prepared_at < supplied_cutoff:
        raise ValueError("incompatible_features: row was prepared before its cutoff")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("artifacts/traverse_model_variant.joblib")
    )
    parser.add_argument(
        "--model-sha256",
        help="Trusted out-of-band SHA-256 to verify before joblib deserialization",
    )
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/traverse_daily.csv"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--date", help="Select a prepared historical day from --dataset")
    source.add_argument("--features", type=Path, help="CSV containing one prepared noon row")
    args = parser.parse_args()

    if args.features:
        row = pd.read_csv(args.features)
        if len(row) != 1:
            raise SystemExit(f"Expected exactly one row in {args.features}, found {len(row)}")
        prediction_date = str(row.iloc[0].get("date", "unknown"))
        try:
            expected_model_sha256 = args.model_sha256
            if expected_model_sha256 is None and "meta_model_sha256" in row:
                expected_model_sha256 = str(row.iloc[0]["meta_model_sha256"])
            loaded_artifact = load_artifact_with_sha256(
                args.model, expected_model_sha256
            )
            validate_prepared_contract(args.model, row, loaded_artifact)
        except (ValueError, KeyError) as error:
            print(json.dumps({"date": prediction_date, "status": str(error)}, indent=2))
            return 2
    else:
        try:
            loaded_artifact = load_artifact_with_sha256(
                args.model, args.model_sha256
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        model_payload = loaded_artifact[0]
        expected_dataset_hash = model_payload.get("provenance", {}).get("dataset_sha256")
        if not expected_dataset_hash or sha256_file(args.dataset) != expected_dataset_hash:
            raise SystemExit("Historical dataset SHA-256 does not match the model artifact")
        frame = pd.read_csv(args.dataset)
        row = frame[frame["date"].astype(str) == args.date]
        if len(row) != 1:
            raise SystemExit(f"Expected exactly one prepared row for {args.date}, found {len(row)}")
        prediction_date = args.date
    try:
        probability, predicted, contributions = predict_loaded(
            loaded_artifact[0], loaded_artifact[1], row
        )
    except ValueError as error:
        print(json.dumps({"date": prediction_date, "status": str(error)}, indent=2))
        return 2
    print(
        json.dumps(
            {
                "date": prediction_date,
                "status": "ok",
                "traverse_probability": float(probability[0]),
                "predict_traverse": bool(predicted[0]),
                "top_logit_contributions": [
                    {"feature": name, "contribution": value}
                    for name, value in contributions[0]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
