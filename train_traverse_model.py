#!/usr/bin/env python3
"""Train a noon Traverse logistic classifier for one feature role."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib import metadata as importlib_metadata
import json
import os
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import threadpoolctl

from build_traverse_dataset import (
    METEO_FRANCE_SOURCE_SCHEMA,
    PIOU_SOURCE_SCHEMA,
    WEATHER_STATIONS,
    label_config_from_payload,
)
from open_meteo_features import (
    OPEN_METEO_LATITUDE,
    OPEN_METEO_LONGITUDE,
    OPEN_METEO_MODEL,
    OPEN_METEO_SOURCE_SCHEMA,
    OPEN_METEO_VARIABLES,
)
from traverse_model import (
    load_daily_dataset,
    save_model_bundle,
    save_json,
    sha256_file,
    sha256_json,
    train_and_evaluate,
)


PROVENANCE_FILES = (
    "build_traverse_dataset.py",
    "open_meteo_features.py",
    "prepare_noon_features.py",
    "traverse_model.py",
    "train_traverse_model.py",
    "predict_traverse.py",
    "compare_traverse_models.py",
    ".python-version",
    "pyproject.toml",
    "uv.lock",
    "tests/test_build_traverse_dataset.py",
    "tests/test_open_meteo_features.py",
    "tests/test_traverse_model.py",
)


def runtime_provenance() -> tuple[dict[str, object], list[str]]:
    packages = sorted(
        {
            f"{distribution.metadata.get('Name', 'unknown')}=={distribution.version}"
            for distribution in importlib_metadata.distributions()
        },
        key=str.casefold,
    )
    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "threadpoolctl": importlib_metadata.version("threadpoolctl"),
        "threadpools": threadpoolctl.threadpool_info(),
        "installed_packages_sha256": sha256_json(packages),
    }
    return runtime, packages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/traverse_daily.csv"))
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Dataset metadata JSON (default: DATASET with .metadata.json suffix)",
    )
    parser.add_argument(
        "--role", choices=("baseline", "variant", "spatial", "nwp"), required=True
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use an earlier reduced split and one L2 value",
    )
    parser.add_argument("--wandb-project", default="pioupiou-traverse")
    parser.add_argument("--wandb-name", required=True)
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument(
        "--wandb-mode", choices=("auto", "online", "offline", "disabled"), default="auto"
    )
    return parser


def flatten_metrics(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        f"{split}/{name}": float(value)
        for split, values in metrics.items()
        for name, value in values.items()
        if np.isfinite(value)
    }


def main() -> int:
    args = build_parser().parse_args()
    metadata_path = args.metadata or args.dataset.with_suffix(".metadata.json")
    if not metadata_path.exists():
        raise SystemExit(f"Dataset metadata is required: {metadata_path}")
    dataset_metadata = json.loads(metadata_path.read_text())
    dataset_sha256 = sha256_file(args.dataset)
    if dataset_metadata.get("output_sha256") != dataset_sha256:
        raise SystemExit("Dataset SHA-256 does not match its metadata")
    frame = load_daily_dataset(args.dataset)
    if int(dataset_metadata.get("row_count", -1)) != len(frame):
        raise SystemExit("Dataset row count does not match its metadata")
    actual_range = [
        frame["date"].min().strftime("%Y-%m-%d"),
        frame["date"].max().strftime("%Y-%m-%d"),
    ]
    if dataset_metadata.get("date_range") != actual_range:
        raise SystemExit("Dataset date range does not match its metadata")
    try:
        label_config_from_payload(dataset_metadata.get("label_config"))
    except ValueError as error:
        raise SystemExit(f"Invalid dataset label contract: {error}") from error
    piou_station_id = dataset_metadata.get("pioupiou", {}).get("station_id")
    weather_station_id = dataset_metadata.get("meteofrance", {}).get("station_id")
    if piou_station_id is None or weather_station_id is None:
        raise SystemExit("Dataset metadata must identify both source stations")
    if dataset_metadata["pioupiou"].get("source_schema") != PIOU_SOURCE_SCHEMA or (
        dataset_metadata["meteofrance"].get("source_schema")
        != METEO_FRANCE_SOURCE_SCHEMA
    ):
        raise SystemExit("Dataset metadata has an unsupported source schema")
    station_manifest = dataset_metadata.get("meteofrance", {}).get("station_manifest")
    expected_station_manifest = [asdict(station) for station in WEATHER_STATIONS]
    if args.role == "spatial" and station_manifest != expected_station_manifest:
        raise SystemExit("Spatial training requires the configured weather station manifest")
    open_meteo_metadata = dataset_metadata.get("open_meteo", {})
    if args.role == "nwp" and (
        not open_meteo_metadata.get("enabled")
        or open_meteo_metadata.get("source_schema") != OPEN_METEO_SOURCE_SCHEMA
        or open_meteo_metadata.get("model") != OPEN_METEO_MODEL
        or open_meteo_metadata.get("variables") != list(OPEN_METEO_VARIABLES)
    ):
        raise SystemExit("NWP training requires the configured ECMWF/Open-Meteo dataset")
    bundle, metrics = train_and_evaluate(
        frame,
        args.role,
        smoke=args.smoke,
        label_config=dataset_metadata["label_config"],
    )
    artifact = bundle["metadata"]
    contract_schema_version = {"spatial": 3, "nwp": 4}.get(args.role, 2)
    artifact["input_contract"] = {
        "schema_version": contract_schema_version,
        "dataset_sha256": dataset_sha256,
        "label_config_sha256": sha256_json(dataset_metadata["label_config"]),
        "feature_schema_sha256": sha256_json(artifact["feature_names"]),
        "piou_station_id": str(piou_station_id),
        "piou_source_schema": PIOU_SOURCE_SCHEMA,
        "weather_station_id": str(weather_station_id),
        "weather_source_schema": METEO_FRANCE_SOURCE_SCHEMA,
    }
    if args.role == "spatial":
        artifact["input_contract"].update(
            {
                "weather_station_manifest": station_manifest,
                "weather_station_manifest_sha256": sha256_json(station_manifest),
            }
        )
    if args.role == "nwp":
        artifact["input_contract"].update(
            {
                "open_meteo_source_schema": OPEN_METEO_SOURCE_SCHEMA,
                "open_meteo_model": OPEN_METEO_MODEL,
                "open_meteo_coordinates": [
                    OPEN_METEO_LATITUDE,
                    OPEN_METEO_LONGITUDE,
                ],
                "open_meteo_variables": list(OPEN_METEO_VARIABLES),
            }
        )
    artifact["dataset"] = str(args.dataset)
    runtime, installed_packages = runtime_provenance()
    artifact["runtime"] = runtime
    artifact["provenance"] = {
        "dataset_sha256": dataset_sha256,
        "dataset_metadata": str(metadata_path),
        "dataset_metadata_sha256": sha256_file(metadata_path),
        "source_sha256": {
            name: sha256_file(Path(name)) for name in PROVENANCE_FILES
        },
    }
    artifact["metrics"] = metrics
    suffix = f"{args.role}{'-smoke' if args.smoke else ''}"
    model_path = args.output_dir / f"traverse_model_{suffix}.joblib"
    model_metadata_path = args.output_dir / f"traverse_model_{suffix}.metadata.json"
    metrics_path = args.output_dir / f"traverse_metrics_{suffix}.json"
    environment_path = args.output_dir / "python_environment.txt"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment_path.write_text("\n".join(installed_packages) + "\n")
    artifact["provenance"]["environment_freeze"] = str(environment_path)
    artifact["provenance"]["environment_freeze_sha256"] = sha256_file(environment_path)
    save_model_bundle(bundle, model_path)
    model_sha256 = sha256_file(model_path)
    save_json(
        {**artifact, "serialized_model_sha256": model_sha256}, model_metadata_path
    )
    save_json(
        {
            "role": args.role,
            "smoke": args.smoke,
            "model_sha256": model_sha256,
            "metrics": metrics,
        },
        metrics_path,
    )

    classifier = bundle["pipeline"].named_steps["classifier"]
    flat = flatten_metrics(metrics)
    flat.update(
        {
            "train/log_loss": metrics["train"]["log_loss"],
            "model/max_abs_coefficient": float(
                np.max(np.abs(classifier.coef_))
            ),
            "data/train_rows": float(artifact["split"]["train_rows"]),
            "model/l2": float(artifact["model"]["l2"]),
            "model/C": float(artifact["model"]["C"]),
            "model/n_iter": float(artifact["model"]["iterations"]),
            "model/converged": 1.0,
            "preprocessing/output_features": float(classifier.coef_.shape[1]),
            "validation/selected_threshold": float(artifact["model"]["threshold"]),
        }
    )
    for station, coverage in dataset_metadata["meteofrance"].get(
        "station_temperature_coverage", {}
    ).items():
        flat[f"data/{station}_temperature_coverage"] = float(coverage["fraction"])
    flat["data/piou_invalid_location_rows"] = float(
        dataset_metadata["pioupiou"]["counters"].get("invalid_location_rows", 0)
    )
    mode = args.wandb_mode
    if mode == "auto":
        mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
    if mode != "disabled":
        import wandb

        configured_l2_candidates = [1.0] if args.smoke else [0.01, 0.1, 1.0, 10.0]
        configured_c_candidates = [1.0 / value for value in configured_l2_candidates]
        selection_years = [2018, 2019] if args.smoke else [2020, 2021, 2022]
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            tags=[tag for tag in args.wandb_tags.split(",") if tag],
            mode=mode,
            config={
                "role": args.role,
                "smoke": args.smoke,
                "dataset": str(args.dataset),
                "dataset_metadata": str(metadata_path),
                "dataset_sha256": artifact["provenance"]["dataset_sha256"],
                "source_sha256": artifact["provenance"]["source_sha256"],
                "features": artifact["feature_names"],
                "model_backend": "sklearn",
                "sklearn_version": sklearn.__version__,
                "estimator_class": "sklearn.linear_model.LogisticRegression",
                "solver": "lbfgs",
                "penalty": "l2",
                "fit_intercept": True,
                "class_weight": None,
                "tol": 1e-8,
                "max_iter": 3000,
                "random_state": 20260807,
                "preprocessing": {
                    "imputer": "median",
                    "keep_empty_features": True,
                    "scaler": "standard_ddof0_values_only",
                    "missing_indicator": "all",
                    "indicator_scaled": False,
                },
                "regularization": {
                    "parameterization": "C",
                    "C_candidates": configured_c_candidates,
                    "equivalent_l2_candidates": configured_l2_candidates,
                    "mapping": "C=1/l2",
                    "selection_metric": "average_precision",
                    "folds": "expanding_year",
                    "validation_years": selection_years,
                    "tie_break": "largest_C",
                },
                "threshold_policy": {
                    "selection_split": "validation",
                    "objective": "balanced_accuracy",
                    "tie_break": "f1_then_closest_to_0.5",
                    "decision_rule": "probability_gte_threshold",
                },
                "metrics_backend": "sklearn.metrics",
                "model_schema_version": 2,
                "input_contract_schema_version": artifact["input_contract"][
                    "schema_version"
                ],
                "model_serialization": "joblib",
                "runtime_versions": {
                    key: runtime[key]
                    for key in (
                        "python",
                        "numpy",
                        "pandas",
                        "scipy",
                        "scikit_learn",
                        "joblib",
                        "threadpoolctl",
                    )
                },
                "environment_freeze_sha256": artifact["provenance"][
                    "environment_freeze_sha256"
                ],
                "split": artifact["split"],
                "label": artifact["label"],
            },
        )
        run.log(flat)
        run.summary["model_path"] = str(model_path)
        run.summary["model_sha256"] = model_sha256
        if not args.smoke:
            dataset_wandb_artifact = wandb.Artifact(
                "pioupiou-traverse-daily",
                type="dataset",
                metadata={
                    "dataset_sha256": artifact["provenance"]["dataset_sha256"],
                    "metadata_sha256": artifact["provenance"][
                        "dataset_metadata_sha256"
                    ],
                },
            )
            dataset_wandb_artifact.add_file(str(args.dataset))
            dataset_wandb_artifact.add_file(str(metadata_path))
            run.log_artifact(dataset_wandb_artifact)
            source_wandb_artifact = wandb.Artifact(
                "pioupiou-traverse-source-sklearn-v2",
                type="code",
                metadata={
                    "source_sha256": artifact["provenance"]["source_sha256"]
                },
            )
            for source_path in PROVENANCE_FILES:
                source_wandb_artifact.add_file(source_path)
            run.log_artifact(source_wandb_artifact)
            model_wandb_artifact = wandb.Artifact(
                f"pioupiou-traverse-{args.role}-sklearn-v2",
                type="model",
                metadata={"model_sha256": model_sha256, "schema_version": 2},
            )
            model_wandb_artifact.add_file(str(model_path))
            model_wandb_artifact.add_file(str(model_metadata_path))
            model_wandb_artifact.add_file(str(metrics_path))
            model_wandb_artifact.add_file(str(environment_path))
            run.log_artifact(model_wandb_artifact)
        run.finish()

    summary = {
        "role": args.role,
        "smoke": args.smoke,
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "threshold": artifact["model"]["threshold"],
        "l2": artifact["model"]["l2"],
        "C": artifact["model"]["C"],
        "test": metrics["test"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
