#!/usr/bin/env python3
"""Train the same-day Traverse classifier."""

from __future__ import annotations

import argparse
from importlib import metadata as importlib_metadata
import json
import os
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import threadpoolctl

from pioupiou.data.daily import label_config_from_payload
from pioupiou.inference.model import (
    load_dataset,
    save_model_bundle,
    save_json,
    sha256_file,
    sha256_json,
    train_and_evaluate,
)


PROVENANCE_FILES = (
    "pioupiou/data/daily.py",
    "pioupiou/data/timestep.py",
    "pioupiou/inference/model.py",
    "scripts/build_timestep_dataset.py",
    "scripts/prepare_timestep.py",
    "scripts/train.py",
    "scripts/predict.py",
    ".python-version",
    "pyproject.toml",
    "uv.lock",
    "tests/test_observation_features.py",
    "tests/test_timestep_traverse.py",
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
    parser.add_argument(
        "--dataset", type=Path, default=Path("artifacts/traverse_timestep.csv")
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Dataset metadata JSON (default: DATASET with .metadata.json suffix)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use an earlier reduced split and one regularization value",
    )
    parser.add_argument("--wandb-project", default="pioupiou-traverse")
    parser.add_argument("--wandb-name", required=True)
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument(
        "--wandb-mode", choices=("auto", "online", "offline", "disabled"), default="auto"
    )
    return parser


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}

    def visit(prefix: str, values: dict[str, Any]) -> None:
        for name, value in values.items():
            key = f"{prefix}/{name}" if prefix else name
            if isinstance(value, dict):
                visit(key, value)
            elif np.isfinite(value):
                flat[key] = float(value)

    visit("", metrics)
    return flat


def main() -> int:
    args = build_parser().parse_args()
    metadata_path = args.metadata or args.dataset.with_suffix(".metadata.json")
    if not metadata_path.exists():
        raise SystemExit(f"Dataset metadata is required: {metadata_path}")
    dataset_metadata = json.loads(metadata_path.read_text())
    dataset_sha256 = sha256_file(args.dataset)
    if dataset_metadata.get("output_sha256") != dataset_sha256:
        raise SystemExit("Dataset SHA-256 does not match its metadata")
    frame = load_dataset(args.dataset)
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
    if "issue_minutes" not in frame.columns:
        raise SystemExit("Training requires a same-day timestep dataset")
    anticipation = dataset_metadata.get("anticipation")
    if not isinstance(anticipation, dict):
        raise SystemExit("Training requires onset-aware dataset metadata")
    required_anticipation_columns = {
        "meta_anticipation_weight",
        "meta_minutes_before_onset",
    }
    missing_anticipation_columns = required_anticipation_columns.difference(
        frame.columns
    )
    if missing_anticipation_columns:
        raise SystemExit(
            "Training requires anticipation columns: "
            f"{sorted(missing_anticipation_columns)}"
        )
    l2_candidates = (1.0, 10.0)
    bundle, metrics = train_and_evaluate(
        frame,
        l2_candidates=l2_candidates,
        smoke=args.smoke,
        label_config=dataset_metadata["label_config"],
    )
    artifact = bundle["metadata"]
    artifact["dataset"] = str(args.dataset)
    artifact["anticipation"] = anticipation
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
    suffix = "-smoke" if args.smoke else ""
    model_path = args.output_dir / f"traverse_model{suffix}.joblib"
    model_metadata_path = args.output_dir / f"traverse_model{suffix}.metadata.json"
    metrics_path = args.output_dir / f"traverse_metrics{suffix}.json"
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
            "smoke": args.smoke,
            "model_sha256": model_sha256,
            "metrics": metrics,
        },
        metrics_path,
    )

    flat = flatten_metrics(metrics)
    flat.update(
        {
            "train/log_loss": metrics["train"]["log_loss"],
            "data/train_rows": float(artifact["split"]["train_rows"]),
            "model/l2": float(artifact["model"]["l2"]),
            "model/n_iter": float(artifact["model"]["iterations"]),
            "preprocessing/output_features": float(len(artifact["feature_names"])),
            "validation/selected_threshold": float(artifact["model"]["threshold"]),
        }
    )
    mode = args.wandb_mode
    if mode == "auto":
        mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
    if mode != "disabled":
        import wandb

        configured_l2_candidates = [1.0] if args.smoke else list(l2_candidates)
        selection_years = [2018, 2019] if args.smoke else [2020, 2021, 2022]
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            tags=[tag for tag in args.wandb_tags.split(",") if tag],
            mode=mode,
            config={
                "smoke": args.smoke,
                "dataset": str(args.dataset),
                "dataset_metadata": str(metadata_path),
                "dataset_sha256": artifact["provenance"]["dataset_sha256"],
                "source_sha256": artifact["provenance"]["source_sha256"],
                "features": artifact["feature_names"],
                "model_backend": "sklearn",
                "sklearn_version": sklearn.__version__,
                "estimator_class": "sklearn.ensemble.HistGradientBoostingClassifier",
                "fit_weighting": artifact["model"]["fit_weighting"],
                "learning_rate": artifact["model"]["learning_rate"],
                "max_iter": 200,
                "max_leaf_nodes": artifact["model"]["max_leaf_nodes"],
                "min_samples_leaf": artifact["model"]["min_samples_leaf"],
                "early_stopping": False,
                "random_state": 20260807,
                "preprocessing": {
                    "imputer": "median",
                    "keep_empty_features": True,
                },
                "regularization": {
                    "parameterization": "l2_regularization",
                    "l2_candidates": configured_l2_candidates,
                    "selection_metric": "event_day_3h_average_precision",
                    "folds": "expanding_year",
                    "validation_years": selection_years,
                    "tie_break": "weaker_regularization",
                },
                "threshold_policy": {
                    "selection_split": "validation",
                    "objective": "event_day_3h_balanced_accuracy",
                    "tie_break": "f1_then_closest_to_0.5",
                    "decision_rule": "probability_gte_threshold",
                },
                "metrics_backend": "sklearn.metrics",
                "model_schema_version": 2,
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
                "pioupiou-traverse-timestep",
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
                "pioupiou-traverse-model-sklearn-v2",
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
        "smoke": args.smoke,
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "threshold": artifact["model"]["threshold"],
        "l2": artifact["model"]["l2"],
        "test": metrics["test"],
        "deployment_later": {
            name: values
            for name, values in metrics.items()
            if name.startswith("deployment_later_")
        },
        "onset_deployment_later": {
            name: values
            for name, values in metrics.items()
            if name.startswith("onset_deployment_later_")
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
