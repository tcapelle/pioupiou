"""Scikit-learn modeling utilities for the noon Traverse classifier."""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import math
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
import tempfile
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_PREFIXES = {
    "baseline": ("cal_",),
    "variant": ("cal_", "piou_", "mf_"),
    "spatial": ("cal_", "piou_", "mf_", "mfs_"),
    "nwp": ("cal_", "piou_", "mf_", "nwp_"),
    "same_day": ("cal_", "piou_", "mf_", "lag_"),
}
RANDOM_STATE = 20260807


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(np.asarray(y_true, dtype=int), scores))


def classification_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    recall = recall_score(y, predicted, zero_division=0)
    specificity = tn / (tn + fp) if tn + fp else 0.0
    curve_precision, curve_recall, _ = precision_recall_curve(y, probabilities)
    eligible_precision = curve_precision[curve_recall >= 0.60]
    return {
        "rows": float(len(y)),
        "prevalence": float(np.mean(y)) if len(y) else float("nan"),
        "average_precision": float(average_precision_score(y, probabilities)),
        "roc_auc": (
            float(roc_auc_score(y, probabilities))
            if np.unique(y).size == 2
            else float("nan")
        ),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "precision_at_recall_0_60": (
            float(np.max(eligible_precision)) if len(eligible_precision) else 0.0
        ),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "f2": float(fbeta_score(y, predicted, beta=2.0, zero_division=0)),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Choose the validation threshold with best balanced accuracy, then F1."""
    candidates = np.unique(np.r_[0.0, probabilities, 1.0])
    best_threshold = 0.5
    best_key = (-math.inf, -math.inf, -math.inf)
    y = np.asarray(y_true, dtype=int)
    for threshold in candidates:
        predicted = (probabilities >= threshold).astype(int)
        key = (
            float(balanced_accuracy_score(y, predicted)),
            float(f1_score(y, predicted, zero_division=0)),
            -abs(float(threshold) - 0.5),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def numeric_feature_frame(frame: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    return frame.reindex(columns=feature_names).apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)


def build_pipeline(l2: float, feature_names: Sequence[str]) -> Pipeline:
    if l2 <= 0:
        raise ValueError("L2 strength must be positive")
    names = list(feature_names)
    if not names:
        raise ValueError("At least one feature is required")
    values = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("values", values, names),
            (
                "missing",
                MissingIndicator(features="all", sparse=False),
                names,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    penalty="l2",
                    C=1.0 / float(l2),
                    solver="lbfgs",
                    max_iter=3000,
                    tol=1e-8,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def fit_pipeline(
    pipeline: Pipeline, frame: pd.DataFrame, target: np.ndarray
) -> Pipeline:
    """Fit and fail loudly if sklearn cannot converge or yields non-finite state."""
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        # Apple's Accelerate BLAS can emit spurious matmul floating-point warnings
        # while lbfgs explores trial weights. Validate the fitted state explicitly.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            pipeline.fit(frame, target)
    transformed = pipeline.named_steps["preprocessor"].transform(frame)
    classifier = pipeline.named_steps["classifier"]
    if not np.isfinite(transformed).all():
        raise ValueError("Non-finite values remain after sklearn preprocessing")
    if not np.isfinite(classifier.coef_).all() or not np.isfinite(
        classifier.intercept_
    ).all():
        raise ValueError("The fitted sklearn classifier contains non-finite parameters")
    return pipeline


def predict_probabilities(
    pipeline: Pipeline, frame: pd.DataFrame, feature_names: Sequence[str]
) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        probability = pipeline.predict_proba(
            numeric_feature_frame(frame, feature_names)
        )[:, 1]
    if not np.isfinite(probability).all():
        raise ValueError("The sklearn classifier produced non-finite probabilities")
    return probability


def feature_names_for_role(frame: pd.DataFrame, role: str) -> list[str]:
    if role not in FEATURE_PREFIXES:
        raise ValueError(f"Unknown role {role!r}; expected one of {sorted(FEATURE_PREFIXES)}")
    prefixes = FEATURE_PREFIXES[role]
    names = sorted(
        column
        for column in frame.columns
        if column.startswith(prefixes)
        and pd.to_numeric(frame[column], errors="coerce").notna().any()
    )
    if not names:
        raise ValueError(f"No feature columns found for {role!r} with prefixes {prefixes}")
    return names


def expanding_year_l2_search(
    training: pd.DataFrame,
    feature_names: Sequence[str],
    candidates: Iterable[float],
) -> tuple[float, dict[str, float]]:
    years = sorted(int(year) for year in training["year"].unique())
    validation_years = years[-3:] if len(years) >= 4 else years[1:]
    scores: dict[str, float] = {}
    for candidate in candidates:
        fold_scores: list[float] = []
        for validation_year in validation_years:
            fold_train = training[training["year"] < validation_year]
            fold_valid = training[training["year"] == validation_year]
            if len(fold_train) == 0 or fold_valid["label"].nunique() < 2:
                continue
            pipeline = build_pipeline(float(candidate), feature_names)
            fit_pipeline(
                pipeline,
                numeric_feature_frame(fold_train, feature_names),
                fold_train["label"].to_numpy(dtype=int),
            )
            probability = predict_probabilities(pipeline, fold_valid, feature_names)
            fold_scores.append(
                float(average_precision_score(fold_valid["label"].to_numpy(), probability))
            )
        scores[str(candidate)] = float(np.mean(fold_scores)) if fold_scores else float("nan")
    finite = [
        (score, -float(candidate), float(candidate))
        for candidate, score in scores.items()
        if np.isfinite(score)
    ]
    if not finite:
        raise ValueError("No valid expanding-year folds for L2 selection")
    return max(finite)[2], scores


def load_daily_dataset(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "year", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    identity = ["date"]
    if "issue_minutes" in frame:
        frame["issue_minutes"] = pd.to_numeric(
            frame["issue_minutes"], errors="raise"
        ).astype(int)
        if (~frame["issue_minutes"].between(0, 1439)).any():
            raise ValueError("Dataset issue_minutes values must be between 0 and 1439")
        identity.append("issue_minutes")
    if frame.duplicated(identity).any():
        duplicates = frame.loc[frame.duplicated(identity), identity].head(5)
        if identity == ["date"]:
            formatted = duplicates["date"].dt.strftime("%Y-%m-%d").tolist()
            raise ValueError(f"Dataset contains duplicate dates: {formatted}")
        raise ValueError(
            f"Dataset contains duplicate sample identities: {duplicates.to_dict('records')}"
        )
    expected_year = frame["date"].dt.year
    if not np.array_equal(frame["year"].to_numpy(), expected_year.to_numpy()):
        raise ValueError("Dataset year column does not match the date column")
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    frame = frame[frame["label"].isin([0, 1])].copy()
    frame["label"] = frame["label"].astype(int)
    return frame.sort_values(identity).reset_index(drop=True)


def split_dataset(
    frame: pd.DataFrame, smoke: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if smoke:
        train = frame[(frame["year"] >= 2017) & (frame["year"] <= 2019)]
        validation = frame[frame["year"] == 2020]
        test = frame[frame["year"] == 2021]
    else:
        train = frame[(frame["year"] >= 2017) & (frame["year"] <= 2022)]
        validation = frame[frame["year"] == 2023]
        test = frame[(frame["year"] >= 2024) & (frame["year"] <= 2025)]
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Chronological split is empty; check dataset coverage")
    return train, validation, test


def train_and_evaluate(
    frame: pd.DataFrame,
    role: str,
    l2_candidates: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    smoke: bool = False,
    label_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    train, validation, test = split_dataset(frame, smoke=smoke)
    feature_names = feature_names_for_role(train, role)
    candidates = (1.0,) if smoke else l2_candidates
    best_l2, cv_scores = expanding_year_l2_search(train, feature_names, candidates)
    pipeline = build_pipeline(best_l2, feature_names)
    fit_pipeline(
        pipeline,
        numeric_feature_frame(train, feature_names), train["label"].to_numpy(dtype=int)
    )
    validation_probability = predict_probabilities(pipeline, validation, feature_names)
    threshold = select_threshold(validation["label"].to_numpy(), validation_probability)
    metrics = {
        split_name: classification_metrics(
            subset["label"].to_numpy(dtype=int),
            predict_probabilities(pipeline, subset, feature_names),
            threshold,
        )
        for split_name, subset in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        )
    }
    for year in sorted(test["year"].unique()):
        subset = test[test["year"] == year]
        metrics[f"test_{int(year)}"] = classification_metrics(
            subset["label"].to_numpy(dtype=int),
            predict_probabilities(pipeline, subset, feature_names),
            threshold,
        )
    if "issue_minutes" in test.columns:
        for issue_minutes in sorted(test["issue_minutes"].unique()):
            subset = test[test["issue_minutes"] == issue_minutes]
            hour, minute = divmod(int(issue_minutes), 60)
            metrics[f"test_time_{hour:02d}{minute:02d}"] = classification_metrics(
                subset["label"].to_numpy(dtype=int),
                predict_probabilities(pipeline, subset, feature_names),
                threshold,
            )
    classifier = pipeline.named_steps["classifier"]
    metadata = {
        "schema_version": 2,
        "artifact_format": "joblib",
        "role": role,
        "feature_prefixes": list(FEATURE_PREFIXES[role]),
        "feature_names": feature_names,
        "estimator": {
            "library": "scikit-learn",
            "library_version": sklearn.__version__,
            "pipeline": [
                "ColumnTransformer(values, missingness)",
                "SimpleImputer(strategy='median', keep_empty_features=True)",
                "StandardScaler(values_only)",
                "MissingIndicator(features='all')",
                "LogisticRegression(penalty='l2', solver='lbfgs')",
            ],
            "random_state": RANDOM_STATE,
        },
        "model": {
            "threshold": threshold,
            "l2": best_l2,
            "C": 1.0 / best_l2,
            "iterations": int(np.max(classifier.n_iter_)),
        },
        "selection": {"l2_average_precision_by_candidate": cv_scores},
        "split": {
            "train_years": sorted(int(value) for value in train["year"].unique()),
            "validation_years": sorted(int(value) for value in validation["year"].unique()),
            "test_years": sorted(int(value) for value in test["year"].unique()),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "prediction_minutes": (
                sorted(int(value) for value in frame["issue_minutes"].unique())
                if "issue_minutes" in frame.columns
                else []
            ),
        },
        "label": label_config or {"status": "not_supplied"},
    }
    return {"metadata": metadata, "pipeline": pipeline}, metrics


def save_json(payload: dict[str, Any], path: Path | str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def save_model_bundle(bundle: dict[str, Any], path: Path | str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=target.parent, prefix=f".{target.name}.", suffix=".part", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        joblib.dump(bundle, temporary, compress=3)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _deserialize_artifact(raw: bytes) -> tuple[dict[str, Any], Pipeline]:
    """Deserialize trusted project output; joblib must never receive untrusted bytes."""
    try:
        bundle = joblib.load(io.BytesIO(raw))
    except Exception as error:
        raise ValueError(
            "Invalid or unsupported Traverse model bundle; retrain as schema-v2 joblib"
        ) from error
    if not isinstance(bundle, dict) or not isinstance(bundle.get("metadata"), dict):
        raise ValueError("Invalid Traverse model bundle")
    metadata = bundle["metadata"]
    if metadata.get("schema_version") != 2 or metadata.get("artifact_format") != "joblib":
        raise ValueError("Unsupported Traverse model schema; expected schema-v2 joblib")
    trained_version = metadata.get("estimator", {}).get("library_version")
    if trained_version != sklearn.__version__:
        raise ValueError(
            "Unsupported scikit-learn version for this model: "
            f"trained with {trained_version!r}, running {sklearn.__version__!r}; retrain it"
        )
    pipeline = bundle.get("pipeline")
    if not isinstance(pipeline, Pipeline):
        raise ValueError("Traverse model bundle does not contain a scikit-learn Pipeline")
    return metadata, pipeline


def load_artifact_with_sha256(
    path: Path | str, expected_sha256: str | None = None
) -> tuple[dict[str, Any], Pipeline, str]:
    """Read once so validation, hashing, and scoring bind to identical bytes."""
    raw = Path(path).read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and not hmac.compare_digest(
        actual_sha256, expected_sha256
    ):
        raise ValueError("Traverse model SHA-256 does not match the trusted value")
    metadata, pipeline = _deserialize_artifact(raw)
    return metadata, pipeline, actual_sha256


def load_artifact(path: Path | str) -> tuple[dict[str, Any], Pipeline]:
    metadata, pipeline, _ = load_artifact_with_sha256(path)
    return metadata, pipeline


def predict_loaded(
    payload: dict[str, Any], pipeline: Pipeline, frame: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[list[tuple[str, float]]]]:
    feature_names = list(payload["feature_names"])
    if payload["role"] in {
        "variant",
        "spatial",
        "nwp",
        "same_day",
    }:
        guard_columns = {
            "piou_observation_count_morning",
            "piou_last_age_minutes",
            "mf_core_observation_count_morning",
            "mf_last_age_minutes",
        }
        if payload["role"] == "nwp":
            guard_columns.update(
                {"nwp_core_observation_count_morning", "nwp_last_age_minutes"}
            )
        artifact_missing = sorted(guard_columns.difference(feature_names))
        if artifact_missing:
            raise ValueError(f"invalid_model: missing feed guard features {artifact_missing}")
        row_missing = sorted(guard_columns.difference(frame.columns))
        if row_missing:
            raise ValueError(f"insufficient_data: missing required feed columns {row_missing}")
        feed_guards = [
            (
                "piou_",
                "piou_observation_count_morning",
                "piou_last_age_minutes",
                "maximum_feature_age_minutes",
                30.0,
            ),
            (
                "mf_",
                "mf_core_observation_count_morning",
                "mf_last_age_minutes",
                "maximum_weather_feature_age_minutes",
                90.0,
            ),
        ]
        if payload["role"] == "nwp":
            feed_guards.append(
                (
                    "nwp_",
                    "nwp_core_observation_count_morning",
                    "nwp_last_age_minutes",
                    "maximum_weather_feature_age_minutes",
                    90.0,
                )
            )
        for prefix, count_name, age_name, maximum_age_key, default_maximum_age in feed_guards:
            physical = [
                name
                for name in feature_names
                if name.startswith(prefix)
                and not name.endswith("_count_morning")
                and not name.endswith("_age_minutes")
            ]
            numeric = numeric_feature_frame(frame, physical)
            missing_rows = ~np.isfinite(numeric.to_numpy(dtype=float)).any(axis=1)
            counts = pd.to_numeric(frame[count_name], errors="coerce").to_numpy(dtype=float)
            missing_rows |= ~np.isfinite(counts) | (counts <= 0)
            if missing_rows.any():
                raise ValueError(
                    f"insufficient_data: unavailable {prefix} feed for row positions "
                    f"{np.flatnonzero(missing_rows).tolist()}"
                )
            ages = pd.to_numeric(frame[age_name], errors="coerce").to_numpy(dtype=float)
            maximum_age = float(
                payload.get("label", {}).get(maximum_age_key, default_maximum_age)
            )
            stale = ~np.isfinite(ages) | (ages < 0) | (ages > maximum_age)
            if stale.any():
                raise ValueError(
                    f"insufficient_data: stale {prefix} feed for row positions "
                    f"{np.flatnonzero(stale).tolist()}"
                )
    numeric = numeric_feature_frame(frame, feature_names)
    probability = predict_probabilities(pipeline, numeric, feature_names)
    threshold = float(payload["model"]["threshold"])
    predicted = probability >= threshold
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(numeric)
    transformed_names = preprocessor.get_feature_names_out()
    coefficients = pipeline.named_steps["classifier"].coef_[0]
    contribution_rows: list[list[tuple[str, float]]] = []
    for transformed_row in transformed:
        contributions = list(zip(transformed_names, transformed_row * coefficients))
        contributions.sort(key=lambda item: abs(item[1]), reverse=True)
        contribution_rows.append([(str(name), float(value)) for name, value in contributions[:8]])
    return probability, predicted, contribution_rows


def predict_frame(
    artifact_path: Path | str, frame: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[list[tuple[str, float]]]]:
    """Convenience wrapper for trusted local artifacts."""
    payload, pipeline = load_artifact(artifact_path)
    return predict_loaded(payload, pipeline, frame)
