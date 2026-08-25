"""Scikit-learn modeling and inference utilities for Traverse classifiers."""

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

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
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


FEATURE_PREFIXES = ("cal_", "piou_", "mf_")
RANDOM_STATE = 20260807
DEFAULT_TRAVERSE_SPEED_KMH = 10.0 * 1.852


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    recall = recall_score(y, predicted, zero_division=0)
    specificity = tn / (tn + fp) if tn + fp else 0.0
    curve_precision, curve_recall, _ = precision_recall_curve(y, probabilities)
    eligible_precision = curve_precision[curve_recall >= 0.60]
    metrics = {
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
    if sample_weight is not None:
        weight = np.asarray(sample_weight, dtype=float)
        metrics.update(
            {
                "lead_weighted_average_precision": float(
                    average_precision_score(y, probabilities, sample_weight=weight)
                ),
                "lead_weighted_brier_score": float(
                    brier_score_loss(y, probabilities, sample_weight=weight)
                ),
                "lead_weighted_log_loss": float(
                    log_loss(y, probabilities, labels=[0, 1], sample_weight=weight)
                ),
            }
        )
    return metrics


def select_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Choose the validation threshold with best balanced accuracy, then F1."""
    candidates = np.unique(np.r_[0.0, probabilities, 1.0])
    best_threshold = 0.5
    best_key = (-math.inf, -math.inf, -math.inf)
    y = np.asarray(y_true, dtype=int)
    for threshold in candidates:
        predicted = (probabilities >= threshold).astype(int)
        key = (
            float(
                balanced_accuracy_score(
                    y, predicted, sample_weight=sample_weight
                )
            ),
            float(
                f1_score(
                    y,
                    predicted,
                    sample_weight=sample_weight,
                    zero_division=0,
                )
            ),
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
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=200,
                    learning_rate=0.05,
                    max_leaf_nodes=7,
                    min_samples_leaf=100,
                    l2_regularization=float(l2),
                    early_stopping=False,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def fit_pipeline(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    target: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> Pipeline:
    """Fit and fail loudly if preprocessing or probabilities are non-finite."""
    fit_parameters = (
        {"classifier__sample_weight": sample_weight}
        if sample_weight is not None
        else {}
    )
    pipeline.fit(frame, target, **fit_parameters)
    transformed = pipeline.named_steps["imputer"].transform(frame)
    if not np.isfinite(transformed).all():
        raise ValueError("Non-finite values remain after sklearn preprocessing")
    probability = pipeline.predict_proba(frame)[:, 1]
    if not np.isfinite(probability).all():
        raise ValueError("The fitted sklearn classifier produced non-finite probabilities")
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


def feature_names(frame: pd.DataFrame) -> list[str]:
    names = sorted(
        column
        for column in frame.columns
        if column.startswith(FEATURE_PREFIXES)
        and pd.to_numeric(frame[column], errors="coerce").notna().any()
    )
    if not names:
        raise ValueError(f"No feature columns found with prefixes {FEATURE_PREFIXES}")
    return names


def onset_evidence(
    frame: pd.DataFrame,
    speed_threshold_kmh: float = DEFAULT_TRAVERSE_SPEED_KMH,
) -> np.ndarray:
    """Latest local westerly component as a 0–1 fraction of Traverse speed."""
    if speed_threshold_kmh <= 0:
        raise ValueError("Traverse speed threshold must be positive")
    required = {"piou_last_wind_avg_kmh", "piou_last_heading_sin"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Onset evidence requires columns: {missing}")
    speed = pd.to_numeric(
        frame["piou_last_wind_avg_kmh"], errors="coerce"
    ).to_numpy(dtype=float)
    heading_sin = pd.to_numeric(
        frame["piou_last_heading_sin"], errors="coerce"
    ).to_numpy(dtype=float)
    evidence = speed * np.clip(-heading_sin, 0.0, 1.0) / speed_threshold_kmh
    evidence[~np.isfinite(evidence)] = 0.0
    return np.clip(evidence, 0.0, 1.0)


def onset_evidence_metrics(
    frame: pd.DataFrame, evidence: np.ndarray
) -> dict[str, float]:
    """Measure whether physical onset evidence rises during an event's final 6 h."""
    required = {"date", "label", "issue_minutes", "meta_minutes_before_onset"}
    if not required.issubset(frame.columns):
        return {}
    scored = frame.loc[:, list(required)].copy()
    scored["onset_evidence"] = np.asarray(evidence, dtype=float)
    scored = scored[
        (scored["label"] == 1)
        & (scored["meta_minutes_before_onset"] <= 360)
    ]
    slopes: list[float] = []
    for _, day in scored.groupby("date", sort=False):
        finite = day[np.isfinite(day["onset_evidence"])]
        if len(finite) < 3 or finite["issue_minutes"].nunique() < 2:
            continue
        slope_per_minute = np.polyfit(
            finite["issue_minutes"].to_numpy(dtype=float),
            finite["onset_evidence"].to_numpy(dtype=float),
            1,
        )[0]
        slopes.append(float(slope_per_minute * 60.0))
    if not slopes:
        return {}
    values = np.asarray(slopes)
    return {
        "events_with_onset_evidence_slope": float(len(values)),
        "event_rising_onset_evidence_rate_final_6h": float(np.mean(values > 0)),
        "median_onset_evidence_slope_per_hour_final_6h": float(np.median(values)),
    }


def anticipation_weights(frame: pd.DataFrame) -> np.ndarray:
    """Return validated lead-time weights, or uniform weights for daily data."""
    if "meta_anticipation_weight" not in frame.columns:
        return np.ones(len(frame), dtype=float)
    weights = pd.to_numeric(
        frame["meta_anticipation_weight"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(weights).all() or np.any(weights <= 0) or np.any(weights > 1):
        raise ValueError("Anticipation weights must be finite and in (0, 1]")
    if {"label", "meta_minutes_before_onset"}.issubset(frame.columns):
        labels = pd.to_numeric(frame["label"], errors="coerce").to_numpy(dtype=float)
        lead = pd.to_numeric(
            frame["meta_minutes_before_onset"], errors="coerce"
        ).to_numpy(dtype=float)
        positive = labels == 1
        if not np.isfinite(lead[positive]).all() or np.any(lead[positive] <= 0):
            raise ValueError("Positive anticipation rows must be strictly before onset")
        if np.isfinite(lead[~positive]).any():
            raise ValueError("Negative anticipation rows must not have an event onset")
        expected = np.ones(len(frame), dtype=float)
        expected[positive] = np.minimum(lead[positive] / 180.0, 1.0)
        if not np.allclose(weights, expected, rtol=0.0, atol=1e-12):
            raise ValueError("Anticipation weights do not match the lead-time policy")
    return weights


def day_normalized_anticipation_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give every day equal fit weight while preserving within-day lead value."""
    weights = anticipation_weights(frame)
    if "date" not in frame.columns:
        return weights
    day_totals = pd.Series(weights, index=frame.index).groupby(
        frame["date"], sort=False
    ).transform("sum").to_numpy(dtype=float)
    if not np.isfinite(day_totals).all() or np.any(day_totals <= 0):
        raise ValueError("Each training day must have positive finite total weight")
    return weights / day_totals


def anticipation_metrics(
    frame: pd.DataFrame, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    """Summarize warning performance by event day instead of pooled rows."""
    required = {"date", "label", "meta_minutes_before_onset"}
    if not required.issubset(frame.columns):
        return {}
    scored = frame.loc[:, ["date", "label", "meta_minutes_before_onset"]].copy()
    scored["alert"] = np.asarray(probabilities, dtype=float) >= threshold
    positive = scored[scored["label"] == 1]
    negative = scored[scored["label"] == 0]
    positive_days = positive["date"].nunique()
    negative_days = negative["date"].nunique()
    output = {
        "positive_days": float(positive_days),
        "negative_days": float(negative_days),
    }
    for hours in (3, 2, 1):
        qualifying_alerts = positive[
            positive["alert"]
            & (positive["meta_minutes_before_onset"] >= hours * 60)
        ]
        output[f"event_alert_rate_lead_{hours}h"] = (
            float(qualifying_alerts["date"].nunique() / positive_days)
            if positive_days
            else float("nan")
        )
    false_alert_days = negative.loc[negative["alert"], "date"].nunique()
    output["false_alert_day_rate"] = (
        float(false_alert_days / negative_days)
        if negative_days
        else float("nan")
    )
    alerted_positive = positive[positive["alert"]]
    warning_by_day = alerted_positive.groupby("date")[
        "meta_minutes_before_onset"
    ].max()
    output["event_alert_rate_any_lead"] = (
        float(len(warning_by_day) / positive_days)
        if positive_days
        else float("nan")
    )
    output["median_warning_minutes"] = (
        float(warning_by_day.median()) if len(warning_by_day) else float("nan")
    )
    for hours in (3, 2, 1):
        day_labels, day_scores = day_alert_scores(
            frame, probabilities, minimum_lead_minutes=hours * 60
        )
        output[f"event_day_{hours}h_average_precision"] = float(
            average_precision_score(day_labels, day_scores)
        )
        output[f"event_day_{hours}h_roc_auc"] = (
            float(roc_auc_score(day_labels, day_scores))
            if np.unique(day_labels).size == 2
            else float("nan")
        )
    output["event_day_3h_balanced_accuracy"] = 0.5 * (
        output["event_alert_rate_lead_3h"]
        + 1.0
        - output["false_alert_day_rate"]
    )
    if {"piou_last_wind_avg_kmh", "piou_last_heading_sin"}.issubset(frame.columns):
        output.update(onset_evidence_metrics(frame, onset_evidence(frame)))
    return output


def day_alert_scores(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    minimum_lead_minutes: float = 180.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce timestep probabilities to one operational alert score per day."""
    required = {"date", "label", "meta_minutes_before_onset"}
    if not required.issubset(frame.columns):
        raise ValueError("Day alert scores require onset-aware timestep rows")
    scored = frame.loc[:, ["date", "label", "meta_minutes_before_onset"]].copy()
    scored["probability"] = np.asarray(probabilities, dtype=float)
    labels: list[int] = []
    scores: list[float] = []
    for _, day in scored.groupby("date", sort=True):
        label = int(day["label"].iloc[0])
        if label == 1:
            eligible = day[
                day["meta_minutes_before_onset"] >= minimum_lead_minutes
            ]
            score = (
                float(eligible["probability"].max())
                if len(eligible)
                else -1.0
            )
        else:
            score = float(day["probability"].max())
        labels.append(label)
        scores.append(score)
    return np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)


def select_anticipation_threshold(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    minimum_lead_minutes: float = 180.0,
) -> float:
    """Balance event-day advance alerts against false-alert days."""
    labels, scores = day_alert_scores(
        frame, probabilities, minimum_lead_minutes
    )
    candidates = np.unique(np.r_[0.0, scores[scores >= 0], 1.0])
    best_threshold = 0.5
    best_key = (-math.inf, -math.inf, -math.inf)
    for threshold in candidates:
        predicted = (scores >= threshold).astype(int)
        key = (
            float(balanced_accuracy_score(labels, predicted)),
            float(f1_score(labels, predicted, zero_division=0)),
            -abs(float(threshold) - 0.5),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


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
                day_normalized_anticipation_weights(fold_train),
            )
            probability = predict_probabilities(pipeline, fold_valid, feature_names)
            if "meta_minutes_before_onset" in fold_valid.columns:
                labels, day_scores = day_alert_scores(fold_valid, probability)
                fold_scores.append(
                    float(average_precision_score(labels, day_scores))
                )
            else:
                fold_scores.append(
                    float(
                        average_precision_score(
                            fold_valid["label"].to_numpy(),
                            probability,
                            sample_weight=anticipation_weights(fold_valid),
                        )
                    )
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


def load_dataset(path: Path | str) -> pd.DataFrame:
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
    l2_candidates: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    smoke: bool = False,
    label_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    train, validation, test = split_dataset(frame, smoke=smoke)
    selected_features = feature_names(train)
    candidates = (1.0,) if smoke else l2_candidates
    best_l2, cv_scores = expanding_year_l2_search(
        train, selected_features, candidates
    )
    pipeline = build_pipeline(best_l2, selected_features)
    fit_pipeline(
        pipeline,
        numeric_feature_frame(train, selected_features),
        train["label"].to_numpy(dtype=int),
        day_normalized_anticipation_weights(train),
    )
    validation_probability = predict_probabilities(
        pipeline, validation, selected_features
    )
    threshold = (
        select_anticipation_threshold(validation, validation_probability)
        if "meta_minutes_before_onset" in validation.columns
        else select_threshold(
            validation["label"].to_numpy(),
            validation_probability,
            anticipation_weights(validation),
        )
    )
    metrics = {}
    for split_name, subset in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        probability = predict_probabilities(pipeline, subset, selected_features)
        metrics[split_name] = {
            **classification_metrics(
                subset["label"].to_numpy(dtype=int),
                probability,
                threshold,
                anticipation_weights(subset),
            ),
            **anticipation_metrics(subset, probability, threshold),
        }
    for year in sorted(test["year"].unique()):
        subset = test[test["year"] == year]
        metrics[f"test_{int(year)}"] = classification_metrics(
            subset["label"].to_numpy(dtype=int),
            predict_probabilities(pipeline, subset, selected_features),
            threshold,
            anticipation_weights(subset),
        )
    if "issue_minutes" in test.columns:
        for issue_minutes in sorted(test["issue_minutes"].unique()):
            subset = test[test["issue_minutes"] == issue_minutes]
            if subset["label"].nunique() < 2:
                continue
            hour, minute = divmod(int(issue_minutes), 60)
            metrics[f"test_time_{hour:02d}{minute:02d}"] = classification_metrics(
                subset["label"].to_numpy(dtype=int),
                predict_probabilities(pipeline, subset, selected_features),
                threshold,
                anticipation_weights(subset),
            )
    classifier = pipeline.named_steps["classifier"]
    metadata = {
        "schema_version": 2,
        "artifact_format": "joblib",
        "feature_names": selected_features,
        "estimator": {
            "library": "scikit-learn",
            "library_version": sklearn.__version__,
            "pipeline": [
                "SimpleImputer(strategy='median', keep_empty_features=True)",
                "HistGradientBoostingClassifier(max_leaf_nodes=7)",
            ],
            "random_state": RANDOM_STATE,
        },
        "model": {
            "threshold": threshold,
            "l2": best_l2,
            "iterations": int(classifier.n_iter_),
            "learning_rate": float(classifier.learning_rate),
            "max_leaf_nodes": int(classifier.max_leaf_nodes),
            "min_samples_leaf": int(classifier.min_samples_leaf),
            "fit_weighting": "equal_total_weight_per_day_within_day_lead_weight",
        },
        "selection": {
            "l2_event_day_3h_average_precision_by_candidate": cv_scores,
            "selection_data": "expanding_year_folds_within_training_years_only",
        },
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
    suffix = "_core_observation_count_morning"
    weather_prefixes = {
        name[: -len(suffix)]
        for name in feature_names
        if name.startswith("mf") and name.endswith(suffix)
    }
    guard_columns = {
        "piou_observation_count_morning",
        "piou_last_age_minutes",
    }
    for prefix in weather_prefixes:
        guard_columns.update(
            {
                f"{prefix}_core_observation_count_morning",
                f"{prefix}_last_age_minutes",
            }
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
    ]
    feed_guards.extend(
        (
            f"{prefix}_",
            f"{prefix}_core_observation_count_morning",
            f"{prefix}_last_age_minutes",
            "maximum_weather_feature_age_minutes",
            90.0,
        )
        for prefix in sorted(weather_prefixes)
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
    # Exact signed logit contributions are a property of the previous linear model.
    # Keep the response shape stable while avoiding misleading tree attributions.
    contribution_rows: list[list[tuple[str, float]]] = [[] for _ in range(len(frame))]
    return probability, predicted, contribution_rows


def predict_frame(
    artifact_path: Path | str, frame: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[list[tuple[str, float]]]]:
    """Convenience wrapper for trusted local artifacts."""
    payload, pipeline = load_artifact(artifact_path)
    return predict_loaded(payload, pipeline, frame)
