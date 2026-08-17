import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from pioupiou.inference.model import (
    build_pipeline,
    classification_metrics,
    feature_names,
    fit_pipeline,
    load_artifact,
    load_artifact_with_sha256,
    load_dataset,
    numeric_feature_frame,
    predict_frame,
    save_model_bundle,
    select_threshold,
    train_and_evaluate,
)
from scripts.prepare_timestep import bind_to_model_schema


def save_fitted_model(path: Path, features: list[str]) -> None:
    training = pd.DataFrame(
        {
            name: np.asarray([-2.0, -1.0, 1.0, 2.0]) + index
            for index, name in enumerate(features)
        }
    )
    pipeline = build_pipeline(1.0, features)
    fit_pipeline(pipeline, training, np.array([0, 0, 1, 1]))
    save_model_bundle(
        {
            "metadata": {
                "schema_version": 2,
                "artifact_format": "joblib",
                "feature_names": features,
                "estimator": {
                    "library": "scikit-learn",
                    "library_version": sklearn.__version__,
                },
                "model": {"threshold": 0.5, "l2": 1.0, "C": 1.0},
            },
            "pipeline": pipeline,
        },
        path,
    )


class TraverseModelTests(unittest.TestCase):
    def test_metrics_confusion_counts(self):
        metrics = classification_metrics(
            np.array([0, 0, 1, 1]), np.array([0.1, 0.7, 0.8, 0.2]), 0.5
        )
        self.assertEqual(
            [metrics[name] for name in ("tp", "fp", "tn", "fn")],
            [1, 1, 1, 1],
        )

    def test_pipeline_imputes_scales_and_marks_missingness(self):
        features = ["a", "b"]
        train = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [3.0] * 4})
        pipeline = build_pipeline(1.0, features)
        fit_pipeline(pipeline, train, np.array([0, 0, 1, 1]))
        transformed = pipeline.named_steps["preprocessor"].transform(
            pd.DataFrame({"a": [np.nan], "b": [3.0]})
        )
        values = pipeline.named_steps["preprocessor"].named_transformers_["values"]
        self.assertIsInstance(pipeline, Pipeline)
        self.assertIsInstance(values.named_steps["imputer"], SimpleImputer)
        self.assertIsInstance(pipeline.named_steps["classifier"], LogisticRegression)
        self.assertEqual(transformed.shape, (1, 4))
        self.assertEqual(list(transformed[0, 2:]), [1.0, 0.0])

    def test_feature_schema_keeps_only_selected_blocks(self):
        frame = pd.DataFrame(
            {
                "cal_doy_sin": [0.0],
                "piou_last_wind_avg_kmh": [5.0],
                "mf_temperature_c_latest": [12.0],
                "other_station_temperature_c_latest": [11.0],
                "external_temperature_c_latest": [10.0],
                "debug_value": [99.0],
            }
        )
        self.assertEqual(
            feature_names(frame),
            [
                "cal_doy_sin",
                "mf_temperature_c_latest",
                "piou_last_wind_avg_kmh",
            ],
        )

    def test_preparer_emits_exact_model_schema(self):
        prepared = {"date": "2025-09-21", "year": 2025, "a": 1, "extra": 2}
        self.assertEqual(
            bind_to_model_schema(prepared, ["a"]),
            {"date": "2025-09-21", "year": 2025, "a": 1},
        )
        with self.assertRaisesRegex(ValueError, "Cannot construct"):
            bind_to_model_schema(prepared, ["missing"])

    def test_numeric_features_preserve_order_and_treat_infinity_as_missing(self):
        result = numeric_feature_frame(
            pd.DataFrame({"b": ["2"], "a": [np.inf]}), ["a", "b"]
        )
        self.assertEqual(list(result.columns), ["a", "b"])
        self.assertTrue(np.isnan(result.iloc[0]["a"]))
        self.assertEqual(result.iloc[0]["b"], 2.0)

    def test_threshold_uses_validation_balanced_accuracy(self):
        y = np.array([0, 0, 1, 1])
        probability = np.array([0.1, 0.4, 0.5, 0.9])
        self.assertEqual(select_threshold(y, probability), 0.5)

    def test_dataset_identity_includes_issue_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            frame = pd.DataFrame(
                {
                    "date": ["2024-01-01", "2024-01-01"],
                    "year": [2024, 2024],
                    "issue_minutes": [600, 630],
                    "label": [0, 0],
                }
            )
            frame.to_csv(path, index=False)
            self.assertEqual(len(load_dataset(path)), 2)
            frame["issue_minutes"] = 600
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate sample identities"):
                load_dataset(path)

    def test_training_has_one_fixed_feature_schema(self):
        rows = []
        for year in range(2017, 2026):
            for label, signal in ((0, -1.0), (1, 1.0)):
                rows.append(
                    {
                        "date": f"{year}-07-{label + 1:02d}",
                        "year": year,
                        "issue_minutes": 720,
                        "label": label,
                        "cal_doy_sin": signal,
                        "piou_last_wind_avg_kmh": signal,
                        "mf_temperature_c_latest": signal,
                        "ignored": 99.0,
                    }
                )
        bundle, metrics = train_and_evaluate(
            pd.DataFrame(rows), l2_candidates=(1.0,)
        )
        self.assertEqual(
            bundle["metadata"]["feature_names"],
            [
                "cal_doy_sin",
                "mf_temperature_c_latest",
                "piou_last_wind_avg_kmh",
            ],
        )
        self.assertNotIn("role", bundle["metadata"])
        self.assertEqual(metrics["test"]["rows"], 4.0)

    def test_joblib_round_trip_and_hash_check(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.joblib"
            save_fitted_model(model, ["cal_doy_sin"])
            metadata, pipeline = load_artifact(model)
            self.assertEqual(metadata["schema_version"], 2)
            self.assertIsInstance(pipeline, Pipeline)
            with self.assertRaisesRegex(ValueError, "trusted value"):
                load_artifact_with_sha256(model, "0" * 64)

    def test_inference_requires_fresh_primary_feeds(self):
        features = [
            "piou_last_wind_avg_kmh",
            "piou_observation_count_morning",
            "piou_last_age_minutes",
            "mf_temperature_c_latest",
            "mf_core_observation_count_morning",
            "mf_last_age_minutes",
        ]
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.joblib"
            save_fitted_model(model, features)
            row = pd.DataFrame(
                {
                    "piou_last_wind_avg_kmh": [5.0],
                    "piou_observation_count_morning": [10],
                    "piou_last_age_minutes": [4],
                    "mf_temperature_c_latest": [12.0],
                    "mf_core_observation_count_morning": [6],
                    "mf_last_age_minutes": [60],
                }
            )
            probability, _, _ = predict_frame(model, row)
            self.assertTrue(np.isfinite(probability).all())
            stale = row.copy()
            stale["mf_last_age_minutes"] = 91
            with self.assertRaisesRegex(ValueError, "stale mf_"):
                predict_frame(model, stale)


if __name__ == "__main__":
    unittest.main()
