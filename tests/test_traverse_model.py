import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from pioupiou.inference.model import (
    anticipation_metrics,
    anticipation_weights,
    build_pipeline,
    classification_metrics,
    day_normalized_anticipation_weights,
    feature_names,
    fit_pipeline,
    load_artifact,
    load_artifact_with_sha256,
    load_bundle,
    load_dataset,
    numeric_feature_frame,
    onset_horizon_labels,
    onset_interval_labels,
    onset_evidence,
    onset_evidence_metrics,
    predict_bundle_onset_probabilities,
    predict_frame,
    save_model_bundle,
    select_anticipation_threshold,
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
                "model": {"threshold": 0.5, "l2": 1.0},
            },
            "pipeline": pipeline,
        },
        path,
    )


class TraverseModelTests(unittest.TestCase):
    def test_anticipation_metrics_count_event_and_false_alert_days(self):
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2024-01-01",
                        "2024-01-01",
                        "2024-01-02",
                        "2024-01-02",
                        "2024-01-03",
                        "2024-01-03",
                    ]
                ),
                "label": [1, 1, 1, 1, 0, 0],
                "meta_minutes_before_onset": [240, 120, 150, 30, np.nan, np.nan],
            }
        )
        metrics = anticipation_metrics(
            frame,
            np.array([0.8, 0.1, 0.8, 0.1, 0.2, 0.7]),
            0.5,
        )
        self.assertEqual(metrics["event_alert_rate_lead_3h"], 0.5)
        self.assertEqual(metrics["event_alert_rate_lead_2h"], 1.0)
        self.assertEqual(metrics["event_alert_rate_lead_1h"], 1.0)
        self.assertEqual(metrics["false_alert_day_rate"], 1.0)
        self.assertEqual(metrics["median_warning_minutes"], 195.0)
        self.assertIn("event_day_2h_average_precision", metrics)
        self.assertIn("event_day_1h_average_precision", metrics)

    def test_anticipation_weights_validate_the_dataset_contract(self):
        frame = pd.DataFrame(
            {
                "label": [0, 1],
                "meta_minutes_before_onset": [np.nan, 90.0],
                "meta_anticipation_weight": [1.0, 0.5],
            }
        )
        np.testing.assert_array_equal(
            anticipation_weights(frame), np.array([1.0, 0.5])
        )
        frame.loc[1, "meta_minutes_before_onset"] = 0.0
        with self.assertRaisesRegex(ValueError, "strictly before onset"):
            anticipation_weights(frame)

    def test_fit_weights_give_each_day_equal_total_weight(self):
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-01-01", "2024-01-01", "2024-01-02"]
                ),
                "label": [1, 1, 0],
                "meta_minutes_before_onset": [180.0, 90.0, np.nan],
                "meta_anticipation_weight": [1.0, 0.5, 1.0],
            }
        )
        weights = day_normalized_anticipation_weights(frame)
        np.testing.assert_allclose(weights, np.array([2 / 3, 1 / 3, 1.0]))
        totals = pd.Series(weights).groupby(frame["date"]).sum()
        np.testing.assert_allclose(totals.to_numpy(), np.ones(2))

    def test_anticipation_threshold_operates_on_day_alerts(self):
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]
                ),
                "label": [1, 1, 0, 0],
                "meta_minutes_before_onset": [240, 120, np.nan, np.nan],
            }
        )
        threshold = select_anticipation_threshold(
            frame, np.array([0.8, 0.95, 0.2, 0.7])
        )
        self.assertGreater(threshold, 0.7)
        self.assertLessEqual(threshold, 0.8)

    def test_metrics_confusion_counts(self):
        metrics = classification_metrics(
            np.array([0, 0, 1, 1]), np.array([0.1, 0.7, 0.8, 0.2]), 0.5
        )
        self.assertEqual(
            [metrics[name] for name in ("tp", "fp", "tn", "fn")],
            [1, 1, 1, 1],
        )

    def test_pipeline_imputes_for_shallow_histogram_booster(self):
        features = ["a", "b"]
        train = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [3.0] * 4})
        pipeline = build_pipeline(1.0, features)
        fit_pipeline(pipeline, train, np.array([0, 0, 1, 1]))
        transformed = pipeline.named_steps["imputer"].transform(
            pd.DataFrame({"a": [np.nan], "b": [3.0]})
        )
        self.assertIsInstance(pipeline, Pipeline)
        self.assertIsInstance(pipeline.named_steps["imputer"], SimpleImputer)
        self.assertIsInstance(
            pipeline.named_steps["classifier"], HistGradientBoostingClassifier
        )
        self.assertEqual(transformed.shape, (1, 2))
        self.assertTrue(np.isfinite(transformed).all())

    def test_onset_evidence_is_normalized_westerly_component(self):
        frame = pd.DataFrame(
            {
                "piou_last_wind_avg_kmh": [18.52, 18.52, 37.04, np.nan],
                "piou_last_heading_sin": [-1.0, 1.0, -1.0, -1.0],
            }
        )
        np.testing.assert_allclose(
            onset_evidence(frame), np.array([1.0, 0.0, 1.0, 0.0])
        )

    def test_onset_evidence_metric_measures_each_event_trajectory(self):
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-07-01"] * 3 + ["2024-07-02"] * 3),
                "label": [1] * 6,
                "issue_minutes": [720, 750, 780] * 2,
                "meta_minutes_before_onset": [180, 150, 120] * 2,
            }
        )
        metrics = onset_evidence_metrics(
            frame, np.array([0.1, 0.2, 0.3, 0.3, 0.2, 0.1])
        )
        self.assertEqual(metrics["event_rising_onset_evidence_rate_final_6h"], 0.5)

    def test_feature_schema_keeps_only_selected_blocks(self):
        frame = pd.DataFrame(
            {
                "cal_doy_sin": [0.0],
                "piou_last_wind_avg_kmh": [5.0],
                "mf_temperature_c_latest": [12.0],
                "mf_belley_temperature_c_latest": [11.5],
                "other_station_temperature_c_latest": [11.0],
                "external_temperature_c_latest": [10.0],
                "debug_value": [99.0],
            }
        )
        self.assertEqual(
            feature_names(frame),
            [
                "cal_doy_sin",
                "mf_belley_temperature_c_latest",
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

    def test_threshold_uses_balanced_accuracy(self):
        y = np.array([0, 0, 1, 1])
        probability = np.array([0.1, 0.4, 0.5, 0.9])
        self.assertEqual(select_threshold(y, probability), 0.5)

    def test_ordered_onset_bundle_round_trip_preserves_horizon_order(self):
        rows = []
        examples = (
            (1, 30.0, 1.0),
            (1, 90.0, 0.6),
            (1, 150.0, 0.2),
            (1, 240.0, -0.4),
            (0, np.nan, -1.0),
        )
        for year in range(2017, 2027):
            for day, (label, lead, signal) in enumerate(examples, start=1):
                rows.append(
                    {
                        "date": f"{year}-07-{day:02d}",
                        "year": year,
                        "issue_minutes": 720,
                        "label": label,
                        "meta_minutes_before_onset": lead,
                        "meta_anticipation_weight": (
                            min(lead / 180.0, 1.0) if label else 1.0
                        ),
                        "cal_doy_sin": signal,
                        "piou_last_wind_avg_kmh": signal,
                        "mf_temperature_c_latest": signal,
                    }
                )
        frame = pd.DataFrame(rows)
        np.testing.assert_array_equal(
            onset_interval_labels(frame.iloc[:5]), np.array([0, 1, 2, 3, 3])
        )
        np.testing.assert_array_equal(
            onset_horizon_labels(frame.iloc[:5], 120), np.array([1, 1, 0, 0, 0])
        )
        invalid = frame.iloc[:5].copy()
        invalid.loc[invalid["label"] == 0, "meta_minutes_before_onset"] = 30.0
        with self.assertRaisesRegex(ValueError, "Negative onset rows"):
            onset_interval_labels(invalid)
        bundle, _ = train_and_evaluate(frame, l2_candidates=(1.0,))
        scored = frame[frame["year"] == 2026]
        probability = predict_bundle_onset_probabilities(bundle, scored)
        self.assertTrue(np.all(probability[60] <= probability[120]))
        self.assertTrue(np.all(probability[120] <= probability[180]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ordered.joblib"
            save_model_bundle(bundle, path)
            loaded, _ = load_bundle(path)
        loaded_probability = predict_bundle_onset_probabilities(loaded, scored)
        np.testing.assert_allclose(loaded_probability[180], probability[180])

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
        for year in range(2017, 2027):
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
        self.assertEqual(
            bundle["metadata"]["model"]["fit_weighting"],
            "equal_total_weight_per_day_within_day_lead_weight",
        )
        self.assertEqual(metrics["test"]["rows"], 2.0)
        self.assertEqual(
            bundle["metadata"]["split"]["train_years"],
            list(range(2017, 2026)),
        )
        self.assertEqual(bundle["metadata"]["split"]["train_rows"], 18)
        self.assertEqual(
            bundle["metadata"]["split"]["cross_validation_years"],
            list(range(2020, 2026)),
        )
        self.assertEqual(metrics["cross_validation"]["rows"], 12.0)

    def test_joblib_round_trip_and_hash_check(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.joblib"
            save_fitted_model(model, ["cal_doy_sin"])
            metadata, pipeline = load_artifact(model)
            self.assertEqual(metadata["schema_version"], 2)
            self.assertIsInstance(pipeline, Pipeline)
            with self.assertRaisesRegex(ValueError, "trusted value"):
                load_artifact_with_sha256(model, "0" * 64)

    def test_inference_requires_fresh_weather_feeds(self):
        features = [
            "piou_last_wind_avg_kmh",
            "piou_observation_count_morning",
            "piou_last_age_minutes",
            "mf_temperature_c_latest",
            "mf_core_observation_count_morning",
            "mf_last_age_minutes",
            "mf_belley_temperature_c_latest",
            "mf_belley_core_observation_count_morning",
            "mf_belley_last_age_minutes",
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
                    "mf_belley_temperature_c_latest": [11.0],
                    "mf_belley_core_observation_count_morning": [6],
                    "mf_belley_last_age_minutes": [60],
                }
            )
            probability, _, _ = predict_frame(model, row)
            self.assertTrue(np.isfinite(probability).all())
            stale = row.copy()
            stale["mf_last_age_minutes"] = 91
            with self.assertRaisesRegex(ValueError, "stale mf_"):
                predict_frame(model, stale)
            stale_belley = row.copy()
            stale_belley["mf_belley_last_age_minutes"] = 91
            with self.assertRaisesRegex(ValueError, "stale mf_belley_"):
                predict_frame(model, stale_belley)


if __name__ == "__main__":
    unittest.main()
