import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from pioupiou.data.daily import (
    METEO_FRANCE_SOURCE_SCHEMA,
    PIOU_SOURCE_SCHEMA,
    LabelConfig,
)
from scripts.evaluate import load_validated_inputs, sha256_file
from scripts.prepare_noon import bind_to_model_schema
from scripts.predict import validate_prepared_contract
from pioupiou.inference.model import (
    average_precision,
    build_pipeline,
    classification_metrics,
    fit_pipeline,
    feature_names_for_role,
    load_artifact,
    load_artifact_with_sha256,
    load_daily_dataset,
    numeric_feature_frame,
    predict_frame,
    save_model_bundle,
    select_threshold,
    sha256_json,
)


def save_fitted_model(
    path: Path, metadata: dict[str, object], feature_names: list[str]
) -> None:
    training = pd.DataFrame(
        {
            name: np.asarray([-2.0, -1.0, 1.0, 2.0]) + index
            for index, name in enumerate(feature_names)
        }
    )
    pipeline = build_pipeline(1.0, feature_names)
    fit_pipeline(pipeline, training, np.array([0, 0, 1, 1]))
    complete_metadata = {
        "schema_version": 2,
        "artifact_format": "joblib",
        "feature_names": feature_names,
        "estimator": {
            "library": "scikit-learn",
            "library_version": sklearn.__version__,
        },
        "model": {"threshold": 0.5, "l2": 1.0, "C": 1.0},
        **metadata,
    }
    save_model_bundle(
        {"metadata": complete_metadata, "pipeline": pipeline}, path
    )


class TraverseModelTests(unittest.TestCase):
    def test_average_precision_is_one_for_perfect_ranking(self):
        y = np.array([0, 1, 0, 1])
        probability = np.array([0.1, 0.8, 0.2, 0.9])
        self.assertAlmostEqual(average_precision(y, probability), 1.0)

    def test_metrics_confusion_counts(self):
        metrics = classification_metrics(
            np.array([0, 0, 1, 1]), np.array([0.1, 0.7, 0.8, 0.2]), 0.5
        )
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)
        self.assertGreaterEqual(metrics["precision_at_recall_0_60"], 0.5)

    def test_sklearn_pipeline_imputes_scales_and_marks_all_missingness(self):
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
        self.assertEqual(transformed[0, 2], 1.0)
        self.assertEqual(transformed[0, 3], 0.0)

    def test_sklearn_logistic_regression_learns_separable_signal(self):
        frame = pd.DataFrame({"signal": [-2.0, -1.0, 1.0, 2.0]})
        target = np.array([0, 0, 1, 1])
        pipeline = build_pipeline(0.1, ["signal"])
        fit_pipeline(pipeline, frame, target)
        probability = pipeline.predict_proba(frame)[:, 1]
        self.assertGreater(probability[2], probability[1])
        self.assertGreater(average_precision(target, probability), 0.99)

    def test_numeric_features_preserve_order_and_treat_infinity_as_missing(self):
        result = numeric_feature_frame(
            pd.DataFrame({"b": ["2"], "a": [np.inf]}), ["a", "b"]
        )
        self.assertEqual(list(result.columns), ["a", "b"])
        self.assertTrue(np.isnan(result.iloc[0]["a"]))
        self.assertEqual(result.iloc[0]["b"], 2.0)

    def test_enriched_roles_add_only_their_feature_blocks(self):
        frame = pd.DataFrame(
            {
                "cal_doy_sin": [0.0],
                "piou_last_wind_avg_kmh": [5.0],
                "mf_temperature_c_latest": [12.0],
                "mfs_belley_temperature_c_latest": [11.0],
                "nwp_temperature_c_latest": [10.0],
                "debug_value": [99.0],
            }
        )
        variant = feature_names_for_role(frame, "variant")
        spatial = feature_names_for_role(frame, "spatial")
        nwp = feature_names_for_role(frame, "nwp")

        self.assertEqual(
            set(spatial).difference(variant),
            {"mfs_belley_temperature_c_latest"},
        )
        self.assertNotIn("debug_value", spatial)
        self.assertEqual(
            set(nwp).difference(variant), {"nwp_temperature_c_latest"}
        )
        self.assertNotIn("mfs_belley_temperature_c_latest", nwp)

    def test_noon_preparer_emits_exact_model_schema(self):
        prepared = {"date": "2025-09-21", "year": 2025, "a": 1, "new_field": 2}
        self.assertEqual(
            bind_to_model_schema(prepared, ["a"]),
            {"date": "2025-09-21", "year": 2025, "a": 1},
        )
        with self.assertRaisesRegex(ValueError, "Cannot construct"):
            bind_to_model_schema(prepared, ["a", "missing"])

    def test_threshold_uses_validation_balanced_accuracy(self):
        y = np.array([0, 0, 1, 1])
        probability = np.array([0.1, 0.4, 0.5, 0.9])
        threshold = select_threshold(y, probability)
        self.assertEqual(threshold, 0.5)

    def test_daily_dataset_requires_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            pd.DataFrame({"date": ["2024-01-01"], "year": [2024]}).to_csv(
                path, index=False
            )
            with self.assertRaisesRegex(ValueError, "label"):
                load_daily_dataset(path)

    def test_daily_dataset_rejects_duplicate_dates_and_bad_year(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            pd.DataFrame(
                {
                    "date": ["2024-01-01", "2024-01-01"],
                    "year": [2024, 2024],
                    "label": [0, 1],
                }
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate dates"):
                load_daily_dataset(path)

            pd.DataFrame(
                {"date": ["2024-01-01"], "year": [2023], "label": [0]}
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "year column"):
                load_daily_dataset(path)

    def test_joblib_round_trip_and_legacy_artifact_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.joblib"
            save_fitted_model(model, {"role": "baseline"}, ["cal_doy_sin"])
            metadata, pipeline = load_artifact(model)
            self.assertEqual(metadata["schema_version"], 2)
            self.assertIsInstance(pipeline, Pipeline)
            with self.assertRaisesRegex(ValueError, "trusted value"):
                load_artifact_with_sha256(model, "0" * 64)

            legacy = root / "legacy.json"
            legacy.write_text('{"schema_version": 1}\n')
            with self.assertRaisesRegex(ValueError, "schema-v2 joblib"):
                load_artifact(legacy)

    def test_variant_inference_fails_closed_for_missing_weather(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            features = [
                "piou_last_wind_avg_kmh",
                "piou_observation_count_morning",
                "piou_last_age_minutes",
                "mf_temperature_c_latest",
                "mf_core_observation_count_morning",
                "mf_last_age_minutes",
            ]
            save_fitted_model(
                path,
                {
                    "role": "variant",
                    "label": {
                        "maximum_feature_age_minutes": 30,
                        "maximum_weather_feature_age_minutes": 90,
                    },
                },
                features,
            )
            row = pd.DataFrame(
                {
                    "piou_last_wind_avg_kmh": [5],
                    "piou_observation_count_morning": [10],
                    "piou_last_age_minutes": [4],
                    "mf_temperature_c_latest": ["not-a-number"],
                    "mf_core_observation_count_morning": [0],
                    "mf_last_age_minutes": [60],
                }
            )
            with self.assertRaisesRegex(ValueError, "unavailable mf_"):
                predict_frame(path, row)

            valid = row.copy()
            valid["mf_temperature_c_latest"] = 12
            valid["mf_core_observation_count_morning"] = 6
            for missing in (
                "piou_observation_count_morning",
                "piou_last_age_minutes",
                "mf_core_observation_count_morning",
                "mf_last_age_minutes",
            ):
                with self.subTest(missing=missing):
                    with self.assertRaisesRegex(
                        ValueError, "missing required feed columns"
                    ):
                        predict_frame(path, valid.drop(columns=[missing]))

            stale = valid.copy()
            stale["mf_last_age_minutes"] = 91
            with self.assertRaisesRegex(ValueError, "stale mf_"):
                predict_frame(path, stale)

    def test_spatial_inference_requires_primary_feeds_but_not_optional_stations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            features = [
                "piou_last_wind_avg_kmh",
                "piou_observation_count_morning",
                "piou_last_age_minutes",
                "mf_temperature_c_latest",
                "mf_core_observation_count_morning",
                "mf_last_age_minutes",
                "mfs_belley_temperature_c_latest",
            ]
            save_fitted_model(
                path,
                {
                    "role": "spatial",
                    "label": {
                        "maximum_feature_age_minutes": 30,
                        "maximum_weather_feature_age_minutes": 90,
                    },
                },
                features,
            )
            primary_only = pd.DataFrame(
                {
                    "piou_last_wind_avg_kmh": [5.0],
                    "piou_observation_count_morning": [10],
                    "piou_last_age_minutes": [4],
                    "mf_temperature_c_latest": [12.0],
                    "mf_core_observation_count_morning": [6],
                    "mf_last_age_minutes": [60],
                }
            )

            probability, _, _ = predict_frame(path, primary_only)
            self.assertTrue(np.isfinite(probability).all())

            unavailable_primary = primary_only.copy()
            unavailable_primary["mf_core_observation_count_morning"] = 0
            with self.assertRaisesRegex(ValueError, "unavailable mf_"):
                predict_frame(path, unavailable_primary)

    def test_nwp_inference_requires_fresh_grid_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            features = [
                "piou_last_wind_avg_kmh",
                "piou_observation_count_morning",
                "piou_last_age_minutes",
                "mf_temperature_c_latest",
                "mf_core_observation_count_morning",
                "mf_last_age_minutes",
                "nwp_temperature_c_latest",
                "nwp_core_observation_count_morning",
                "nwp_last_age_minutes",
            ]
            save_fitted_model(
                path,
                {
                    "role": "nwp",
                    "label": {
                        "maximum_feature_age_minutes": 30,
                        "maximum_weather_feature_age_minutes": 90,
                    },
                },
                features,
            )
            row = pd.DataFrame(
                {
                    "piou_last_wind_avg_kmh": [5.0],
                    "piou_observation_count_morning": [10],
                    "piou_last_age_minutes": [4],
                    "mf_temperature_c_latest": [12.0],
                    "mf_core_observation_count_morning": [6],
                    "mf_last_age_minutes": [60],
                    "nwp_temperature_c_latest": [11.0],
                    "nwp_core_observation_count_morning": [6],
                    "nwp_last_age_minutes": [60],
                }
            )
            probability, _, _ = predict_frame(path, row)
            self.assertTrue(np.isfinite(probability).all())

            unavailable = row.copy()
            unavailable["nwp_core_observation_count_morning"] = 0
            with self.assertRaisesRegex(ValueError, "unavailable nwp_"):
                predict_frame(path, unavailable)

            stale = row.copy()
            stale["nwp_last_age_minutes"] = 91
            with self.assertRaisesRegex(ValueError, "stale nwp_"):
                predict_frame(path, stale)

    def test_comparison_binds_actual_dataset_and_model_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "daily.csv"
            pd.DataFrame(
                {
                    "date": ["2024-01-01"],
                    "year": [2024],
                    "label": [0],
                    "cal_doy_sin": [0.0],
                }
            ).to_csv(dataset, index=False)
            dataset_hash = sha256_file(dataset)
            common = {
                "split": {"test_years": [2024], "test_rows": 1},
                "provenance": {
                    "dataset_sha256": dataset_hash,
                    "source_sha256": {"pioupiou/inference/model.py": "same"},
                },
            }
            baseline = root / "baseline.joblib"
            variant = root / "variant.joblib"
            spatial = root / "spatial.joblib"
            save_fitted_model(
                baseline, {**common, "role": "baseline"}, ["cal_doy_sin"]
            )
            save_fitted_model(
                variant, {**common, "role": "variant"}, ["cal_doy_sin"]
            )
            save_fitted_model(
                spatial, {**common, "role": "spatial"}, ["cal_doy_sin"]
            )
            load_validated_inputs(dataset, baseline, variant)
            _, _, reference, candidate, _ = load_validated_inputs(
                dataset, variant, spatial
            )
            self.assertEqual(reference[0]["role"], "variant")
            self.assertEqual(candidate[0]["role"], "spatial")

            changed_source = {
                **common,
                "provenance": {
                    **common["provenance"],
                    "source_sha256": {"pioupiou/inference/model.py": "changed"},
                },
                "role": "spatial",
            }
            save_fitted_model(spatial, changed_source, ["cal_doy_sin"])
            with self.assertRaisesRegex(ValueError, "source revisions"):
                load_validated_inputs(dataset, variant, spatial)

            changed = pd.read_csv(dataset)
            changed["label"] = 1
            changed.to_csv(dataset, index=False)
            with self.assertRaisesRegex(ValueError, "Supplied dataset SHA-256"):
                load_validated_inputs(dataset, baseline, variant)

    def test_prepared_row_is_bound_to_model_and_stations(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.joblib"
            label = asdict(LabelConfig())
            features = ["cal_doy_sin"]
            contract = {
                "schema_version": 2,
                "dataset_sha256": "dataset-hash",
                "label_config_sha256": sha256_json(label),
                "feature_schema_sha256": sha256_json(features),
                "piou_station_id": "456",
                "piou_source_schema": PIOU_SOURCE_SCHEMA,
                "weather_station_id": "73329001",
                "weather_source_schema": METEO_FRANCE_SOURCE_SCHEMA,
            }
            save_fitted_model(
                model_path,
                {
                    "role": "variant",
                    "label": label,
                    "input_contract": contract,
                },
                features,
            )
            row = pd.DataFrame(
                {
                    "date": ["2024-07-01"],
                    "cal_doy_sin": [0.0],
                    "meta_contract_schema_version": [2],
                    "meta_dataset_sha256": ["dataset-hash"],
                    "meta_feature_cutoff_local": ["2024-07-01T12:00:00+02:00"],
                    "meta_feature_prepared_at_utc": ["2024-07-01T10:00:00+00:00"],
                    "meta_feature_schema_sha256": [contract["feature_schema_sha256"]],
                    "meta_label_config_sha256": [contract["label_config_sha256"]],
                    "meta_model_sha256": [sha256_file(model_path)],
                    "meta_piou_source_schema": [contract["piou_source_schema"]],
                    "meta_piou_station_id": ["456"],
                    "meta_weather_source_schema": [contract["weather_source_schema"]],
                    "meta_weather_station_id": ["73329001"],
                }
            )
            validate_prepared_contract(model_path, row)
            incompatible = row.copy()
            incompatible["meta_weather_station_id"] = "99999999"
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                validate_prepared_contract(model_path, incompatible)
            with self.assertRaisesRegex(ValueError, "modeled column mismatch"):
                validate_prepared_contract(model_path, row.drop(columns=["cal_doy_sin"]))

            old_contract = dict(contract)
            old_contract["piou_source_schema"] = "pioupiou-archive-v1-kmh"
            save_fitted_model(
                model_path,
                {
                    "role": "variant",
                    "label": label,
                    "input_contract": old_contract,
                },
                features,
            )
            with self.assertRaisesRegex(ValueError, "unsupported source schema"):
                validate_prepared_contract(model_path, row)

    def test_spatial_prepared_row_is_bound_to_station_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "spatial.joblib"
            label = asdict(LabelConfig())
            features = [
                "cal_doy_sin",
                "mfs_belley_temperature_c_latest",
            ]
            manifest = [{"slug": "airport", "station_id": "73329001"}]
            contract = {
                "schema_version": 3,
                "dataset_sha256": "dataset-hash",
                "label_config_sha256": sha256_json(label),
                "feature_schema_sha256": sha256_json(features),
                "piou_station_id": "456",
                "piou_source_schema": "pioupiou-archive-v2-kmh-location-guard",
                "weather_station_id": "73329001",
                "weather_source_schema": METEO_FRANCE_SOURCE_SCHEMA,
                "weather_station_manifest": manifest,
                "weather_station_manifest_sha256": sha256_json(manifest),
            }
            save_fitted_model(
                model_path,
                {"role": "spatial", "label": label, "input_contract": contract},
                features,
            )
            row = pd.DataFrame(
                {
                    "date": ["2024-07-01"],
                    "cal_doy_sin": [0.0],
                    "mfs_belley_temperature_c_latest": [18.0],
                    "meta_contract_schema_version": [3],
                    "meta_dataset_sha256": ["dataset-hash"],
                    "meta_feature_cutoff_local": ["2024-07-01T12:00:00+02:00"],
                    "meta_feature_prepared_at_utc": ["2024-07-01T10:00:00+00:00"],
                    "meta_feature_schema_sha256": [contract["feature_schema_sha256"]],
                    "meta_label_config_sha256": [contract["label_config_sha256"]],
                    "meta_model_sha256": [sha256_file(model_path)],
                    "meta_piou_source_schema": [contract["piou_source_schema"]],
                    "meta_piou_station_id": ["456"],
                    "meta_weather_source_schema": [contract["weather_source_schema"]],
                    "meta_weather_station_id": ["73329001"],
                    "meta_weather_station_manifest_sha256": [
                        contract["weather_station_manifest_sha256"]
                    ],
                }
            )
            validate_prepared_contract(model_path, row)
            incompatible = row.copy()
            incompatible["meta_weather_station_manifest_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                validate_prepared_contract(model_path, incompatible)


if __name__ == "__main__":
    unittest.main()
