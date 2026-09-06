import tempfile
import unittest
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import sklearn

from pioupiou.data.daily import LabelConfig, PRIMARY_WEATHER_STATION
from pioupiou.inference.model import load_bundle, predict_loaded, save_model_bundle
from realtime_inference import predict_now
from scripts.remaining_wind import export_deployment
from tests.test_timestep_traverse import observation
from tests.test_traverse_model import save_fitted_model


class RemainingWindDeploymentTests(unittest.TestCase):
    def test_export_preserves_scores_and_live_forecast_after_onset_until_1930(self):
        features = ["piou_observation_count_morning", "piou_last_age_minutes", "piou_last_wind_avg_kmh"]
        config = LabelConfig(piou_morning_start_hour=6)
        with tempfile.TemporaryDirectory() as directory:
            source, deployed = Path(directory) / "research.joblib", Path(directory) / "live.joblib"
            save_fitted_model(source, features)
            research = joblib.load(source)
            research["metadata"] = {
                "model_kind": "remaining_wind_research", "selected_variant": "remaining_wind",
                "target_config": asdict(config), "feature_names": features,
                "sklearn_version": sklearn.__version__, "l2": 10,
                "fit_weighting": "equal_per_day_uniform_within_day",
                "fit_years": [2025], "dataset_sha256": "test",
            }
            save_model_bundle(research, source)
            export_deployment(source, deployed)
            bundle, _ = load_bundle(deployed)
            row = pd.DataFrame([{features[0]: 8, features[1]: 5, features[2]: 20, "issue_minutes": 1170}])
            probabilities, decisions, _ = predict_loaded(bundle["metadata"], bundle["pipeline"], row)
            np.testing.assert_array_equal(probabilities, research["pipeline"].predict_proba(row[features])[:, 1])
            self.assertIsNone(decisions[0])
            with self.assertRaisesRegex(ValueError, "outside_prediction_window"):
                predict_loaded(bundle["metadata"], bundle["pipeline"], row.assign(issue_minutes=1171))

            local = ZoneInfo("Europe/Paris")
            cutoff = datetime(2026, 7, 1, 19, 30, tzinfo=local)
            observations = [observation(cutoff - timedelta(minutes=5 * index)) for index in range(10, 0, -1)]
            meteo = ({"mf_last_age_minutes": 30}, {PRIMARY_WEATHER_STATION.slug: [{"timestamp_local": cutoff - timedelta(minutes=30)}]})
            with patch.dict("os.environ", {"METEOFRANCE_TOKEN": "test"}), patch(
                "realtime_inference.datetime"
            ) as clock, patch("realtime_inference.fetch_json"), patch(
                "realtime_inference.piou_observations_from_archive_payload", return_value=observations
            ), patch("realtime_inference.current_weather_features", return_value=meteo):
                clock.now.return_value = cutoff
                clock.combine.side_effect = datetime.combine
                result = predict_now(deployed)
                self.assertEqual(result["status"], "remaining_wind")
                self.assertIsNotNone(result["observed_wind_onset"])
                self.assertGreater(result["traverse_probability"], 0)
                self.assertIsNone(result["predict_traverse"])
                self.assertEqual(result["onset_within_probabilities"], {})
                self.assertEqual(result["target_window"]["start"], cutoff.isoformat())
                self.assertEqual(result["target_window"]["end"], cutoff.replace(hour=20, minute=0).isoformat())
                clock.now.return_value = cutoff + timedelta(minutes=1)
                result = predict_now(deployed)
                self.assertEqual(result["status"], "outside_prediction_window")
                self.assertNotIn("traverse_probability", result)
