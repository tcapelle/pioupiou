import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from scripts.publish_live import prediction_document, update_history


class PublishLiveTests(unittest.TestCase):
    def test_success_document_includes_public_metadata(self):
        with patch(
            "scripts.publish_live.predict_now",
            return_value={"status": "pre_onset", "traverse_probability": 0.42},
        ), patch("scripts.publish_live.ensure_deployment_model"):
            result = prediction_document(Path("model.joblib"), Path("manifest.json"))
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["status"], "pre_onset")
        self.assertEqual(result["traverse_probability"], 0.42)
        self.assertIn("published_at", result)

    def test_failure_document_can_still_be_published(self):
        with patch(
            "scripts.publish_live.predict_now",
            side_effect=ValueError("observations are stale"),
        ), patch("scripts.publish_live.ensure_deployment_model"):
            result = prediction_document(Path("model.joblib"), Path("manifest.json"))
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["message"], "observations are stale")

    def test_closed_prediction_window_is_not_reported_as_an_outage(self):
        with patch(
            "scripts.publish_live.predict_now",
            side_effect=ValueError(
                "Current time is outside the model's 06:30-19:59 window"
            ),
        ), patch("scripts.publish_live.ensure_deployment_model"):
            result = prediction_document(Path("model.joblib"), Path("manifest.json"))
        self.assertEqual(result["status"], "outside_prediction_window")

    def test_update_history_preserves_daily_points_and_rebuilds_index(self):
        first = {
            "schema_version": 1,
            "prediction_time": "2026-08-29T12:00:00+02:00",
            "status": "pre_onset",
        }
        second = {
            **first,
            "prediction_time": "2026-08-29T12:05:00+02:00",
        }
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory)
            update_history(checkout, first)
            update_history(checkout, second)
            day = json.loads((checkout / "days" / "2026-08-29.json").read_text())
            dates = json.loads((checkout / "dates.json").read_text())
        self.assertEqual(len(day["points"]), 2)
        self.assertEqual(day["source"], "live_published")
        self.assertEqual(dates["dates"][0]["points"], 2)


if __name__ == "__main__":
    unittest.main()
