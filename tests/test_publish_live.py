import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.publish_live import prediction_document


class PublishLiveTests(unittest.TestCase):
    def test_success_document_includes_public_metadata(self):
        with patch(
            "scripts.publish_live.predict_now",
            return_value={"status": "pre_onset", "traverse_probability": 0.42},
        ):
            result = prediction_document(Path("model.joblib"))
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["status"], "pre_onset")
        self.assertEqual(result["traverse_probability"], 0.42)
        self.assertIn("published_at", result)

    def test_failure_document_can_still_be_published(self):
        with patch(
            "scripts.publish_live.predict_now",
            side_effect=ValueError("observations are stale"),
        ):
            result = prediction_document(Path("model.joblib"))
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["message"], "observations are stale")

    def test_closed_prediction_window_is_not_reported_as_an_outage(self):
        with patch(
            "scripts.publish_live.predict_now",
            side_effect=ValueError(
                "Current time is outside the model's 06:30-19:59 window"
            ),
        ):
            result = prediction_document(Path("model.joblib"))
        self.assertEqual(result["status"], "outside_prediction_window")


if __name__ == "__main__":
    unittest.main()
