import tempfile
import unittest
from pathlib import Path

from pioupiou.dashboard import (
    DASHBOARD_END_MINUTES,
    DASHBOARD_START_MINUTES,
    kmh_to_knots,
    load_predictions,
)


class DashboardTests(unittest.TestCase):
    def test_kmh_to_knots_uses_exact_conversion(self):
        self.assertAlmostEqual(kmh_to_knots(18.52), 10.0)

    def test_dashboard_display_window_is_noon_through_21h(self):
        self.assertEqual((DASHBOARD_START_MINUTES, DASHBOARD_END_MINUTES), (720, 1260))

    def test_load_predictions_groups_timesteps_by_day(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            path.write_text(
                "date,year,issue_minutes,label,event_onset_minutes,traverse_probability,onset_evidence,predict_traverse,threshold,probability_onset_within_60m,probability_onset_within_120m,probability_onset_within_180m\n"
                "2026-06-01,2026,390,1,780.5,0.25,0.2,1,0.2,0.1,0.2,0.3\n"
                "2026-06-01,2026,420,1,780.5,0.5,0.4,1,0.2,0.2,0.4,0.6\n"
            )
            result = load_predictions(path)
        self.assertEqual(len(result["days"]), 1)
        self.assertTrue(result["days"][0]["label"])
        self.assertEqual(result["days"][0]["event_onset_minutes"], 780.5)
        self.assertEqual(result["days"][0]["predictions"][1]["probability"], 0.5)
        self.assertEqual(result["days"][0]["predictions"][1]["onset_evidence"], 0.4)
        self.assertEqual(result["days"][0]["predictions"][1][60], 0.2)
        self.assertEqual(result["days"][0]["predictions"][1][120], 0.4)
        self.assertEqual(result["days"][0]["predictions"][1][180], 0.6)


if __name__ == "__main__":
    unittest.main()
