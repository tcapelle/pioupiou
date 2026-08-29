import unittest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from pioupiou.data.daily import PiouObservation
from scripts.build_web_history import audit_summary, build_day


class WebHistoryTests(unittest.TestCase):
    def test_audit_summary_reports_event_and_false_alert_day_counts(self):
        result = audit_summary(
            {
                "year": 2026,
                "metrics": {
                    "positive_days": 10.0,
                    "negative_days": 77.0,
                    "event_alert_rate_lead_3h": 0.5,
                    "false_alert_day_rate": 12 / 77,
                    "event_day_3h_average_precision": 0.267,
                    "event_day_3h_roc_auc": 0.658,
                    "median_warning_minutes": 366.0,
                },
            },
            "2026-08-17",
        )
        self.assertEqual(result["events_alerted_at_least_3h"], 5)
        self.assertEqual(result["events_missed_at_least_3h"], 5)
        self.assertEqual(result["false_alert_days"], 12)
        self.assertEqual(result["through"], "2026-08-17")

    def test_build_day_marks_reconstruction_and_uses_strictly_prior_wind(self):
        local_timezone = ZoneInfo("Europe/Paris")
        before = datetime(2026, 7, 1, 11, 55, tzinfo=local_timezone)
        at_issue = datetime(2026, 7, 1, 12, 0, tzinfo=local_timezone)

        def observation(when, speed):
            return PiouObservation(
                timestamp_utc=when.astimezone(timezone.utc),
                timestamp_local=when,
                wind_speed_min_kmh=speed - 1,
                wind_speed_avg_kmh=speed,
                wind_speed_max_kmh=speed + 1,
                wind_heading_degrees=270,
            )

        rows = pd.DataFrame(
            [
                {
                    "issue_minutes": 720,
                    "label": 1,
                    "event_onset_minutes": 900,
                    "traverse_probability": 0.4,
                    "onset_evidence": 0.2,
                    "predict_traverse": 1,
                    "threshold": 0.275,
                    "probability_onset_within_60m": 0.1,
                    "probability_onset_within_120m": 0.2,
                    "probability_onset_within_180m": 0.3,
                }
            ]
        )
        result = build_day(
            "2026-07-01",
            rows,
            [observation(before, 18.0), observation(at_issue, 99.0)],
            local_timezone,
        )
        self.assertEqual(result["source"], "retrospective_reconstruction")
        self.assertEqual(result["points"][0]["current_wind"]["average_kmh"], 18.0)
        self.assertEqual(result["points"][0]["traverse_probability"], 0.4)


if __name__ == "__main__":
    unittest.main()
