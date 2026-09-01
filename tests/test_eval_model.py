import unittest

import numpy as np
import pandas as pd

from eval_model import evaluate_2026
from pioupiou.inference.model import train_and_evaluate


class EvalModelTests(unittest.TestCase):
    def test_evaluate_2026_reports_event_scores_and_negative_days(self):
        rows = []
        for year in range(2017, 2027):
            for day, issue_minutes, label, lead, signal in (
                (1, 930, 1, 30.0, 1.0),
                (1, 870, 1, 90.0, 0.7),
                (1, 810, 1, 150.0, 0.4),
                (1, 720, 1, 240.0, 0.1),
                (2, 720, 0, np.nan, -1.0),
            ):
                rows.append(
                    {
                        "date": f"{year}-07-{day:02d}",
                        "year": year,
                        "issue_minutes": issue_minutes,
                        "label": label,
                        "meta_minutes_before_onset": lead,
                        "meta_anticipation_weight": (
                            min(lead / 180.0, 1.0) if label else 1.0
                        ),
                        "cal_doy_sin": signal,
                    }
                )
        frame = pd.DataFrame(rows)
        bundle, _ = train_and_evaluate(frame, l2_candidates=(1.0,))

        report = evaluate_2026(bundle, frame)

        self.assertEqual(report["year"], 2026)
        self.assertEqual(report["days"], 2)
        self.assertEqual(report["metrics"]["positive_days"], 1.0)
        self.assertEqual(report["metrics"]["negative_days"], 1.0)
        confusion = report["event_day_confusion"]["at_least_3h"]
        self.assertEqual(confusion["positive_event_days"], 1)
        self.assertEqual(confusion["negative_days"], 1)
        self.assertEqual(
            confusion["detected_events"] + confusion["false_negative_events"], 1
        )
        self.assertEqual(
            confusion["false_positive_days"] + confusion["true_negative_days"], 1
        )
        self.assertEqual(len(report["events"]), 1)
        self.assertEqual(report["events"][0]["onset_minutes"], 960)
        self.assertIn("max_probability_at_least_3h", report["events"][0])


if __name__ == "__main__":
    unittest.main()
