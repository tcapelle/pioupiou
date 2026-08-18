import unittest
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from pioupiou.data.timestep import (
    cutoff_for_minutes,
    traverse_progress_features,
)
from pioupiou.data.daily import LabelConfig, PiouObservation, piou_features


LOCAL = ZoneInfo("Europe/Paris")


def observation(local_time: datetime, speed: float = 20.0, heading: float = 270.0):
    return PiouObservation(
        timestamp_utc=local_time.astimezone(timezone.utc),
        timestamp_local=local_time,
        wind_speed_min_kmh=max(0.0, speed - 2.0),
        wind_speed_avg_kmh=speed,
        wind_speed_max_kmh=speed + 2.0,
        wind_heading_degrees=heading,
    )


class TimestepTraverseTests(unittest.TestCase):
    def test_piou_features_accept_an_arbitrary_minute_cutoff(self):
        local_day = date(2024, 7, 1)
        config = LabelConfig(minimum_target_coverage=0.0)
        before = observation(datetime(2024, 7, 1, 13, 14, tzinfo=LOCAL), speed=7.0)
        at_cutoff = observation(datetime(2024, 7, 1, 13, 17, tzinfo=LOCAL), speed=99.0)
        cutoff = cutoff_for_minutes(local_day, 13 * 60 + 17, LOCAL)
        features = piou_features(
            local_day, [before, at_cutoff], config, cutoff_local=cutoff
        )
        self.assertEqual(features["piou_last_wind_avg_kmh"], 7.0)
        self.assertEqual(features["piou_last_age_minutes"], 3.0)

    def test_progress_uses_only_event_evidence_seen_so_far(self):
        local_day = date(2024, 7, 1)
        config = LabelConfig(minimum_target_coverage=0.0)
        start = datetime(2024, 7, 1, 12, 0, tzinfo=LOCAL)
        values = [observation(start + timedelta(minutes=5 * index)) for index in range(8)]
        before_event = traverse_progress_features(
            local_day,
            values,
            config,
            datetime(2024, 7, 1, 11, 0, tzinfo=LOCAL),
        )
        after_event = traverse_progress_features(
            local_day,
            values,
            config,
            datetime(2024, 7, 1, 12, 40, tzinfo=LOCAL),
        )
        self.assertEqual(before_event["piou_wind_event_observed_so_far"], 0.0)
        self.assertEqual(after_event["piou_wind_event_observed_so_far"], 1.0)
        self.assertGreaterEqual(
            after_event["piou_wind_event_qualifying_minutes_so_far"], 30.0
        )

if __name__ == "__main__":
    unittest.main()
