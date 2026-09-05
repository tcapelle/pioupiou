import unittest
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from pioupiou.data.daily import LabelConfig
from scripts.remaining_wind import remaining_label
from tests.test_timestep_traverse import observation


class RemainingWindTests(unittest.TestCase):
    def test_past_wind_does_not_count_but_continuing_or_second_spells_do(self):
        local = ZoneInfo("Europe/Paris")
        day = date(2024, 7, 1)
        config = LabelConfig(minimum_target_coverage=0.99, target_end_hour=21)
        start = datetime(2024, 7, 1, 12, tzinfo=local)
        values = [
            observation(start + timedelta(minutes=5 * index), speed=20 if index < 12 else 5)
            for index in range(108)
        ]
        cutoff = datetime(2024, 7, 1, 16, 30, tzinfo=local)
        self.assertEqual(remaining_label(day, values, start, config)["label"], 1)
        self.assertEqual(remaining_label(day, values, cutoff, config)["label"], 0)
        for index in range(53, 63):
            values[index] = observation(values[index].timestamp_local, speed=20)
        target = remaining_label(day, values, cutoff, config)
        self.assertEqual(target["label"], 1)
        self.assertEqual(target["meta_current_qualifies"], 1)
        self.assertEqual(target["meta_future_window_minutes"], 270)
        # Removing most future readings makes the target unknown, not negative.
        self.assertIsNone(remaining_label(day, values[:60], cutoff, config))
