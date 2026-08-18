import json
import math
import tempfile
import unittest
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from realtime_inference import meteofrance_observations
from pioupiou.data.daily import (
    LabelConfig,
    PiouObservation,
    deduplicate_piou_observations,
    discover_daily_weather_resources,
    discover_weather_resources,
    label_config_from_payload,
    piou_features,
    piou_observations_from_archive_payload,
    target_label,
    valid_piou_location,
    weather_features,
    weather_value,
)


LOCAL = ZoneInfo("Europe/Paris")


def observation(local_time, speed=20.0, heading=270.0):
    return PiouObservation(
        timestamp_utc=local_time.astimezone(timezone.utc),
        timestamp_local=local_time,
        wind_speed_min_kmh=max(0.0, speed - 2.0),
        wind_speed_avg_kmh=speed,
        wind_speed_max_kmh=speed + 2.0,
        wind_heading_degrees=heading,
    )


class ObservationFeatureTests(unittest.TestCase):
    def setUp(self):
        self.day = date(2024, 7, 1)
        self.config = LabelConfig(minimum_target_coverage=0.0)

    def test_target_requires_duration_and_consecutive_run(self):
        start = datetime(2024, 7, 1, 12, 0, tzinfo=LOCAL)
        values = [observation(start + timedelta(minutes=4 * index)) for index in range(8)]
        result = target_label(self.day, values, self.config, 30.0)
        self.assertEqual(result["label"], 1)
        self.assertEqual(result["meta_target_hot_day"], 1)
        self.assertEqual(result["meta_target_wind_event"], 1)
        self.assertGreaterEqual(result["meta_target_qualifying_minutes"], 30.0)
        self.assertEqual(result["meta_target_longest_qualifying_run"], 8)

    def test_target_requires_hot_day_inside_may_through_september(self):
        start = datetime(2024, 7, 1, 12, 0, tzinfo=LOCAL)
        values = [observation(start + timedelta(minutes=4 * index)) for index in range(8)]
        self.assertEqual(target_label(self.day, values, self.config, 25.0)["label"], 0)
        winter = date(2024, 12, 1)
        self.assertIsNone(target_label(winter, values, self.config, 30.0))

    def test_piou_features_exclude_the_cutoff(self):
        before = datetime(2024, 7, 1, 11, 56, tzinfo=LOCAL)
        at_noon = datetime(2024, 7, 1, 12, 0, tzinfo=LOCAL)
        features = piou_features(
            self.day,
            [observation(before, speed=7.0), observation(at_noon, speed=99.0)],
            self.config,
        )
        self.assertEqual(features["piou_last_wind_avg_kmh"], 7.0)

    def test_piou_rows_are_deduplicated_and_location_guarded(self):
        timestamp = datetime(2024, 7, 1, 11, 0, tzinfo=LOCAL)
        first = observation(timestamp, speed=7.0)
        self.assertEqual(len(deduplicate_piou_observations([first, first])), 1)
        with self.assertRaisesRegex(ValueError, "Conflicting PiouPiou rows"):
            deduplicate_piou_observations([first, observation(timestamp, speed=8.0)])
        self.assertTrue(valid_piou_location(45.701731, 5.883505))
        self.assertFalse(valid_piou_location(None, None))
        payload = {
            "legend": [
                "time",
                "latitude",
                "longitude",
                "wind_speed_min",
                "wind_speed_avg",
                "wind_speed_max",
                "wind_heading",
            ],
            "data": [
                ["2025-08-16T12:00:29.000Z", 45.70171, 5.883485, 2, 4, 6, 270],
                ["2025-08-20T10:27:15.000Z", None, None, 8, 20, 30, 270],
            ],
        }
        self.assertEqual(len(piou_observations_from_archive_payload(payload, LOCAL)), 1)

    def test_weather_features_preserve_units_and_freshness(self):
        valid = {
            "timestamp_local": datetime(2024, 7, 1, 11, 0, tzinfo=LOCAL),
            "T": 10.0,
            "TD": 4.0,
            "U": 60.0,
            "RR1": 1.0,
            "PMER": 1012.0,
            "PSTAT": 985.0,
            "N": 4.0,
            "VV": 20000.0,
            "GLO": 360.0,
            "DIR": 180.0,
            "DIF": 180.0,
            "INS": 30.0,
            "FF": 3.0,
            "DD": 270.0,
        }
        features = weather_features(
            [valid],
            cutoff=datetime(2024, 7, 1, 12, 0, tzinfo=LOCAL),
            maximum_age_minutes=90,
        )
        self.assertEqual(features["mf_last_age_minutes"], 60.0)
        self.assertEqual(features["mf_dewpoint_depression_c_latest"], 6.0)
        self.assertAlmostEqual(features["mf_global_radiation_mean_w_m2"], 1000.0)
        belley = weather_features(
            [valid],
            cutoff=datetime(2024, 7, 1, 12, 0, tzinfo=LOCAL),
            maximum_age_minutes=90,
            prefix="mf_belley",
        )
        self.assertEqual(belley["mf_belley_temperature_c_latest"], 10.0)

    def test_realtime_weather_mapping_preserves_units_and_cutoff(self):
        row = {
            "geo_id_insee": "73329001",
            "lat": 45.641333,
            "lon": 5.877833,
            "validity_time": "2024-07-01T09:00:00Z",
            "t": 293.15,
            "td": 283.15,
            "u": 60,
            "rr1": 1,
            "pmer": 101200,
            "pres": 98500,
            "vv": 20000,
            "n": 4,
            "insolh": 30,
            "ray_glo01": 3_600_000,
            "ff": 3,
            "dd": 270,
        }
        at_cutoff = {**row, "validity_time": "2024-07-01T10:00:00Z"}
        observations = meteofrance_observations(
            [row, at_cutoff], datetime(2024, 7, 1, 12, 0, tzinfo=LOCAL)
        )
        self.assertEqual(len(observations), 1)
        self.assertAlmostEqual(observations[0]["T"], 20.0)
        self.assertAlmostEqual(observations[0]["PMER"], 1012.0)
        self.assertAlmostEqual(observations[0]["GLO"], 360.0)

    def test_weather_quality_is_an_explicit_whitelist(self):
        self.assertEqual(weather_value({"T": "12.3", "QT": "0"}, "T"), 12.3)
        self.assertEqual(weather_value({"T": "12.3", "QT": "9"}, "T"), 12.3)
        self.assertTrue(math.isnan(weather_value({"T": "12.3", "QT": "2"}, "T")))

    def test_resource_discovery_keeps_only_the_primary_department(self):
        resources = [
            {
                "id": f"id-{department}",
                "title": f"HOR_departement_{department}_periode_2020-2024",
                "url": f"https://example.invalid/{department}.csv.gz",
                "last_modified": "2026-08-01T00:00:00+00:00",
            }
            for department in ("01", "73", "74")
        ]
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            (cache_dir / "meteofrance_hourly_resources.json").write_text(
                json.dumps({"resources": resources})
            )
            selected = discover_weather_resources(
                cache_dir, 2021, 2023, offline=True, departments=("73",)
            )
        self.assertEqual([resource.department for resource in selected], ["73"])

    def test_daily_resource_discovery_selects_temperature_archive(self):
        resources = [
            {
                "id": "daily-73",
                "title": "QUOT_departement_73_periode_1950-2024_RR-T-Vent",
                "url": "https://example.invalid/73.csv.gz",
                "last_modified": "2026-08-01T00:00:00+00:00",
            },
            {
                "id": "other-73",
                "title": "QUOT_departement_73_periode_1950-2024_autres-parametres",
                "url": "https://example.invalid/other.csv.gz",
                "last_modified": "2026-08-01T00:00:00+00:00",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            (cache_dir / "meteofrance_daily_resources.json").write_text(
                json.dumps({"resources": resources})
            )
            selected = discover_daily_weather_resources(
                cache_dir, 2021, 2023, offline=True
            )
        self.assertEqual([resource.resource_id for resource in selected], ["daily-73"])

    def test_label_config_payload_must_be_complete(self):
        payload = asdict(LabelConfig())
        self.assertEqual(label_config_from_payload(payload), LabelConfig())
        del payload["heading_min_degrees"]
        with self.assertRaisesRegex(ValueError, "missing=.*heading_min_degrees"):
            label_config_from_payload(payload)


if __name__ == "__main__":
    unittest.main()
