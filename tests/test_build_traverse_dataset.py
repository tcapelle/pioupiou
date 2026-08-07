import csv
import gzip
import json
import math
import tempfile
import unittest
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from build_traverse_dataset import (
    LabelConfig,
    PiouObservation,
    WEATHER_CACHE_FIELDS,
    WEATHER_RAW_FIELDS,
    WeatherResource,
    cache_station_resource,
    deduplicate_piou_observations,
    elapsed_minutes,
    filtered_resource_path,
    label_config_from_payload,
    local_boundary,
    piou_features,
    sha256_file,
    target_label,
    weather_value,
    weather_features,
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


class TraverseDatasetTests(unittest.TestCase):
    def setUp(self):
        self.day = date(2024, 7, 1)
        self.config = LabelConfig(minimum_target_coverage=0.0)

    def test_target_requires_duration_and_consecutive_run(self):
        start = datetime(2024, 7, 1, 12, 0, tzinfo=LOCAL)
        values = [observation(start + timedelta(minutes=4 * index)) for index in range(8)]
        result = target_label(self.day, values, self.config)
        self.assertEqual(result["label"], 1)
        self.assertGreaterEqual(result["meta_target_qualifying_minutes"], 30.0)
        self.assertEqual(result["meta_target_longest_qualifying_run"], 8)

    def test_target_rejects_short_or_nonwesterly_signal(self):
        start = datetime(2024, 7, 1, 12, 0, tzinfo=LOCAL)
        short = [observation(start + timedelta(minutes=4 * index)) for index in range(3)]
        self.assertEqual(target_label(self.day, short, self.config)["label"], 0)
        east = [
            observation(start + timedelta(minutes=4 * index), heading=90.0)
            for index in range(20)
        ]
        self.assertEqual(target_label(self.day, east, self.config)["label"], 0)

    def test_target_coverage_can_make_label_unknown(self):
        strict = LabelConfig(minimum_target_coverage=0.75)
        start = datetime(2024, 7, 1, 12, 0, tzinfo=LOCAL)
        sparse = [observation(start), observation(start + timedelta(hours=7))]
        self.assertIsNone(target_label(self.day, sparse, strict))

    def test_piou_features_are_strictly_before_noon(self):
        before = datetime(2024, 7, 1, 11, 56, tzinfo=LOCAL)
        at_noon = datetime(2024, 7, 1, 12, 0, tzinfo=LOCAL)
        features = piou_features(
            self.day,
            [observation(before, speed=7.0), observation(at_noon, speed=99.0)],
            self.config,
        )
        self.assertEqual(features["piou_last_wind_avg_kmh"], 7.0)

    def test_piou_duplicate_conflicts_are_rejected(self):
        timestamp = datetime(2024, 7, 1, 11, 0, tzinfo=LOCAL)
        first = observation(timestamp, speed=7.0)
        self.assertEqual(len(deduplicate_piou_observations([first, first])), 1)
        with self.assertRaisesRegex(ValueError, "Conflicting PiouPiou rows"):
            deduplicate_piou_observations(
                [first, observation(timestamp, speed=8.0)]
            )

    def test_weather_quality_features_and_units(self):
        observation_time = datetime(2024, 7, 1, 11, 0, tzinfo=LOCAL)
        values = [
            {
                "timestamp_local": observation_time,
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
        ]
        features = weather_features(
            values, cutoff=datetime(2024, 7, 1, 12, 0, tzinfo=LOCAL)
        )
        self.assertEqual(features["mf_last_age_minutes"], 60.0)
        self.assertEqual(features["mf_dewpoint_depression_c_latest"], 6.0)
        self.assertEqual(features["mf_precipitation_morning_mm"], 1.0)
        self.assertAlmostEqual(features["mf_global_radiation_mean_w_m2"], 1000.0)
        self.assertAlmostEqual(features["mf_west_component_mean_ms"], 3.0)

    def test_weather_quality_is_an_explicit_whitelist(self):
        self.assertEqual(weather_value({"T": "12.3", "QT": "0"}, "T"), 12.3)
        self.assertEqual(weather_value({"T": "12.3", "QT": "9"}, "T"), 12.3)
        self.assertTrue(math.isnan(weather_value({"T": "12.3", "QT": "2"}, "T")))
        self.assertTrue(math.isnan(weather_value({"T": "12.3", "QT": ""}, "T")))

    def test_invalid_newest_weather_row_does_not_look_fresh(self):
        valid = {field: 1.0 for field in WEATHER_RAW_FIELDS}
        valid["timestamp_local"] = datetime(2024, 7, 1, 6, 0, tzinfo=LOCAL)
        invalid = {field: float("nan") for field in WEATHER_RAW_FIELDS}
        invalid["timestamp_local"] = datetime(2024, 7, 1, 11, 0, tzinfo=LOCAL)
        features = weather_features(
            [valid, invalid],
            cutoff=datetime(2024, 7, 1, 12, 0, tzinfo=LOCAL),
            maximum_age_minutes=90,
        )
        self.assertEqual(features["mf_last_age_minutes"], 360.0)
        self.assertEqual(features["mf_core_observation_count_morning"], 1.0)
        self.assertTrue(math.isnan(features["mf_temperature_c_latest"]))

    def test_dst_elapsed_time_and_end_hour_24(self):
        spring_day = date(2024, 3, 31)
        spring_start = local_boundary(spring_day, 0, LOCAL)
        spring_end = local_boundary(spring_day, 4, LOCAL)
        self.assertEqual(elapsed_minutes(spring_end, spring_start), 180.0)

        autumn_day = date(2024, 10, 27)
        autumn_start = local_boundary(autumn_day, 0, LOCAL)
        autumn_end = local_boundary(autumn_day, 4, LOCAL)
        self.assertEqual(elapsed_minutes(autumn_end, autumn_start), 300.0)

        midnight = local_boundary(self.day, 24, LOCAL)
        self.assertEqual(midnight.date(), date(2024, 7, 2))
        self.assertEqual(midnight.hour, 0)

    def test_label_config_payload_must_be_complete(self):
        payload = asdict(LabelConfig())
        self.assertEqual(label_config_from_payload(payload), LabelConfig())
        del payload["heading_min_degrees"]
        with self.assertRaisesRegex(ValueError, "missing=.*heading_min_degrees"):
            label_config_from_payload(payload)

    def test_weather_cache_requires_and_checks_sidecar_hash(self):
        resource = WeatherResource(
            resource_id="resource-id",
            title="HOR_departement_73_periode_2024-2024",
            url="https://example.invalid/weather.csv.gz",
            start_year=2024,
            end_year=2024,
            last_modified="2024-01-02T00:00:00+00:00",
            filesize=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            target = filtered_resource_path(cache_dir, resource, "73329001")
            with gzip.open(target, "wt", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=WEATHER_CACHE_FIELDS)
                writer.writeheader()
                writer.writerow({name: "" for name in WEATHER_CACHE_FIELDS})
            with self.assertRaisesRegex(ValueError, "Unverified legacy weather cache"):
                cache_station_resource(cache_dir, resource, "73329001", False, True)

            sidecar = target.with_name(target.name + ".metadata.json")
            sidecar.write_text(
                json.dumps(
                    {
                        "resource_id": resource.resource_id,
                        "last_modified": resource.last_modified,
                        "url": resource.url,
                        "station_id": "73329001",
                        "row_count": 1,
                        "cache_sha256": sha256_file(target),
                    }
                )
            )
            _, rows = cache_station_resource(
                cache_dir, resource, "73329001", False, True
            )
            self.assertEqual(rows, 1)
            with target.open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                cache_station_resource(cache_dir, resource, "73329001", False, True)


if __name__ == "__main__":
    unittest.main()
