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

from pioupiou.data.daily import (
    LabelConfig,
    PRIMARY_WEATHER_STATION,
    PiouObservation,
    WEATHER_CACHE_FIELDS,
    WEATHER_RAW_FIELDS,
    WEATHER_STATIONS,
    WeatherResource,
    build_weather_days,
    cache_weather_resource,
    compact_station_features,
    deduplicate_piou_observations,
    discover_weather_resources,
    elapsed_minutes,
    filtered_resources_path,
    label_config_from_payload,
    local_boundary,
    piou_observations_from_archive_payload,
    piou_features,
    resource_from_payload,
    sha256_file,
    spatial_weather_features,
    target_label,
    valid_piou_location,
    valid_weather_station_location,
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

    def test_piou_rows_must_be_at_the_lake_site(self):
        self.assertTrue(valid_piou_location(45.701731, 5.883505))
        self.assertFalse(valid_piou_location(None, None))
        self.assertFalse(valid_piou_location(45.739329, 5.728061))
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
                ["2025-08-20T10:31:15.000Z", 45.739329, 5.728061, 8, 20, 30, 270],
            ],
        }
        result = piou_observations_from_archive_payload(payload, LOCAL)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].wind_speed_avg_kmh, 4.0)

    def test_weather_rows_must_match_the_station_site(self):
        self.assertTrue(
            valid_weather_station_location(
                45.641333, 5.878000, PRIMARY_WEATHER_STATION
            )
        )
        self.assertFalse(
            valid_weather_station_location(
                45.739329, 5.728061, PRIMARY_WEATHER_STATION
            )
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

    def test_compact_spatial_features(self):
        def station_weather(temperature, dewpoint, west, pressure=1011.0):
            return {
                "mf_observation_count_morning": 6.0,
                "mf_core_observation_count_morning": 6.0,
                "mf_last_age_minutes": 60.0,
                "mf_temperature_c_latest": temperature,
                "mf_temperature_c_delta_morning": 3.0,
                "mf_dewpoint_c_latest": dewpoint,
                "mf_relative_humidity_pct_latest": 60.0,
                "mf_wind_speed_10m_ms_latest": 4.0,
                "mf_wind_direction_latest_sin": -1.0,
                "mf_wind_direction_latest_cos": 0.0,
                "mf_west_component_mean_ms": west,
                "mf_pressure_msl_hpa_latest": pressure,
            }

        optional = {
            "mont_du_chat": station_weather(12.0, 6.0, 3.0),
            "belley": station_weather(18.0, 8.0, 4.0),
            "meythet": station_weather(17.0, 7.0, 2.0, pressure=1010.0),
            "montmelian": station_weather(21.0, 11.0, 0.0),
        }
        result = spatial_weather_features(optional, 90.0)
        self.assertEqual(
            result["mfs_mont_du_chat_temperature_c_latest"], 12.0
        )
        self.assertEqual(
            result["mfs_belley_west_component_mean_ms"], 4.0
        )
        self.assertEqual(
            result["mfs_meythet_pressure_msl_hpa_latest"], 1010.0
        )

    def test_stale_optional_station_is_missing_without_becoming_required(self):
        features = compact_station_features(
            WEATHER_STATIONS[1],
            {
                "mf_observation_count_morning": 6.0,
                "mf_core_observation_count_morning": 6.0,
                "mf_last_age_minutes": 180.0,
                "mf_temperature_c_latest": 10.0,
            },
            90.0,
        )
        self.assertEqual(features["mfs_mont_du_chat_core_observation_count_morning"], 6.0)
        self.assertNotIn("mfs_mont_du_chat_observation_count_morning", features)
        self.assertEqual(features["mfs_mont_du_chat_last_age_minutes"], 180.0)
        self.assertTrue(
            math.isnan(features["mfs_mont_du_chat_temperature_c_latest"])
        )

    def test_weather_resources_include_department(self):
        resource = resource_from_payload(
            {
                "id": "id-01",
                "title": "HOR_departement_01_periode_2020-2024",
                "url": "https://example.invalid/01.csv.gz",
                "last_modified": "2026-08-01T00:00:00+00:00",
                "filesize": 123,
            }
        )
        self.assertEqual(resource.department, "01")
        self.assertEqual((resource.start_year, resource.end_year), (2020, 2024))

    def test_weather_resource_discovery_selects_requested_departments(self):
        resources = [
            {
                "id": f"id-{department}",
                "title": f"HOR_departement_{department}_periode_2020-2024",
                "url": f"https://example.invalid/{department}.csv.gz",
                "last_modified": "2026-08-01T00:00:00+00:00",
            }
            for department in ("01", "38", "73", "74")
        ]
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            (cache_dir / "meteofrance_hourly_resources.json").write_text(
                json.dumps({"resources": resources})
            )
            selected = discover_weather_resources(
                cache_dir,
                2021,
                2023,
                offline=True,
                departments=("01", "73", "74"),
            )
        self.assertEqual(
            [resource.department for resource in selected], ["01", "73", "74"]
        )

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

    def test_each_station_is_filtered_strictly_before_noon(self):
        def cached_row(station_id, timestamp, temperature):
            row = {name: "" for name in WEATHER_CACHE_FIELDS}
            row.update(
                {
                    "NUM_POSTE": station_id,
                    "LAT": "45.641333",
                    "LON": "5.878000",
                    "ALTI": "235",
                    "AAAAMMJJHH": timestamp,
                    "T": str(temperature),
                    "QT": "0",
                    "U": "60",
                    "QU": "0",
                    "FF": "3",
                    "QFF": "0",
                    "DD": "270",
                    "QDD": "0",
                }
            )
            return row

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stations.csv.gz"
            with gzip.open(path, "wt", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=WEATHER_CACHE_FIELDS)
                writer.writeheader()
                writer.writerow(cached_row("73329001", "2024070109", 10.0))
                writer.writerow(cached_row("99999999", "2024070109", 77.0))
                moved = cached_row("73329001", "2024070108", 88.0)
                moved["LAT"] = "45.739329"
                moved["LON"] = "5.728061"
                writer.writerow(moved)
                writer.writerow(cached_row("73329001", "2024070110", 99.0))
            days = build_weather_days(
                [path], LabelConfig(), 2024, 2024, station=PRIMARY_WEATHER_STATION
            )
        features = days[date(2024, 7, 1)]
        self.assertEqual(features["mf_temperature_c_latest"], 10.0)
        self.assertEqual(features["mf_observation_count_morning"], 1.0)

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
            department="73",
        )
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            station_ids = ("73329001",)
            target = filtered_resources_path(cache_dir, resource, station_ids)
            with gzip.open(target, "wt", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=WEATHER_CACHE_FIELDS)
                writer.writeheader()
                row = {name: "" for name in WEATHER_CACHE_FIELDS}
                row["NUM_POSTE"] = "73329001"
                writer.writerow(row)
            with self.assertRaisesRegex(ValueError, "Unverified legacy weather cache"):
                cache_weather_resource(cache_dir, resource, station_ids, False, True)

            sidecar = target.with_name(target.name + ".metadata.json")
            sidecar.write_text(
                json.dumps(
                    {
                        "resource_id": resource.resource_id,
                        "last_modified": resource.last_modified,
                        "url": resource.url,
                        "station_ids": list(station_ids),
                        "row_count": 1,
                        "resource_rows": {"73329001": 1},
                        "cache_sha256": sha256_file(target),
                    }
                )
            )
            _, rows = cache_weather_resource(
                cache_dir, resource, station_ids, False, True
            )
            self.assertEqual(rows, {"73329001": 1})
            with target.open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                cache_weather_resource(cache_dir, resource, station_ids, False, True)


if __name__ == "__main__":
    unittest.main()
