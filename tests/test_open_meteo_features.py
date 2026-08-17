import math
import unittest
from datetime import date

from pioupiou.feature_eng.open_meteo import (
    EXPECTED_UNITS,
    OPEN_METEO_VARIABLES,
    open_meteo_days_from_payload,
    open_meteo_url,
)


def payload(times: list[str], rows: list[dict[str, float | None]]):
    hourly = {"time": times}
    for variable in OPEN_METEO_VARIABLES:
        hourly[variable] = [row.get(variable) for row in rows]
    return {
        "latitude": 45.6875,
        "longitude": 5.875,
        "timezone": "Europe/Paris",
        "hourly_units": {"time": "iso8601", **EXPECTED_UNITS},
        "hourly": hourly,
    }


class OpenMeteoFeatureTests(unittest.TestCase):
    def test_quartz_weather_block_is_strictly_before_noon(self):
        base = {
            "temperature_2m": 10.0,
            "relative_humidity_2m": 60.0,
            "dew_point_2m": 4.0,
            "precipitation": 0.5,
            "surface_pressure": 985.0,
            "pressure_msl": 1012.0,
            "cloud_cover": 50.0,
            "cloud_cover_low": 40.0,
            "cloud_cover_mid": 30.0,
            "cloud_cover_high": 20.0,
            "wind_speed_10m": 18.0,
            "wind_direction_10m": 270.0,
            "is_day": 1.0,
            "direct_radiation": 200.0,
            "diffuse_radiation": 100.0,
        }
        result = open_meteo_days_from_payload(
            payload(
                [
                    "2024-07-01T06:00",
                    "2024-07-01T11:00",
                    "2024-07-01T12:00",
                ],
                [
                    base,
                    {**base, "temperature_2m": 16.0},
                    {**base, "temperature_2m": 99.0},
                ],
            ),
            "Europe/Paris",
            6,
            12,
        )[date(2024, 7, 1)]

        self.assertEqual(result["nwp_observation_count_morning"], 2.0)
        self.assertEqual(result["nwp_core_observation_count_morning"], 2.0)
        self.assertEqual(result["nwp_last_age_minutes"], 60.0)
        self.assertEqual(result["nwp_temperature_c_latest"], 16.0)
        self.assertEqual(result["nwp_temperature_c_delta_morning"], 6.0)
        self.assertEqual(result["nwp_precipitation_morning_mm"], 1.0)
        self.assertEqual(result["nwp_daylight_fraction_morning"], 1.0)
        self.assertAlmostEqual(result["nwp_west_component_mean_kmh"], 18.0)
        self.assertAlmostEqual(result["nwp_south_component_mean_kmh"], 0.0, places=12)
        self.assertAlmostEqual(result["nwp_wind_direction_latest_sin"], -1.0)
        self.assertAlmostEqual(result["nwp_wind_direction_latest_cos"], 0.0, places=12)

    def test_missing_core_fields_make_feed_age_unknown(self):
        row = {variable: None for variable in OPEN_METEO_VARIABLES}
        result = open_meteo_days_from_payload(
            payload(["2024-07-01T11:00"], [row]),
            "Europe/Paris",
            6,
            12,
        )[date(2024, 7, 1)]
        self.assertEqual(result["nwp_core_observation_count_morning"], 0.0)
        self.assertTrue(math.isnan(result["nwp_last_age_minutes"]))

    def test_request_binds_model_location_variables_and_timezone(self):
        url = open_meteo_url(
            date(2024, 1, 1), date(2024, 12, 31), "Europe/Paris"
        )
        self.assertIn("historical-forecast-api.open-meteo.com", url)
        self.assertIn("models=ecmwf_ifs", url)
        self.assertIn("timezone=Europe%2FParis", url)
        self.assertIn("wind_direction_10m", url)


if __name__ == "__main__":
    unittest.main()
