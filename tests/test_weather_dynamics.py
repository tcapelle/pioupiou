import math
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from pioupiou.data.daily import LabelConfig
from scripts.weather_dynamics_ablation import dynamic_weather_features


class WeatherDynamicsTests(unittest.TestCase):
    def test_changes_use_exact_past_hours_and_simultaneous_wind_components(self):
        local = ZoneInfo("Europe/Paris")

        def at(hour, minute=0):
            return datetime(2024, 7, 1, hour, minute, tzinfo=local)

        observations = {
            ("airport", at(8)): {"T": 12.0, "FF": 2.0, "DD": 90.0},
            ("airport", at(10)): {"T": 18.0, "FF": 3.0, "DD": 270.0},
            ("airport", at(11)): {"T": 17.0, "FF": 5.0, "DD": 270.0, "GLO": 360.0},
            ("airport", at(12)): {"T": 999.0, "FF": 999.0, "DD": 0.0},
            ("belley", at(11)): {"T": 15.0, "FF": 7.0, "DD": 270.0},
            # A nonadjacent record cannot substitute for a missing 10:00 slot.
            ("belley", at(9)): {"T": 10.0, "FF": 4.0, "DD": 270.0},
            ("mont_du_chat", at(11)): {"FF": 8.0},
        }
        result = dynamic_weather_features(observations, at(12), LabelConfig())
        self.assertEqual(result["mf_dyn_thermal_airport_temperature_delta_1h_c"], -1.0)
        self.assertEqual(result["mf_dyn_thermal_airport_temperature_delta_3h_c"], 5.0)
        self.assertAlmostEqual(result["mf_dyn_thermal_airport_radiation_latest_w_m2"], 1000.0)
        self.assertAlmostEqual(result["mf_dyn_wind_change_airport_u_delta_3h_ms"], 7.0)
        self.assertAlmostEqual(result["mf_dyn_wind_snapshot_belley_minus_airport_u_ms"], 2.0)
        self.assertTrue(math.isnan(result["mf_dyn_thermal_belley_temperature_delta_1h_c"]))
        self.assertTrue(math.isnan(result["mf_dyn_wind_snapshot_mont_du_chat_u_ms"]))
        later = dynamic_weather_features(observations, at(12, 30), LabelConfig())
        self.assertEqual(later["mf_dyn_thermal_airport_temperature_delta_1h_c"], 982.0)
        early = dynamic_weather_features(
            {("airport", at(5)): {"T": 1.0}, ("airport", at(6)): {"T": 2.0}},
            at(6, 30), LabelConfig(),
        )
        self.assertTrue(math.isnan(early["mf_dyn_thermal_airport_temperature_delta_1h_c"]))
