import unittest
from unittest.mock import patch

from api import app, dashboard, dashboard_data, predict


class ApiTests(unittest.TestCase):
    def test_api_endpoints(self):
        self.assertEqual(
            [(route.path, route.methods) for route in app.routes],
            [
                ("/predict", {"GET"}),
                ("/dashboard", {"GET"}),
                ("/dashboard/data", {"GET"}),
            ],
        )
        expected = {"prediction_time": "2026-08-17T18:08:00+02:00"}
        with patch("api.predict_now", return_value=expected):
            self.assertEqual(predict(), expected)

    def test_dashboard_page_and_data(self):
        self.assertIn("Traverse archive", dashboard())
        expected = {"years": [], "events": [], "metadata": {}}
        with patch("api.build_dashboard_data", return_value=expected):
            self.assertEqual(dashboard_data(), expected)


if __name__ == "__main__":
    unittest.main()
