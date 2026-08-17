import unittest
from unittest.mock import patch

from api import app, predict


class ApiTests(unittest.TestCase):
    def test_predict_is_the_only_endpoint(self):
        self.assertEqual(
            [(route.path, route.methods) for route in app.routes],
            [("/predict", {"GET"})],
        )
        expected = {"prediction_time": "2026-08-17T18:08:00+02:00"}
        with patch("api.predict_now", return_value=expected):
            self.assertEqual(predict(), expected)


if __name__ == "__main__":
    unittest.main()
