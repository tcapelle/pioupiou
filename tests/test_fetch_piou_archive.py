import tempfile
import unittest
from datetime import date
from pathlib import Path

from scripts.fetch_piou_archive import next_month, write_month


class FetchPiouArchiveTests(unittest.TestCase):
    def test_next_month_crosses_year_boundary(self):
        self.assertEqual(next_month(date(2026, 12, 1)), date(2027, 1, 1))

    def test_write_month_uses_repository_csv_shape(self):
        payload = {
            "license": "https://example.test/license",
            "attribution": "station contributors",
            "legend": ["time", "latitude"],
            "units": ["utc", "degrees"],
            "data": [["2026-05-01T00:00:00.000Z", 45.7]],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "2026-05.csv"
            self.assertEqual(write_month(payload, output), 1)
            contents = output.read_text()
        self.assertIn('"License","https://example.test/license"', contents)
        self.assertIn('"time","latitude"', contents)


if __name__ == "__main__":
    unittest.main()
