#!/usr/bin/env python3
"""Download a station archive into the repository's monthly CSV format."""

from __future__ import annotations

import argparse
import csv
import json
import urllib.parse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from pioupiou.data.daily import open_url


ARCHIVE_URL = "https://api.pioupiou.fr/v1/archive/{station_id}"


def next_month(value: date) -> date:
    return date(value.year + (value.month == 12), value.month % 12 + 1, 1)


def fetch_month(station_id: int, start: date, stop: date) -> dict:
    query = urllib.parse.urlencode(
        {
            "start": datetime.combine(
                start, datetime.min.time(), timezone.utc
            ).isoformat(),
            "stop": datetime.combine(
                stop, datetime.min.time(), timezone.utc
            ).isoformat(),
        }
    )
    url = f"{ARCHIVE_URL.format(station_id=station_id)}?{query}"
    with open_url(url, timeout=120) as response:
        return json.load(response)


def write_month(payload: dict, output: Path) -> int:
    rows = payload.get("data", [])
    if not rows:
        return 0
    legend = payload["legend"]
    units = payload["units"]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(
            handle, quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n"
        )
        writer.writerow(["License", payload["license"]])
        writer.writerow(["Attribution", payload["attribution"]])
        writer.writerow(legend)
        writer.writerow(units)
        writer.writerows(sorted(rows, key=lambda row: row[0]))
    temporary.replace(output)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station-id", type=int, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("pioudata"))
    args = parser.parse_args()

    tomorrow_utc = datetime.now(timezone.utc).date() + timedelta(days=1)
    start = date(args.year, 1, 1)
    requested_stop = date(args.year + 1, 1, 1)
    stop = min(requested_stop, tomorrow_utc)
    if start >= stop:
        raise SystemExit(f"No completed observations are available for {args.year}")

    total = 0
    month = start
    while month < stop:
        month_stop = min(next_month(month), stop)
        payload = fetch_month(args.station_id, month, month_stop)
        output = args.output_dir / f"{month:%Y-%m}.csv"
        count = write_month(payload, output)
        if count:
            print(f"{output}: {count:,} observations")
            total += count
        month = next_month(month)
    print(f"Fetched {total:,} observations for station {args.station_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
