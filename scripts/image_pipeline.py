"""Image pipeline for inventorying and downloading Grand Port webcam frames."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PAGE_URL = "https://www.skaping.com/grandlac/grandport"
TIMEZONE = ZoneInfo("Europe/Paris")
QUALITIES = ("thumb", "mini", "small", "large", "hd", "4k")
ARTIFACTS = Path("artifacts/webcam_grandport")
INVENTORY_FIELDS = (
    "media_id",
    "captured_at_local",
    "bestshot",
    *(f"url_{quality}" for quality in QUALITIES),
)
SELECTION_FIELDS = (
    "media_id",
    "captured_at_local",
    "quality",
    "image_url",
    "relative_path",
)


def fetch(url: str, data: bytes | None = None) -> bytes:
    headers = {
        "User-Agent": "pioupiou-research",
        "Referer": PAGE_URL,
        "Accept": "application/json,image/jpeg,text/html",
    }
    if data is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def player_config() -> tuple[str, str]:
    html = fetch(PAGE_URL).decode()
    match = re.search(
        r"SkapingAPI\.setConfig\(\s*['\"]([^'\"]+)['\"]\s*,\s*"
        r"['\"]([^'\"]+)['\"]\s*\)",
        html,
    )
    if not match:
        raise RuntimeError("Could not find the Skaping API configuration")
    api_url, api_key = match.groups()
    return api_url.replace("http://", "https://").rstrip("/"), api_key


def month_starts(start: date, end: date):
    current = start.replace(day=1)
    while current <= end:
        yield current
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)


def next_month(month: date) -> date:
    if month.month == 12:
        return month.replace(year=month.year + 1, month=1)
    return month.replace(month=month.month + 1)


def load_month(month: date, api_url: str, api_key: str) -> dict:
    cache = ARTIFACTS / "api" / f"{month:%Y-%m}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    following = next_month(month)
    end = date.fromordinal(following.toordinal() - 1)
    form = urllib.parse.urlencode(
        {
            "api_key": api_key,
            "types": "image",
            "start": f"{month} 00:00:00",
            "end": f"{end} 23:59:59",
            "_n": "5000",
        }
    ).encode()
    payload = json.loads(fetch(f"{api_url}/media/search", form))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload))
    return payload


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def inventory(args: argparse.Namespace) -> None:
    api_url, api_key = player_config()
    rows = []
    for month in month_starts(args.start_date, args.end_date):
        for media in load_month(month, api_url, api_key)["medias"]:
            captured = datetime.strptime(
                media["date"], "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=TIMEZONE)
            if not args.start_date <= captured.date() <= args.end_date:
                continue
            rows.append(
                {
                    "media_id": media["id"],
                    "captured_at_local": captured.isoformat(),
                    "bestshot": media.get("bestshot", "0"),
                    **{
                        f"url_{quality}": media["urls"].get(quality, "")
                        for quality in QUALITIES
                    },
                }
            )

    rows.sort(key=lambda row: row["captured_at_local"])
    write_csv(args.output, INVENTORY_FIELDS, rows)
    print(f"Wrote {len(rows)} captures to {args.output}")


def minutes(clock: str) -> int:
    if clock == "24:00":
        return 24 * 60
    parsed = datetime.strptime(clock, "%H:%M")
    return parsed.hour * 60 + parsed.minute


def plan(args: argparse.Namespace) -> None:
    by_day = defaultdict(list)
    start_minute = minutes(args.from_time)
    end_minute = minutes(args.until_time)

    for row in read_csv(args.inventory):
        captured = datetime.fromisoformat(row["captured_at_local"])
        minute = captured.hour * 60 + captured.minute
        if args.start_date and captured.date() < args.start_date:
            continue
        if args.end_date and captured.date() > args.end_date:
            continue
        if start_minute <= minute < end_minute:
            by_day[captured.date()].append(row)

    selected = []
    for day in sorted(by_day):
        for row in sorted(by_day[day], key=lambda item: item["captured_at_local"])[
            :: args.stride
        ]:
            captured = datetime.fromisoformat(row["captured_at_local"])
            selected.append(
                {
                    "media_id": row["media_id"],
                    "captured_at_local": row["captured_at_local"],
                    "quality": args.quality,
                    "image_url": row[f"url_{args.quality}"],
                    "relative_path": str(
                        Path("images")
                        / args.quality
                        / f"{captured:%Y/%m/%d/%H-%M-%S}_{row['media_id']}.jpg"
                    ),
                }
            )
            if args.limit and len(selected) >= args.limit:
                break
        if args.limit and len(selected) >= args.limit:
            break

    write_csv(args.output, SELECTION_FIELDS, selected)
    print(f"Wrote {len(selected)} selected captures to {args.output}")


def download(args: argparse.Namespace) -> None:
    downloaded = 0
    cached = 0
    for row in read_csv(args.selection):
        destination = args.output_dir / row["relative_path"]
        if destination.exists():
            cached += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(fetch(row["image_url"]))
        downloaded += 1
        time.sleep(args.delay)
    print(f"Downloaded {downloaded} images; reused {cached} existing files")


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("inventory")
    command.add_argument("--start-date", type=parse_date, required=True)
    command.add_argument("--end-date", type=parse_date, required=True)
    command.add_argument("--output", type=Path, default=ARTIFACTS / "inventory.csv")
    command.set_defaults(run=inventory)

    command = commands.add_parser("plan")
    command.add_argument("--inventory", type=Path, default=ARTIFACTS / "inventory.csv")
    command.add_argument("--output", type=Path, default=ARTIFACTS / "selection.csv")
    command.add_argument("--start-date", type=parse_date)
    command.add_argument("--end-date", type=parse_date)
    command.add_argument("--from-time", default="00:00")
    command.add_argument("--until-time", default="24:00")
    command.add_argument("--stride", type=int, default=1)
    command.add_argument("--quality", choices=QUALITIES, default="mini")
    command.add_argument("--limit", type=int)
    command.set_defaults(run=plan)

    command = commands.add_parser("download")
    command.add_argument("--selection", type=Path, default=ARTIFACTS / "selection.csv")
    command.add_argument("--output-dir", type=Path, default=ARTIFACTS)
    command.add_argument("--delay", type=float, default=0.25)
    command.set_defaults(run=download)

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.run(arguments)
