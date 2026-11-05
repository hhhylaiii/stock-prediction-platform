"""Time the eleven read-only API endpoints in process and print a markdown table."""

import math
import statistics
import time
from typing import Any

from django.core.management.base import BaseCommand, CommandParser
from django.test import Client

ENDPOINTS = (
    "/api/stocks/{stock}",
    "/api/stocks/{stock}/prices",
    "/api/stocks/{stock}/predictions",
    "/api/stocks/{stock}/features",
    "/api/stocks/{stock}/history",
    "/api/status",
    "/api/predictions",
    "/api/models",
    "/api/models/ic-monthly",
    "/api/etl/runs",
    "/api/data-quality",
)


def timed(client: Client, path: str, repeat: int) -> tuple[int, list[float]]:
    """Status of a warm-up call, then the milliseconds of each repeated call."""
    status = client.get(path).status_code
    spent = []
    for _ in range(repeat):
        start = time.perf_counter()
        client.get(path)
        spent.append((time.perf_counter() - start) * 1000)
    return status, spent


class Command(BaseCommand):
    help = "Time each API endpoint in process; the throttle allows about 300 calls a minute."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--stock", default="2330")
        parser.add_argument("--repeat", type=int, default=10)

    def handle(self, *args: Any, **options: Any) -> None:
        client = Client(HTTP_HOST="localhost")
        repeat = max(1, options["repeat"])
        self.stdout.write("| 端點 | 狀態 | 中位數 ms | p95 ms | 最大 ms |")
        self.stdout.write("| --- | --- | --- | --- | --- |")
        for template in ENDPOINTS:
            path = template.format(stock=options["stock"])
            status, spent = timed(client, path, repeat)
            p95 = sorted(spent)[math.ceil(0.95 * repeat) - 1]
            cells = (statistics.median(spent), p95, max(spent))
            self.stdout.write(
                f"| `{path}` | {status} | " + " | ".join(f"{v:.1f}" for v in cells) + " |"
            )
