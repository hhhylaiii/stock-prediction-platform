"""Time the eleven read-only API endpoints in process and print a markdown table."""

import math
import statistics
import time
from collections import Counter
from typing import Any

from django.conf import settings
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


def timed(client: Client, path: str, repeat: int) -> tuple[str, list[float]]:
    """Statuses of a warm-up call and the repeats, then the milliseconds of each repeat."""
    first = client.get(path).status_code
    others: Counter[int] = Counter()
    spent = []
    for _ in range(repeat):
        start = time.perf_counter()
        status = client.get(path).status_code
        spent.append((time.perf_counter() - start) * 1000)
        if status != first:
            others[status] += 1
    return "/".join([str(first), *(f"{code}×{n}" for code, n in sorted(others.items()))]), spent


class Command(BaseCommand):
    help = f"Time each API endpoint in process; the throttle allows {settings.API_THROTTLE_RATE}."

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
