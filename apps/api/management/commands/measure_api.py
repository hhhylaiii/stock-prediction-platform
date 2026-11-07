"""Time the thirteen read-only API endpoints in process and print a markdown table."""

import math
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

from django.conf import settings
from django.core.cache import cache
from django.core.management.base import BaseCommand, CommandParser
from django.test import Client

from apps.api.ops import CORRECTIONS_CACHE_KEY

ENDPOINTS = (
    "/api/stocks/{stock}",
    "/api/stocks/{stock}/prices",
    "/api/stocks/{stock}/predictions",
    "/api/stocks/{stock}/features",
    "/api/stocks/{stock}/history",
    "/api/stocks?q={stock}",
    "/api/status",
    "/api/predictions",
    "/api/models",
    "/api/models/ic-monthly",
    "/api/etl/runs",
    "/api/data-quality",
    "/api/data-quality/corrections",
)
CACHED = {"/api/data-quality/corrections": CORRECTIONS_CACHE_KEY}
HEADER = ("| 端點 | 狀態 | 中位數 ms | p95 ms | 最大 ms |", "| --- | --- | --- | --- | --- |")


@dataclass(frozen=True)
class Timing:
    path: str
    status: str
    median: float
    p95: float
    slowest: float

    def line(self) -> str:
        cells = (self.median, self.p95, self.slowest)
        return f"| `{self.path}` | {self.status} | " + " | ".join(f"{v:.1f}" for v in cells) + " |"


def timed(client: Client, path: str, repeat: int) -> tuple[str, list[float]]:
    """Statuses of a warm-up call and the repeats, then each repeat's ms on a cold cache."""
    first = client.get(path).status_code
    others: Counter[int] = Counter()
    spent = []
    for _ in range(repeat):
        if path in CACHED:
            cache.delete(CACHED[path])
        start = time.perf_counter()
        status = client.get(path).status_code
        spent.append((time.perf_counter() - start) * 1000)
        if status != first:
            others[status] += 1
    return "/".join([str(first), *(f"{code}×{n}" for code, n in sorted(others.items()))]), spent


def measure(client: Client, stock: str, repeat: int) -> list[Timing]:
    """One timing per endpoint, at least one repeat each."""
    repeat = max(1, repeat)
    timings = []
    for template in ENDPOINTS:
        path = template.format(stock=stock)
        status, spent = timed(client, path, repeat)
        p95 = sorted(spent)[math.ceil(0.95 * repeat) - 1]
        timings.append(Timing(path, status, statistics.median(spent), p95, max(spent)))
    return timings


class Command(BaseCommand):
    help = f"Time each API endpoint in process; the throttle allows {settings.API_THROTTLE_RATE}."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--stock", default="2330")
        parser.add_argument("--repeat", type=int, default=10)

    def handle(self, *args: Any, **options: Any) -> None:
        timings = measure(Client(HTTP_HOST="localhost"), options["stock"], options["repeat"])
        for line in (*HEADER, *(timing.line() for timing in timings)):
            self.stdout.write(line)
