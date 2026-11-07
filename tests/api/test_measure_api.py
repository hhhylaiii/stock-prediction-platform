from collections.abc import Iterator
from io import StringIO
from itertools import count, cycle
from typing import Any

import pytest
from django.core.management import call_command
from django.test import Client

from apps.api.management.commands import measure_api
from tests.api.conftest import Runs

pytestmark = pytest.mark.django_db


def ticks(durations: list[float]) -> Iterator[float]:
    """perf_counter readings whose start and stop pairs are the given durations, in turn."""
    starts = count(start=0, step=10)
    for spent in cycle(durations):
        start = float(next(starts))
        yield start
        yield start + spent


def rows_of(out: StringIO) -> list[list[str]]:
    return [line.strip("| ").split(" | ") for line in out.getvalue().splitlines()[2:]]


def test_every_endpoint_gets_its_median_p95_and_max(
    runs: Runs, calendar_in_db: list[object], monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = ticks([0.001] * 10 + [0.002] * 9 + [0.05])
    monkeypatch.setattr(measure_api.time, "perf_counter", lambda: next(clock))
    calls = []
    real_get = Client.get

    def counted(self: Client, path: str, *args: Any, **kwargs: Any) -> Any:
        calls.append(path)
        return real_get(self, path, *args, **kwargs)

    monkeypatch.setattr(Client, "get", counted)
    out = StringIO()

    call_command("measure_api", "--repeat", "20", stdout=out)

    rows = rows_of(out)
    assert len(rows) == 12
    assert rows[0] == ["`/api/stocks/2330`", "200", "1.5", "2.0", "50.0"]
    assert all(row[1:] == ["200", "1.5", "2.0", "50.0"] for row in rows)
    assert len(calls) == 12 * (20 + 1)


def test_statuses_other_than_the_first_are_counted(
    runs: Runs, calendar_in_db: list[object], monkeypatch: pytest.MonkeyPatch
) -> None:
    real_get = Client.get
    seen = count()

    def throttled(self: Client, path: str, *args: Any, **kwargs: Any) -> Any:
        response = real_get(self, path, *args, **kwargs)
        if path == "/api/etl/runs" and next(seen) > 0:
            response.status_code = 429
        return response

    monkeypatch.setattr(Client, "get", throttled)
    out = StringIO()

    call_command("measure_api", "--repeat", "2", stdout=out)

    assert [row[1] for row in rows_of(out) if row[0] == "`/api/etl/runs`"] == ["200/429×2"]


def test_the_stock_and_repeat_are_options(runs: Runs, calendar_in_db: list[object]) -> None:
    out = StringIO()

    call_command("measure_api", "--stock", "6488", "--repeat", "0", stdout=out)

    assert "| `/api/stocks/6488/history` |" in out.getvalue()
