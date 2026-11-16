from collections.abc import Iterator
from io import StringIO
from itertools import count, cycle
from typing import Any

import pytest
from django.core.management import call_command
from django.db import connection
from django.test import Client
from django.test.utils import CaptureQueriesContext

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
    runs: Runs, calendar_in_db: list[object], monkeypatch: pytest.MonkeyPatch, settings: Any
) -> None:
    settings.API_THROTTLE_RATE = "100000/min"
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
    assert len(rows) == len(measure_api.PATHS)
    assert rows[0] == ["`/api/stocks/2330`", "200", "1.5", "2.0", "50.0"]
    assert all(row[1:] == ["200", "1.5", "2.0", "50.0"] for row in rows)
    assert len(calls) == len(measure_api.PATHS) * (20 + 1)


def test_the_paths_add_the_five_day_reads_and_each_served_ordering() -> None:
    assert measure_api.PATHS[: len(measure_api.ENDPOINTS)] == measure_api.ENDPOINTS
    assert len(measure_api.ENDPOINTS) == 13
    assert measure_api.PATHS[len(measure_api.ENDPOINTS) :] == (
        "/api/predictions?horizon=5",
        "/api/stocks/{stock}/features?horizon=5",
        "/api/stocks/{stock}/history?horizon=5",
        "/api/models/ic-monthly?horizon=5",
        "/api/predictions?horizon=5&ordering=-close",
        "/api/predictions?horizon=5&ordering=-ret_pred",
        "/api/predictions?horizon=5&ordering=-price_pred",
        "/api/predictions?horizon=5&ordering=-prev_rank",
        "/api/predictions?horizon=20&ordering=-close",
        "/api/predictions?horizon=20&ordering=-ret_pred",
        "/api/predictions?horizon=20&ordering=-price_pred",
        "/api/predictions?horizon=20&ordering=-prob_up_5pct",
        "/api/predictions?horizon=20&ordering=-prev_rank",
    )


def test_each_sorted_path_is_the_ordering_the_table_applies(
    client: Client, runs: Runs, calendar_in_db: list[object]
) -> None:
    sorted_paths = [path for path in measure_api.PATHS if "&ordering=" in path]

    applied = {path: client.get(path).json()["meta"]["ordering"] for path in sorted_paths}

    assert applied == {path: path.rsplit("=", 1)[1] for path in sorted_paths if "=20&" in path} | {
        "/api/predictions?horizon=5&ordering=-close": "-close",
        "/api/predictions?horizon=5&ordering=-ret_pred": "-ret_pred",
        "/api/predictions?horizon=5&ordering=-price_pred": "-price_pred",
        "/api/predictions?horizon=5&ordering=-prev_rank": "rank",
    }


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


def test_the_corrections_summary_is_timed_without_its_cache(client: Client) -> None:
    with CaptureQueriesContext(connection) as queries:
        measure_api.timed(client, "/api/data-quality/corrections", 3)

    summaries = [q for q in queries.captured_queries if "GROUP BY" in q["sql"].upper()]
    assert len(summaries) == 1 + 3
