from collections.abc import Iterator

import pytest

from tests.api import conftest
from tests.api.conftest import TIMED_RUNS, fastest


def clock(readings: list[float]) -> Iterator[float]:
    yield from readings


def test_the_shortest_of_the_runs_is_the_time_and_the_last_result_is_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = clock([0.0, 0.9, 1.0, 1.2, 2.0, 2.7])
    monkeypatch.setattr(conftest.time, "perf_counter", lambda: next(ticks))
    calls = iter(["first", "second", "third"])

    result, spent = fastest(lambda: next(calls))

    assert (result, round(spent, 6)) == ("third", 0.2)


def test_every_run_calls_once() -> None:
    seen: list[int] = []

    fastest(lambda: seen.append(1))

    assert len(seen) == TIMED_RUNS == 3
