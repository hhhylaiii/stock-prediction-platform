import math
from datetime import date, timedelta
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.market.models import TradingCalendar
from apps.research import holding_inputs
from apps.research.holding import Market
from apps.research.management.commands import holding_sim

pytestmark = pytest.mark.django_db
NAN = math.nan
FIRST = date(2016, 1, 4)


@pytest.fixture(autouse=True)
def calendar_rows() -> list[date]:
    days = [FIRST + timedelta(days=n) for n in range(60)]
    TradingCalendar.objects.bulk_create(
        TradingCalendar(date=day, is_trading_day=day.weekday() < 5) for day in days
    )
    return [day for day in days if day.weekday() < 5]


def book_market(days: list[date], score: np.ndarray) -> Market:
    count, names = score.shape
    prices = 10 * 1.001 ** np.arange(count)[:, None] * np.ones((count, names))
    return Market(
        days=days[:count],
        stocks=[str(1000 + k) for k in range(names)],
        opens=prices,
        closes=prices,
        score=score,
        in_universe=np.ones((count, names), dtype=bool),
        blocked=np.zeros((count, names), dtype=bool),
        delisted=np.full(names, count, dtype=np.int64),
        limit_down_open=np.zeros((count, names), dtype=bool),
    )


@pytest.fixture
def fed(monkeypatch: pytest.MonkeyPatch, calendar_rows: list[date]) -> dict[str, Any]:
    """The command's inputs replaced by a ten-stock market; records what it was asked for."""
    asked: dict[str, Any] = {}
    score = np.tile(np.arange(1.0, 11.0), (20, 1))
    score[10:] = score[10:, ::-1]

    def scores(run_key: str, folds: list[int], until: date) -> pd.DataFrame:
        asked.update(run_key=run_key, folds=folds, until=until)
        return pd.DataFrame({"date": [calendar_rows[0]]})

    def stored(start: date, until: date) -> pd.DataFrame:
        asked.update(start=start, until=until)
        return pd.DataFrame({"date": [calendar_rows[0]]})

    def market(frame: pd.DataFrame, end: date, calendar: Any) -> Market:
        asked.update(end=end)
        return book_market(calendar_rows, score)

    levels = pd.Series(100.0, index=calendar_rows)
    monkeypatch.setattr(holding_inputs, "_cached_scores", scores)
    monkeypatch.setattr(holding_sim, "stored_scores", stored)
    monkeypatch.setattr(holding_sim, "load_market", market)
    monkeypatch.setattr(holding_sim, "index_levels", lambda start, end: levels)
    monkeypatch.setattr(holding_sim, "latest_price_date", lambda: calendar_rows[19])
    monkeypatch.setattr(holding_inputs, "holds_holdout", lambda key, model, index: index >= 44)
    monkeypatch.setattr(holding_sim, "labels_settled", lambda day: True)
    monkeypatch.setattr(holding_sim, "SHADOW_START", calendar_rows[0])
    return asked


def run(*args: str) -> str:
    out = StringIO()
    call_command("holding_sim", *args, stdout=out)
    return out.getvalue()


RESEARCH = ["--run-key", "rank-cand60", "--folds", "8-43", "--until", "2016-01-28"]


def test_the_research_book_prints_its_summary_counts_and_years(fed: dict[str, Any]) -> None:
    text = run(*RESEARCH, "--end", "2016-01-29", "--by-year", "--size", "2")

    assert fed == {
        "run_key": "rank-cand60",
        "folds": list(range(8, 44)),
        "until": date(2016, 1, 28),
        "end": date(2016, 1, 29),
    }
    assert text.splitlines()[0].startswith("H70/N2 live, rank-cand60 folds 8-43, valued 2016-01-04")
    assert "net_mo" in text
    assert "sells 2, settled 0, stuck days 0, sells opening at limit down 0" in text
    assert "\n2016 " in text
    assert "has no close" not in text


PREDICTION = ["--source", "prediction", "--start", "2016-01-04"]


def test_the_list_shows_the_last_day_keeps_and_entries(fed: dict[str, Any]) -> None:
    text = run(*PREDICTION, "--size", "2", "--list")

    listed = text.split("list of 2016-01-29:")[1].strip().splitlines()
    assert listed[0].split() == ["stock_id", "action", "priority", "score", "pct"]
    assert [line.split()[:2] for line in listed[1:]] == [["1000", "keep"], ["1001", "keep"]]


def test_a_last_day_without_predictions_says_so_instead_of_listing(
    fed: dict[str, Any], monkeypatch: pytest.MonkeyPatch, calendar_rows: list[date]
) -> None:
    score = np.tile(np.arange(1.0, 11.0), (20, 1))
    score[19] = NAN
    unscored = book_market(calendar_rows, score)
    monkeypatch.setattr(holding_sim, "load_market", lambda frame, end, calendar: unscored)

    text = run(*PREDICTION, "--list")

    assert "no prediction stored for 2016-01-29; the book holds still" in text
    assert "list of" not in text


def test_stored_predictions_default_to_the_latest_price_day(fed: dict[str, Any]) -> None:
    text = run("--source", "prediction", "--start", "2016-01-04", "--settle", "known")

    assert fed["start"] == date(2016, 1, 4)
    assert fed["until"] == fed["end"] == date(2016, 1, 29)
    assert text.splitlines()[0].startswith("H70/N30 known, prediction 2016-01-04..2016-01-29")


def test_stored_predictions_start_on_the_first_shadow_signal_day_by_default(
    fed: dict[str, Any],
) -> None:
    run("--source", "prediction")

    assert fed["start"] == date(2016, 1, 4)


def test_stored_predictions_before_the_first_shadow_signal_day_are_refused(
    fed: dict[str, Any], monkeypatch: pytest.MonkeyPatch, calendar_rows: list[date]
) -> None:
    monkeypatch.setattr(holding_sim, "SHADOW_START", calendar_rows[5])

    with pytest.raises(CommandError, match="first signal day 2016-01-11"):
        run(*PREDICTION)


def test_the_first_signal_day_alone_still_prints_its_list(
    fed: dict[str, Any], monkeypatch: pytest.MonkeyPatch, calendar_rows: list[date]
) -> None:
    one = book_market(calendar_rows, np.arange(1.0, 11.0)[None, :])
    monkeypatch.setattr(holding_sim, "load_market", lambda frame, end, calendar: one)

    text = run(*PREDICTION, "--size", "2", "--list")

    assert "no trade yet; the first signal day is the last day" in text
    listed = text.split("list of 2016-01-04:")[1].strip().splitlines()
    assert [line.split()[:3] for line in listed[1:]] == [
        ["1009", "buy", "1"],
        ["1008", "buy", "2"],
    ]


def test_the_research_source_refuses_a_list(fed: dict[str, Any]) -> None:
    with pytest.raises(CommandError, match="use --source prediction"):
        run(*RESEARCH, "--end", "2016-01-29", "--list")


def test_days_without_a_taiex_tr_close_are_counted(
    fed: dict[str, Any], monkeypatch: pytest.MonkeyPatch, calendar_rows: list[date]
) -> None:
    levels = pd.Series(100.0, index=calendar_rows[:17])
    monkeypatch.setattr(holding_sim, "index_levels", lambda start, end: levels)

    text = run(*RESEARCH, "--end", "2016-01-29")

    assert "TAIEX_TR has no close on 3 of 20 days; carried over" in text


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--folds", "8-43", "--until", "2016-01-28", "--end", "2016-01-29"], "--run-key"),
        (["--run-key", "rank-cand60", "--until", "2016-01-28", "--end", "2016-01-29"], "--folds"),
        ([*RESEARCH[:3], "8-x", *RESEARCH[4:], "--end", "2016-01-29"], "look like 8-43"),
        ([*RESEARCH[:3], "43-8", *RESEARCH[4:], "--end", "2016-01-29"], "runs backwards"),
        ([*RESEARCH, "--end", "2016-01-28"], "after --until"),
        ([*RESEARCH, "--end", "2025-01-02"], "before the holdout period"),
        ([*RESEARCH[:3], "8-44", *RESEARCH[4:], "--end", "2016-01-29"], "run them apart"),
        ([*RESEARCH[:3], "8-44", *RESEARCH[4:], "--end", "2016-01-29", "--rule", "H50"], "44"),
        ([*RESEARCH, "--end", "2016-01-29", "--rule", "P5"], "rule must look like"),
    ],
)
def test_bad_or_forbidden_requests_are_refused(
    fed: dict[str, Any], args: list[str], message: str
) -> None:
    with pytest.raises(CommandError, match=message):
        run(*args)


def test_stored_predictions_wait_for_the_labels_of_the_day_before_the_last(
    fed: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[date] = []

    def unsettled(day: date) -> bool:
        asked.append(day)
        return False

    monkeypatch.setattr(holding_sim, "labels_settled", unsettled)

    with pytest.raises(CommandError, match="untradable_t1 of 2016-01-28 is not settled"):
        run(*PREDICTION)

    assert asked == [date(2016, 1, 28)]


def test_a_one_day_prediction_book_needs_no_settled_labels(
    fed: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(holding_sim, "labels_settled", lambda day: False)

    text = run("--source", "prediction", "--start", "2016-01-29", "--list")

    assert "list of 2016-01-29:" in text


def test_holdout_folds_may_price_on_holdout_days_for_the_three_rules(fed: dict[str, Any]) -> None:
    args = ["--run-key", "rank-holdout-20261112", "--folds", "44-51", "--until", "2026-11-04"]

    text = run(*args, "--end", "2026-11-11", "--rule", "H90", "--size", "100")

    assert text.splitlines()[0].startswith("H90/N100 live, rank-holdout-20261112 folds 44-51")


def test_a_book_that_never_trades_is_reported_as_such(
    fed: dict[str, Any], monkeypatch: pytest.MonkeyPatch, calendar_rows: list[date]
) -> None:
    empty = book_market(calendar_rows, np.full((20, 3), NAN))
    monkeypatch.setattr(holding_sim, "load_market", lambda frame, end, calendar: empty)

    with pytest.raises(CommandError, match="never traded"):
        run(*RESEARCH, "--end", "2016-01-29")
