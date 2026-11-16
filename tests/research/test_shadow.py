from datetime import date, datetime, time, timedelta
from typing import Any

import numpy as np
import pytest
from django.db import OperationalError, connection, transaction
from django.db.models import F
from django.utils import timezone

from apps.etl.models import DataQualityLog
from apps.market.calendar import TradingDays
from apps.market.models import DailyPrice, IndexDaily, StockMaster, TradingCalendar
from apps.research import shadow, tasks
from apps.research.holding import Rule, simulate
from apps.research.holding_inputs import index_levels, load_market, stored_scores
from apps.research.holding_report import index_returns, universe_returns
from apps.research.models import (
    BookPhase,
    BookStatus,
    HoldingBookAction,
    HoldingBookDay,
    LabelDaily,
    ModelRun,
    Prediction,
)
from apps.research.shadow import SHADOW_CHECK, Outcome, Spec, budget, record_books

pytestmark = pytest.mark.django_db
FIRST = date(2026, 11, 16)
STOCKS = [str(1000 + k) for k in range(20)]
GONE = "1019"
QUIET = 6
BLOCKED = ("1003", 4)
RULE = Rule.parse("H70", 3)
BOOK = {"T": Spec(RULE, FIRST)}
COUNT = 14


@pytest.fixture
def days() -> list[date]:
    span = [FIRST + timedelta(days=n) for n in range(40)]
    TradingCalendar.objects.bulk_create(
        TradingCalendar(date=day, is_trading_day=day.weekday() < 5) for day in span
    )
    return [day for day in span if day.weekday() < 5][:COUNT]


@pytest.fixture
def market(days: list[date]) -> list[date]:
    """Twenty stocks; GONE leads, stops trading after day 5, delists on day 8; QUIET is unscored."""
    rng = np.random.default_rng(7)
    for stock in STOCKS:
        delist = days[8] if stock == GONE else None
        StockMaster.objects.create(
            stock_id=stock,
            name=stock,
            market="TWSE",
            industry="半導體業",
            list_date=date(2020, 1, 2),
            delist_date=delist,
        )
    run = ModelRun.objects.create(
        run_id="rank.h5",
        run_key="prod",
        model_type="lightgbm",
        objective="rank",
        version="2026-11",
        train_start=date(2015, 1, 5),
        train_end=date(2025, 12, 31),
        lib_versions={},
        feature_hash="0" * 64,
        status="active",
        params={},
    )
    bars, predictions, labels = [], [], []
    for stock in STOCKS:
        close = 50.0
        for n, day in enumerate(days):
            opened = close * (1 + rng.uniform(-0.02, 0.02))
            close = opened * (1 + rng.uniform(-0.04, 0.04))
            if stock != GONE or n <= 5:
                bars.append(price(stock, day, opened, close))
                score = 9.0 if stock == GONE else float(rng.normal())
                if n != QUIET:
                    predictions.append(
                        Prediction(
                            run=run,
                            stock_id=stock,
                            date=day,
                            horizon=5,
                            score=score,
                            in_universe=(stock, n) != ("1005", 2),
                        )
                    )
            labels.append(
                LabelDaily(stock_id=stock, date=day, horizon=5, untradable_t1=(stock, n) == BLOCKED)
            )
    DailyPrice.objects.bulk_create(bars)
    Prediction.objects.bulk_create(predictions)
    LabelDaily.objects.bulk_create(labels)
    IndexDaily.objects.bulk_create(
        IndexDaily(index_id="TAIEX_TR", date=day, close=100 + n) for n, day in enumerate(days)
    )
    return days


def price(stock: str, day: date, opened: float, close: float) -> DailyPrice:
    return DailyPrice(
        stock_id=stock,
        date=day,
        open=round(opened, 2),
        high=round(max(opened, close) * 1.01, 2),
        low=round(min(opened, close) * 0.99, 2),
        close=round(close, 2),
        volume=1_000,
        turnover=50_000,
        adj_factor=1.0,
    )


def at(day: date, clock: time) -> datetime:
    return timezone.make_aware(datetime.combine(day, clock))


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> dict[str, datetime]:
    """Each morning written at 06:30 of the day after its session, unless a test moves it."""
    now = {"at": at(FIRST, time(6, 30))}
    monkeypatch.setattr(shadow.timezone, "now", lambda: now["at"])
    return now


def mornings(
    days: list[date], clock: dict[str, datetime], book: dict[str, Spec] | None = None
) -> list[Any]:
    outcomes = []
    for session in days:
        clock["at"] = at(session + timedelta(days=1), time(6, 30))
        outcomes.append(record_books(session, book or BOOK))
    return outcomes


def fills(book: str = "T") -> list[HoldingBookDay]:
    rows = HoldingBookDay.objects.filter(book=book, phase=BookPhase.FILL)
    return list(rows.order_by("signal_date"))


def lines(book: str, signal: date, phase: str) -> list[tuple[Any, ...]]:
    rows = HoldingBookAction.objects.filter(book=book, signal_date=signal, phase=phase)
    fields = ("stock_id", "action", "priority", "score", "pct", "amount", "value", "cost")
    return list(rows.order_by("stock_id").values_list(*fields))


def test_mornings_one_by_one_equal_one_simulation_of_the_same_predictions(
    market: list[date], clock: dict[str, datetime]
) -> None:
    outcomes = mornings(market, clock)

    assert {outcome.severity for outcome in outcomes} == {"info"}
    last = market[-1]
    book = simulate(load_market(stored_scores(FIRST, last), last, TradingDays.from_db()), RULE)
    daily = book.daily.iloc[1:]
    rows = fills()
    assert [row.trade_date for row in rows] == list(daily["date"])
    for column in ("value", "cash", "paid", "traded"):
        got = [getattr(row, column) for row in rows]
        assert got == pytest.approx(list(daily[column]), rel=1e-12, abs=1e-15)
    assert [row.held for row in rows] == list(daily["held"])
    grid = load_market(stored_scores(FIRST, last), last, TradingDays.from_db())
    levels = index_levels(FIRST, last)
    assert [row.universe_return for row in rows] == pytest.approx(
        list(universe_returns(grid)[1:]), rel=1e-12, abs=1e-15
    )
    assert [row.index_return for row in rows] == pytest.approx(
        list(index_returns(levels, grid.days)[1:]), rel=1e-12
    )
    moved = HoldingBookAction.objects.filter(book="T", phase=BookPhase.FILL).exclude(action="keep")
    stored = sorted(
        (
            row.signal_date,
            row.stock_id,
            row.action,
            row.amount if row.amount is not None else row.value,
        )
        for row in moved
    )
    expected = sorted(
        (line.signal_date, line.stock_id, line.action, line.amount)
        for line in book.actions.itertuples()
    )
    assert [line[:3] for line in stored] == [line[:3] for line in expected]
    assert [line[3] for line in stored] == pytest.approx([line[3] for line in expected], rel=1e-12)
    assert {line[2] for line in stored} >= {"buy", "sell", "stuck", "settle"}


def test_each_list_counts_its_lines_and_a_quiet_day_lists_nothing_and_only_values(
    market: list[date], clock: dict[str, datetime]
) -> None:
    mornings(market[: QUIET + 2], clock)

    quiet = HoldingBookDay.objects.get(book="T", signal_date=market[QUIET], phase=BookPhase.LIST)
    assert quiet.status == BookStatus.NO_PREDICTION
    assert lines("T", market[QUIET], BookPhase.LIST) == []
    idle = HoldingBookDay.objects.get(book="T", signal_date=market[QUIET], phase=BookPhase.FILL)
    assert idle.status == BookStatus.IDLE
    assert idle.paid == idle.traded == 0.0
    assert {row[1] for row in lines("T", market[QUIET], BookPhase.FILL)} == {"keep"}
    first = HoldingBookDay.objects.get(book="T", signal_date=FIRST, phase=BookPhase.LIST)
    listed = lines("T", FIRST, BookPhase.LIST)
    assert (first.status, first.run_id, first.held_before) == (BookStatus.LISTED, "rank.h5", 0)
    assert first.buys == RULE.slots == len([row for row in listed if row[1] == "buy"])
    assert first.backups == len([row for row in listed if row[1] == "backup"])
    assert listed[-1][0] == GONE
    assert listed[-1][2] == 1
    assert len(first.universe) == len(STOCKS)


def test_a_morning_rerun_writes_nothing_more(
    market: list[date], clock: dict[str, datetime]
) -> None:
    mornings(market[:4], clock)
    counts = (HoldingBookDay.objects.count(), HoldingBookAction.objects.count())

    again = record_books(market[3], BOOK)

    assert (HoldingBookDay.objects.count(), HoldingBookAction.objects.count()) == counts
    assert again.detail["books"]["T"] == {"listed": [], "filled": [], "waiting": "2026-11-19"}
    assert again.summary.startswith("T: nothing due;")


def test_a_late_first_morning_backfills_and_marks_what_was_due_before_it(
    market: list[date], clock: dict[str, datetime]
) -> None:
    clock["at"] = at(market[5], time(7, 0))

    outcome = record_books(market[4], BOOK)

    assert outcome.detail["books"]["T"]["listed"] == [day.isoformat() for day in market[:5]]
    marked = HoldingBookDay.objects.filter(book="T", backfilled=True)
    assert (
        sorted(marked.filter(phase=BookPhase.LIST).values_list("signal_date", flat=True))
        == market[:4]
    )
    assert (
        sorted(marked.filter(phase=BookPhase.FILL).values_list("signal_date", flat=True))
        == market[:3]
    )
    assert all(row.recorded_at == clock["at"] for row in HoldingBookDay.objects.all())


def test_a_fill_waits_for_its_labels_and_the_next_list_waits_with_it(
    market: list[date], clock: dict[str, datetime]
) -> None:
    LabelDaily.objects.filter(date=market[2], stock_id="1000").update(untradable_t1=None)

    outcome = mornings(market[:4], clock)[-1]

    assert outcome.detail["books"]["T"] == {"listed": [], "filled": [], "waiting": "2026-11-18"}
    assert [row.signal_date for row in fills()] == market[:2]
    listed = HoldingBookDay.objects.filter(book="T", phase=BookPhase.LIST)
    assert sorted(listed.values_list("signal_date", flat=True)) == market[:3]


def test_a_fill_follows_its_list_after_the_predictions_are_rewritten(
    market: list[date], clock: dict[str, datetime]
) -> None:
    mornings(market[:3], clock)
    rewritten = Prediction.objects.filter(date=market[2])
    rewritten.update(score=-F("score"))
    rewritten.filter(stock_id="1007").update(in_universe=False)

    warned = mornings(market[3:4], clock)[0]

    rewritten.update(score=-F("score"))
    rewritten.filter(stock_id="1007").update(in_universe=True)
    mornings(market[:4], clock, {"R": Spec(RULE, FIRST)})
    note = warned.detail["notes"][0]
    assert (warned.severity, note["signal_date"]) == ("warning", "2026-11-18")
    assert note["predictions_rewritten"]["scores_changed"] > 0
    assert note["predictions_rewritten"]["universe_removed"] == 1
    for phase in (BookPhase.LIST, BookPhase.FILL):
        assert lines("T", market[2], phase) == lines("R", market[2], phase)
    rewrote, kept = fills("T")[2], fills("R")[2]
    assert (rewrote.value, rewrote.universe_return) == (kept.value, kept.universe_return)


def test_a_held_stock_whose_predictions_are_all_gone_still_fills_and_leaves(
    market: list[date], clock: dict[str, datetime]
) -> None:
    mornings(market[:3], clock)
    held = HoldingBookAction.objects.filter(book="T", signal_date=market[1], phase=BookPhase.FILL)
    assert GONE in set(held.values_list("stock_id", flat=True))
    Prediction.objects.filter(stock_id=GONE).delete()

    outcome = mornings(market[3:4], clock)[0]

    assert outcome.detail["books"]["T"]["filled"] == ["2026-11-18"]
    listed = HoldingBookAction.objects.get(
        book="T", signal_date=market[3], phase=BookPhase.LIST, stock_id=GONE
    )
    assert (listed.action, listed.score, listed.pct) == ("sell", None, None)


def test_a_universe_stock_without_a_score_is_noted_and_the_list_still_written(
    market: list[date], clock: dict[str, datetime]
) -> None:
    Prediction.objects.filter(date=FIRST, stock_id="1001").update(score=None)

    outcome = record_books(FIRST, BOOK)

    assert outcome.severity == "warning"
    assert outcome.detail["notes"] == [
        {
            "book": "T",
            "signal_date": "2026-11-16",
            "in_universe_without_score": 1,
            "first": ["1001"],
        }
    ]
    assert "1001" in HoldingBookDay.objects.get(book="T", phase=BookPhase.LIST).universe


def test_any_failure_keeps_nothing_and_names_the_book_and_day(
    market: list[date], clock: dict[str, datetime], monkeypatch: pytest.MonkeyPatch
) -> None:
    real, calls = shadow.step, []

    def breaks(*args: Any) -> Any:
        calls.append(1)
        if len(calls) == 2:
            raise ArithmeticError("bad day")
        return real(*args)

    monkeypatch.setattr(shadow, "step", breaks)

    outcome = record_books(market[3], BOOK)

    assert outcome.severity == "warning"
    assert outcome.detail["book"] == "T"
    assert outcome.detail["signal_date"] == "2026-11-17"
    assert outcome.detail["error"] == "ArithmeticError: bad day"
    assert outcome.summary.startswith("failed at T 2026-11-17, see research_shadow;")
    assert not HoldingBookDay.objects.exists()
    assert not HoldingBookAction.objects.exists()


def test_a_spent_budget_stops_the_morning_and_keeps_nothing(
    market: list[date], clock: dict[str, datetime], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shadow, "BUDGET_SECONDS", 0.0)

    outcome = record_books(market[3], BOOK)

    assert outcome.detail["error"] == "ShadowTimeoutError: the 0 s shadow budget ran out"
    assert not HoldingBookDay.objects.exists()


def show_timeout() -> str:
    with connection.cursor() as cursor:
        cursor.execute("SHOW statement_timeout")
        (value,) = cursor.fetchone()
    return str(value)


def test_the_budget_bounds_each_statement_and_restores_the_setting_after() -> None:
    before = show_timeout()
    with transaction.atomic(), budget(30.0):
        inside = show_timeout()
    assert inside.endswith("s")
    assert inside != before
    assert show_timeout() == before
    with (
        pytest.raises(OperationalError, match="statement timeout"),
        transaction.atomic(),
        budget(0.3),
        connection.cursor() as cursor,
    ):
        cursor.execute("SELECT pg_sleep(2)")
    assert show_timeout() == before


def test_no_book_is_due_before_its_start(days: list[date]) -> None:
    outcome = record_books(FIRST - timedelta(days=3), BOOK)

    assert (outcome.severity, outcome.summary) == (
        "info",
        f"no book due; {outcome.detail['seconds']} s",
    )


def test_the_morning_step_records_the_books_without_alerting_when_all_is_well(
    market: list[date], clock: dict[str, datetime], monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[DataQualityLog] = []
    monkeypatch.setattr(tasks, "dispatch", sent.extend)

    summary = tasks.STEPS["shadow"](market[1])

    assert summary.startswith("H70N30-C0: listed 2026-11-16, 2026-11-17, filled 2026-11-16; ")
    row = DataQualityLog.objects.get(date=market[1], check_name=SHADOW_CHECK)
    assert row.severity == "info"
    assert row.detail["books"]["H70N30-C0"]["waiting"] == "2026-11-17"
    assert sent == []


def test_the_morning_step_alerts_a_failure_and_never_raises(
    days: list[date], monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[DataQualityLog] = []
    monkeypatch.setattr(tasks, "dispatch", sent.extend)
    failed = Outcome("warning", {"book": "T", "signal_date": "2026-11-16"}, "failed at T")
    monkeypatch.setattr(tasks, "record_books", lambda day: failed)

    assert tasks.STEPS["shadow"](FIRST) == "failed at T"
    row = DataQualityLog.objects.get(date=FIRST, check_name=SHADOW_CHECK)
    assert (row.severity, row.detail) == ("warning", failed.detail)
    assert sent == [row]

    def gone(day: date) -> Outcome:
        raise ConnectionError("database gone")

    monkeypatch.setattr(tasks, "record_books", gone)

    assert tasks.STEPS["shadow"](FIRST) == "failed to record (ConnectionError), see the worker log"


def test_a_held_stock_outside_the_grid_is_an_error(market: list[date]) -> None:
    inputs = shadow.Inputs(FIRST, market[2], TradingDays.from_db(), set)

    assert inputs.mask(["1000", "1002"]).nonzero()[0].tolist() == [0, 2]
    with pytest.raises(ValueError, match=r"without a prediction row since 2026-11-16: \['9999'\]"):
        inputs.mask(["1000", "9999"])
