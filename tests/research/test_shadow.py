import hashlib
import pickle
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from django.db import OperationalError, connection, transaction
from django.db.models import F
from django.utils import timezone

from apps.api.ops import STOCK_CODES
from apps.etl.models import DataQualityLog
from apps.market.calendar import TradingDays
from apps.market.models import DailyPrice, IndexDaily, StockMaster, TradingCalendar
from apps.research import candidate_scores, shadow, tasks
from apps.research.candidate_scores import Scored
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
    ShadowArtifact,
    ShadowScore,
)
from apps.research.shadow import SHADOW_CHECK, Outcome, Spec, all_books, budget, record_books

pytestmark = pytest.mark.django_db
FIRST = date(2026, 11, 16)
STOCKS = [str(1000 + k) for k in range(20)]
GONE = "1019"
DROPPED = "1018"
FORCED = {0: 8.0, 1: 8.0, 2: 8.0, 3: 8.0, 4: 8.0, 5: -9.0}
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
            if (stock, n) == (DROPPED, QUIET):
                opened = close * 0.88
            close = opened * (1 + rng.uniform(-0.04, 0.04))
            if stock != GONE or n <= 5:
                bars.append(price(stock, day, opened, close))
                score = 9.0 if stock == GONE else float(rng.normal())
                if stock == DROPPED:
                    score = FORCED.get(n, score)
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
    sells = moved.filter(action="sell").order_by("signal_date", "stock_id")
    opened_down = book.actions[book.actions["action"] == "sell"].sort_values(
        ["signal_date", "stock_id"]
    )
    assert [row.limit_down_open for row in sells] == list(opened_down["limit_down_open"])
    assert sells.get(stock_id=DROPPED, signal_date=market[5]).limit_down_open is True
    for row in rows:
        costs = HoldingBookAction.objects.filter(
            book="T", signal_date=row.signal_date, phase=BookPhase.FILL, cost__isnull=False
        ).values_list("cost", flat=True)
        assert sum(cost or 0.0 for cost in costs) == pytest.approx(row.paid, rel=1e-12, abs=1e-15)
    listed = HoldingBookDay.objects.filter(book="T", phase=BookPhase.LIST)
    run_ids = dict(listed.values_list("signal_date", "run_id"))
    fill_lines = HoldingBookAction.objects.filter(book="T", phase=BookPhase.FILL)
    assert {row.run_id for row in rows} == {"rank.h5", ""}
    assert all(row.run_id == run_ids[row.signal_date] for row in rows)
    assert all(line.run_id == run_ids[line.signal_date] for line in fill_lines)


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
    assert note["predictions_rewritten"][STOCK_CODES][0] == "1007"
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
            STOCK_CODES: ["1001"],
        }
    ]
    assert "1001" in HoldingBookDay.objects.get(book="T", phase=BookPhase.LIST).universe


def test_a_failing_book_keeps_nothing_of_the_morning_and_is_noted_by_book_and_day(
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
    assert outcome.detail["books"] == {}
    assert outcome.detail["notes"] == [
        {
            "book": "T",
            "role": "main",
            "signal_date": "2026-11-17",
            "book_failed": "ArithmeticError: bad day",
        }
    ]
    assert outcome.summary.startswith("T: failed at 2026-11-17, 1 notes, see research_shadow;")
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


def close_day(day: date) -> None:
    """The calendar now says the day was closed, and nothing traded on it."""
    TradingCalendar.objects.filter(date=day).update(is_trading_day=False)
    for model in (DailyPrice, Prediction, LabelDaily, IndexDaily):
        model.objects.filter(date=day).delete()


def test_a_trade_day_closed_after_its_list_moves_the_fill_to_the_next_trading_day(
    market: list[date], clock: dict[str, datetime]
) -> None:
    mornings(market[:3], clock)
    close_day(market[3])

    outcome = mornings(market[4:5], clock)[0]

    assert outcome.detail["notes"] == [
        {
            "book": "T",
            "signal_date": "2026-11-18",
            "trade_date_moved": {"listed": "2026-11-19", "filled": "2026-11-20"},
        }
    ]
    moved = fills("T")[2]
    assert (moved.trade_date, moved.backfilled) == (market[4], False)
    mornings([*market[:3], market[4]], clock, {"R": Spec(RULE, FIRST)})
    for phase in (BookPhase.LIST, BookPhase.FILL):
        assert lines("T", market[2], phase) == lines("R", market[2], phase)
    assert [row.value for row in fills("T")] == [row.value for row in fills("R")]


def test_a_day_opened_under_the_book_stops_it_and_names_the_day(
    market: list[date], clock: dict[str, datetime]
) -> None:
    close_day(market[2])
    mornings([*market[:2], *market[3:5]], clock)
    TradingCalendar.objects.filter(date=market[2]).update(is_trading_day=True)
    counts = (HoldingBookDay.objects.count(), HoldingBookAction.objects.count())

    outcome = record_books(market[4], BOOK)

    (note,) = outcome.detail["notes"]
    assert (note["book"], note["signal_date"]) == ("T", "2026-11-18")
    assert "already listed 2026-11-20: the calendar opened a day" in note["book_failed"]
    assert (HoldingBookDay.objects.count(), HoldingBookAction.objects.count()) == counts


def test_a_missing_taiex_tr_close_leaves_the_index_return_empty_and_noted(
    market: list[date], clock: dict[str, datetime]
) -> None:
    IndexDaily.objects.filter(date=market[3]).delete()

    outcomes = mornings(market[:5], clock)

    assert [row.index_return is None for row in fills()] == [False, False, True, True]
    assert outcomes[3].detail["notes"] == [
        {"book": "T", "signal_date": "2026-11-18", "no_taiex_tr": ["2026-11-19"]}
    ]
    assert outcomes[4].detail["notes"][0]["no_taiex_tr"] == ["2026-11-19"]


def test_a_quiet_rerun_keeps_the_day_s_first_row_and_its_notes(
    market: list[date], clock: dict[str, datetime], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tasks, "dispatch", lambda rows: None)
    Prediction.objects.filter(date=FIRST, stock_id="1001").update(score=None)
    tasks.STEPS["shadow"](FIRST)

    again = tasks.STEPS["shadow"](FIRST)

    assert again.endswith("; kept the day's earlier research_shadow row")
    row = DataQualityLog.objects.get(date=FIRST, check_name=SHADOW_CHECK)
    assert row.severity == "warning"
    assert row.detail["notes"][0][STOCK_CODES] == ["1001"]


def test_a_universe_stock_whose_predictions_are_all_gone_still_counts_in_the_fill(
    market: list[date], clock: dict[str, datetime]
) -> None:
    both = {"T": Spec(RULE, FIRST), "R": Spec(RULE, FIRST)}
    mornings(market[:3], clock, both)
    acted = set(HoldingBookAction.objects.values_list("stock_id", flat=True))
    universe = HoldingBookDay.objects.get(book="T", signal_date=market[2], phase=BookPhase.LIST)
    quiet = sorted(set(universe.universe) - acted)[0]
    clock["at"] = at(market[3] + timedelta(days=1), time(6, 30))
    record_books(market[3], {"R": both["R"]})
    Prediction.objects.filter(stock_id=quiet).delete()

    outcome = record_books(market[3], {"T": both["T"]})

    assert outcome.detail["books"]["T"]["filled"] == ["2026-11-18"]
    assert fills("T")[2].universe_return == fills("R")[2].universe_return


def frozen(name: str, start: date, folder: Path, stored: bool = True) -> ShadowArtifact:
    """A frozen artifact row of the candidate from start; its file is kept only when stored."""
    path = f"shadow/{name.lower()}/prod/g1.pkl"
    data = pickle.dumps({"candidate": name})
    if stored:
        (folder / path).parent.mkdir(parents=True, exist_ok=True)
        (folder / path).write_bytes(data)
    return ShadowArtifact.objects.create(
        candidate=name,
        serving_run=ModelRun.objects.get(run_id="rank.h5"),
        generation=1,
        as_of=FIRST - timedelta(days=3),
        train_start=date(2021, 10, 14),
        train_end=date(2026, 10, 13),
        fold={},
        path=path,
        sha256=hashlib.sha256(data).hexdigest(),
        recipe={},
        first_signal=start,
    )


def scored_against_c0(name: str) -> None:
    """Candidate scores that reverse C0's, on the same universe."""
    ShadowScore.objects.bulk_create(
        ShadowScore(
            candidate=name,
            date=row.date,
            stock_id=row.stock_id,
            serving_run_id=row.run_id,
            generation=1,
            score=None if row.score is None else -row.score,
            raw=None if row.score is None else -row.score,
            in_universe=row.in_universe,
        )
        for row in Prediction.objects.all()
    )


def written(book: str) -> list[tuple[Any, ...]]:
    """Every column of the book's day and action rows, in key order."""
    days = HoldingBookDay.objects.filter(book=book).order_by("signal_date", "phase")
    acts = HoldingBookAction.objects.filter(book=book).order_by("signal_date", "phase", "stock_id")
    day_fields = [field.name for field in HoldingBookDay._meta.concrete_fields]
    act_fields = [field.name for field in HoldingBookAction._meta.concrete_fields]
    return [*days.values_list(*day_fields), *acts.values_list(*act_fields)]


CANDIDATE_BOOKS = {"T": Spec(RULE, FIRST), "T-C1": Spec(RULE, date(2026, 11, 18), "C1")}


def alone_then_cleared(market: list[date], clock: dict[str, datetime], last: int) -> list[Any]:
    mornings(market[:last], clock)
    kept = written("T")
    HoldingBookAction.objects.all().delete()
    HoldingBookDay.objects.all().delete()
    return kept


def test_all_books_adds_each_frozen_candidate_and_one_c0_book_from_its_first_day(
    market: list[date], tmp_path: Path
) -> None:
    assert list(all_books()) == ["H70N30-C0"]
    frozen("C1", market[2], tmp_path)
    frozen("C2", market[2], tmp_path)

    found = all_books()

    assert list(found) == ["H70N30-C0", "H70N30-C0-20261118", "H70N30-C1", "H70N30-C2"]
    assert [(spec.start, spec.source, spec.role) for spec in found.values()] == [
        (FIRST, None, "main"),
        (market[2], None, "paired"),
        (market[2], "C1", "candidate"),
        (market[2], "C2", "candidate"),
    ]


def test_a_failing_c0_book_turns_the_morning_into_a_warning_and_sends_the_alert(
    market: list[date], clock: dict[str, datetime], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        tasks, "dispatch", lambda rows: sent.extend((r.check_name, r.severity) for r in rows)
    )
    frozen("C1", market[0], tmp_path)
    scored_against_c0("C1")
    monkeypatch.setattr(shadow, "score_candidate", lambda *args: Scored())
    real = shadow.order_list

    def breaks(held: Any, score: Any, *args: Any) -> Any:
        if np.nanmax(score) > 0:
            raise ArithmeticError("bad C0 day")
        return real(held, score, *args)

    monkeypatch.setattr(shadow, "order_list", breaks)
    Prediction.objects.filter(score__lt=0).update(score=F("score") * -1)
    ShadowScore.objects.filter(score__gt=0).update(score=F("score") * -1)
    clock["at"] = at(FIRST + timedelta(days=1), time(6, 30))

    said = tasks.STEPS["shadow"](FIRST)

    row = DataQualityLog.objects.get(date=FIRST, check_name=SHADOW_CHECK)
    assert row.severity == "warning"
    assert sent == [(SHADOW_CHECK, "warning")]
    failed = [note for note in row.detail["notes"] if "book_failed" in note]
    assert failed == [
        {
            "book": "H70N30-C0",
            "role": "main",
            "signal_date": "2026-11-16",
            "book_failed": "ArithmeticError: bad C0 day",
        },
        {
            "book": "H70N30-C0-20261116",
            "role": "paired",
            "signal_date": "2026-11-16",
            "book_failed": "ArithmeticError: bad C0 day",
        },
    ]
    assert list(row.detail["books"]) == ["H70N30-C1"]
    assert said.startswith("H70N30-C1: listed 2026-11-16; H70N30-C0: failed at 2026-11-16;")


def test_a_candidate_book_follows_its_own_scores_and_c0_s_rows_stay_as_they_were(
    market: list[date], clock: dict[str, datetime], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    alone = alone_then_cleared(market, clock, 6)
    frozen("C1", market[2], tmp_path)
    scored_against_c0("C1")
    monkeypatch.setattr(
        shadow, "score_candidate", lambda name, session, *args: Scored([session.isoformat()])
    )

    outcomes = mornings(market[:6], clock, CANDIDATE_BOOKS)

    assert written("T") == alone
    head = HoldingBookDay.objects.filter(book="T-C1", phase=BookPhase.LIST).order_by("signal_date")
    assert head[0].signal_date == market[2]
    assert {row.run_id for row in head} == {"C1:rank.h5:g1"}
    served = Prediction.objects.filter(date=market[2]).exclude(score=None)
    reversed_scores = {row.stock_id: -(row.score or 0.0) for row in served}
    for stock, _, _, score, *_ in lines("T-C1", market[2], BookPhase.LIST):
        assert score == reversed_scores[stock]
    assert outcomes[3].detail["scores"] == {"C1": {"scored": ["2026-11-19"], "artifacts": []}}
    assert outcomes[3].severity == "info"


def test_a_candidate_whose_scores_fail_leaves_c0_alone_and_skips_its_own_book(
    market: list[date], clock: dict[str, datetime], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    alone = alone_then_cleared(market, clock, 4)
    frozen("C1", market[2], tmp_path)

    def broken(*args: Any) -> Scored:
        raise ArithmeticError("no features")

    monkeypatch.setattr(shadow, "score_candidate", broken)

    outcome = mornings(market[:4], clock, CANDIDATE_BOOKS)[-1]

    assert written("T") == alone
    assert not HoldingBookDay.objects.filter(book="T-C1").exists()
    assert outcome.severity == "warning"
    assert outcome.detail["notes"] == [
        {"book": "C1", "role": "candidate", "scores_failed": "ArithmeticError: no features"},
        {
            "book": "T-C1",
            "role": "candidate",
            "signal_date": None,
            "book_failed": "skipped: C1 has no scores this morning",
        },
    ]
    assert "T-C1" not in outcome.detail["books"]


def test_a_failing_candidate_book_rolls_back_its_earlier_day_of_the_same_morning(
    market: list[date], clock: dict[str, datetime], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mornings(market[:3], clock)
    frozen("C1", market[2], tmp_path)
    scored_against_c0("C1")
    monkeypatch.setattr(shadow, "score_candidate", lambda *args: Scored())
    real = shadow.order_list
    lists: list[int] = []

    def breaks(held: Any, score: Any, *args: Any) -> Any:
        if np.nanmax(score) < 0:
            lists.append(1)
            if len(lists) == 2:
                raise ArithmeticError("bad candidate day")
        return real(held, score, *args)

    monkeypatch.setattr(shadow, "order_list", breaks)
    Prediction.objects.filter(score__lt=0).update(score=F("score") * -1)
    ShadowScore.objects.filter(score__gt=0).update(score=F("score") * -1)
    clock["at"] = at(market[3] + timedelta(days=1), time(6, 30))

    outcome = record_books(market[3], CANDIDATE_BOOKS)

    assert len(lists) == 2
    assert not HoldingBookDay.objects.filter(book="T-C1").exists()
    assert not HoldingBookAction.objects.filter(book="T-C1").exists()
    assert outcome.detail["notes"] == [
        {
            "book": "T-C1",
            "role": "candidate",
            "signal_date": "2026-11-19",
            "book_failed": "ArithmeticError: bad candidate day",
        }
    ]
    assert outcome.detail["books"]["T"]["listed"] == ["2026-11-19"]


@pytest.mark.django_db(transaction=True)
def test_one_candidate_spending_the_budget_leaves_the_other_s_book_written(
    market: list[date], clock: dict[str, datetime], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frozen("C1", market[2], tmp_path)
    frozen("C2", market[2], tmp_path)
    scored_against_c0("C1")
    scored_against_c0("C2")
    books = {**CANDIDATE_BOOKS, "T-C2": Spec(RULE, market[2], "C2")}
    ticks = {"now": 0.0}
    monkeypatch.setattr(shadow, "monotonic", lambda: ticks["now"])

    def scores(name: str, *args: Any) -> Scored:
        if name == "C2":
            ticks["now"] += shadow.BUDGET_SECONDS + 1
        return Scored()

    monkeypatch.setattr(shadow, "score_candidate", scores)
    clock["at"] = at(market[3], time(6, 30))

    outcome = record_books(market[2], books)

    assert list(outcome.detail["books"]) == ["T", "T-C1"]
    assert outcome.detail["notes"] == [
        {
            "book": "T-C2",
            "role": "candidate",
            "signal_date": None,
            "book_failed": "skipped: the shadow budget ran out",
        }
    ]


@pytest.mark.django_db(transaction=True)
def test_a_spent_budget_skips_the_candidate_book_and_the_next_morning_backfills_it(
    market: list[date], clock: dict[str, datetime], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frozen("C1", market[2], tmp_path)
    scored_against_c0("C1")
    mornings(market[:2], clock, CANDIDATE_BOOKS)
    ticks = {"now": 0.0}
    monkeypatch.setattr(shadow, "monotonic", lambda: ticks["now"])

    def slow(*args: Any) -> Scored:
        ticks["now"] += shadow.BUDGET_SECONDS + 1
        return Scored()

    monkeypatch.setattr(shadow, "score_candidate", slow)
    clock["at"] = at(market[3], time(6, 30))
    skipped = record_books(market[2], CANDIDATE_BOOKS)
    monkeypatch.setattr(shadow, "score_candidate", lambda *args: Scored())
    clock["at"] = at(market[4], time(6, 30))
    later = record_books(market[3], CANDIDATE_BOOKS)

    assert skipped.detail["books"]["T"]["listed"] == ["2026-11-18"]
    assert skipped.detail["notes"] == [
        {
            "book": "T-C1",
            "role": "candidate",
            "signal_date": None,
            "book_failed": "skipped: the shadow budget ran out",
        }
    ]
    assert later.detail["books"]["T-C1"]["listed"] == ["2026-11-18", "2026-11-19"]
    candidate = HoldingBookDay.objects.filter(book="T-C1", phase=BookPhase.LIST)
    assert dict(candidate.values_list("signal_date", "backfilled")) == {
        market[2]: True,
        market[3]: False,
    }


def test_a_missing_artifact_file_scores_nothing_and_its_book_lists_no_prediction(
    market: list[date],
    clock: dict[str, datetime],
    monkeypatch: pytest.MonkeyPatch,
    settings: Any,
    tmp_path: Path,
) -> None:
    settings.MODEL_DIR = tmp_path
    monkeypatch.setattr(tasks, "dispatch", lambda rows: None)
    alone = alone_then_cleared(market, clock, 4)
    frozen("C1", market[2], tmp_path, stored=False)
    for session in market[:3]:
        clock["at"] = at(session + timedelta(days=1), time(6, 30))
        record_books(session, CANDIDATE_BOOKS)
    clock["at"] = at(market[3] + timedelta(days=1), time(6, 30))
    monkeypatch.setattr(shadow, "all_books", lambda: CANDIDATE_BOOKS)

    said = tasks.STEPS["shadow"](market[3])

    assert written("T") == alone
    assert not ShadowScore.objects.exists()
    candidate = HoldingBookDay.objects.filter(book="T-C1", phase=BookPhase.LIST)
    assert set(candidate.values_list("status", flat=True)) == {BookStatus.NO_PREDICTION}
    row = DataQualityLog.objects.get(date=market[3], check_name=SHADOW_CHECK)
    assert row.severity == "warning"
    (note,) = row.detail["notes"]
    assert (note["book"], note["signal_date"]) == ("C1", "2026-11-19")
    assert "has no file at shadow/c1/prod/g1.pkl" in note["artifact_invalid"]
    assert "1 notes" in said


def test_a_file_found_again_scores_only_the_days_its_book_has_not_listed(
    market: list[date],
    clock: dict[str, datetime],
    monkeypatch: pytest.MonkeyPatch,
    settings: Any,
    tmp_path: Path,
) -> None:
    settings.MODEL_DIR = tmp_path
    monkeypatch.setitem(
        candidate_scores.SCORERS,
        "C1",
        lambda model, row, day, base: base.assign(raw=-base["base"], score=-base["base"]),
    )
    row = frozen("C1", market[2], tmp_path, stored=False)
    mornings(market[:4], clock, CANDIDATE_BOOKS)
    (tmp_path / row.path).parent.mkdir(parents=True)
    (tmp_path / row.path).write_bytes(pickle.dumps({"candidate": "C1"}))
    clock["at"] = at(market[4] + timedelta(days=1), time(6, 30))

    outcome = record_books(market[4], CANDIDATE_BOOKS)

    assert set(ShadowScore.objects.values_list("date", flat=True)) == {market[4]}
    lists = HoldingBookDay.objects.filter(book="T-C1", phase=BookPhase.LIST).order_by("signal_date")
    assert [(day.signal_date, day.status) for day in lists] == [
        (market[2], BookStatus.NO_PREDICTION),
        (market[3], BookStatus.NO_PREDICTION),
        (market[4], BookStatus.LISTED),
    ]
    assert outcome.detail["scores"] == {
        "C1": {"scored": ["2026-11-20"], "artifacts": ["rank.h5 g1"]}
    }
    assert outcome.severity == "info"


def test_scores_the_budget_cut_short_skip_the_candidate_book_until_they_are_complete(
    market: list[date], clock: dict[str, datetime], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frozen("C1", market[2], tmp_path)
    stopped = {
        "book": "C1",
        "role": "candidate",
        "signal_date": "2026-11-19",
        "scores_failed": "stopped: the shadow budget ran out after 1 days",
    }
    monkeypatch.setattr(
        shadow,
        "score_candidate",
        lambda *args: Scored(["2026-11-18"], notes=[stopped], complete=False),
    )

    outcome = mornings(market[:3], clock, CANDIDATE_BOOKS)[-1]

    assert "T-C1" not in outcome.detail["books"]
    assert outcome.detail["notes"] == [
        stopped,
        {
            "book": "T-C1",
            "role": "candidate",
            "signal_date": None,
            "book_failed": "skipped: C1 has no scores this morning",
        },
    ]
