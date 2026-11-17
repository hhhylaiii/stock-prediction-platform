"""Shadow holding books written each morning: a list before the open, a fill after the close.

Rows are only added. A wrong book is never edited or deleted: a new book name is filled again
from its start, marked backfilled, and the old rows stay (holding-shadow-prereg.md §4).
"""

import copy
import math
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time
from functools import cached_property
from time import monotonic
from typing import Any

import numpy as np
import pandas as pd
from django.db import connection, transaction
from django.utils import timezone

from apps.etl.models import DataQualityLog
from apps.market.calendar import TradingDays
from apps.research.candidate_scores import Scored, score_candidate
from apps.research.candidate_scores import first_signal as candidate_start
from apps.research.candidates import CANDIDATES
from apps.research.holding import (
    BACKUP,
    BUY,
    FULL_COSTS,
    KEEP,
    SELL,
    SETTLE,
    STUCK,
    Market,
    Mask,
    Matrix,
    Rule,
    Settle,
    Signal,
    Stepped,
    order_list,
    settle_days,
    step,
    trade_day,
)
from apps.research.holding_inputs import (
    HORIZON,
    SHADOW_START,
    index_levels,
    labels_settled,
    load_market,
    score_grids,
    stored_scores,
)
from apps.research.holding_report import mean_returns
from apps.research.models import (
    BookPhase,
    BookStatus,
    HoldingBookAction,
    HoldingBookDay,
    ModelObjective,
    Prediction,
    ShadowScore,
)

SHADOW_CHECK = "research_shadow"
BUDGET_SECONDS = 60.0
OPEN = time(9, 0)
TOLERANCE = 1e-12
SHOWN = 10
LIMIT_SQL = "SELECT set_config('statement_timeout', %s, true)"
HELD = (KEEP, BUY, STUCK)
Severity = DataQualityLog.Severity


@dataclass(frozen=True)
class Spec:
    """A book's rule, first signal day, and the candidate whose scores it follows (None for C0)."""

    rule: Rule
    start: date
    source: str | None = None


RULE = Rule.parse("H70", 30)
PREFIX = "H70N30-"
BOOKS: dict[str, Spec] = {f"{PREFIX}C0": Spec(RULE, SHADOW_START)}
SCORE_KEYS = ["date", "stock_id", "in_universe", "score"]


@dataclass(frozen=True)
class Ledger:
    """A book after its last fill: the cash and each held stock's value at that close."""

    cash: float
    values: dict[str, float]


@dataclass(frozen=True)
class Progress:
    listed: list[str]
    filled: list[str]
    waiting: str | None
    notes: list[dict[str, Any]]

    def said(self) -> str:
        parts = [("listed", self.listed), ("filled", self.filled)]
        shown = [f"{word} {', '.join(days)}" for word, days in parts if days]
        return ", ".join(shown) or "nothing due"


@dataclass(frozen=True)
class Outcome:
    severity: str
    detail: dict[str, Any]
    summary: str
    wrote: bool = False


class ShadowTimeoutError(RuntimeError):
    """The morning's shadow budget ran out before a statement could start."""


class ShadowStuckError(RuntimeError):
    """A book could not write a signal day; that book keeps nothing of the morning."""

    def __init__(self, book: str, signal: date | None) -> None:
        super().__init__(f"{book} stuck on {signal or 'its start'}")
        self.book = book
        self.signal = signal


class Inputs:
    """What a morning reads once, the first time a book needs it: grids from start to session."""

    def __init__(
        self, start: date, session: date, calendar: TradingDays, stored: Callable[[], set[str]]
    ) -> None:
        self.start, self.session, self.calendar, self.stored = start, session, calendar, stored
        self.source: str | None = None

    def scored_by(self, source: str) -> "Inputs":
        """The same grids with a candidate's stored scores and universe in place of C0's."""
        market = self.market
        rows = ShadowScore.objects.filter(
            candidate=source, date__range=(self.start, self.session)
        ).values_list(*SCORE_KEYS)
        frame = pd.DataFrame(list(rows), columns=SCORE_KEYS)
        score, in_universe = score_grids(
            frame.astype({"score": "float64", "in_universe": "bool"}), market.days, market.stocks
        )
        other = copy.copy(self)
        other.source = source
        other.__dict__["market"] = replace(market, score=score, in_universe=in_universe)
        return other

    @cached_property
    def market(self) -> Market:
        """Scored stocks and every stock the books hold or listed, so a fill never needs a score."""
        scores = stored_scores(self.start, self.session)
        extra = self.stored()
        return load_market(scores, self.session, self.calendar, start=self.start, extra=extra)

    @cached_property
    def marks(self) -> Matrix:
        return self.market.marks

    @cached_property
    def traded(self) -> Mask:
        return self.market.traded

    @cached_property
    def settles(self) -> Mask:
        return settle_days(self.market, Settle.LIVE, self.traded)

    @cached_property
    def at(self) -> dict[date, int]:
        return {day: t for t, day in enumerate(self.market.days)}

    @cached_property
    def column(self) -> dict[str, int]:
        return {stock: k for k, stock in enumerate(self.market.stocks)}

    @cached_property
    def levels(self) -> dict[date, float]:
        levels = index_levels(self.start, self.session)
        return dict(zip(levels.index, levels.to_numpy(dtype=float).tolist(), strict=True))

    def mask(self, stocks: list[str]) -> Mask:
        """The stocks as a mask over the grid's columns; a stock outside it is an error."""
        outside = sorted(set(stocks) - set(self.column))
        if outside:
            raise ValueError(f"stocks without a prediction row since {self.start}: {outside}")
        chosen = np.zeros(len(self.market.stocks), dtype=bool)
        chosen[[self.column[stock] for stock in stocks]] = True
        return chosen


@contextmanager
def budget(seconds: float) -> Iterator[Callable[[], float]]:
    """Inside a transaction, each statement may run only for what is left of the seconds."""
    deadline = monotonic() + seconds
    with connection.cursor() as cursor:
        cursor.execute("SELECT current_setting('statement_timeout')")
        (before,) = cursor.fetchone()

    def bounded(
        execute: Callable[..., Any], sql: str, params: Any, many: bool, context: dict[str, Any]
    ) -> Any:
        left = deadline - monotonic()
        if left <= 0:
            raise ShadowTimeoutError(f"the {seconds:g} s shadow budget ran out")
        context["cursor"].cursor.execute(LIMIT_SQL, [f"{math.ceil(left * 1000)}ms"])
        return execute(sql, params, many, context)

    with connection.execute_wrapper(bounded):
        yield lambda: deadline - monotonic()
    with connection.cursor() as cursor:
        cursor.execute(LIMIT_SQL, [before])


@dataclass
class Morning:
    books: dict[str, Progress] = field(default_factory=dict)
    scores: dict[str, Scored] = field(default_factory=dict)
    notes: list[dict[str, Any]] = field(default_factory=list)


def all_books() -> dict[str, Spec]:
    """The fixed books, then each frozen candidate's book and a C0 book from the same day."""
    found = dict(BOOKS)
    for name in CANDIDATES:
        start = candidate_start(name)
        if start is not None:
            found.setdefault(f"{PREFIX}C0-{start:%Y%m%d}", Spec(RULE, start))
            found[f"{PREFIX}{name}"] = Spec(RULE, start, name)
    return found


def record_books(session: date, books: Mapping[str, Spec] | None = None) -> Outcome:
    """Each book and each candidate's scores in its own transaction under one budget; never raises.

    Books on C0's scores go first and commit first, so a candidate never holds them back.
    """
    began = monotonic()
    try:
        chosen = all_books() if books is None else books
        with budget(BUDGET_SECONDS) as left:
            done = _morning(session, chosen, left)
    except Exception as error:
        return _failed(began, error)
    seconds = round(monotonic() - began, 2)
    written = {
        name: {"listed": got.listed, "filled": got.filled, "waiting": got.waiting}
        for name, got in done.books.items()
    }
    scores = {
        name: {"scored": got.scored, "artifacts": got.artifacts}
        for name, got in done.scores.items()
    }
    detail = {"books": written, **({"scores": scores} if scores else {}), "notes": done.notes}
    detail["seconds"] = seconds
    shown = [f"{name}: {progress.said()}" for name, progress in done.books.items()]
    shown += [_said_failed(note) for note in done.notes if "book_failed" in note]
    shown += [
        f"{name}: scored {', '.join(got.scored)}" for name, got in done.scores.items() if got.scored
    ]
    warned = f", {len(done.notes)} notes, see {SHADOW_CHECK}" if done.notes else ""
    severity = Severity.WARNING if done.notes else Severity.INFO
    wrote = any(got.listed or got.filled for got in done.books.values())
    wrote = wrote or any(got.scored for got in done.scores.values())
    summary = f"{'; '.join(shown) or 'no book due'}{warned}; {seconds} s"
    return Outcome(severity, detail, summary, wrote)


def busy(error: BaseException) -> Outcome:
    """A morning that never got the research lock: nothing written, one warning."""
    return _failed(monotonic(), error)


def _failed(
    began: float, error: BaseException, book: str | None = None, signal: date | None = None
) -> Outcome:
    seconds = round(monotonic() - began, 2)
    detail = {
        "book": book,
        "signal_date": None if signal is None else signal.isoformat(),
        "error": f"{type(error).__name__}: {error}",
        "written": "nothing; the next morning starts again from the same day",
        "seconds": seconds,
    }
    where = f" at {book} {signal}" if book else ""
    return Outcome(Severity.WARNING, detail, f"failed{where}, see {SHADOW_CHECK}; {seconds} s")


def _said_failed(note: dict[str, Any]) -> str:
    at = f" at {note['signal_date']}" if note["signal_date"] else ""
    return f"{note['book']}: failed{at}"


def _book_failed(book: str, signal: date | None, error: BaseException | str) -> dict[str, Any]:
    said = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
    shown = None if signal is None else signal.isoformat()
    return {"book": book, "signal_date": shown, "book_failed": said}


def _book(
    done: Morning, name: str, spec: Spec, session: date, inputs: Inputs, left: Callable[[], float]
) -> None:
    """One book in its own transaction; a failure rolls back that book alone and notes it."""
    if left() <= 0:
        done.notes.append(_book_failed(name, None, "skipped: the shadow budget ran out"))
        return
    try:
        with transaction.atomic():
            view = inputs if spec.source is None else inputs.scored_by(spec.source)
            progress = _advance(name, spec, session, view)
    except ShadowStuckError as stuck:
        done.notes.append(_book_failed(name, stuck.signal, stuck.__cause__ or stuck))
        return
    except Exception as error:
        done.notes.append(_book_failed(name, None, error))
        return
    done.books[name] = progress
    done.notes.extend(progress.notes)


def _scores(
    done: Morning, name: str, session: date, calendar: TradingDays, left: Callable[[], float]
) -> bool:
    """One candidate's due scores in their own transaction; False when none could be written."""
    if left() <= 0:
        done.notes.append({"book": name, "scores_failed": "skipped: the shadow budget ran out"})
        return False
    try:
        with transaction.atomic():
            scored = score_candidate(name, session, calendar)
    except Exception as error:
        done.notes.append({"book": name, "scores_failed": f"{type(error).__name__}: {error}"})
        return False
    done.scores[name] = scored
    done.notes.extend(scored.notes)
    return True


def _morning(session: date, books: Mapping[str, Spec], left: Callable[[], float]) -> Morning:
    """C0's books, then each candidate's scores, then the candidates' books."""
    done = Morning()
    due = {name: spec for name, spec in books.items() if spec.start <= session}
    if not due:
        return done
    with transaction.atomic():
        calendar = TradingDays.from_db()
    start = min(spec.start for spec in due.values())
    inputs = Inputs(start, session, calendar, lambda: _stored_stocks(list(due)))
    for name, spec in due.items():
        if spec.source is None:
            _book(done, name, spec, session, inputs, left)
    sources = list(dict.fromkeys(spec.source for spec in due.values() if spec.source))
    ready = {source for source in sources if _scores(done, source, session, calendar, left)}
    for name, spec in due.items():
        if spec.source is None:
            continue
        if spec.source in ready:
            _book(done, name, spec, session, inputs, left)
        else:
            reason = f"skipped: {spec.source} has no scores this morning"
            done.notes.append(_book_failed(name, None, reason))
    return done


def _stored_stocks(books: list[str]) -> set[str]:
    """Every stock the books ever listed or held, and the universes of lists not filled yet."""
    lines = HoldingBookAction.objects.filter(book__in=books).order_by()
    stocks = set(lines.values_list("stock_id", flat=True).distinct())
    open_lists = HoldingBookDay.objects.filter(book__in=books, phase=BookPhase.LIST).exclude(
        signal_date__in=HoldingBookDay.objects.filter(book__in=books, phase=BookPhase.FILL).values(
            "signal_date"
        )
    )
    for universe in open_lists.values_list("universe", flat=True):
        stocks |= set(universe)
    return stocks


def _advance(book: str, spec: Spec, session: date, inputs: Inputs) -> Progress:
    """Fill then list each signal day in order until a fill must wait for its prices or labels."""
    stored = HoldingBookDay.objects.filter(book=book)
    lists = {row.signal_date: row for row in stored.filter(phase=BookPhase.LIST)}
    filled = set(stored.filter(phase=BookPhase.FILL).values_list("signal_date", flat=True))
    ledger = _ledger(book)
    listed: list[str] = []
    done: list[str] = []
    notes: list[dict[str, Any]] = []
    for signal in inputs.calendar.between(spec.start, session):
        if signal in filled:
            continue
        trade = inputs.calendar.shift(signal, 1)
        try:
            _check_order(signal, lists)
            if signal not in lists:
                lists[signal], note = _write_list(book, spec.rule, signal, trade, ledger, inputs)
                listed.append(signal.isoformat())
                notes += note
            if trade > session or not labels_settled(signal):
                return Progress(listed, done, signal.isoformat(), notes)
            ledger, note = _write_fill(book, spec.rule, lists[signal], ledger, inputs)
        except Exception as error:
            raise ShadowStuckError(book, signal) from error
        done.append(signal.isoformat())
        notes += note
    return Progress(listed, done, None, notes)


def _check_order(signal: date, lists: Mapping[date, HoldingBookDay]) -> None:
    """The first day without a fill is the last listed; else the calendar opened a day under it."""
    later = [day for day in lists if day > signal]
    if later:
        raise ValueError(
            f"{signal} has no fill but the book already listed {max(later)}: the calendar "
            "opened a day under the book; start a new book name"
        )


def _ledger(book: str) -> Ledger:
    """The cash and held values of the book's last fill; a new book holds 1.0 in cash."""
    last = (
        HoldingBookDay.objects.filter(book=book, phase=BookPhase.FILL)
        .order_by("-signal_date")
        .first()
    )
    if last is None:
        return Ledger(1.0, {})
    if last.cash is None:
        raise ValueError(f"the fill of {last.signal_date} has no cash")
    rows = HoldingBookAction.objects.filter(
        book=book, signal_date=last.signal_date, phase=BookPhase.FILL, action__in=HELD
    ).values_list("stock_id", "value")
    return Ledger(last.cash, {stock: float(value or 0.0) for stock, value in rows})


def _opening(day: date) -> datetime:
    return timezone.make_aware(datetime.combine(day, OPEN))


def _run_id(signal: date, source: str | None) -> str:
    """The serving run's run_id, or the candidate artifacts' labels, behind the day's scores."""
    if source is not None:
        found = ShadowScore.objects.filter(candidate=source, date=signal).order_by()
        pairs = set(found.values_list("serving_run_id", "generation"))
        return ",".join(sorted(f"{source}:{run}:g{generation}" for run, generation in pairs))
    rows = Prediction.objects.filter(
        run__objective=ModelObjective.RANK, horizon=HORIZON, date=signal
    ).order_by()
    return ",".join(sorted(set(rows.values_list("run_id", flat=True))))


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _write_list(
    book: str, rule: Rule, signal: date, trade: date, ledger: Ledger, inputs: Inputs
) -> tuple[HoldingBookDay, list[dict[str, Any]]]:
    """The signal day's list from the stored predictions, written before any fill reads it."""
    market, at = inputs.market, inputs.at[signal]
    held = inputs.mask(list(ledger.values))
    score, inside = market.score[at], market.in_universe[at]
    order = order_list(held, score, inside, market.stocks, rule)
    scored = inside & np.isfinite(score)
    universe = [market.stocks[k] for k in np.flatnonzero(inside)]
    run_id, now = _run_id(signal, inputs.source), timezone.now()
    actions = order["action"]
    header = HoldingBookDay(
        book=book,
        signal_date=signal,
        phase=BookPhase.LIST,
        trade_date=trade,
        status=BookStatus.LISTED if scored.any() else BookStatus.NO_PREDICTION,
        run_id=run_id,
        universe=universe,
        held_before=int(held.sum()),
        sells=int((actions == SELL).sum()),
        buys=int((actions == BUY).sum()),
        backups=int((actions == BACKUP).sum()),
        backfilled=now >= _opening(trade),
        recorded_at=now,
    )
    rows = [
        HoldingBookAction(
            book=book,
            signal_date=signal,
            trade_date=trade,
            stock_id=line["stock_id"],
            phase=BookPhase.LIST,
            action=line["action"],
            priority=int(line["priority"]),
            score=_finite(line["score"]),
            pct=_finite(line["pct"]),
            run_id=run_id,
            recorded_at=now,
        )
        for line in order.to_dict("records")
    ]
    HoldingBookDay.objects.bulk_create([header])
    HoldingBookAction.objects.bulk_create(rows)
    unscored = [market.stocks[k] for k in np.flatnonzero(inside & ~scored)]
    if not unscored:
        return header, []
    note = {
        "book": book,
        "signal_date": signal.isoformat(),
        "in_universe_without_score": len(unscored),
        "first": unscored[:SHOWN],
    }
    return header, [note]


def _signal(lines: list[HoldingBookAction], inputs: Inputs) -> Signal:
    """The list's scores and percentiles on the grid; a day without predictions lists none."""
    count = len(inputs.market.stocks)
    score, pct = np.full(count, np.nan), np.full(count, np.nan)
    for line in lines:
        k = inputs.column[line.stock_id]
        score[k] = np.nan if line.score is None else line.score
        pct[k] = np.nan if line.pct is None else line.pct
    return score, pct, np.isfinite(pct)


def _write_fill(
    book: str, rule: Rule, header: HoldingBookDay, ledger: Ledger, inputs: Inputs
) -> tuple[Ledger, list[dict[str, Any]]]:
    """The next trading day after the stored list: its sells, settlements, buys and values.

    The trade day comes from the calendar now; one closed since the list was written moves it.
    """
    market, at = inputs.market, inputs.at[header.signal_date]
    t = at + 1
    trade = market.days[t]
    lines = list(
        HoldingBookAction.objects.filter(
            book=book, signal_date=header.signal_date, phase=BookPhase.LIST
        )
    )
    held = inputs.mask(list(ledger.values))
    values = np.zeros(len(market.stocks))
    for stock, value in ledger.values.items():
        values[inputs.column[stock]] = value
    day = trade_day(market, _signal(lines, inputs), inputs.marks, inputs.traded, inputs.settles, t)
    result = step(ledger.cash, values, held, day, rule, FULL_COSTS)
    universe = inputs.mask(list(header.universe))
    now = timezone.now()
    filled = HoldingBookDay(
        book=book,
        signal_date=header.signal_date,
        phase=BookPhase.FILL,
        trade_date=trade,
        status=BookStatus.TRADED if day.scored.any() else BookStatus.IDLE,
        run_id=header.run_id,
        value=result.cash + float(result.values.sum()),
        cash=result.cash,
        paid=result.paid,
        traded=result.traded,
        held=int(result.held.sum()),
        universe_return=float(mean_returns(inputs.marks[at : t + 1], np.vstack([universe] * 2))[0]),
        index_return=_index_return(inputs.levels, market.days[at], trade),
        backfilled=now >= _opening(inputs.calendar.shift(trade, 1)),
        recorded_at=now,
    )
    HoldingBookDay.objects.bulk_create([filled])
    HoldingBookAction.objects.bulk_create(_fill_rows(filled, held, result, market, t))
    kept = {market.stocks[k]: float(result.values[k]) for k in np.flatnonzero(result.held)}
    notes = [_rewritten(book, header, lines, inputs), _moved(header, trade)]
    notes.append(_unindexed(header, inputs.levels, market.days[at], trade))
    return Ledger(result.cash, kept), [note for note in notes if note is not None]


def _index_return(levels: Mapping[date, float], before: date, after: date) -> float | None:
    """TAIEX_TR close to close; NULL when either close is missing, never a carried zero."""
    if before not in levels or after not in levels:
        return None
    return levels[after] / levels[before] - 1


def _unindexed(
    header: HoldingBookDay, levels: Mapping[date, float], before: date, after: date
) -> dict[str, Any] | None:
    missing = [day.isoformat() for day in (before, after) if day not in levels]
    if not missing:
        return None
    signal = header.signal_date.isoformat()
    return {"book": header.book, "signal_date": signal, "no_taiex_tr": missing}


def _moved(header: HoldingBookDay, trade: date) -> dict[str, Any] | None:
    if trade == header.trade_date:
        return None
    return {
        "book": header.book,
        "signal_date": header.signal_date.isoformat(),
        "trade_date_moved": {"listed": header.trade_date.isoformat(), "filled": trade.isoformat()},
    }


def _fill_rows(
    filled: HoldingBookDay, held: Mask, result: Stepped, market: Market, t: int
) -> list[HoldingBookAction]:
    moves = result.moves
    kept = held & result.held & ~moves.stuck
    found: list[tuple[str, Mask, Matrix | None, float]] = [
        (SELL, moves.sold, moves.proceeds, FULL_COSTS.sell),
        (SETTLE, moves.settled, moves.proceeds, FULL_COSTS.sell),
        (STUCK, moves.stuck, None, 0.0),
        (KEEP, kept, None, 0.0),
        (BUY, moves.bought, moves.entered, FULL_COSTS.buy),
    ]
    return [
        HoldingBookAction(
            book=filled.book,
            signal_date=filled.signal_date,
            trade_date=filled.trade_date,
            stock_id=market.stocks[k],
            phase=BookPhase.FILL,
            action=action,
            run_id=filled.run_id,
            amount=None if amounts is None else float(amounts[k]),
            value=float(result.values[k]),
            cost=None if amounts is None else float(amounts[k]) * rate,
            limit_down_open=bool(market.limit_down_open[t, k]) if action == SELL else None,
            recorded_at=filled.recorded_at,
        )
        for action, chosen, amounts, rate in found
        for k in np.flatnonzero(chosen)
    ]


def _same(stored: float | None, now: float) -> bool:
    if stored is None or not math.isfinite(now):
        return stored is None and not math.isfinite(now)
    return math.isclose(stored, now, rel_tol=TOLERANCE)


def _rewritten(
    book: str, header: HoldingBookDay, lines: list[HoldingBookAction], inputs: Inputs
) -> dict[str, Any] | None:
    """How the stored predictions now differ from the list; the fill still follows the list."""
    market, at = inputs.market, inputs.at[header.signal_date]
    now = {market.stocks[k] for k in np.flatnonzero(market.in_universe[at])}
    then = set(header.universe)
    changed = [
        line.stock_id
        for line in lines
        if not _same(line.score, float(market.score[at, inputs.column[line.stock_id]]))
    ]
    if now == then and not changed:
        return None
    added, removed = sorted(now - then), sorted(then - now)
    return {
        "book": book,
        "signal_date": header.signal_date.isoformat(),
        "predictions_rewritten": {
            "universe_added": len(added),
            "universe_removed": len(removed),
            "scores_changed": len(changed),
            "first": (added + removed + changed)[:SHOWN],
        },
    }
