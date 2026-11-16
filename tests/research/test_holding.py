import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from apps.research.holding import (
    BACKUP,
    BUY,
    FULL_COSTS,
    KEEP,
    NO_COSTS,
    SELL,
    SETTLE,
    STUCK,
    Book,
    Market,
    Matrix,
    Rule,
    Settle,
    order_list,
    percentiles,
    simulate,
)
from apps.research.metrics.deciles import BUY_COST, SELL_COST
from tests.research.conftest import weekday_calendar

NAN = math.nan
CALENDAR = weekday_calendar(date(2016, 1, 4), date(2016, 6, 30))
DAYS = list(CALENDAR.between(date(2016, 1, 4), date(2016, 6, 30)))


def market(
    opens: Matrix,
    closes: Matrix,
    score: Matrix,
    in_universe: np.ndarray | None = None,
    blocked: np.ndarray | None = None,
    delisted: list[int] | None = None,
    limit_down: np.ndarray | None = None,
) -> Market:
    days, names = closes.shape
    shape = (days, names)
    return Market(
        days=DAYS[:days],
        stocks=[str(1000 + k) for k in range(names)],
        opens=np.asarray(opens, dtype=np.float64),
        closes=np.asarray(closes, dtype=np.float64),
        score=np.asarray(score, dtype=np.float64),
        in_universe=np.ones(shape, dtype=bool) if in_universe is None else in_universe,
        blocked=np.zeros(shape, dtype=bool) if blocked is None else blocked,
        delisted=np.array(delisted or [days] * names, dtype=np.int64),
        limit_down_open=np.zeros(shape, dtype=bool) if limit_down is None else limit_down,
    )


def flat(days: int, names: int, price: float = 10.0) -> Matrix:
    return np.full((days, names), price)


def ranked(days: int, scores: list[float]) -> Matrix:
    return np.tile(np.array(scores, dtype=np.float64), (days, 1))


def returns(book: Book) -> Matrix:
    value = book.daily["value"].to_numpy(dtype=np.float64)
    out: Matrix = value[1:] / value[:-1] - 1
    return out


def done(book: Book, day: int, action: str) -> list[str]:
    rows = book.actions[(book.actions["date"] == DAYS[day]) & (book.actions["action"] == action)]
    return sorted(rows["stock_id"])


def test_a_costless_buy_and_hold_book_equals_the_mean_close_over_entry_open() -> None:
    rng = np.random.default_rng(3)
    closes = 10 * np.cumprod(1 + rng.normal(0, 0.02, (30, 6)), axis=0)
    opens = closes * (1 + rng.normal(0, 0.01, (30, 6)))
    every = Rule(exit_below=0.0, slots=3, entry_at=0.0)

    book = simulate(market(opens, closes, ranked(30, [6, 5, 4, 3, 2, 1])), every, NO_COSTS)

    direct = float((closes[29, :3] / opens[1, :3]).mean())
    assert book.daily["value"].iloc[-1] == pytest.approx(direct, rel=1e-12)
    assert book.daily["date"].iloc[0] == DAYS[0]
    assert book.daily["value"].iloc[0] == 1.0


def test_a_book_that_holds_through_the_nights_earns_every_overnight_gain() -> None:
    closes = 10 * 1.01 ** np.arange(12)[:, None] * np.ones((12, 4))
    every = Rule(exit_below=0.0, slots=4, entry_at=0.0)

    book = simulate(market(closes, closes, ranked(12, [4, 3, 2, 1])), every, NO_COSTS)

    daily = returns(book)
    assert daily[0] == pytest.approx(0.0, abs=1e-12)
    assert daily[1:] == pytest.approx(np.full(len(daily) - 1, 0.01), rel=1e-12)


def test_costs_are_charged_on_what_trades_and_holdings_cost_nothing() -> None:
    prices = flat(10, 2)
    score = np.array([[2.0, 1.0]] * 5 + [[1.0, 2.0]] * 5)

    book = simulate(market(prices, prices, score), Rule.parse("H70", 1), FULL_COSTS)

    first = 1 / (1 + BUY_COST)
    second = first * (1 - SELL_COST) / (1 + BUY_COST)
    paid = book.daily.set_index("date")["paid"]
    assert paid[DAYS[1]] == pytest.approx(first * BUY_COST)
    assert paid[DAYS[6]] == pytest.approx(first * SELL_COST + second * BUY_COST)
    assert (paid.drop([DAYS[1], DAYS[6]]) == 0).all()
    assert book.daily.set_index("date")["cash"][DAYS[6]] == pytest.approx(0.0, abs=1e-12)
    assert done(book, 6, SELL) == ["1000"]
    assert done(book, 6, BUY) == ["1001"]


def test_a_name_at_the_exit_percentile_stays_and_entries_need_the_entry_percentile() -> None:
    prices = flat(5, 10)
    score = ranked(5, [1, 2, 3, 4, 5, 8, 10, 9, 7, 6])
    score[0] = np.arange(1, 11)

    book = simulate(market(prices, prices, score), Rule.parse("H70", 3), NO_COSTS)

    assert done(book, 1, BUY) == ["1008", "1009"]
    assert book.daily.set_index("date")["cash"][DAYS[1]] == pytest.approx(1 / 3)
    assert done(book, 2, SELL) == ["1009"]
    assert done(book, 2, BUY) == ["1006", "1007"]
    assert book.daily.set_index("date")["held"][DAYS[2]] == 3


def test_names_untradable_the_next_open_are_skipped_for_the_next_best() -> None:
    prices = flat(4, 10)
    blocked = np.zeros((4, 10), dtype=bool)
    blocked[0, 9] = True
    rule = Rule(exit_below=0.7, slots=1, entry_at=0.8)

    book = simulate(market(prices, prices, ranked(4, list(range(1, 11))), blocked=blocked), rule)

    assert done(book, 1, BUY) == ["1008"]


def test_names_without_a_trade_the_next_day_cannot_be_bought() -> None:
    prices = flat(4, 10)
    prices[1, 9] = NAN
    rule = Rule(exit_below=0.7, slots=1, entry_at=0.8)

    book = simulate(market(prices, prices, ranked(4, list(range(1, 11)))), rule)

    assert done(book, 1, BUY) == ["1008"]


def halted_exit() -> tuple[Matrix, Matrix, np.ndarray]:
    """1000 is bought, leaves the universe on day 3 and has no trade on days 4 and 5."""
    opens, closes = flat(9, 2), flat(9, 2)
    opens[4:6, 0] = closes[4:6, 0] = NAN
    opens[6:, 0] = 12.0
    closes[6:, 0] = 12.0
    universe = np.ones((9, 2), dtype=bool)
    universe[3:, 0] = False
    return opens, closes, universe


def test_an_exit_without_a_trade_waits_at_its_last_close_and_sells_when_it_trades() -> None:
    opens, closes, universe = halted_exit()
    limit_down = np.zeros((9, 2), dtype=bool)
    limit_down[6, 0] = True
    held = market(opens, closes, ranked(9, [2, 1]), in_universe=universe, limit_down=limit_down)

    book = simulate(held, Rule.parse("H70", 1), NO_COSTS)

    assert done(book, 4, STUCK) == ["1000"]
    assert done(book, 5, STUCK) == ["1000"]
    assert done(book, 4, BUY) == []
    assert done(book, 6, SELL) == ["1000"]
    assert done(book, 6, BUY) == ["1001"]
    values = book.daily.set_index("date")["value"]
    assert values[DAYS[5]] == pytest.approx(1.0)
    assert values[DAYS[6]] == pytest.approx(1.2)
    sale = book.actions[book.actions["action"] == SELL].iloc[0]
    assert sale["amount"] == pytest.approx(1.2)
    assert bool(sale["limit_down_open"]) is True
    assert sale["signal_date"] == DAYS[5]


def gone_forever() -> Market:
    opens, closes, universe = halted_exit()
    opens[4:, 0] = closes[4:, 0] = NAN
    return market(opens, closes, ranked(9, [2, 1]), in_universe=universe, delisted=[7, 9])


def test_known_settlement_closes_a_name_that_never_trades_again_on_its_exit_day() -> None:
    book = simulate(gone_forever(), Rule.parse("H70", 1), FULL_COSTS, Settle.KNOWN)

    assert done(book, 4, SETTLE) == ["1000"]
    assert done(book, 4, BUY) == ["1001"]
    assert book.actions["action"].eq(STUCK).sum() == 0


def test_live_settlement_waits_for_the_day_after_delisting() -> None:
    book = simulate(gone_forever(), Rule.parse("H70", 1), FULL_COSTS, Settle.LIVE)

    assert done(book, 4, STUCK) == ["1000"]
    assert done(book, 6, STUCK) == ["1000"]
    assert done(book, 7, SETTLE) == ["1000"]
    assert done(book, 7, BUY) == ["1001"]
    settled = book.actions[book.actions["action"] == SETTLE].iloc[0]
    paid = book.daily.set_index("date")["paid"][DAYS[7]]
    bought = book.actions[book.actions["action"] == BUY]["amount"].iloc[-1]
    assert paid == pytest.approx(settled["amount"] * SELL_COST + bought * BUY_COST)


def test_live_settlement_never_closes_a_name_that_was_not_delisted() -> None:
    opens, closes, universe = halted_exit()
    opens[4:, 0] = closes[4:, 0] = NAN
    held = market(opens, closes, ranked(9, [2, 1]), in_universe=universe)

    book = simulate(held, Rule.parse("H70", 1), NO_COSTS, Settle.LIVE)

    assert book.actions["action"].eq(SETTLE).sum() == 0
    assert (book.daily["held"].iloc[1:] == 1).all()
    assert book.daily["value"].iloc[-1] == pytest.approx(1.0)


def test_percentiles_rank_only_universe_names_with_a_score_and_average_ties() -> None:
    score = np.array([[5.0, 9.0, NAN, 1.0, 9.0], [3.0, 3.0, 1.0, NAN, NAN]])
    universe = np.array([[True, False, True, True, True], [True, True, True, False, False]])

    pct = percentiles(score, universe)

    assert pct[0] == pytest.approx(np.array([2 / 3, NAN, NAN, 1 / 3, 1.0]), nan_ok=True)
    assert pct[1] == pytest.approx(np.array([2.5 / 3, 2.5 / 3, 1 / 3, NAN, NAN]), nan_ok=True)


def test_an_out_of_universe_score_never_pushes_holdings_below_the_exit() -> None:
    prices = flat(4, 4)
    score = ranked(4, [1, 2, 3, 4])
    score[1:, 0] = 100.0
    universe = np.ones((4, 4), dtype=bool)
    universe[:, 0] = False

    book = simulate(market(prices, prices, score, in_universe=universe), Rule.parse("H70", 1))

    assert done(book, 1, BUY) == ["1003"]
    assert book.actions["action"].eq(SELL).sum() == 0


def test_a_day_without_any_score_trades_nothing() -> None:
    prices = flat(6, 2)
    score = ranked(6, [2, 1])
    score[2] = NAN

    book = simulate(market(prices, prices, score), Rule.parse("H70", 1), NO_COSTS)

    assert book.actions["action"].eq(SELL).sum() == 0
    assert (book.daily["held"].iloc[1:] == 1).all()


def test_a_holding_without_a_score_on_a_scored_day_leaves() -> None:
    prices = flat(6, 2)
    score = ranked(6, [2, 1])
    score[2, 0] = NAN

    book = simulate(market(prices, prices, score), Rule.parse("H70", 1), NO_COSTS)

    assert done(book, 3, SELL) == ["1000"]
    assert done(book, 3, BUY) == ["1001"]


def test_equal_scores_enter_in_stock_order() -> None:
    prices = flat(3, 3)
    every = Rule(exit_below=0.0, slots=1, entry_at=0.0)

    book = simulate(market(prices, prices, ranked(3, [1, 5, 5])), every)

    assert done(book, 1, BUY) == ["1001"]


def test_the_book_waits_for_its_first_scored_signal_day() -> None:
    prices = flat(5, 2)
    score = ranked(5, [2, 1])
    score[:2] = NAN

    book = simulate(market(prices, prices, score), Rule.parse("H70", 1), NO_COSTS)

    assert book.daily["date"].iloc[0] == DAYS[2]
    assert done(book, 3, BUY) == ["1000"]


def test_the_list_names_sells_keeps_then_buys_in_priority_with_backups() -> None:
    held = np.zeros(10, dtype=bool)
    held[[0, 9]] = True
    stocks = [str(1000 + k) for k in range(10)]
    rule = Rule(exit_below=0.7, slots=2, entry_at=0.8)

    everyone, nobody = np.ones(10, dtype=bool), np.zeros(10, dtype=bool)

    listed = order_list(held, np.arange(1.0, 11.0), everyone, nobody, stocks, rule)

    assert listed[["stock_id", "action", "priority"]].values.tolist() == [
        ["1000", SELL, 0],
        ["1009", KEEP, 0],
        ["1008", BUY, 1],
        ["1007", BACKUP, 2],
    ]
    assert listed["pct"].tolist() == pytest.approx([0.1, 1.0, 0.9, 0.8])


def test_the_list_is_empty_on_a_day_without_scores() -> None:
    held = np.array([True, False])
    stocks = ["1000", "1001"]
    nothing = np.array([NAN, NAN])

    everyone, nobody = np.ones(2, dtype=bool), np.zeros(2, dtype=bool)

    listed = order_list(held, nothing, everyone, nobody, stocks, Rule.parse("H70", 1))

    assert listed.empty
    assert isinstance(listed, pd.DataFrame)


@pytest.mark.parametrize(
    ("name", "slots", "exit_below", "label"),
    [("H70", 30, 0.70, "H70/N30"), ("H90", 100, 0.90, "H90/N100"), ("H5", 1, 0.05, "H5/N1")],
)
def test_rules_parse_their_exit_percentile(
    name: str, slots: int, exit_below: float, label: str
) -> None:
    rule = Rule.parse(name, slots)

    assert rule.exit_below == exit_below
    assert rule.entry_at == 0.90
    assert rule.name == label


@pytest.mark.parametrize(("name", "slots"), [("P5", 30), ("H700", 30), ("h70", 30), ("H70", 0)])
def test_unknown_rules_and_empty_books_are_refused(name: str, slots: int) -> None:
    with pytest.raises(ValueError, match=r"rule must look like|at least one slot"):
        Rule.parse(name, slots)
