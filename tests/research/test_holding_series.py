from datetime import date

import pytest

from apps.research.holding_series import Fill, Point, cumulative

S0 = date(2026, 11, 16)
D0, D1, D2 = date(2026, 11, 17), date(2026, 11, 18), date(2026, 11, 19)
D5 = date(2026, 11, 23)


def run(*days: tuple[date, float, float | None, float | None]) -> list[Fill]:
    """Fills that each sign on the trade day before, the first on S0."""
    signals = [S0, *(day for day, *_ in days[:-1])]
    return [
        Fill(signal, day, value, universe, close)
        for signal, (day, value, universe, close) in zip(signals, days, strict=True)
    ]


def test_no_fills_is_no_series() -> None:
    assert cumulative([]) == []


@pytest.mark.parametrize("universe", [0.05, None])
def test_the_first_fill_is_one_for_all_three_whatever_its_universe_return(
    universe: float | None,
) -> None:
    fills = run((D0, 0.99, universe, 25000.0), (D1, 0.99, 0.01, 25000.0))

    first, second = cumulative(fills)

    assert first == Point(D0, 1.0, 1.0, 1.0)
    assert second.universe == pytest.approx(1.01)


def test_the_book_is_its_value_over_the_first_value() -> None:
    fills = run((D0, 0.99, 0.0, 100.0), (D1, 1.01, 0.0, 100.0), (D2, 0.97, 0.0, 100.0))

    books = [point.book for point in cumulative(fills)]

    assert books == pytest.approx([1.0, 1.01 / 0.99, 0.97 / 0.99])


def test_the_universe_chains_from_the_day_after_the_first() -> None:
    fills = run((D0, 1.0, 0.3, 100.0), (D1, 1.0, 0.01, 100.0), (D2, 1.0, -0.02, 100.0))

    universe = [point.universe for point in cumulative(fills)]

    assert universe == pytest.approx([1.0, 1.01, 1.01 * 0.98])


def test_the_index_is_its_close_over_the_first_close() -> None:
    fills = run((D0, 1.0, 0.0, 20000.0), (D1, 1.0, 0.0, 20400.0), (D2, 1.0, 0.0, 20502.0))

    index = [point.index for point in cumulative(fills)]

    assert index == pytest.approx([1.0, 1.02, 20502 / 20000])


def test_a_missing_close_blanks_only_its_own_day() -> None:
    fills = run((D0, 1.0, 0.0, 20000.0), (D1, 1.0, 0.0, None), (D2, 1.0, 0.0, 20400.0))

    assert [point.index for point in cumulative(fills)] == [1.0, None, pytest.approx(1.02)]


def test_without_a_first_close_the_index_stays_blank() -> None:
    fills = run((D0, 1.0, 0.0, None), (D1, 1.0, 0.0, 20400.0))

    assert [point.index for point in cumulative(fills)] == [None, None]


def test_a_missing_universe_return_blanks_the_universe_from_that_day_on() -> None:
    fills = run((D0, 1.0, 0.0, 1.0), (D1, 1.0, None, 1.0), (D2, 1.0, 0.01, 1.0))

    assert [point.universe for point in cumulative(fills)] == [1.0, None, None]


def test_fills_are_taken_in_trade_day_order() -> None:
    first, second, third = run((D0, 1.0, 0.5, 1.0), (D1, 1.05, 0.01, 1.0), (D2, 1.1, 0.02, 1.0))

    points = cumulative([third, first, second])

    assert [point.trade_date for point in points] == [D0, D1, D2]
    assert [point.universe for point in points] == pytest.approx([1.0, 1.01, 1.01 * 1.02])


def test_a_fill_moved_past_a_closed_day_still_follows_the_one_before() -> None:
    fills = run((D0, 1.0, 0.0, 1.0), (D1, 1.0, 0.01, 1.0), (D5, 1.0, 0.02, 1.0))

    points = cumulative(fills)

    assert [point.trade_date for point in points] == [D0, D1, D5]
    assert points[-1].universe == pytest.approx(1.01 * 1.02)


def test_a_missing_fill_is_refused_instead_of_chaining_two_days_as_one() -> None:
    first, _, third = run((D0, 1.0, 0.0, 1.0), (D1, 1.0, 0.01, 1.0), (D2, 1.0, 0.01, 1.0))

    with pytest.raises(ValueError, match="gap before 2026-11-19: it signed on 2026-11-18"):
        cumulative([first, third])


@pytest.mark.parametrize(
    ("fills", "said"),
    [
        ([Fill(S0, D0, 1.0, 0.0, 1.0), Fill(S0, D0, 1.0, 0.0, 1.0)], "same trade day"),
        ([Fill(S0, D0, 0.0, 0.0, 1.0)], "not positive"),
        ([Fill(S0, D0, float("nan"), 0.0, 1.0)], "not positive"),
        ([Fill(S0, D0, 1.0, float("inf"), 1.0)], "universe return inf"),
        ([Fill(S0, D0, 1.0, -1.0, 1.0)], "universe return -1.0"),
        ([Fill(S0, D0, 1.0, float("nan"), 1.0)], "universe return nan"),
        ([Fill(S0, D0, 1.0, 0.0, 0.0)], "index close 0.0"),
        ([Fill(S0, D0, 1.0, 0.0, float("nan"))], "index close nan"),
        ([Fill(S0, D0, 1.0, 0.0, float("inf"))], "index close inf"),
    ],
)
def test_a_fill_that_cannot_be_a_book_day_is_refused(fills: list[Fill], said: str) -> None:
    with pytest.raises(ValueError, match=said):
        cumulative(fills)
