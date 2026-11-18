from datetime import date

import pytest

from apps.research.holding_series import Fill, Point, cumulative

D0, D1, D2, D3 = date(2026, 11, 17), date(2026, 11, 18), date(2026, 11, 19), date(2026, 11, 20)


def test_no_fills_is_no_series() -> None:
    assert cumulative([]) == []


def test_the_first_fill_is_one_for_all_three_whatever_its_returns() -> None:
    assert cumulative([Fill(D0, 0.9987, 0.05, None)]) == [Point(D0, 1.0, 1.0, 1.0)]


def test_the_book_is_its_value_over_the_first_value_not_a_chain() -> None:
    fills = [
        Fill(D0, 0.9987, 0.0, 0.0),
        Fill(D1, 1.0187, 0.0, 0.0),
        Fill(D2, 0.9887, 0.0, 0.0),
    ]

    books = [point.book for point in cumulative(fills)]

    assert books == pytest.approx([1.0, 1.0187 / 0.9987, 0.9887 / 0.9987])


def test_the_universe_and_index_chain_from_the_day_after_the_first() -> None:
    fills = [
        Fill(D0, 1.0, 0.3, -0.3),
        Fill(D1, 1.0, 0.01, 0.02),
        Fill(D2, 1.0, -0.02, 0.005),
    ]

    points = cumulative(fills)

    assert [point.universe for point in points] == pytest.approx([1.0, 1.01, 1.01 * 0.98])
    assert [point.index for point in points] == pytest.approx([1.0, 1.02, 1.02 * 1.005])


def test_a_missing_index_close_blanks_the_index_from_that_day_on() -> None:
    fills = [
        Fill(D0, 1.0, 0.0, 0.0),
        Fill(D1, 1.0, 0.01, 0.01),
        Fill(D2, 1.0, 0.01, None),
        Fill(D3, 1.0, 0.01, 0.01),
    ]

    points = cumulative(fills)

    assert [point.index for point in points] == pytest.approx([1.0, 1.01, None, None])
    assert points[-1].universe == pytest.approx(1.01**3)


def test_a_missing_universe_return_blanks_the_universe_instead_of_counting_as_zero() -> None:
    fills = [Fill(D0, 1.0, 0.0, 0.0), Fill(D1, 1.0, None, 0.01), Fill(D2, 1.0, 0.01, 0.01)]

    assert [point.universe for point in cumulative(fills)] == [1.0, None, None]


def test_fills_are_taken_in_trade_day_order() -> None:
    fills = [Fill(D2, 1.1, 0.02, 0.0), Fill(D0, 1.0, 0.5, 0.0), Fill(D1, 1.05, 0.01, 0.0)]

    points = cumulative(fills)

    assert [point.trade_date for point in points] == [D0, D1, D2]
    assert [point.universe for point in points] == pytest.approx([1.0, 1.01, 1.01 * 1.02])


@pytest.mark.parametrize(
    ("fills", "said"),
    [
        ([Fill(D0, 1.0, 0.0, 0.0), Fill(D0, 1.0, 0.0, 0.0)], "same trade day"),
        ([Fill(D0, 0.0, 0.0, 0.0)], "not positive"),
        ([Fill(D0, float("nan"), 0.0, 0.0)], "not positive"),
        ([Fill(D0, 1.0, 0.0, 0.0), Fill(D1, 1.0, float("inf"), 0.0)], "has return inf"),
        ([Fill(D0, 1.0, 0.0, 0.0), Fill(D1, 1.0, 0.0, -1.0)], "has return -1.0"),
    ],
)
def test_a_fill_that_cannot_be_a_book_day_is_refused(fills: list[Fill], said: str) -> None:
    with pytest.raises(ValueError, match=said):
        cumulative(fills)
