from collections.abc import Mapping
from datetime import date

import pytest
from django.db import IntegrityError, transaction

from apps.market.market_history import Span, market_on, market_spans, merge_spans, record_markets
from apps.market.models import Market, MarketHistory

TWSE = Market.TWSE.value
TPEX = Market.TPEX.value
BOTH = frozenset({TWSE, TPEX})


def _day(
    listed: Mapping[str, frozenset[str]], day: date
) -> tuple[date, Mapping[str, frozenset[str]]]:
    return day, listed


def _only(market: str) -> frozenset[str]:
    return frozenset({market})


def _spans() -> list[tuple[str, date, date | None, str]]:
    rows = MarketHistory.objects.order_by("stock_id", "start_date")
    return list(rows.values_list("stock_id", "start_date", "end_date", "market"))


def test_a_stock_on_one_market_has_one_open_span_from_its_first_day() -> None:
    days = [
        _day({"2330": _only(TWSE)}, date(2026, 10, 1)),
        _day({"2330": _only(TWSE)}, date(2026, 10, 2)),
    ]
    assert market_spans(days) == [Span("2330", date(2026, 10, 1), None, TWSE)]


def test_the_day_both_markets_list_a_stock_stays_on_the_old_one() -> None:
    days = [
        _day({"6177": _only(TPEX)}, date(2013, 7, 30)),
        _day({"6177": BOTH}, date(2013, 7, 31)),
        _day({"6177": _only(TWSE)}, date(2013, 8, 1)),
        _day({"6177": _only(TWSE)}, date(2013, 8, 2)),
    ]
    assert market_spans(days) == [
        Span("6177", date(2013, 7, 30), date(2013, 7, 31), TPEX),
        Span("6177", date(2013, 8, 1), None, TWSE),
    ]


def test_a_move_without_a_shared_day_ends_the_old_span_the_calendar_day_before() -> None:
    days = [
        _day({"6423": _only(TWSE)}, date(2026, 1, 16)),
        _day({"6423": _only(TPEX)}, date(2026, 1, 19)),
    ]
    assert market_spans(days) == [
        Span("6423", date(2026, 1, 16), date(2026, 1, 18), TWSE),
        Span("6423", date(2026, 1, 19), None, TPEX),
    ]


def test_days_a_stock_is_missing_keep_its_market() -> None:
    days = [
        _day({"1101": _only(TWSE)}, date(2026, 10, 1)),
        _day({}, date(2026, 10, 2)),
        _day({"1101": _only(TWSE)}, date(2026, 10, 5)),
    ]
    assert market_spans(days) == [Span("1101", date(2026, 10, 1), None, TWSE)]


def test_a_first_day_on_both_markets_takes_the_next_single_market() -> None:
    days = [
        _day({"8908": BOTH}, date(2026, 10, 14)),
        _day({"8908": _only(TPEX)}, date(2026, 10, 15)),
    ]
    assert market_spans(days) == [Span("8908", date(2026, 10, 14), None, TPEX)]


def test_days_are_read_in_date_order_whatever_order_they_come_in() -> None:
    days = [
        _day({"6423": _only(TPEX)}, date(2026, 1, 19)),
        _day({"6423": _only(TWSE)}, date(2026, 1, 16)),
    ]
    assert [span.market for span in market_spans(days)] == [TWSE, TPEX]


def test_stored_boundaries_on_days_the_pages_did_not_see_are_kept() -> None:
    pages = [Span("6177", date(2026, 10, 1), None, TPEX)]
    stored = [
        Span("6177", date(2026, 10, 1), date(2026, 10, 26), TPEX),
        Span("6177", date(2026, 10, 27), None, TWSE),
    ]
    seen = {date(2026, 10, 1), date(2026, 10, 2), date(2026, 11, 2)}
    assert merge_spans(pages, stored, seen) == stored


def test_stored_boundaries_on_days_the_pages_saw_give_way_to_the_pages() -> None:
    pages = [
        Span("6177", date(2026, 10, 1), date(2026, 10, 27), TPEX),
        Span("6177", date(2026, 10, 28), None, TWSE),
    ]
    stored = [
        Span("6177", date(2026, 10, 1), date(2026, 10, 28), TPEX),
        Span("6177", date(2026, 10, 29), None, TWSE),
    ]
    seen = {date(2026, 10, 1), date(2026, 10, 28), date(2026, 10, 29)}
    assert merge_spans(pages, stored, seen) == pages


def test_a_stored_start_before_the_first_page_day_it_could_not_see_is_kept() -> None:
    pages = [Span("2249", date(2026, 11, 2), None, TWSE)]
    stored = [Span("2249", date(2026, 10, 22), None, TWSE)]
    assert merge_spans(pages, stored, {date(2026, 11, 2)}) == stored


@pytest.mark.django_db
def test_market_on_reads_the_span_covering_each_day() -> None:
    MarketHistory.objects.bulk_create(
        [
            MarketHistory(
                stock_id="6177",
                start_date=date(2010, 1, 4),
                end_date=date(2013, 7, 31),
                market=TPEX,
            ),
            MarketHistory(stock_id="6177", start_date=date(2013, 8, 1), market=TWSE),
            MarketHistory(stock_id="2330", start_date=date(2010, 1, 4), market=TWSE),
            MarketHistory(
                stock_id="1101",
                start_date=date(2010, 1, 4),
                end_date=date(2013, 7, 31),
                market=TWSE,
            ),
        ]
    )
    pairs = [
        ("6177", date(2013, 7, 31)),
        ("6177", date(2013, 8, 1)),
        ("6177", date(2026, 11, 6)),
        ("6177", date(2009, 12, 31)),
        ("2330", date(2026, 11, 6)),
        ("9999", date(2026, 11, 6)),
        ("6177", date(2010, 1, 4)),
        ("1101", date(2013, 8, 1)),
    ]
    codes, days = zip(*pairs, strict=True)
    assert market_on(codes, days) == [TPEX, TWSE, TWSE, None, TWSE, None, TPEX, None]


@pytest.mark.django_db
def test_a_quote_on_another_market_closes_the_open_span_and_opens_one_from_that_day() -> None:
    MarketHistory.objects.create(stock_id="3293", start_date=date(2010, 1, 4), market=TWSE)
    MarketHistory.objects.create(stock_id="2330", start_date=date(2010, 1, 4), market=TWSE)
    opened = record_markets(date(2026, 10, 28), {"3293": TPEX, "2330": TWSE, "6488": TPEX})
    assert opened == ["3293", "6488"]
    assert _spans() == [
        ("2330", date(2010, 1, 4), None, TWSE),
        ("3293", date(2010, 1, 4), date(2026, 10, 27), TWSE),
        ("3293", date(2026, 10, 28), None, TPEX),
        ("6488", date(2026, 10, 28), None, TPEX),
    ]


@pytest.mark.django_db
def test_recording_the_same_day_again_changes_nothing() -> None:
    MarketHistory.objects.create(stock_id="3293", start_date=date(2010, 1, 4), market=TWSE)
    record_markets(date(2026, 10, 28), {"3293": TPEX})
    before = _spans()
    assert record_markets(date(2026, 10, 28), {"3293": TPEX}) == []
    assert _spans() == before


@pytest.mark.django_db
def test_an_open_span_starting_on_or_after_the_day_is_left_alone() -> None:
    MarketHistory.objects.create(stock_id="3293", start_date=date(2026, 10, 28), market=TPEX)
    assert record_markets(date(2026, 10, 28), {"3293": TWSE}) == []
    assert _spans() == [("3293", date(2026, 10, 28), None, TPEX)]


@pytest.mark.django_db
@pytest.mark.parametrize(
    "second",
    [
        MarketHistory(
            stock_id="6177", start_date=date(2010, 1, 4), end_date=date(2012, 1, 1), market=TWSE
        ),
        MarketHistory(stock_id="6177", start_date=date(2013, 8, 1), market=TWSE),
        MarketHistory(
            stock_id="2330", start_date=date(2013, 8, 1), end_date=date(2013, 7, 31), market=TWSE
        ),
    ],
    ids=["same-start", "second-open-span", "ends-before-start"],
)
def test_the_table_refuses_a_repeated_start_a_second_open_span_and_a_reversed_span(
    second: MarketHistory,
) -> None:
    MarketHistory.objects.create(stock_id="6177", start_date=date(2010, 1, 4), market=TPEX)
    with pytest.raises(IntegrityError), transaction.atomic():
        second.save()
