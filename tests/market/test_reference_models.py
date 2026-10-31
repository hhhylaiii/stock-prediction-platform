from datetime import date
from decimal import Decimal

import pytest
from django.db import IntegrityError, transaction

from apps.market.models import (
    DailyValuation,
    DispositionEvent,
    DispositionType,
    IndexDaily,
    IndexId,
    MacroIndicator,
    ShareChangeReason,
    SharesOutstanding,
)
from tests.market.schema import index_columns, primary_key_columns

DAY = date(2026, 10, 28)


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("table", "columns"),
    [
        ("index_daily", ["index_id", "date"]),
        ("macro_indicator", ["indicator_id", "period"]),
        ("shares_outstanding", ["stock_id", "effective_date"]),
        ("daily_valuation", ["stock_id", "date"]),
        ("disposition_event", ["stock_id", "start_date", "type"]),
    ],
)
def test_primary_keys(table: str, columns: list[str]) -> None:
    assert primary_key_columns(table) == [columns]


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("table", "columns"),
    [
        ("daily_valuation", ["date"]),
        ("disposition_event", ["start_date"]),
        ("macro_indicator", ["announce_date"]),
    ],
)
def test_secondary_indexes(table: str, columns: list[str]) -> None:
    assert columns in index_columns(table)


def test_index_ids_cover_the_spec_series() -> None:
    assert set(IndexId.values) == {"TAIEX", "TPEX", "SOX", "TSM_ADR", "USDTWD"}


@pytest.mark.django_db
def test_index_daily_keeps_fx_without_volume() -> None:
    IndexDaily.objects.create(index_id=IndexId.USDTWD, date=DAY, close=Decimal("32.1450"))
    IndexDaily.objects.create(
        index_id=IndexId.TAIEX, date=DAY, close=Decimal("27850.12"), volume=8_200_000_000
    )
    assert IndexDaily.objects.get(pk=(IndexId.USDTWD, DAY)).volume is None
    assert IndexDaily.objects.get(pk=(IndexId.TAIEX, DAY)).close == Decimal("27850.1200")


@pytest.mark.django_db
@pytest.mark.parametrize(
    "values",
    [
        {"index_id": "TAIEX", "close": Decimal("0")},
        {"index_id": "TAIEX", "close": Decimal("100"), "volume": -1},
        {"index_id": "NASDAQ", "close": Decimal("100")},
    ],
)
def test_index_daily_check_constraints(values: dict[str, object]) -> None:
    with pytest.raises(IntegrityError), transaction.atomic():
        IndexDaily.objects.create(date=DAY, **values)


@pytest.mark.django_db
def test_macro_indicator_round_trip_and_required_announce_date() -> None:
    MacroIndicator.objects.create(
        indicator_id="business_signal",
        period="2026-09",
        value=31.0,
        announce_date=date(2026, 10, 27),
    )
    assert MacroIndicator.objects.get(pk=("business_signal", "2026-09")).value == 31.0
    with pytest.raises(IntegrityError), transaction.atomic():
        MacroIndicator.objects.create(indicator_id="business_signal", period="2026-10", value=1.0)


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("period", "value"),
    [("2026-13", 1.0), ("202609", 1.0), ("2026-Q3", 1.0), ("2026-09", float("nan"))],
)
def test_macro_indicator_check_constraints(period: str, value: float) -> None:
    with pytest.raises(IntegrityError), transaction.atomic():
        MacroIndicator.objects.create(
            indicator_id="business_signal", period=period, value=value, announce_date=DAY
        )


@pytest.mark.django_db
def test_shares_outstanding_defaults_reason_to_unknown() -> None:
    SharesOutstanding.objects.create(
        stock_id="2330", effective_date=date(2010, 1, 4), shares=25_930_380_458
    )
    row = SharesOutstanding.objects.get(pk=("2330", date(2010, 1, 4)))
    assert (row.shares, row.reason) == (25_930_380_458, ShareChangeReason.UNKNOWN)


@pytest.mark.django_db
@pytest.mark.parametrize(
    "values",
    [{"shares": 0}, {"shares": 1, "reason": "split"}],
)
def test_shares_outstanding_check_constraints(values: dict[str, object]) -> None:
    with pytest.raises(IntegrityError), transaction.atomic():
        SharesOutstanding.objects.create(stock_id="2330", effective_date=DAY, **values)


def test_share_change_reasons() -> None:
    assert set(ShareChangeReason.values) == {
        "unknown",
        "capital_increase",
        "capital_reduction",
        "treasury_cancellation",
    }


@pytest.mark.django_db
def test_daily_valuation_allows_blank_pe_for_loss_makers() -> None:
    DailyValuation.objects.create(stock_id="2330", date=DAY, pe=None, pb=6.1, dividend_yield=0.0145)
    assert DailyValuation.objects.get(pk=("2330", DAY)).pe is None


@pytest.mark.django_db
@pytest.mark.parametrize(
    "values",
    [{"pe": 0.0}, {"pb": -1.0}, {"dividend_yield": -0.01}, {"pb": float("nan")}],
)
def test_daily_valuation_check_constraints(values: dict[str, object]) -> None:
    with pytest.raises(IntegrityError), transaction.atomic():
        DailyValuation.objects.create(stock_id="2330", date=DAY, **values)


@pytest.mark.django_db
def test_disposition_event_types_share_a_start_date() -> None:
    for kind in (DispositionType.ATTENTION, DispositionType.DISPOSITION):
        DispositionEvent.objects.create(
            stock_id="6488", start_date=DAY, type=kind, end_date=date(2026, 11, 10)
        )
    attention = DispositionEvent.objects.get(pk=("6488", DAY, DispositionType.ATTENTION))
    assert attention.end_date == date(2026, 11, 10)
    assert DispositionEvent.objects.count() == 2


@pytest.mark.django_db
def test_disposition_event_end_date_may_be_open() -> None:
    DispositionEvent.objects.create(
        stock_id="6488", start_date=DAY, type=DispositionType.FULL_DELIVERY
    )
    assert DispositionEvent.objects.get().end_date is None


@pytest.mark.django_db
@pytest.mark.parametrize(
    "values",
    [
        {"type": "attention", "end_date": date(2026, 10, 27)},
        {"type": "suspended"},
    ],
)
def test_disposition_event_check_constraints(values: dict[str, object]) -> None:
    with pytest.raises(IntegrityError), transaction.atomic():
        DispositionEvent.objects.create(stock_id="6488", start_date=DAY, **values)


def test_disposition_types() -> None:
    assert set(DispositionType.values) == {"attention", "disposition", "full_delivery"}


def test_string_representations() -> None:
    assert str(IndexDaily(index_id="TAIEX", date=DAY)) == "TAIEX 2026-10-28"
    assert str(MacroIndicator(indicator_id="business_signal", period="2026-09")) == (
        "business_signal 2026-09"
    )
    assert str(SharesOutstanding(stock_id="2330", effective_date=DAY)) == "2330 from 2026-10-28"
    assert str(DailyValuation(stock_id="2330", date=DAY)) == "2330 2026-10-28"
    assert str(DispositionEvent(stock_id="6488", start_date=DAY, type="attention")) == (
        "6488 attention from 2026-10-28"
    )
