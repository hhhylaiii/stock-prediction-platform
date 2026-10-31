from datetime import date
from decimal import Decimal

import pytest
from django.db import IntegrityError, connection, transaction

from apps.market.models import DailyPrice, Market, StockMaster, TradingCalendar


def _index_columns(table: str) -> dict[str, list[str]]:
    with connection.cursor() as cursor:
        constraints = connection.introspection.get_constraints(cursor, table)
    return {name: info["columns"] for name, info in constraints.items() if info["index"]}


def _price(**overrides: object) -> DailyPrice:
    values: dict[str, object] = {
        "stock_id": "2330",
        "date": date(2026, 10, 28),
        "open": Decimal("1000.5"),
        "high": Decimal("1010.25"),
        "low": Decimal("995.125"),
        "close": Decimal("1005.4567"),
        "volume": 25_000_000,
        "turnover": 25_100_000_000,
    }
    values.update(overrides)
    return DailyPrice(**values)


@pytest.mark.django_db
def test_daily_price_primary_key_is_stock_and_date() -> None:
    with connection.cursor() as cursor:
        constraints = connection.introspection.get_constraints(cursor, "daily_price")
    primary = [info["columns"] for info in constraints.values() if info["primary_key"]]
    assert primary == [["stock_id", "date"]]
    assert ["date"] in _index_columns("daily_price").values()


def test_daily_price_composite_pk_fields() -> None:
    assert [f.name for f in DailyPrice._meta.pk_fields] == ["stock_id", "date"]


@pytest.mark.django_db
def test_daily_price_rejects_duplicate_stock_date() -> None:
    _price().save()
    with pytest.raises(IntegrityError), transaction.atomic():
        DailyPrice.objects.create(stock_id="2330", date=date(2026, 10, 28), volume=1, turnover=1)


@pytest.mark.django_db
def test_daily_price_round_trips_by_composite_pk() -> None:
    _price().save()
    row = DailyPrice.objects.get(pk=("2330", date(2026, 10, 28)))
    assert (row.open, row.high, row.low, row.close) == (
        Decimal("1000.5"),
        Decimal("1010.25"),
        Decimal("995.125"),
        Decimal("1005.4567"),
    )
    assert row.adj_factor == 1.0


@pytest.mark.django_db
def test_daily_price_keeps_fractional_adj_factor() -> None:
    _price(adj_factor=0.912345678).save()
    row = DailyPrice.objects.get(pk=("2330", date(2026, 10, 28)))
    assert row.adj_factor == pytest.approx(0.912345678, rel=1e-12)


@pytest.mark.django_db
def test_daily_price_upserts_on_the_composite_key() -> None:
    _price().save()
    updated = _price(close=Decimal("1010.5"), volume=30_000_000)
    added = _price(stock_id="2317", close=Decimal("251.5"))
    DailyPrice.objects.bulk_create(
        [updated, added],
        update_conflicts=True,
        unique_fields=["stock_id", "date"],
        update_fields=["close", "volume"],
    )
    assert DailyPrice.objects.count() == 2
    row = DailyPrice.objects.get(pk=("2330", date(2026, 10, 28)))
    assert (row.close, row.volume) == (Decimal("1010.5"), 30_000_000)


@pytest.mark.django_db
def test_daily_price_volume_and_turnover_hold_large_share_and_twd_counts() -> None:
    _price(volume=3_000_000_000, turnover=900_000_000_000).save()
    row = DailyPrice.objects.get(pk=("2330", date(2026, 10, 28)))
    assert (row.volume, row.turnover) == (3_000_000_000, 900_000_000_000)
    assert DailyPrice._meta.get_field("volume").verbose_name == "成交量（股）"
    assert DailyPrice._meta.get_field("turnover").verbose_name == "成交金額（元）"


@pytest.mark.django_db
def test_daily_price_allows_missing_prices_on_no_trade_days() -> None:
    _price(open=None, high=None, low=None, close=None, volume=0, turnover=0).save()
    assert DailyPrice.objects.get(pk=("2330", date(2026, 10, 28))).close is None


@pytest.mark.django_db
@pytest.mark.parametrize(
    "overrides",
    [
        {"volume": -1},
        {"turnover": -1},
        {"adj_factor": 0.0},
        {"adj_factor": -0.5},
        {"adj_factor": float("nan")},
        {"adj_factor": float("inf")},
    ],
)
def test_daily_price_check_constraints(overrides: dict[str, object]) -> None:
    with pytest.raises(IntegrityError), transaction.atomic():
        _price(**overrides).save()


@pytest.mark.django_db
def test_stock_master_keeps_delisted_and_listed_rows() -> None:
    StockMaster.objects.create(stock_id="2330", name="台積電", market=Market.TWSE)
    StockMaster.objects.create(
        stock_id="9999", name="已下市測試股", market=Market.TPEX, delist_date=date(2019, 12, 31)
    )
    assert StockMaster.objects.get(stock_id="2330").delist_date is None
    assert StockMaster.objects.filter(delist_date__isnull=False).count() == 1


@pytest.mark.django_db
def test_trading_calendar_primary_key_is_date() -> None:
    TradingCalendar.objects.create(date=date(2015, 9, 29), is_trading_day=False, note="颱風假")
    with pytest.raises(IntegrityError), transaction.atomic():
        TradingCalendar.objects.create(date=date(2015, 9, 29), is_trading_day=True)


def test_market_choices_cover_twse_and_tpex() -> None:
    assert set(Market.values) == {"TWSE", "TPEx"}


def test_string_representations() -> None:
    assert str(StockMaster(stock_id="2330", name="台積電")) == "2330 台積電"
    assert str(TradingCalendar(date=date(2015, 9, 29), is_trading_day=False)) == "2015-09-29 closed"
    assert str(TradingCalendar(date=date(2015, 9, 30), is_trading_day=True)) == "2015-09-30 open"
    assert str(_price()) == "2330 2026-10-28"
