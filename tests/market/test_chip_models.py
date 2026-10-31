from datetime import date
from decimal import Decimal

import pandas as pd
import pytest
from django.db import IntegrityError, models, transaction

from apps.etl.upsert import upsert
from apps.market.models import DailyChip, DividendEvent, WeeklyHolding
from tests.market.schema import index_columns, primary_key_columns

DAY = date(2026, 10, 28)


def _dividend(**overrides: object) -> DividendEvent:
    values: dict[str, object] = {
        "stock_id": "2330",
        "ex_date": date(2026, 9, 16),
        "cash_dividend": Decimal("5.0"),
    }
    values.update(overrides)
    return DividendEvent(**values)


def _holding(**overrides: object) -> WeeklyHolding:
    values: dict[str, object] = {
        "stock_id": "2330",
        "data_date": date(2026, 10, 23),
        "level": 15,
        "holders": 1_500,
        "shares": 22_000_000_000,
        "pct": 84.71,
        "publish_date": date(2026, 10, 24),
    }
    values.update(overrides)
    return WeeklyHolding(**values)


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("table", "columns"),
    [
        ("dividend_event", ["stock_id", "ex_date"]),
        ("daily_chip", ["stock_id", "date"]),
        ("weekly_holding", ["stock_id", "data_date", "level"]),
    ],
)
def test_primary_keys(table: str, columns: list[str]) -> None:
    assert primary_key_columns(table) == [columns]


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("table", "columns"),
    [("daily_chip", ["date"]), ("weekly_holding", ["publish_date"])],
)
def test_cross_section_indexes(table: str, columns: list[str]) -> None:
    assert columns in index_columns(table)


@pytest.mark.django_db
def test_dividend_defaults_and_unset_adj_factor() -> None:
    _dividend().save()
    row = DividendEvent.objects.get(pk=("2330", date(2026, 9, 16)))
    assert (row.cash_dividend, row.stock_dividend, row.adj_factor) == (
        Decimal("5.0"),
        Decimal("0"),
        None,
    )


@pytest.mark.django_db
def test_dividend_keeps_eight_decimal_places() -> None:
    _dividend(cash_dividend=Decimal("2.50001234"), stock_dividend=Decimal("0.12345678")).save()
    row = DividendEvent.objects.get(pk=("2330", date(2026, 9, 16)))
    assert (row.cash_dividend, row.stock_dividend) == (
        Decimal("2.50001234"),
        Decimal("0.12345678"),
    )


@pytest.mark.django_db
@pytest.mark.parametrize(
    "overrides",
    [
        {"cash_dividend": Decimal("-0.1")},
        {"stock_dividend": Decimal("-1")},
        {"adj_factor": 0.0},
        {"adj_factor": float("nan")},
        {"adj_factor": float("inf")},
        {"announce_date": date(2026, 9, 17)},
    ],
)
def test_dividend_check_constraints(overrides: dict[str, object]) -> None:
    with pytest.raises(IntegrityError), transaction.atomic():
        _dividend(**overrides).save()


@pytest.mark.django_db
def test_dividend_announced_on_or_before_ex_date() -> None:
    _dividend(announce_date=date(2026, 6, 10)).save()
    _dividend(ex_date=date(2027, 3, 18), announce_date=date(2027, 3, 18)).save()
    assert DividendEvent.objects.filter(announce_date__isnull=False).count() == 2


@pytest.mark.django_db
def test_daily_chip_sources_upsert_their_own_columns() -> None:
    keys = ("stock_id", "date")
    flows = pd.DataFrame(
        [{"stock_id": "2330", "date": DAY, "fi_net": -3_500_000, "it_net": -120_000}]
    )
    margin = pd.DataFrame([{"stock_id": "2330", "date": DAY, "margin_balance": 25_000_000}])
    upsert(DailyChip, flows, keys, ("fi_net", "it_net"))
    upsert(DailyChip, margin, keys, ("margin_balance",))
    row = DailyChip.objects.get(pk=("2330", DAY))
    assert (row.fi_net, row.it_net, row.margin_balance) == (-3_500_000, -120_000, 25_000_000)
    assert row.day_trade_volume is None


@pytest.mark.django_db
def test_daily_chip_flows_may_be_negative() -> None:
    DailyChip.objects.create(
        stock_id="2330",
        date=DAY,
        fi_net=-1,
        it_net=-2,
        dealer_self_net=-3,
        dealer_hedge_net=-4,
        margin_balance=0,
        sbl_short_balance=0,
    )
    row = DailyChip.objects.get(pk=("2330", DAY))
    assert (row.dealer_self_net, row.dealer_hedge_net) == (-3, -4)


@pytest.mark.django_db
@pytest.mark.parametrize("pct", [0.0, 72.35, 100.0])
def test_daily_chip_fi_holding_pct_is_a_percent(pct: float) -> None:
    DailyChip.objects.create(stock_id="2330", date=DAY, fi_holding_pct=pct)
    assert DailyChip.objects.get(pk=("2330", DAY)).fi_holding_pct == pct


def test_daily_chip_share_columns_are_labelled_in_shares() -> None:
    share_columns = [
        "fi_net",
        "it_net",
        "dealer_self_net",
        "dealer_hedge_net",
        "margin_balance",
        "margin_limit",
        "short_balance",
        "sbl_short_balance",
        "day_trade_volume",
    ]
    for name in share_columns:
        field = DailyChip._meta.get_field(name)
        assert isinstance(field, models.Field)
        assert "股" in str(field.verbose_name), name


@pytest.mark.django_db
@pytest.mark.parametrize(
    "values",
    [
        {"margin_balance": -1},
        {"margin_limit": -1},
        {"short_balance": -1},
        {"sbl_short_balance": -1},
        {"day_trade_volume": -1},
        {"fi_holding_pct": -0.01},
        {"fi_holding_pct": 100.01},
        {"fi_holding_pct": float("nan")},
    ],
)
def test_daily_chip_check_constraints(values: dict[str, object]) -> None:
    with pytest.raises(IntegrityError), transaction.atomic():
        DailyChip.objects.create(stock_id="2330", date=DAY, **values)


@pytest.mark.django_db
def test_weekly_holding_publish_date_is_required() -> None:
    with pytest.raises(IntegrityError), transaction.atomic():
        _holding(publish_date=None).save()


@pytest.mark.django_db
@pytest.mark.parametrize(
    "overrides",
    [
        {"level": 0},
        {"level": 18},
        {"publish_date": date(2026, 10, 23)},
        {"holders": -1},
        {"shares": -1},
        {"pct": -0.1},
        {"pct": float("nan")},
        {"level": 16, "pct": float("nan")},
        {"level": 16, "pct": float("-inf")},
        {"level": 17, "shares": -1},
    ],
)
def test_weekly_holding_check_constraints(overrides: dict[str, object]) -> None:
    with pytest.raises(IntegrityError), transaction.atomic():
        _holding(**overrides).save()


@pytest.mark.django_db
def test_weekly_holding_adjustment_level_may_be_negative() -> None:
    _holding(level=16, holders=0, shares=-12_345, pct=-0.01).save()
    _holding(level=17, holders=1_200_000, shares=25_930_000_000, pct=100.0).save()
    _holding(level=1, holders=0, shares=0, pct=0.0).save()
    assert WeeklyHolding.objects.count() == 3


@pytest.mark.django_db
def test_weekly_holding_upserts_on_its_three_column_key() -> None:
    frame = pd.DataFrame(
        [
            {
                "stock_id": "2330",
                "data_date": date(2026, 10, 23),
                "level": level,
                "holders": 100 * level,
                "shares": 1_000 * level,
                "pct": float(level),
                "publish_date": date(2026, 10, 24),
            }
            for level in (1, 2, 15)
        ]
    )
    keys = ("stock_id", "data_date", "level")
    fields = ("holders", "shares", "pct", "publish_date")
    upsert(WeeklyHolding, frame, keys, fields)
    upsert(WeeklyHolding, frame.assign(holders=frame["holders"] + 1), keys, fields)
    assert WeeklyHolding.objects.count() == 3
    assert WeeklyHolding.objects.get(pk=("2330", date(2026, 10, 23), 15)).holders == 1_501


def test_string_representations() -> None:
    assert str(_dividend()) == "2330 ex 2026-09-16"
    assert str(DailyChip(stock_id="2330", date=DAY)) == "2330 2026-10-28"
    assert str(_holding()) == "2330 2026-10-23 L15"
