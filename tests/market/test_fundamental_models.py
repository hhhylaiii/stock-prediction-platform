from collections.abc import Callable
from datetime import date
from decimal import Decimal

import pytest
from django.db import IntegrityError, transaction

from apps.market.models import FinancialStatement, FinType, MonthlyRevenue
from tests.market.schema import index_columns, primary_key_columns


def _revenue(**overrides: object) -> MonthlyRevenue:
    values: dict[str, object] = {
        "stock_id": "2330",
        "period": "2026-09",
        "revenue": 330_000_000_000,
        "yoy": 0.312,
        "mom": -0.025,
        "announce_date": date(2026, 10, 10),
    }
    values.update(overrides)
    return MonthlyRevenue(**values)


def _statement(**overrides: object) -> FinancialStatement:
    values: dict[str, object] = {
        "stock_id": "2330",
        "period": "2026-Q3",
        "eps": Decimal("17.5612"),
        "revenue": 990_000_000_000,
        "gross_margin": 0.592,
        "op_margin": 0.495,
        "net_income": 455_000_000_000,
        "roe": 0.081,
        "debt_ratio": 0.33,
        "operating_cf": 600_000_000_000,
        "announce_date": date(2026, 11, 14),
    }
    values.update(overrides)
    return FinancialStatement(**values)


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("table", "columns"),
    [
        ("monthly_revenue", ["stock_id", "period"]),
        ("financial_statement", ["stock_id", "period", "fin_type"]),
    ],
)
def test_primary_keys(table: str, columns: list[str]) -> None:
    assert primary_key_columns(table) == [columns]


@pytest.mark.django_db
@pytest.mark.parametrize("table", ["monthly_revenue", "financial_statement"])
def test_announce_date_is_indexed(table: str) -> None:
    assert ["announce_date"] in index_columns(table)


@pytest.mark.django_db
@pytest.mark.parametrize("factory", [_revenue, _statement])
def test_announce_date_is_required(
    factory: Callable[..., MonthlyRevenue | FinancialStatement],
) -> None:
    with pytest.raises(IntegrityError), transaction.atomic():
        factory(announce_date=None).save()


@pytest.mark.django_db
def test_monthly_revenue_round_trip() -> None:
    _revenue().save()
    row = MonthlyRevenue.objects.get(pk=("2330", "2026-09"))
    assert (row.revenue, row.yoy, row.mom) == (330_000_000_000, 0.312, -0.025)


@pytest.mark.django_db
def test_monthly_revenue_ratios_may_be_missing() -> None:
    _revenue(yoy=None, mom=None).save()
    assert MonthlyRevenue.objects.get(pk=("2330", "2026-09")).yoy is None


@pytest.mark.django_db
@pytest.mark.parametrize(
    "overrides",
    [
        {"period": "2026-13"},
        {"period": "2026-00"},
        {"period": "202609"},
        {"period": "2026-9"},
        {"yoy": float("nan")},
    ],
)
def test_monthly_revenue_check_constraints(overrides: dict[str, object]) -> None:
    with pytest.raises(IntegrityError), transaction.atomic():
        _revenue(**overrides).save()


@pytest.mark.django_db
def test_financial_statement_defaults_to_consolidated_and_keeps_both_types() -> None:
    _statement().save()
    _statement(fin_type=FinType.INDIVIDUAL, eps=Decimal("17.5600")).save()
    consolidated = FinancialStatement.objects.get(pk=("2330", "2026-Q3", FinType.CONSOLIDATED))
    assert consolidated.eps == Decimal("17.5612")
    assert FinancialStatement.objects.count() == 2


@pytest.mark.django_db
def test_financial_statement_allows_missing_line_items() -> None:
    _statement(gross_margin=None, op_margin=None, operating_cf=None).save()
    row = FinancialStatement.objects.get(pk=("2330", "2026-Q3", FinType.CONSOLIDATED))
    assert row.gross_margin is None
    assert row.eps == Decimal("17.5612")


@pytest.mark.django_db
def test_financial_statement_allows_losses() -> None:
    _statement(eps=Decimal("-1.2300"), net_income=-5_000_000, roe=-0.02).save()
    row = FinancialStatement.objects.get(pk=("2330", "2026-Q3", FinType.CONSOLIDATED))
    assert row.eps == Decimal("-1.2300")


@pytest.mark.django_db
@pytest.mark.parametrize(
    "overrides",
    [
        {"period": "2026-Q5"},
        {"period": "2026-Q0"},
        {"period": "2026Q3"},
        {"period": "2026-09"},
        {"fin_type": "annual"},
        {"debt_ratio": -0.1},
        {"roe": float("inf")},
    ],
)
def test_financial_statement_check_constraints(overrides: dict[str, object]) -> None:
    with pytest.raises(IntegrityError), transaction.atomic():
        _statement(**overrides).save()


def test_fin_type_choices() -> None:
    assert set(FinType.values) == {"consolidated", "individual"}


def test_string_representations() -> None:
    assert str(_revenue()) == "2330 2026-09"
    assert str(_statement()) == "2330 2026-Q3 consolidated"
