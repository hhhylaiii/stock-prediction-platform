from datetime import date
from decimal import Decimal

import pytest
from django.core.management import CommandError, call_command

from apps.etl.models import DataQualityLog
from apps.etl.quality import CHECKS, run_checks, should_run_prediction
from apps.market.calendar import TradingDays
from apps.market.models import DailyChip, DailyPrice, DailyValuation, DividendEvent, StockMaster

PREV = date(2026, 10, 27)
DAY = date(2026, 10, 28)
CODES = [f"{1100 + n}" for n in range(100)]
WEBHOOK = "https://hooks.example.test/alerts"
ERROR = DataQualityLog.Severity.ERROR
WARNING = DataQualityLog.Severity.WARNING
INFO = DataQualityLog.Severity.INFO


def _prices(day: date, codes: list[str], close: str = "10") -> None:
    DailyPrice.objects.bulk_create(
        DailyPrice(stock_id=code, date=day, close=Decimal(close), volume=1000, turnover=10000)
        for code in codes
    )


def _chips(day: date, codes: list[str]) -> None:
    DailyChip.objects.bulk_create(
        DailyChip(stock_id=code, date=day, fi_net=1, margin_balance=1000) for code in codes
    )


def _valuations(day: date, codes: list[str]) -> None:
    DailyValuation.objects.bulk_create(
        DailyValuation(stock_id=code, date=day, pe=12.0, pb=1.5, dividend_yield=0.03)
        for code in codes
    )


def _two_clean_days() -> None:
    for day in (PREV, DAY):
        _prices(day, CODES)
        _chips(day, CODES)
        _valuations(day, CODES)


def _results(day: date = DAY) -> dict[tuple[str, str], DataQualityLog]:
    return {
        (row.check_name, row.detail.get("table", "")): row
        for row in DataQualityLog.objects.filter(date=day)
    }


def _severity(check: str, table: str = "") -> str:
    return _results()[(check, table)].severity


@pytest.fixture
def calendar(october_2026: TradingDays) -> TradingDays:
    return october_2026


@pytest.mark.django_db
def test_clean_days_pass_every_check_and_allow_the_prediction(calendar: TradingDays) -> None:
    _two_clean_days()
    run_checks(DAY, calendar)
    assert {row.severity for row in DataQualityLog.objects.all()} == {INFO}
    assert {row.check_name for row in DataQualityLog.objects.all()} == set(CHECKS)
    assert should_run_prediction(DAY)


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("kept", "severity"),
    [(94, WARNING), (96, INFO), (0, ERROR)],
    ids=["six-percent", "four-percent", "empty"],
)
def test_a_day_with_a_different_stock_count_is_flagged(
    calendar: TradingDays, kept: int, severity: str
) -> None:
    _two_clean_days()
    DailyPrice.objects.filter(date=DAY, stock_id__in=CODES[kept:]).delete()
    run_checks(DAY, calendar)
    row = _results()[("row_count", "daily_price")]
    assert row.severity == severity
    assert (row.detail["count"], row.detail["previous"]) == (kept, 100)
    assert should_run_prediction(DAY) is (severity != ERROR)


@pytest.mark.django_db
def test_tables_empty_on_both_days_are_not_an_error(calendar: TradingDays) -> None:
    for day in (PREV, DAY):
        _prices(day, CODES)
    run_checks(DAY, calendar)
    assert _severity("row_count", "daily_chip") == INFO
    assert should_run_prediction(DAY)


@pytest.mark.django_db
def test_big_moves_without_a_dividend_are_noted_for_review(calendar: TradingDays) -> None:
    _two_clean_days()
    DailyPrice.objects.filter(date=DAY, stock_id=CODES[0]).update(close=Decimal("11.2"))
    DailyPrice.objects.filter(date=DAY, stock_id=CODES[1]).update(close=Decimal("8.9"))
    DailyPrice.objects.filter(date=DAY, stock_id=CODES[2]).update(close=Decimal("11.0"))
    DailyPrice.objects.filter(date=DAY, stock_id=CODES[3]).update(close=Decimal("7.0"))
    DividendEvent.objects.create(stock_id=CODES[3], ex_date=DAY, cash_dividend=Decimal("3"))
    run_checks(DAY, calendar)
    row = _results()[("big_move", "daily_price")]
    assert row.severity == INFO
    assert row.detail["stocks"] == {CODES[0]: 0.12, CODES[1]: -0.11}
    assert should_run_prediction(DAY)


@pytest.mark.django_db
def test_rows_gone_since_yesterday_count_as_missing_but_delisted_ones_do_not(
    calendar: TradingDays,
) -> None:
    _two_clean_days()
    StockMaster.objects.create(stock_id=CODES[2], name="x", market="TWSE", delist_date=DAY)
    DailyPrice.objects.filter(date=DAY, stock_id__in=CODES[:3]).delete()
    run_checks(DAY, calendar)
    row = _results()[("missing_values", "daily_price")]
    assert row.severity == WARNING
    assert row.detail["columns"] == {"row": {"missing": 2, "population": 99}}


@pytest.mark.django_db
def test_a_missing_close_is_no_trade_not_a_gap(calendar: TradingDays) -> None:
    _two_clean_days()
    DailyPrice.objects.filter(date=DAY, stock_id__in=CODES[:5]).update(close=None)
    run_checks(DAY, calendar)
    assert _severity("missing_values", "daily_price") == INFO


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("model", "column", "table"),
    [
        (DailyChip, "margin_balance", "daily_chip"),
        (DailyValuation, "pb", "daily_valuation"),
    ],
)
def test_a_column_that_went_empty_since_yesterday_is_flagged(
    calendar: TradingDays, model: type[DailyChip | DailyValuation], column: str, table: str
) -> None:
    _two_clean_days()
    model.objects.filter(date=DAY, stock_id__in=CODES[:2]).update(**{column: None})
    run_checks(DAY, calendar)
    row = _results()[("missing_values", table)]
    assert row.severity == WARNING
    assert row.detail["columns"][column] == {"missing": 2, "population": 100}


@pytest.mark.django_db
def test_columns_with_one_missing_stock_stay_under_the_limit(calendar: TradingDays) -> None:
    _two_clean_days()
    DailyChip.objects.filter(date=DAY, stock_id=CODES[0]).update(margin_balance=None)
    run_checks(DAY, calendar)
    assert _severity("missing_values", "daily_chip") == INFO


@pytest.mark.django_db
def test_a_missing_pe_means_a_loss_not_a_gap(calendar: TradingDays) -> None:
    _two_clean_days()
    DailyValuation.objects.filter(date=DAY, stock_id__in=CODES[:10]).update(pe=None)
    run_checks(DAY, calendar)
    assert _severity("missing_values", "daily_valuation") == INFO


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("model", "table"),
    [(DailyPrice, "daily_price"), (DailyChip, "daily_chip"), (DailyValuation, "daily_valuation")],
)
def test_rows_on_a_closed_day_since_the_last_session_are_an_error(
    calendar: TradingDays, model: type[DailyPrice | DailyChip | DailyValuation], table: str
) -> None:
    _two_clean_days()
    holiday = date(2026, 10, 26)
    if model is DailyPrice:
        _prices(holiday, CODES[:1])
    elif model is DailyChip:
        _chips(holiday, CODES[:1])
    else:
        _valuations(holiday, CODES[:1])
    run_checks(PREV, calendar)
    row = DataQualityLog.objects.get(date=PREV, check_name="trading_calendar")
    assert row.severity == ERROR
    assert row.detail["rows"] == {table: {"2026-10-26": 1}}
    assert not should_run_prediction(PREV)


@pytest.mark.django_db
def test_the_first_session_of_the_calendar_has_nothing_to_compare(calendar: TradingDays) -> None:
    first = date(2026, 10, 1)
    _prices(first, CODES)
    run_checks(first, calendar)
    row = DataQualityLog.objects.get(
        date=first, check_name="row_count", detail__table="daily_price"
    )
    assert row.severity == INFO
    assert row.detail["previous"] is None


@pytest.mark.django_db
def test_rerunning_replaces_only_these_checks(calendar: TradingDays) -> None:
    _two_clean_days()
    other = DataQualityLog.objects.create(
        date=DAY, check_name="interval_width", severity=WARNING, detail={}
    )
    run_checks(DAY, calendar)
    first = DataQualityLog.objects.filter(date=DAY).count()
    run_checks(DAY, calendar)
    assert DataQualityLog.objects.filter(date=DAY).count() == first
    assert DataQualityLog.objects.filter(pk=other.pk).exists()


@pytest.mark.django_db
def test_no_prediction_without_every_check_or_with_any_error(calendar: TradingDays) -> None:
    assert not should_run_prediction(DAY)
    _two_clean_days()
    run_checks(DAY, calendar)
    DataQualityLog.objects.filter(date=DAY, check_name="big_move").delete()
    assert not should_run_prediction(DAY)
    run_checks(DAY, calendar)
    DataQualityLog.objects.create(date=DAY, check_name="source_conflict", severity=ERROR, detail={})
    assert not should_run_prediction(DAY)


@pytest.mark.usefixtures("october_2026_in_db")
def test_check_quality_command_writes_the_results(capsys: pytest.CaptureFixture[str]) -> None:
    _two_clean_days()
    call_command("check_quality", date="2026-10-28")
    assert should_run_prediction(DAY)
    assert "trading_calendar" in capsys.readouterr().out


@pytest.mark.usefixtures("october_2026_in_db")
def test_check_quality_refuses_a_closed_day() -> None:
    with pytest.raises(CommandError, match="not a trading day"):
        call_command("check_quality", date="2026-10-25")


@pytest.mark.django_db
def test_check_quality_needs_a_loaded_calendar() -> None:
    with pytest.raises(CommandError, match="trading_calendar is empty"):
        call_command("check_quality", date="2026-10-28")
