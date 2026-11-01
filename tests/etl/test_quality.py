from collections.abc import Callable
from datetime import date
from decimal import Decimal

import pytest
from django.core.management import CommandError, call_command

from apps.etl.models import DataQualityLog
from apps.etl.quality import CHECKS, run_checks, should_run_prediction
from apps.market.calendar import NotATradingDayError, OutsideCalendarError, TradingDays
from apps.market.models import (
    DailyChip,
    DailyPrice,
    DailyValuation,
    DividendEvent,
    IndexDaily,
    StockMaster,
)

PREV = date(2026, 10, 27)
DAY = date(2026, 10, 28)
CODES = [f"{1100 + n}" for n in range(100)]
EARLIER = date(2026, 10, 22)
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


def _indices(day: date) -> None:
    us_day = date.fromordinal(day.toordinal() - 1)
    IndexDaily.objects.bulk_create(
        [
            *(
                IndexDaily(index_id=name, date=day, close=Decimal(100))
                for name in ("TAIEX", "TPEX")
            ),
            *(
                IndexDaily(index_id=name, date=us_day, close=Decimal(100))
                for name in ("SOX", "TSM_ADR", "USDTWD")
            ),
        ]
    )


def _two_clean_days() -> None:
    for day in (PREV, DAY):
        _prices(day, CODES)
        _chips(day, CODES)
        _valuations(day, CODES)
    _indices(DAY)


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
    [(94, WARNING), (95, INFO), (96, INFO), (0, ERROR), (106, WARNING)],
    ids=["six-percent", "five-percent", "four-percent", "empty", "six-percent-more"],
)
def test_a_day_with_a_different_stock_count_is_flagged(
    calendar: TradingDays, kept: int, severity: str
) -> None:
    _two_clean_days()
    DailyPrice.objects.filter(date=DAY, stock_id__in=CODES[kept:]).delete()
    _prices(DAY, [f"{1300 + n}" for n in range(max(kept - 100, 0))])
    run_checks(DAY, calendar)
    row = _results()[("row_count", "daily_price")]
    assert row.severity == severity
    assert (row.detail["count"], row.detail["previous"]) == (kept, 100)
    assert should_run_prediction(DAY) is (severity != ERROR)


@pytest.mark.django_db
def test_a_table_that_stays_empty_keeps_blocking_the_prediction(calendar: TradingDays) -> None:
    _two_clean_days()
    _prices(EARLIER, CODES)
    DailyPrice.objects.filter(date__in=[PREV, DAY]).delete()
    run_checks(DAY, calendar)
    assert _severity("row_count", "daily_price") == ERROR
    assert _severity("missing_values", "daily_price") == ERROR
    assert not should_run_prediction(DAY)


@pytest.mark.django_db
def test_tables_never_loaded_are_not_an_error(calendar: TradingDays) -> None:
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
    gap = row.detail["columns"]["row"]
    assert (gap["missing"], gap["population"]) == (2, 99)


@pytest.mark.django_db
@pytest.mark.parametrize(("emptied", "severity"), [(5, INFO), (6, WARNING), (100, ERROR)])
def test_missing_closes_only_count_beyond_the_no_trade_level(
    calendar: TradingDays, emptied: int, severity: str
) -> None:
    _two_clean_days()
    DailyPrice.objects.filter(date=DAY, stock_id__in=CODES[:emptied]).update(close=None)
    run_checks(DAY, calendar)
    assert _severity("missing_values", "daily_price") == severity


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
    gap = row.detail["columns"][column]
    assert (gap["missing"], gap["population"]) == (2, 100)


@pytest.mark.django_db
@pytest.mark.parametrize(("emptied", "severity"), [(50, WARNING), (51, ERROR)])
def test_a_column_mostly_gone_means_its_dataset_failed(
    calendar: TradingDays, emptied: int, severity: str
) -> None:
    _two_clean_days()
    DailyChip.objects.filter(date=DAY, stock_id__in=CODES[:emptied]).update(margin_balance=None)
    run_checks(DAY, calendar)
    assert _severity("missing_values", "daily_chip") == severity
    assert should_run_prediction(DAY) is (severity != ERROR)


@pytest.mark.django_db
def test_a_column_that_stays_empty_keeps_failing(calendar: TradingDays) -> None:
    _two_clean_days()
    _chips(EARLIER, CODES)
    DailyChip.objects.filter(date__in=[PREV, DAY]).update(margin_balance=None)
    run_checks(DAY, calendar)
    gap = _results()[("missing_values", "daily_chip")].detail["columns"]["margin_balance"]
    assert (gap["missing"], gap["population"]) == (100, 100)
    assert not should_run_prediction(DAY)


def _two_markets() -> None:
    StockMaster.objects.bulk_create(
        StockMaster(stock_id=code, name=code, market="TWSE" if n < 58 else "TPEx")
        for n, code in enumerate(CODES)
    )


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("lost", "market"), [(CODES[:58], "TWSE"), (CODES[58:], "TPEx")], ids=["twse", "tpex"]
)
def test_one_market_losing_its_rows_blocks_every_day_it_lasts(
    calendar: TradingDays, lost: list[str], market: str
) -> None:
    _two_markets()
    _two_clean_days()
    _prices(date(2026, 10, 29), CODES)
    DailyPrice.objects.filter(date__in=[DAY, date(2026, 10, 29)], stock_id__in=lost).delete()
    for day in (DAY, date(2026, 10, 29)):
        run_checks(day, calendar)
        row = DataQualityLog.objects.get(
            date=day, check_name="missing_values", detail__table="daily_price"
        )
        assert row.severity == ERROR
        assert row.detail["columns"]["row"]["markets"][market]["missing"] == len(lost)
        assert not should_run_prediction(day)


@pytest.mark.django_db
def test_a_partly_emptied_column_keeps_warning_on_later_days(calendar: TradingDays) -> None:
    _two_clean_days()
    _chips(date(2026, 10, 29), CODES)
    DailyChip.objects.filter(date__in=[DAY, date(2026, 10, 29)], stock_id__in=CODES[:10]).update(
        margin_balance=None
    )
    run_checks(date(2026, 10, 29), calendar)
    row = DataQualityLog.objects.get(
        date=date(2026, 10, 29), check_name="missing_values", detail__table="daily_chip"
    )
    assert row.severity == WARNING
    assert row.detail["columns"]["margin_balance"]["missing"] == 10


@pytest.mark.django_db
def test_a_table_empty_for_over_a_month_still_blocks(calendar: TradingDays) -> None:
    _prices(date(2026, 9, 1), CODES)
    run_checks(DAY, calendar)
    assert _severity("row_count", "daily_price") == ERROR


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
    for holiday in (date(2026, 10, 23), date(2026, 10, 26)):
        if model is DailyPrice:
            _prices(holiday, CODES[:1])
        elif model is DailyChip:
            _chips(holiday, CODES[:1])
        else:
            _valuations(holiday, CODES[:1])
    run_checks(PREV, calendar)
    row = DataQualityLog.objects.get(date=PREV, check_name="trading_calendar")
    assert row.severity == ERROR
    assert row.detail["rows"] == {table: {"2026-10-23": 1, "2026-10-26": 1}}
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


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("day", "error"),
    [(date(2026, 10, 25), NotATradingDayError), (date(2026, 11, 2), OutsideCalendarError)],
)
def test_checks_refuse_a_closed_or_uncovered_day(
    calendar: TradingDays, day: date, error: type[Exception]
) -> None:
    with pytest.raises(error):
        run_checks(day, calendar)
    assert not DataQualityLog.objects.exists()


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("change", "stale"),
    [
        (lambda: None, set()),
        (lambda: IndexDaily.objects.filter(index_id="TAIEX").delete(), {"TAIEX"}),
        (lambda: IndexDaily.objects.filter(index_id="TPEX").update(date=PREV), {"TPEX"}),
        (
            lambda: IndexDaily.objects.filter(index_id="SOX").update(date=date(2026, 10, 23)),
            {"SOX"},
        ),
        (
            lambda: IndexDaily.objects.filter(index_id="USDTWD").update(date=date(2026, 10, 24)),
            set(),
        ),
        (lambda: IndexDaily.objects.all().delete(), {"TAIEX", "TPEX", "SOX", "TSM_ADR", "USDTWD"}),
    ],
    ids=[
        "fresh",
        "taiex-missing",
        "tpex-yesterday",
        "sox-five-days",
        "usdtwd-four-days",
        "none-loaded",
    ],
)
def test_market_series_that_stopped_updating_are_flagged(
    calendar: TradingDays, change: Callable[[], object], stale: set[str]
) -> None:
    _two_clean_days()
    change()
    run_checks(DAY, calendar)
    row = DataQualityLog.objects.get(date=DAY, check_name="index_freshness")
    assert set(row.detail["stale"]) == stale
    assert row.severity == (WARNING if stale else INFO)
