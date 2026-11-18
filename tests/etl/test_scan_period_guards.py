"""Filing re-dates and FinMind backfills keep out of the periods the daily scans date themselves."""

import logging
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from apps.etl.adapters.finmind import FinmindAdapter
from apps.etl.adapters.mops import STATEMENT_FILING, MopsAdapter
from apps.etl.filings import FILING_COLUMNS, apply_statement_filings
from apps.etl.models import EtlRun
from apps.etl.ratelimit import RateLimit, RateLimiter
from apps.etl.scan_start import (
    REVENUE_SCAN,
    STATEMENT_SCAN,
    first_scanned,
    first_scanned_month,
    first_scanned_quarter,
)
from apps.market.calendar import TradingDays
from apps.market.models import FinancialStatement, FinType, Market, StockMaster

TAIPEI = ZoneInfo("Asia/Taipei")
LIMIT = RateLimit(min_interval=12.0, max_retries=2, backoff_base=1.0, backoff_max=2.0, cooldown=5.0)
CUTOFF = time(22, 40)
CONSOLIDATED = FinType.CONSOLIDATED.value


def weekdays(start: date, end: date) -> TradingDays:
    span = (start + timedelta(days=offset) for offset in range((end - start).days + 1))
    return TradingDays([day for day in span if day.weekday() < 5], start, end)


CALENDAR = weekdays(date(2009, 1, 1), date(2026, 12, 31))


@pytest.fixture
def limiter() -> RateLimiter:
    return RateLimiter(LIMIT, sleep=lambda seconds: None)


def scanned(dataset: str, day: date, status: str = EtlRun.Status.SUCCESS) -> None:
    EtlRun.objects.create(
        source="mops", dataset=dataset, start_date=day, end_date=day, status=status
    )


def statements(*periods: tuple[str, date]) -> None:
    for period, announced in periods:
        FinancialStatement.objects.create(
            stock_id="2330",
            period=period,
            fin_type=CONSOLIDATED,
            eps=Decimal("1"),
            announce_date=announced,
        )


def filings(*uploads: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            (
                "2330",
                period,
                CONSOLIDATED,
                datetime.fromisoformat(at).replace(tzinfo=TAIPEI),
                False,
                False,
            )
            for period, at in uploads
        ],
        columns=FILING_COLUMNS,
    )


def dates() -> dict[str, date]:
    return {row.period: row.announce_date for row in FinancialStatement.objects.all()}


Q2_UPLOAD = ("2026-Q2", "2026-08-07 15:00")
Q3_UPLOAD = ("2026-Q3", "2026-11-03 15:00")
STORED = (("2026-Q2", date(2026, 9, 1)), ("2026-Q3", date(2026, 11, 5)))


@pytest.mark.django_db
def test_no_scan_has_run_yet_so_no_period_is_the_scans() -> None:
    assert (first_scanned(), first_scanned_quarter(), first_scanned_month()) == (None, None, None)


@pytest.mark.django_db
def test_the_first_scans_date_the_quarter_and_month_still_being_filed() -> None:
    scanned(STATEMENT_SCAN, date(2026, 11, 5))
    scanned(REVENUE_SCAN, date(2026, 11, 5))
    scanned(REVENUE_SCAN, date(2026, 11, 6))
    assert (first_scanned(), first_scanned_quarter(), first_scanned_month()) == (
        "2026-Q3",
        "2026-Q3",
        "2026-10",
    )


@pytest.mark.django_db
def test_a_failed_statement_scan_starts_only_the_k18_bound() -> None:
    scanned(STATEMENT_SCAN, date(2026, 9, 1), status=EtlRun.Status.FAILED)
    scanned(STATEMENT_SCAN, date(2026, 11, 5))
    assert (first_scanned(), first_scanned_quarter()) == ("2026-Q2", "2026-Q3")


@pytest.mark.django_db
def test_a_failed_revenue_scan_does_not_start_the_scanned_months() -> None:
    scanned(REVENUE_SCAN, date(2026, 10, 1), status=EtlRun.Status.FAILED)
    scanned(REVENUE_SCAN, date(2026, 11, 5))
    assert first_scanned_month() == "2026-10"


@pytest.mark.django_db
def test_filing_dates_leave_the_periods_the_scan_dated_alone() -> None:
    statements(*STORED)
    changed = apply_statement_filings(filings(Q2_UPLOAD, Q3_UPLOAD), CALENDAR, CUTOFF, "2026-Q3")
    assert changed == 1
    assert dates() == {"2026-Q2": date(2026, 8, 7), "2026-Q3": date(2026, 11, 5)}


@pytest.mark.django_db
def test_scan_periods_get_filing_dates_only_when_asked_and_are_named_first(
    caplog: pytest.LogCaptureFixture,
) -> None:
    statements(*STORED)
    frame = filings(Q2_UPLOAD, Q3_UPLOAD)
    with caplog.at_level(logging.WARNING, logger="apps.etl.filings"):
        changed = apply_statement_filings(
            frame, CALENDAR, CUTOFF, "2026-Q3", include_scan_periods=True
        )
    assert changed == 2
    assert dates() == {"2026-Q2": date(2026, 8, 7), "2026-Q3": date(2026, 11, 3)}
    warnings = [
        record.getMessage() for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert warnings == [
        "filing dates replace daily-scan sessions in {'2026-Q3': 1} rows per period"
    ]


@pytest.mark.django_db
def test_the_scan_periods_are_named_before_any_date_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statements(*STORED)
    seen: list[dict[str, date]] = []
    monkeypatch.setattr("apps.etl.filings.logger.warning", lambda *args: seen.append(dates()))
    apply_statement_filings(
        filings(Q2_UPLOAD, Q3_UPLOAD), CALENDAR, CUTOFF, "2026-Q3", include_scan_periods=True
    )
    assert seen == [dict(STORED)]


@pytest.mark.django_db
def test_before_any_scan_every_period_takes_its_filing_date() -> None:
    statements(*STORED)
    changed = apply_statement_filings(filings(Q2_UPLOAD, Q3_UPLOAD), CALENDAR, CUTOFF, None)
    assert changed == 2
    assert dates() == {"2026-Q2": date(2026, 8, 7), "2026-Q3": date(2026, 11, 3)}


@pytest.mark.django_db
@pytest.mark.parametrize(("include", "q3"), [(False, date(2026, 11, 5)), (True, date(2026, 11, 3))])
def test_the_mops_adapter_reads_the_scan_start_and_its_own_flag(
    limiter: RateLimiter, include: bool, q3: date
) -> None:
    scanned(STATEMENT_SCAN, date(2026, 11, 5))
    statements(*STORED)
    adapter = MopsAdapter(CALENDAR, limiter, include_scan_periods=include)
    adapter.upsert(STATEMENT_FILING, filings(Q2_UPLOAD, Q3_UPLOAD))
    assert dates() == {"2026-Q2": date(2026, 8, 7), "2026-Q3": q3}


@pytest.mark.django_db
def test_k17_filled_quarters_before_the_first_successful_scan_still_take_filing_dates(
    limiter: RateLimiter,
) -> None:
    scanned(STATEMENT_SCAN, date(2026, 9, 1), status=EtlRun.Status.FAILED)
    scanned(STATEMENT_SCAN, date(2026, 11, 5))
    statements(*STORED)
    MopsAdapter(CALENDAR, limiter).upsert(STATEMENT_FILING, filings(Q2_UPLOAD, Q3_UPLOAD))
    assert dates() == {"2026-Q2": date(2026, 8, 7), "2026-Q3": date(2026, 11, 5)}


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    flags: list[bool] = []
    monkeypatch.setattr(
        "apps.etl.management.commands.reapply_filings.TradingDays.from_db", lambda: CALENDAR
    )

    def reapply(adapter: MopsAdapter, stocks: Any) -> Any:
        flags.append(adapter.include_scan_periods)
        return type("Result", (), {"changed": 0, "stocks": 0, "rejected": {}})()

    monkeypatch.setattr("apps.etl.management.commands.reapply_filings.reapply_filings", reapply)
    return flags


@pytest.mark.django_db
def test_reapply_filings_reaches_scan_periods_only_with_the_flag(seen: list[bool]) -> None:
    call_command("reapply_filings")
    call_command("reapply_filings", "--include-scan-periods")
    assert seen == [False, True]


@pytest.fixture
def listed(db: None, monkeypatch: pytest.MonkeyPatch) -> None:
    StockMaster.objects.create(stock_id="2330", name="台積電", market=Market.TWSE)
    monkeypatch.setattr("apps.etl.adapters.TradingDays.from_db", lambda: CALENDAR)


@pytest.mark.usefixtures("listed")
def test_backfill_takes_the_flag_for_mops_filing_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    flags: list[bool] = []

    def backfill(self: MopsAdapter, dataset: str, *args: Any, **kwargs: Any) -> EtlRun:
        flags.append(self.include_scan_periods)
        return EtlRun.objects.create(
            source="mops", dataset=dataset, start_date=date(2026, 1, 1), end_date=date(2026, 1, 1)
        )

    monkeypatch.setattr(MopsAdapter, "backfill", backfill)
    options = ("--source", "mops", "--dataset", STATEMENT_FILING, "--from", "2026-01-01")
    call_command("backfill", *options, "--stock", "2330")
    call_command("backfill", *options, "--stock", "2330", "--include-scan-periods")
    assert flags == [False, True]


@pytest.mark.usefixtures("listed")
@pytest.mark.parametrize(
    ("source", "dataset"), [("finmind", "financial_statement"), ("mops", "statement_viewer")]
)
def test_the_flag_belongs_to_mops_filing_dates_only(source: str, dataset: str) -> None:
    with pytest.raises(CommandError, match="--include-scan-periods"):
        call_command(
            "backfill",
            "--source",
            source,
            "--dataset",
            dataset,
            "--from",
            "2026-01-01",
            "--stock",
            "2330",
            "--include-scan-periods",
        )


@pytest.fixture
def finmind(limiter: RateLimiter, db: None) -> FinmindAdapter:
    return FinmindAdapter(CALENDAR, limiter)


CASES: list[tuple[str, str, date, date, str]] = [
    ("financial_statement", STATEMENT_SCAN, date(2026, 9, 30), date(2026, 9, 29), "2026-Q3"),
    ("monthly_revenue", REVENUE_SCAN, date(2026, 11, 1), date(2026, 10, 31), "2026-10"),
]


@pytest.mark.parametrize(("dataset", "scan", "reaching", "last", "period"), CASES)
def test_finmind_backfills_end_before_the_first_period_the_scan_dates(
    finmind: FinmindAdapter, dataset: str, scan: str, reaching: date, last: date, period: str
) -> None:
    scanned(scan, date(2026, 11, 5))
    with pytest.raises(ValueError, match=f"{period}.*{last.isoformat()}"):
        finmind.backfill(dataset, ["2330"], date(2010, 1, 1), reaching)
    run = finmind.backfill(dataset, [], date(2010, 1, 1), last)
    assert run.status == EtlRun.Status.SUCCESS


@pytest.mark.parametrize(("dataset", "scan", "reaching", "last", "period"), CASES)
def test_finmind_backfills_are_unbounded_until_a_scan_has_run(
    finmind: FinmindAdapter, dataset: str, scan: str, reaching: date, last: date, period: str
) -> None:
    run = finmind.backfill(dataset, [], date(2010, 1, 1), date(2026, 11, 17))
    assert run.status == EtlRun.Status.SUCCESS


def test_a_failed_statement_scan_does_not_move_the_finmind_bound(finmind: FinmindAdapter) -> None:
    scanned(STATEMENT_SCAN, date(2026, 9, 1), status=EtlRun.Status.FAILED)
    scanned(STATEMENT_SCAN, date(2026, 11, 5))
    run = finmind.backfill("financial_statement", [], date(2010, 1, 1), date(2026, 9, 29))
    assert run.status == EtlRun.Status.SUCCESS


@pytest.mark.usefixtures("listed")
def test_the_backfill_command_names_the_last_day_it_may_reach() -> None:
    scanned(STATEMENT_SCAN, date(2026, 11, 5))
    with pytest.raises(CommandError, match="2026-09-29"):
        call_command(
            "backfill",
            "--source",
            "finmind",
            "--dataset",
            "financial_statement",
            "--from",
            "2010-01-01",
            "--to",
            "2026-09-30",
            "--stock",
            "2330",
        )


def test_other_finmind_datasets_are_not_bounded_by_the_scans(finmind: FinmindAdapter) -> None:
    scanned(STATEMENT_SCAN, date(2026, 11, 5))
    scanned(REVENUE_SCAN, date(2026, 11, 5))
    run = finmind.backfill("daily_price", [], date(2026, 1, 1), date(2026, 11, 17))
    assert run.status == EtlRun.Status.SUCCESS
