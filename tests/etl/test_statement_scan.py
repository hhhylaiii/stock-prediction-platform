import re
from collections.abc import Callable, Iterator
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
import responses
import time_machine
from django.conf import settings
from responses import matchers

from apps.etl.adapters.mops import STATEMENT_SCAN, MopsAdapter
from apps.etl.errors import SchemaChangedError
from apps.etl.models import DataQualityLog, EtlRun, RawMops
from apps.etl.ratelimit import RateLimit, RateLimiter
from apps.etl.statement_scan import SCAN_COLUMNS, SOURCE_MISMATCH_CHECK
from apps.etl.statement_summary import (
    BALANCE,
    CASH_FLOW,
    INCOME,
    TYPE_MISMATCH_CHECK,
    open_periods,
    previous_period,
)
from apps.market.calendar import TradingDays
from apps.market.models import FinancialStatement, FinType

TAIPEI = ZoneInfo("Asia/Taipei")
LIMIT = RateLimit(min_interval=6.0, max_retries=2, backoff_base=1.0, backoff_max=2.0, cooldown=5.0)
SCAN_DAY = date(2026, 11, 5)
CONSOLIDATED = FinType.CONSOLIDATED.value
INDIVIDUAL = FinType.INDIVIDUAL.value
FILERS = {"1215", "1232", "1570", "6026"}


@pytest.mark.parametrize(
    ("day", "periods"),
    [
        (date(2026, 9, 30), []),
        (date(2026, 10, 1), ["2026-Q3"]),
        (SCAN_DAY, ["2026-Q3"]),
        (date(2026, 12, 5), ["2026-Q3"]),
        (date(2026, 12, 6), []),
        (date(2026, 1, 1), ["2025-Q4"]),
        (date(2026, 4, 2), ["2025-Q4", "2026-Q1"]),
        (date(2026, 4, 6), ["2026-Q1"]),
        (date(2026, 6, 5), ["2026-Q1"]),
        (date(2026, 6, 6), []),
    ],
)
def test_a_period_is_scanned_from_its_quarter_end_to_days_past_the_latest_deadline(
    day: date, periods: list[str]
) -> None:
    assert open_periods(day, 5) == periods


@pytest.mark.parametrize(
    ("period", "previous"), [("2026-Q1", "2025-Q4"), ("2026-Q3", "2026-Q2"), ("2025-Q4", "2025-Q3")]
)
def test_the_previous_period_is_the_quarter_before(period: str, previous: str) -> None:
    assert previous_period(period) == previous


def weekdays(start: date, end: date) -> TradingDays:
    span = (start + timedelta(days=offset) for offset in range((end - start).days + 1))
    return TradingDays([day for day in span if day.weekday() < 5], start, end)


@pytest.fixture
def adapter(fake_sleep: Callable[[float], None]) -> MopsAdapter:
    calendar = weekdays(date(2026, 1, 1), date(2026, 12, 31))
    return MopsAdapter(calendar, RateLimiter(LIMIT, sleep=fake_sleep))


@pytest.fixture
def summary(fixture_path: Callable[[str], Path]) -> Callable[..., bytes]:
    return lambda report, market, season, roc=115: fixture_path(
        f"mops/summary/{report}_{market}_{roc}_{season:02d}.html"
    ).read_bytes()


def serve(report: str, market: str, season: int, body: bytes, roc: int = 115) -> None:
    form = {
        "encodeURIComponent": "1",
        "step": "1",
        "firstin": "1",
        "off": "1",
        "isQuery": "Y",
        "TYPEK": market,
        "year": str(roc),
        "season": f"{season:02d}",
    }
    responses.post(
        f"{settings.MOPS_BASE_URL}/mops/web/ajax_{report}",
        body=body,
        match=[matchers.urlencoded_params_matcher(form)],
    )


@pytest.fixture
def published(summary: Callable[[str, str, int], bytes]) -> Iterator[None]:
    responses.start()
    for report in (INCOME, BALANCE, CASH_FLOW):
        for market in ("sii", "otc"):
            for season in (3, 2):
                serve(report, market, season, summary(report, market, season))
    yield
    responses.stop()
    responses.reset()


def scan_at(adapter: MopsAdapter, moment: str) -> EtlRun:
    with time_machine.travel(datetime.fromisoformat(moment).replace(tzinfo=TAIPEI), tick=False):
        return adapter.run(STATEMENT_SCAN, SCAN_DAY, SCAN_DAY)


def stored() -> dict[str, FinancialStatement]:
    return {row.stock_id: row for row in FinancialStatement.objects.filter(period="2026-Q3")}


@pytest.mark.django_db
@pytest.mark.usefixtures("published")
def test_a_scan_adds_the_quarters_new_filings_as_of_its_session(adapter: MopsAdapter) -> None:
    run = scan_at(adapter, "2026-11-05 19:00:00")
    rows = stored()
    assert set(rows) == FILERS
    assert {row.announce_date for row in rows.values()} == {SCAN_DAY}
    assert {row.fin_type for row in rows.values()} == {CONSOLIDATED}
    first = rows["1215"]
    assert (first.revenue, first.net_income, first.operating_cf) == (
        7_464_783_000,
        658_561_000,
        547_073_000,
    )
    assert first.eps == Decimal("2.2300")
    assert first.gross_margin == pytest.approx(0.191920, abs=1e-6)
    assert first.debt_ratio == pytest.approx(0.645841, abs=1e-6)
    assert first.roe == pytest.approx(0.058029, abs=1e-6)
    assert rows["6026"].revenue is None
    assert (run.status, run.rows) == ("success", 4)
    assert RawMops.objects.filter(dataset=STATEMENT_SCAN).count() == 12


@pytest.mark.django_db
@pytest.mark.usefixtures("published")
@pytest.mark.parametrize(
    ("moment", "session"),
    [
        (f"2026-11-05 {settings.ETL_AT['STATEMENTS_LATE']}:30", SCAN_DAY),
        (f"2026-11-05 {settings.ETL_AT['CHECK_QUALITY']}:00", date(2026, 11, 6)),
    ],
)
def test_the_late_scan_still_counts_for_the_session_but_not_after_the_evening_check(
    adapter: MopsAdapter, moment: str, session: date
) -> None:
    scan_at(adapter, moment)
    assert {row.announce_date for row in stored().values()} == {session}


@pytest.mark.django_db
@pytest.mark.usefixtures("published")
def test_a_filing_seen_before_keeps_its_first_values_and_date(adapter: MopsAdapter) -> None:
    FinancialStatement.objects.create(
        stock_id="1215",
        period="2026-Q3",
        fin_type=CONSOLIDATED,
        eps=Decimal("9.9900"),
        announce_date=date(2026, 11, 3),
    )
    run = scan_at(adapter, "2026-11-05 19:00:00")
    kept = stored()["1215"]
    assert (kept.eps, kept.revenue, kept.announce_date) == (
        Decimal("9.9900"),
        None,
        date(2026, 11, 3),
    )
    assert run.rows == 3
    differing = DataQualityLog.objects.get(check_name=SOURCE_MISMATCH_CHECK)
    assert differing.severity == "info"
    [entry] = differing.detail["statements"]
    assert (entry["stock_id"], entry["fin_type"]) == ("1215", CONSOLIDATED)
    assert entry["fields"]["eps"] == ["9.9900", "2.2300"]
    assert entry["fields"]["revenue"] == [None, 7_464_783_000]
    rerun = scan_at(adapter, "2026-11-05 22:40:00")
    assert DataQualityLog.objects.filter(check_name=SOURCE_MISMATCH_CHECK).count() == 1
    assert rerun.rows == 0


@pytest.mark.django_db
@pytest.mark.usefixtures("published")
def test_a_filing_stored_as_the_other_report_type_is_logged_not_doubled(
    adapter: MopsAdapter,
) -> None:
    FinancialStatement.objects.create(
        stock_id="1232", period="2026-Q3", fin_type=INDIVIDUAL, announce_date=date(2026, 11, 3)
    )
    scan_at(adapter, "2026-11-05 19:00:00")
    assert FinancialStatement.objects.filter(stock_id="1232").count() == 1
    log = DataQualityLog.objects.get(check_name=TYPE_MISMATCH_CHECK)
    assert (log.date, log.severity) == (SCAN_DAY, "warning")
    assert log.detail == {
        "statements": [
            {"stock_id": "1232", "period": "2026-Q3", "stored": INDIVIDUAL, "found": CONSOLIDATED}
        ]
    }
    scan_at(adapter, "2026-11-05 22:40:00")
    assert DataQualityLog.objects.filter(check_name=TYPE_MISMATCH_CHECK).count() == 1


@pytest.mark.django_db
@responses.activate
def test_a_quarter_nobody_filed_yet_adds_nothing(
    adapter: MopsAdapter, summary: Callable[[str, str, int], bytes]
) -> None:
    empty = summary(INCOME, "sii", 4)
    for report in (INCOME, BALANCE, CASH_FLOW):
        for market in ("sii", "otc"):
            serve(report, market, 3, empty)
            serve(report, market, 2, summary(report, market, 2))
    run = scan_at(adapter, "2026-11-05 19:00:00")
    assert (run.status, run.rows) == ("success", 0)
    assert not FinancialStatement.objects.exists()


@pytest.mark.django_db
@responses.activate
def test_a_broken_summary_is_kept_as_raw_before_failing(adapter: MopsAdapter) -> None:
    serve(INCOME, "sii", 3, b"<html><body>FOR SECURITY REASONS</body></html>")
    with pytest.raises(SchemaChangedError):
        scan_at(adapter, "2026-11-05 19:00:00")
    assert "SECURITY" in RawMops.objects.get(dataset=STATEMENT_SCAN).payload["html"]
    assert EtlRun.objects.get(dataset=STATEMENT_SCAN).status == "failed"


@pytest.mark.django_db
@responses.activate
def test_a_company_in_both_markets_fails(
    adapter: MopsAdapter, summary: Callable[[str, str, int], bytes]
) -> None:
    serve(INCOME, "sii", 3, summary(INCOME, "sii", 3))
    serve(INCOME, "otc", 3, summary(INCOME, "otc", 3).replace(b">6026<", b">1215<"))
    with pytest.raises(SchemaChangedError, match="twice"):
        scan_at(adapter, "2026-11-05 19:00:00")


def test_a_row_without_any_figure_is_dropped(adapter: MopsAdapter) -> None:
    rows = pd.DataFrame(
        [
            {"stock_id": "1215", "period": "2026-Q3", "fin_type": CONSOLIDATED, "eps": None},
            {"stock_id": "1232", "period": "2026-Q3", "fin_type": CONSOLIDATED, "eps": Decimal(1)},
        ],
        columns=SCAN_COLUMNS,
        dtype=object,
    )
    assert adapter.normalize(STATEMENT_SCAN, rows)["stock_id"].tolist() == ["1232"]


@pytest.mark.django_db
def test_a_scan_only_sees_today(adapter: MopsAdapter) -> None:
    with time_machine.travel(datetime(2026, 11, 6, 19, 0, tzinfo=TAIPEI), tick=False):
        with pytest.raises(ValueError, match="only today"):
            adapter.run(STATEMENT_SCAN, SCAN_DAY, SCAN_DAY)
        with pytest.raises(ValueError, match="whole market"):
            adapter.fetch(STATEMENT_SCAN, SCAN_DAY, SCAN_DAY, "1215")


@pytest.mark.django_db
@responses.activate
def test_an_empty_previous_quarter_fails_instead_of_blanking_the_flows(
    adapter: MopsAdapter, summary: Callable[..., bytes]
) -> None:
    empty = summary(INCOME, "sii", 4)
    for report in (INCOME, BALANCE, CASH_FLOW):
        for market in ("sii", "otc"):
            serve(report, market, 3, summary(report, market, 3))
            serve(report, market, 2, empty if report == CASH_FLOW else summary(report, market, 2))
    with pytest.raises(SchemaChangedError, match="came back empty"):
        scan_at(adapter, "2026-11-05 19:00:00")
    assert not FinancialStatement.objects.exists()


@pytest.mark.django_db
@responses.activate
def test_early_april_scans_the_annual_report_and_the_first_quarter_once_each(
    adapter: MopsAdapter, summary: Callable[..., bytes]
) -> None:
    for report in (INCOME, BALANCE, CASH_FLOW):
        for market in ("sii", "otc"):
            serve(report, market, 1, summary(report, market, 1))
            for season in (4, 3):
                serve(report, market, season, summary(report, market, season, 114), 114)
    day = date(2026, 4, 2)
    with time_machine.travel(datetime(2026, 4, 2, 19, 0, tzinfo=TAIPEI), tick=False):
        adapter.run(STATEMENT_SCAN, day, day)
    assert len(responses.calls) == 18
    periods = set(FinancialStatement.objects.values_list("period", flat=True))
    assert periods == {"2025-Q4", "2026-Q1"}
    q4 = FinancialStatement.objects.get(stock_id="2330", period="2025-Q4")
    assert (q4.eps, q4.announce_date) == (Decimal("19.5100"), day)


@pytest.mark.django_db
@responses.activate
def test_a_page_differing_only_in_its_export_name_is_stored_once(
    adapter: MopsAdapter, summary: Callable[..., bytes]
) -> None:
    for name in (b"t163_20261105_190000001.csv", b"t163_20261105_224000002.csv"):
        for report in (INCOME, BALANCE, CASH_FLOW):
            for market in ("sii", "otc"):
                for season in (3, 2):
                    page = summary(report, market, season)
                    renamed = re.sub(
                        rb"name='filename' value='[^']*'",
                        b"name='filename' value='" + name + b"'",
                        page,
                    )
                    serve(report, market, season, renamed)
    scan_at(adapter, "2026-11-05 19:00:00")
    scan_at(adapter, "2026-11-05 22:40:00")
    assert len(responses.calls) == 24
    assert RawMops.objects.filter(dataset=STATEMENT_SCAN).count() == 12


@pytest.mark.django_db
@pytest.mark.usefixtures("published")
def test_rows_written_after_the_evening_check_count_for_the_next_session(
    adapter: MopsAdapter,
) -> None:
    with time_machine.travel(datetime(2026, 11, 5, 22, 40, tzinfo=TAIPEI), tick=False):
        frame = adapter.normalize(STATEMENT_SCAN, adapter.fetch(STATEMENT_SCAN, SCAN_DAY, SCAN_DAY))
    with time_machine.travel(datetime(2026, 11, 5, 22, 55, tzinfo=TAIPEI), tick=False):
        adapter.upsert(STATEMENT_SCAN, frame)
    assert {row.announce_date for row in stored().values()} == {date(2026, 11, 6)}


@pytest.mark.django_db
@responses.activate
def test_a_company_missing_from_one_summary_waits_and_is_named(
    adapter: MopsAdapter, summary: Callable[..., bytes], caplog: pytest.LogCaptureFixture
) -> None:
    rows = re.compile(rb"<tr class='(?:even|odd)'>(?:(?!</tr>).)*?>1215<.*?</tr>", re.S)
    for report in (INCOME, BALANCE, CASH_FLOW):
        for market in ("sii", "otc"):
            page = summary(report, market, 3)
            if (report, market) == (BALANCE, "sii"):
                page = rows.sub(b"", page)
                assert b">1215<" not in page
            serve(report, market, 3, page)
            serve(report, market, 2, summary(report, market, 2))
    scan_at(adapter, "2026-11-05 19:00:00")
    assert set(stored()) == FILERS - {"1215"}
    assert any("2026-Q3 scan waits for 1215" in row.getMessage() for row in caplog.records)


@pytest.mark.django_db
@responses.activate
@pytest.mark.parametrize(
    ("day", "empty"),
    [(date(2026, 4, 2), (BALANCE, "sii", 4, 114)), (date(2026, 7, 2), (INCOME, "sii", 1, 115))],
)
def test_a_quarter_past_its_deadline_cannot_come_back_empty_even_when_still_open(
    adapter: MopsAdapter,
    summary: Callable[..., bytes],
    day: date,
    empty: tuple[str, str, int, int],
) -> None:
    pages = {(1, 115), (4, 114), (3, 114)} if day.month == 4 else {(2, 115), (1, 115), (4, 114)}
    nothing = summary(INCOME, "sii", 4)
    for report in (INCOME, BALANCE, CASH_FLOW):
        for market in ("sii", "otc"):
            for season, roc in pages:
                body = nothing if (report, market, season, roc) == empty else None
                serve(report, market, season, body or summary(report, market, season, roc), roc)
    with (
        time_machine.travel(
            datetime.combine(day, datetime.min.time(), TAIPEI).replace(hour=19), tick=False
        ),
        pytest.raises(SchemaChangedError, match="came back empty"),
    ):
        adapter.run(STATEMENT_SCAN, day, day)
    assert not FinancialStatement.objects.exists()
