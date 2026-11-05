from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
import responses
from django.conf import settings
from django.core.management import call_command
from django.core.management.base import CommandError
from responses import matchers

from apps.etl.adapters import ADAPTERS
from apps.etl.adapters.mops import STATEMENT_FILING, MopsAdapter, parse_listing
from apps.etl.errors import SchemaChangedError
from apps.etl.filings import FILED_LATER_CHECK, filing_session
from apps.etl.models import DataQualityLog, EtlRun, RawMops
from apps.etl.ratelimit import RateLimit, RateLimiter
from apps.market.calendar import TradingDays
from apps.market.models import FinancialStatement, FinType, Market, StockMaster, TradingCalendar

URL = "https://doc.twse.com.tw/server-java/t57sb01"
TAIPEI = ZoneInfo("Asia/Taipei")
LIMIT = RateLimit(min_interval=3.5, max_retries=2, backoff_base=1.0, backoff_max=2.0, cooldown=5.0)
CONSOLIDATED = FinType.CONSOLIDATED.value
INDIVIDUAL = FinType.INDIVIDUAL.value


def weekdays(start: date, end: date) -> TradingDays:
    span = (start + timedelta(days=offset) for offset in range((end - start).days + 1))
    return TradingDays([day for day in span if day.weekday() < 5], start, end)


def at(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=TAIPEI)


@pytest.fixture
def page(fixture_path: Callable[[str], Path]) -> Callable[[str], bytes]:
    return lambda name: fixture_path(f"mops/t57sb01_{name}_A.html").read_bytes()


def text(body: bytes) -> str:
    return body.decode("cp950", errors="replace")


def serve(code: str, roc_year: int, body: bytes) -> None:
    params = {
        "step": "1",
        "colorchg": "1",
        "co_id": code,
        "year": str(roc_year),
        "seamon": "",
        "mtype": "A",
    }
    responses.get(URL, body=body, match=[matchers.query_param_matcher(params)])


@pytest.fixture
def calendar() -> TradingDays:
    return weekdays(date(2009, 1, 1), date(2026, 12, 31))


@pytest.fixture
def adapter(calendar: TradingDays, fake_sleep: Callable[[float], None]) -> MopsAdapter:
    return MopsAdapter(calendar, RateLimiter(LIMIT, sleep=fake_sleep))


def statement(code: str, period: str, fin_type: str, announced: date) -> FinancialStatement:
    return FinancialStatement.objects.create(
        stock_id=code, period=period, fin_type=fin_type, eps=Decimal("1"), announce_date=announced
    )


def announced(code: str) -> dict[tuple[str, str], date]:
    rows = FinancialStatement.objects.filter(stock_id=code)
    return {(row.period, row.fin_type): row.announce_date for row in rows}


def test_ifrs_reports_keep_the_chinese_uploads_by_statement_type(
    page: Callable[[str], bytes],
) -> None:
    filings = parse_listing(text(page("2330_113")), "2330", 113)
    found = {(row.period, row.fin_type): row.uploaded_at for row in filings}
    assert found[("2024-Q1", CONSOLIDATED)] == at("2024-05-15 14:25:29")
    assert found[("2024-Q3", CONSOLIDATED)] == at("2024-11-14 18:03:20")
    assert found[("2024-Q4", CONSOLIDATED)] == at("2025-02-27 14:01:14")
    assert found[("2024-Q4", INDIVIDUAL)] == at("2025-02-27 14:03:03")
    assert len(filings) == 5
    assert not any(row.corrected for row in filings)


def test_english_versions_filed_weeks_later_are_not_the_announcement(
    page: Callable[[str], bytes],
) -> None:
    filings = parse_listing(text(page("5483_113")), "5483", 113)
    q1 = [row for row in filings if row.period == "2024-Q1"]
    assert [(row.fin_type, row.uploaded_at) for row in q1] == [
        (CONSOLIDATED, at("2024-05-10 19:07:20"))
    ]
    assert q1[0].corrected
    q4 = {row.fin_type: row.corrected for row in filings if row.period == "2024-Q4"}
    assert q4 == {CONSOLIDATED: False, INDIVIDUAL: False}


def test_the_old_format_maps_parent_and_group_reports(page: Callable[[str], bytes]) -> None:
    filings = parse_listing(text(page("2330_99")), "2330", 99)
    found = {(row.period, row.fin_type): row.uploaded_at for row in filings}
    assert found[("2010-Q1", INDIVIDUAL)] == at("2010-04-29 10:58:10")
    assert found[("2010-Q1", CONSOLIDATED)] == at("2010-04-29 10:59:26")
    assert {period for period, _ in found} == {"2010-Q1", "2010-Q2", "2010-Q3", "2010-Q4"}


def test_a_delisted_company_keeps_its_reports(page: Callable[[str], bytes]) -> None:
    filings = parse_listing(text(page("1606_100")), "1606", 100)
    found = {(row.period, row.fin_type): row.uploaded_at for row in filings}
    assert found == {
        ("2011-Q2", INDIVIDUAL): at("2011-08-30 18:32:16"),
        ("2011-Q2", CONSOLIDATED): at("2011-08-30 18:33:26"),
        ("2011-Q4", INDIVIDUAL): at("2012-04-30 16:13:53"),
        ("2011-Q4", CONSOLIDATED): at("2012-04-28 17:07:45"),
    }


def test_no_reports_is_an_empty_answer(page: Callable[[str], bytes]) -> None:
    assert parse_listing(text(page("2330_80")), "2330", 80) == []


def test_a_page_without_the_report_table_fails(page: Callable[[str], bytes]) -> None:
    with pytest.raises(SchemaChangedError, match="report table"):
        parse_listing(text(page("blank_113")), "", 113)


@pytest.mark.parametrize(("code", "roc_year"), [("2303", 113), ("2330", 112)])
def test_rows_of_another_company_or_year_fail(
    page: Callable[[str], bytes], code: str, roc_year: int
) -> None:
    with pytest.raises(SchemaChangedError):
        parse_listing(text(page("2330_113")), code, roc_year)


def test_an_unreadable_upload_time_fails(page: Callable[[str], bytes]) -> None:
    broken = text(page("2330_113")).replace("113/05/15 14:25:29", "113/05/15")
    with pytest.raises(SchemaChangedError, match="upload"):
        parse_listing(broken, "2330", 113)


@pytest.mark.parametrize(
    ("uploaded", "session"),
    [
        ("2024-11-14 18:03:20", date(2024, 11, 14)),
        ("2024-11-14 23:10:00", date(2024, 11, 15)),
        ("2024-11-16 10:00:00", date(2024, 11, 18)),
    ],
)
def test_a_filing_counts_from_the_session_whose_evening_run_can_see_it(
    calendar: TradingDays, uploaded: str, session: date
) -> None:
    assert filing_session(at(uploaded), calendar, time(23, 0)) == session


@pytest.fixture
def listed(db: None) -> None:
    StockMaster.objects.bulk_create(
        [
            StockMaster(stock_id="2330", name="台積電", market=Market.TWSE),
            StockMaster(stock_id="0050", name="元大台灣50", market=Market.TWSE),
        ]
    )


@pytest.mark.usefixtures("listed")
@responses.activate
def test_actual_dates_only_move_announcements_earlier_and_log_the_later_ones(
    adapter: MopsAdapter, page: Callable[[str], bytes]
) -> None:
    serve("2330", 113, page("2330_113"))
    statement("2330", "2024-Q1", CONSOLIDATED, date(2024, 6, 3))
    statement("2330", "2024-Q3", CONSOLIDATED, date(2024, 11, 14))
    statement("2330", "2024-Q4", CONSOLIDATED, date(2025, 2, 25))
    statement("2330", "2024-Q4", INDIVIDUAL, date(2025, 4, 1))
    statement("2330", "2024-Q2", INDIVIDUAL, date(2024, 9, 2))
    raw = adapter.fetch(STATEMENT_FILING, date(2024, 1, 1), date(2024, 12, 31), "2330")
    written = adapter.upsert(STATEMENT_FILING, adapter.normalize(STATEMENT_FILING, raw))
    assert announced("2330") == {
        ("2024-Q1", CONSOLIDATED): date(2024, 5, 15),
        ("2024-Q3", CONSOLIDATED): date(2024, 11, 14),
        ("2024-Q4", CONSOLIDATED): date(2025, 2, 25),
        ("2024-Q4", INDIVIDUAL): date(2025, 2, 27),
        ("2024-Q2", INDIVIDUAL): date(2024, 8, 14),
    }
    assert written == 3
    log = DataQualityLog.objects.get(check_name=FILED_LATER_CHECK)
    assert log.severity == "warning"
    assert log.detail == {
        "stock_id": "2330",
        "filed_later": [
            {
                "period": "2024-Q4",
                "fin_type": CONSOLIDATED,
                "announce_date": "2025-02-25",
                "filed": "2025-02-27",
                "corrected": False,
            }
        ],
    }
    assert RawMops.objects.get().request_params["co_id"] == "2330"


@pytest.mark.usefixtures("listed")
@responses.activate
def test_a_rerun_changes_nothing_and_keeps_one_log_row(
    adapter: MopsAdapter, page: Callable[[str], bytes]
) -> None:
    serve("2330", 113, page("2330_113"))
    statement("2330", "2024-Q4", CONSOLIDATED, date(2025, 2, 25))
    statement("2330", "2024-Q1", CONSOLIDATED, date(2024, 6, 3))

    def load() -> int:
        raw = adapter.fetch(STATEMENT_FILING, date(2024, 1, 1), date(2024, 12, 31), "2330")
        return adapter.upsert(STATEMENT_FILING, adapter.normalize(STATEMENT_FILING, raw))

    assert load() == 1
    assert load() == 0
    assert DataQualityLog.objects.filter(check_name=FILED_LATER_CHECK).count() == 1
    assert RawMops.objects.count() == 1


@pytest.mark.usefixtures("listed")
@responses.activate
def test_only_years_with_statements_are_asked_for(
    adapter: MopsAdapter, page: Callable[[str], bytes]
) -> None:
    serve("2330", 113, page("2330_113"))
    statement("2330", "2024-Q1", CONSOLIDATED, date(2024, 6, 3))
    statement("2330", "2008-Q1", CONSOLIDATED, date(2008, 6, 2))
    adapter.fetch(STATEMENT_FILING, date(2009, 1, 1), date(2026, 11, 4), "2330")
    assert adapter.fetch(STATEMENT_FILING, date(2009, 1, 1), date(2026, 11, 4), "0050").empty
    assert len(responses.calls) == 1


@pytest.mark.usefixtures("listed")
@responses.activate
def test_a_broken_page_is_kept_as_raw_before_failing(adapter: MopsAdapter) -> None:
    serve("2330", 113, "<html><body>系統維護中</body></html>".encode("cp950"))
    statement("2330", "2024-Q1", CONSOLIDATED, date(2024, 6, 3))
    with pytest.raises(SchemaChangedError):
        adapter.fetch(STATEMENT_FILING, date(2024, 1, 1), date(2024, 12, 31), "2330")
    assert "系統維護中" in RawMops.objects.get().payload["html"]


def test_a_dataset_run_needs_a_stock(adapter: MopsAdapter) -> None:
    with pytest.raises(ValueError, match="stock by stock"):
        adapter.run(STATEMENT_FILING, date(2024, 1, 1), date(2024, 12, 31))


def test_normalize_keeps_the_first_upload_of_each_report_type(
    adapter: MopsAdapter,
) -> None:
    frame = pd.DataFrame(
        [
            ("2330", "2010-Q1", INDIVIDUAL, at("2010-04-20 10:00:00"), False),
            ("2330", "2010-Q1", INDIVIDUAL, at("2010-04-29 10:00:00"), False),
            ("2330", "2010-Q2", CONSOLIDATED, at("2010-08-27 10:00:00"), True),
        ],
        columns=["stock_id", "period", "fin_type", "uploaded_at", "corrected"],
    )
    normalized = adapter.normalize(STATEMENT_FILING, frame)
    assert normalized.to_dict("records") == [
        {
            "stock_id": "2330",
            "period": "2010-Q1",
            "fin_type": INDIVIDUAL,
            "uploaded_at": at("2010-04-20 10:00:00"),
            "corrected": False,
        },
        {
            "stock_id": "2330",
            "period": "2010-Q2",
            "fin_type": CONSOLIDATED,
            "uploaded_at": at("2010-08-27 10:00:00"),
            "corrected": True,
        },
    ]


@pytest.mark.django_db
def test_a_statement_without_its_report_type_uses_the_other_report(
    adapter: MopsAdapter,
) -> None:
    statement("2330", "2010-Q1", CONSOLIDATED, date(2010, 6, 1))
    statement("2330", "2010-Q2", CONSOLIDATED, date(2010, 9, 1))
    frame = pd.DataFrame(
        [("2330", "2010-Q1", INDIVIDUAL, at("2010-04-20 10:00:00"), False)],
        columns=["stock_id", "period", "fin_type", "uploaded_at", "corrected"],
    )
    assert adapter.upsert(STATEMENT_FILING, frame) == 1
    assert announced("2330") == {
        ("2010-Q1", CONSOLIDATED): date(2010, 4, 20),
        ("2010-Q2", CONSOLIDATED): date(2010, 9, 1),
    }


@pytest.fixture
def calendar_in_db(db: None, calendar: TradingDays) -> None:
    TradingCalendar.objects.bulk_create(
        TradingCalendar(date=day, is_trading_day=is_open) for day, is_open in calendar.all_days()
    )


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)


@pytest.mark.usefixtures("listed", "calendar_in_db", "no_sleep")
@responses.activate
def test_the_backfill_command_walks_stock_by_stock(page: Callable[[str], bytes]) -> None:
    serve("2330", 113, page("2330_113"))
    statement("2330", "2024-Q1", CONSOLIDATED, date(2024, 6, 3))
    call_command(
        "backfill",
        "--source",
        "mops",
        "--dataset",
        STATEMENT_FILING,
        "--from",
        "2024-01-01",
        "--to",
        "2024-12-31",
    )
    run = EtlRun.objects.get(source="mops")
    assert (run.status, run.rows, run.checkpoint["last_stock_id"]) == ("success", 1, "2330")
    assert announced("2330") == {("2024-Q1", CONSOLIDATED): date(2024, 5, 15)}


@pytest.mark.usefixtures("calendar_in_db")
def test_a_single_dataset_fetch_is_refused() -> None:
    with pytest.raises(CommandError, match="stock by stock"):
        call_command("fetch", source="mops", dataset=STATEMENT_FILING, date="2024-11-14")


def test_the_source_is_registered_with_its_own_limits() -> None:
    assert ADAPTERS["mops"] is MopsAdapter
    assert settings.ETL_RATE_LIMITS["mops"]["min_interval"] >= 3.0
    assert settings.FILING_VISIBLE_BEFORE == "23:00"
