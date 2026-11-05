import io
from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
import responses
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.core.management import call_command
from django.core.management.base import CommandError
from responses import matchers

from apps.etl.adapters import ADAPTERS
from apps.etl.adapters.mops import (
    MOPS_MIN_INTERVAL,
    STATEMENT_FILING,
    MopsAdapter,
    parse_listing,
)
from apps.etl.errors import SchemaChangedError
from apps.etl.filings import (
    FILED_LATE_CHECK,
    FILED_LATER_CHECK,
    FILING_COLUMNS,
    RESTATED_CHECK,
    filing_session,
)
from apps.etl.models import DataQualityLog, EtlRun, RawMops
from apps.etl.ratelimit import RateLimit, RateLimiter, RetryableError
from apps.market.calendar import TradingDays
from apps.market.models import FinancialStatement, FinType, Market, StockMaster, TradingCalendar
from tests.etl.conftest import RecordingSleep

URL = "https://doc.twse.com.tw/server-java/t57sb01"
TAIPEI = ZoneInfo("Asia/Taipei")
LIMIT = RateLimit(min_interval=6.0, max_retries=2, backoff_base=1.0, backoff_max=2.0, cooldown=5.0)
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


def test_a_report_table_wins_over_a_stray_no_data_note(page: Callable[[str], bytes]) -> None:
    noted = text(page("2330_113")).replace("</body>", "查無所需資料</body>")
    assert len(parse_listing(noted, "2330", 113)) == 5


def test_a_row_with_another_cell_count_fails(page: Callable[[str], bytes]) -> None:
    trimmed = text(page("2330_113")).replace("<td align='center'>無</td>", "", 1)
    assert trimmed != text(page("2330_113"))
    with pytest.raises(SchemaChangedError, match="cells"):
        parse_listing(trimmed, "2330", 113)


def test_a_company_off_the_calendar_year_fails(page: Callable[[str], bytes]) -> None:
    shifted = text(page("2330_113")).replace("曆年制", "非曆年制（7月制）", 1)
    with pytest.raises(SchemaChangedError, match="fiscal year"):
        parse_listing(shifted, "2330", 113)


@pytest.mark.parametrize(("code", "roc_year"), [("2303", 113), ("2330", 112)])
def test_rows_of_another_company_or_year_fail(
    page: Callable[[str], bytes], code: str, roc_year: int
) -> None:
    with pytest.raises(SchemaChangedError):
        parse_listing(text(page("2330_113")), code, roc_year)


@pytest.mark.parametrize("shown", ["113/05/15", "113/02/30 14:25:29"])
def test_an_unreadable_upload_time_fails(page: Callable[[str], bytes], shown: str) -> None:
    broken = text(page("2330_113")).replace("113/05/15 14:25:29", shown)
    with pytest.raises(SchemaChangedError, match="upload"):
        parse_listing(broken, "2330", 113)


def test_restated_reports_are_marked_with_their_own_upload(page: Callable[[str], bytes]) -> None:
    found = parse_listing(text(page("1213_107")), "1213", 107)
    q4 = [(f.fin_type, f.uploaded_at, f.restated) for f in found if f.period == "2018-Q4"]
    assert q4 == [
        (CONSOLIDATED, at("2019-04-08 18:18:18"), False),
        (INDIVIDUAL, at("2019-04-08 18:20:07"), False),
        (CONSOLIDATED, at("2019-09-18 10:38:45"), True),
        (INDIVIDUAL, at("2019-09-18 10:40:57"), True),
    ]
    assert not any(f.restated for f in found if f.period != "2018-Q4")


def test_old_restatements_named_without_brackets_are_restated(page: Callable[[str], bytes]) -> None:
    found = parse_listing(text(page("1310_98")), "1310", 98)
    q1 = [(f.fin_type, f.uploaded_at, f.restated) for f in found if f.period == "2009-Q1"]
    assert q1 == [
        (INDIVIDUAL, at("2012-01-04 21:36:43"), False),
        (CONSOLIDATED, at("2012-01-04 21:37:30"), False),
        (INDIVIDUAL, at("2012-01-07 14:42:13"), True),
        (CONSOLIDATED, at("2012-01-07 14:43:06"), True),
    ]
    assert {f.period for f in found if f.restated} == {"2009-Q1", "2009-Q2"}


@pytest.mark.django_db
def test_a_restated_period_is_announced_by_its_restatement_and_logged(
    adapter: MopsAdapter, page: Callable[[str], bytes]
) -> None:
    statement("1213", "2018-Q3", CONSOLIDATED, date(2018, 12, 3))
    statement("1213", "2018-Q4", CONSOLIDATED, date(2019, 4, 2))
    listing = pd.DataFrame(parse_listing(text(page("1213_107")), "1213", 107))
    normalized = adapter.normalize(STATEMENT_FILING, listing)
    q4 = normalized[normalized["period"] == "2018-Q4"]
    assert q4[["fin_type", "uploaded_at", "restated"]].values.tolist() == [
        [CONSOLIDATED, at("2019-09-18 10:38:45"), True],
        [INDIVIDUAL, at("2019-09-18 10:40:57"), True],
    ]
    adapter.upsert(STATEMENT_FILING, normalized)
    assert announced("1213") == {
        ("2018-Q3", CONSOLIDATED): date(2018, 11, 15),
        ("2018-Q4", CONSOLIDATED): date(2019, 9, 18),
    }
    assert logged(RESTATED_CHECK) == {
        "1213": [("2018-Q4", CONSOLIDATED, "2019-04-02", "2019-09-18")]
    }
    assert FILED_LATE_CHECK not in {row.check_name for row in DataQualityLog.objects.all()}


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("uploaded", "session"),
    [("2025-09-10 10:00:00", date(2025, 9, 10)), ("2025-03-20 10:00:00", date(2025, 3, 20))],
)
def test_a_restatement_sets_the_date_even_when_corrected_or_early(
    adapter: MopsAdapter, uploaded: str, session: date
) -> None:
    statement("2330", "2024-Q4", CONSOLIDATED, date(2025, 4, 1))
    frame = pd.DataFrame(
        [("2330", "2024-Q4", CONSOLIDATED, at(uploaded), True, True)], columns=FILING_COLUMNS
    )
    adapter.upsert(STATEMENT_FILING, frame)
    assert announced("2330") == {("2024-Q4", CONSOLIDATED): session}
    assert logged(RESTATED_CHECK) == {
        "2330": [("2024-Q4", CONSOLIDATED, "2025-04-01", session.isoformat())]
    }


@pytest.mark.parametrize(
    "supplement", ["關係企業合併報表", "關係企業合併報表重編", "關係企業合併報表(重編)"]
)
def test_the_affiliates_report_is_not_an_announcement(
    page: Callable[[str], bytes], supplement: str
) -> None:
    listing = text(page("2330_99"))
    assert listing.count("英文版-母公司財報") == 4
    supplemented = listing.replace("英文版-母公司財報", supplement, 1)
    assert parse_listing(supplemented, "2330", 99) == parse_listing(listing, "2330", 99)


def test_companies_without_subsidiaries_file_individual_reports(
    page: Callable[[str], bytes],
) -> None:
    found = parse_listing(text(page("2820_113")), "2820", 113)
    assert [(f.period, f.fin_type) for f in found] == [
        (f"2024-Q{quarter}", INDIVIDUAL) for quarter in (1, 2, 3, 4)
    ]


@pytest.mark.parametrize(
    ("old", "new"),
    [("財務報告書</td>", "股東會年報</td>"), ("IFRSs個體財報", "IFRSs簡明財報")],
)
def test_an_unknown_kind_of_report_fails(page: Callable[[str], bytes], old: str, new: str) -> None:
    listing = text(page("2330_113"))
    assert old in listing
    with pytest.raises(SchemaChangedError, match="unknown report"):
        parse_listing(listing.replace(old, new, 1), "2330", 113)


@pytest.mark.parametrize(
    ("uploaded", "session"),
    [
        ("2024-11-14 18:03:20", date(2024, 11, 14)),
        ("2024-11-14 23:10:00", date(2024, 11, 15)),
        ("2024-11-16 10:00:00", date(2024, 11, 18)),
        ("2024-11-14 22:40:00", date(2024, 11, 15)),
        ("2024-11-14 22:39:59", date(2024, 11, 14)),
    ],
)
def test_a_filing_counts_from_the_session_whose_evening_run_can_see_it(
    calendar: TradingDays, uploaded: str, session: date
) -> None:
    assert filing_session(at(uploaded), calendar, time(22, 40)) == session


@pytest.fixture
def listed(db: None) -> None:
    StockMaster.objects.bulk_create(
        [
            StockMaster(stock_id="2330", name="台積電", market=Market.TWSE),
            StockMaster(stock_id="0050", name="元大台灣50", market=Market.TWSE),
        ]
    )


def filings(*rows: tuple[str, str, str, str, bool]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            (code, period, kind, at(uploaded), corrected, False)
            for code, period, kind, uploaded, corrected in rows
        ],
        columns=FILING_COLUMNS,
    )


def logged(check: str) -> dict[str, list[tuple[str, str, str, str]]]:
    return {
        row.detail["stock_id"]: [
            (entry["period"], entry["fin_type"], entry["deadline"], entry["filed"])
            for entry in row.detail["filings"]
        ]
        for row in DataQualityLog.objects.filter(check_name=check)
    }


@pytest.mark.usefixtures("listed")
@responses.activate
def test_filings_on_time_set_the_announcement_either_way(
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
        ("2024-Q4", CONSOLIDATED): date(2025, 2, 27),
        ("2024-Q4", INDIVIDUAL): date(2025, 2, 27),
        ("2024-Q2", INDIVIDUAL): date(2024, 8, 14),
    }
    assert written == 4
    assert not DataQualityLog.objects.exists()
    assert RawMops.objects.get().request_params["co_id"] == "2330"


@pytest.mark.usefixtures("listed")
@responses.activate
def test_a_late_uncorrected_report_moves_the_announcement_later(
    adapter: MopsAdapter, page: Callable[[str], bytes]
) -> None:
    serve("1606", 100, page("1606_100"))
    statement("1606", "2011-Q4", CONSOLIDATED, date(2012, 4, 3))
    statement("1606", "2011-Q4", INDIVIDUAL, date(2012, 4, 3))

    def load() -> int:
        raw = adapter.fetch(STATEMENT_FILING, date(2011, 1, 1), date(2011, 12, 31), "1606")
        return adapter.upsert(STATEMENT_FILING, adapter.normalize(STATEMENT_FILING, raw))

    assert load() == 2
    assert announced("1606") == {
        ("2011-Q4", CONSOLIDATED): date(2012, 4, 30),
        ("2011-Q4", INDIVIDUAL): date(2012, 4, 30),
    }
    late = [
        ("2011-Q4", CONSOLIDATED, "2012-04-03", "2012-04-30"),
        ("2011-Q4", INDIVIDUAL, "2012-04-03", "2012-04-30"),
    ]
    assert logged(FILED_LATE_CHECK) == {"1606": late}
    assert DataQualityLog.objects.get(check_name=FILED_LATE_CHECK).severity == "info"
    assert load() == 0
    assert logged(FILED_LATE_CHECK) == {"1606": late}
    assert RawMops.objects.count() == 1


@pytest.mark.django_db
def test_a_late_corrected_report_keeps_the_deadline_and_warns(adapter: MopsAdapter) -> None:
    statement("5483", "2024-Q4", CONSOLIDATED, date(2025, 4, 1))
    statement("5483", "2024-Q3", CONSOLIDATED, date(2024, 12, 2))
    written = adapter.upsert(
        STATEMENT_FILING,
        filings(
            ("5483", "2024-Q4", CONSOLIDATED, "2025-05-07 08:53:42", True),
            ("5483", "2024-Q3", CONSOLIDATED, "2024-11-11 17:38:10", True),
        ),
    )
    assert written == 1
    assert announced("5483") == {
        ("2024-Q4", CONSOLIDATED): date(2025, 4, 1),
        ("2024-Q3", CONSOLIDATED): date(2024, 11, 11),
    }
    assert logged(FILED_LATER_CHECK) == {
        "5483": [("2024-Q4", CONSOLIDATED, "2025-04-01", "2025-05-07")]
    }
    assert DataQualityLog.objects.get(check_name=FILED_LATER_CHECK).severity == "warning"
    assert not DataQualityLog.objects.filter(check_name=FILED_LATE_CHECK).exists()


@pytest.mark.django_db
def test_each_statement_type_takes_its_own_report(adapter: MopsAdapter) -> None:
    statement("2330", "2010-Q1", CONSOLIDATED, date(2010, 6, 1))
    statement("2330", "2010-Q1", INDIVIDUAL, date(2010, 6, 1))
    adapter.upsert(
        STATEMENT_FILING,
        filings(
            ("2330", "2010-Q1", INDIVIDUAL, "2010-04-20 10:00:00", False),
            ("2330", "2010-Q1", CONSOLIDATED, "2010-04-29 10:00:00", False),
        ),
    )
    assert announced("2330") == {
        ("2010-Q1", CONSOLIDATED): date(2010, 4, 29),
        ("2010-Q1", INDIVIDUAL): date(2010, 4, 20),
    }


@pytest.mark.django_db
def test_each_stock_keeps_its_own_log_row_and_a_narrower_rerun_keeps_the_rest(
    adapter: MopsAdapter,
) -> None:
    for code in ("1606", "3369"):
        statement(code, "2011-Q2", CONSOLIDATED, date(2011, 9, 1))
        statement(code, "2011-Q4", CONSOLIDATED, date(2012, 4, 3))
    both = [
        (code, period, CONSOLIDATED, uploaded, False)
        for code in ("1606", "3369")
        for period, uploaded in (
            ("2011-Q2", "2011-09-05 10:00:00"),
            ("2011-Q4", "2012-04-27 10:00:00"),
        )
    ]
    adapter.upsert(STATEMENT_FILING, filings(*both))
    assert set(logged(FILED_LATE_CHECK)) == {"1606", "3369"}
    adapter.upsert(
        STATEMENT_FILING, filings(("1606", "2011-Q2", CONSOLIDATED, "2011-08-30 10:00:00", False))
    )
    assert logged(FILED_LATE_CHECK)["1606"] == [
        ("2011-Q4", CONSOLIDATED, "2012-04-03", "2012-04-27")
    ]
    assert len(logged(FILED_LATE_CHECK)["3369"]) == 2


@pytest.mark.django_db
@pytest.mark.parametrize(
    "uploaded",
    ["2024-06-14 10:00:00", "2024-06-28 23:30:00", "2024-06-29 10:00:00", "2024-06-30 10:00:00"],
)
def test_a_filing_before_the_quarter_ended_rejects_the_stock(
    adapter: MopsAdapter, uploaded: str
) -> None:
    statement("2330", "2024-Q2", CONSOLIDATED, date(2024, 9, 2))
    with pytest.raises(SchemaChangedError, match="before the quarter ended"):
        adapter.upsert(
            STATEMENT_FILING, filings(("2330", "2024-Q2", CONSOLIDATED, uploaded, False))
        )
    assert announced("2330") == {("2024-Q2", CONSOLIDATED): date(2024, 9, 2)}


@pytest.mark.django_db
def test_a_filing_off_the_calendar_rejects_the_stock(adapter: MopsAdapter) -> None:
    statement("2330", "2026-Q3", CONSOLIDATED, date(2026, 11, 16))
    with pytest.raises(SchemaChangedError, match="off the calendar"):
        adapter.upsert(
            STATEMENT_FILING,
            filings(("2330", "2026-Q3", CONSOLIDATED, "2027-01-05 10:00:00", False)),
        )


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
def test_a_page_asking_to_query_less_is_retried_after_a_pause(
    adapter: MopsAdapter, page: Callable[[str], bytes], fake_sleep: RecordingSleep
) -> None:
    serve("2330", 113, page("throttled"))
    serve("2330", 113, page("2330_113"))
    statement("2330", "2024-Q1", CONSOLIDATED, date(2024, 6, 3))
    found = adapter.fetch(STATEMENT_FILING, date(2024, 1, 1), date(2024, 12, 31), "2330")
    assert len(responses.calls) == 2
    assert len(found) == 5
    assert RawMops.objects.count() == 1
    assert "查詢過量" not in RawMops.objects.get().payload["html"]
    assert LIMIT.backoff(0) in fake_sleep.calls
    assert LIMIT.cooldown not in fake_sleep.calls


@pytest.mark.usefixtures("listed")
@responses.activate
def test_asking_too_often_fails_the_run_once_retries_run_out(
    adapter: MopsAdapter, page: Callable[[str], bytes]
) -> None:
    serve("2330", 113, page("throttled"))
    statement("2330", "2024-Q1", CONSOLIDATED, date(2024, 6, 3))
    with pytest.raises(RetryableError, match="query less"):
        adapter.fetch(STATEMENT_FILING, date(2024, 1, 1), date(2024, 12, 31), "2330")
    assert len(responses.calls) == LIMIT.max_retries + 1
    assert not RawMops.objects.exists()


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


def test_the_last_restatement_dates_the_period(adapter: MopsAdapter) -> None:
    frame = pd.DataFrame(
        [
            ("1213", "2018-Q4", CONSOLIDATED, at("2019-04-08 18:18:18"), False, False),
            ("1213", "2018-Q4", CONSOLIDATED, at("2019-10-02 09:00:00"), False, True),
            ("1213", "2018-Q4", CONSOLIDATED, at("2019-09-18 10:38:45"), False, True),
        ],
        columns=FILING_COLUMNS,
    )
    normalized = adapter.normalize(STATEMENT_FILING, frame)
    assert normalized[["uploaded_at", "restated"]].values.tolist() == [
        [at("2019-10-02 09:00:00"), True]
    ]


def test_normalize_keeps_the_first_upload_of_each_report_type(
    adapter: MopsAdapter,
) -> None:
    frame = pd.DataFrame(
        [
            ("2330", "2010-Q1", INDIVIDUAL, at("2010-04-20 10:00:00"), False, False),
            ("2330", "2010-Q1", INDIVIDUAL, at("2010-04-29 10:00:00"), False, False),
            ("2330", "2010-Q2", CONSOLIDATED, at("2010-08-27 10:00:00"), True, False),
        ],
        columns=FILING_COLUMNS,
    )
    normalized = adapter.normalize(STATEMENT_FILING, frame)
    assert normalized.to_dict("records") == [
        {
            "stock_id": "2330",
            "period": "2010-Q1",
            "fin_type": INDIVIDUAL,
            "uploaded_at": at("2010-04-20 10:00:00"),
            "corrected": False,
            "restated": False,
        },
        {
            "stock_id": "2330",
            "period": "2010-Q2",
            "fin_type": CONSOLIDATED,
            "uploaded_at": at("2010-08-27 10:00:00"),
            "corrected": True,
            "restated": False,
        },
    ]


@pytest.mark.django_db
def test_a_statement_without_its_report_type_uses_the_other_report(
    adapter: MopsAdapter,
) -> None:
    statement("2330", "2010-Q1", CONSOLIDATED, date(2010, 6, 1))
    statement("2330", "2010-Q2", CONSOLIDATED, date(2010, 9, 1))
    frame = pd.DataFrame(
        [("2330", "2010-Q1", INDIVIDUAL, at("2010-04-20 10:00:00"), False, False)],
        columns=FILING_COLUMNS,
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


def test_doc_twse_refuses_requests_closer_than_five_seconds() -> None:
    limiter = RateLimiter(RateLimit(4.9, 1, 1.0, 1.0, 1.0))
    with pytest.raises(ImproperlyConfigured, match="at least 5 seconds"):
        MopsAdapter(weekdays(date(2024, 1, 1), date(2024, 12, 31)), limiter)


def test_the_source_is_registered_with_its_own_limits() -> None:
    assert ADAPTERS["mops"] is MopsAdapter
    assert settings.ETL_RATE_LIMITS["mops"]["min_interval"] >= MOPS_MIN_INTERVAL
    assert settings.FILING_VISIBLE_BEFORE == "22:40"


@pytest.mark.usefixtures("listed", "calendar_in_db")
@responses.activate
def test_reapply_reruns_the_rules_on_the_latest_stored_page(
    adapter: MopsAdapter, page: Callable[[str], bytes]
) -> None:
    serve("2330", 113, page("2330_113"))
    statement("2330", "2024-Q1", CONSOLIDATED, date(2024, 6, 3))
    adapter.fetch(STATEMENT_FILING, date(2024, 1, 1), date(2024, 12, 31), "2330")
    newer = text(page("2330_113")).replace("113/05/15 14:25:29", "113/05/16 09:00:00")
    RawMops.objects.create(
        dataset=STATEMENT_FILING,
        request_params={"co_id": "2330", "year": "113", "mtype": "A"},
        payload={"html": newer},
        source_hash="newer",
    )
    RawMops.objects.create(
        dataset=STATEMENT_FILING,
        request_params={"co_id": "1606", "year": "100", "mtype": "A"},
        payload={"html": "<html>維護中</html>"},
        source_hash="broken",
    )
    out, err = io.StringIO(), io.StringIO()
    call_command("reapply_filings", stdout=out, stderr=err)
    assert announced("2330") == {("2024-Q1", CONSOLIDATED): date(2024, 5, 16)}
    assert "1 statements changed across 2 stocks" in out.getvalue()
    assert "rejected 1606" in err.getvalue()
    assert len(responses.calls) == 1


@pytest.mark.usefixtures("listed", "calendar_in_db")
@responses.activate
def test_reapply_takes_the_page_seen_last_even_when_it_was_seen_before(
    adapter: MopsAdapter, page: Callable[[str], bytes]
) -> None:
    statement("2330", "2024-Q1", CONSOLIDATED, date(2024, 6, 3))
    serve("2330", 113, page("2330_113"))
    serve("2330", 113, "<html><body>系統維護中</body></html>".encode("cp950"))
    serve("2330", 113, page("2330_113"))
    adapter.fetch(STATEMENT_FILING, date(2024, 1, 1), date(2024, 12, 31), "2330")
    with pytest.raises(SchemaChangedError):
        adapter.fetch(STATEMENT_FILING, date(2024, 1, 1), date(2024, 12, 31), "2330")
    adapter.fetch(STATEMENT_FILING, date(2024, 1, 1), date(2024, 12, 31), "2330")
    out, err = io.StringIO(), io.StringIO()
    call_command("reapply_filings", stdout=out, stderr=err)
    assert RawMops.objects.count() == 2
    assert err.getvalue() == ""
    assert announced("2330") == {("2024-Q1", CONSOLIDATED): date(2024, 5, 15)}
