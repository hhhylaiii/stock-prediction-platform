import re
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import responses
import time_machine
from django.conf import settings

from apps.etl.adapters.mops import REVENUE_SCAN, MopsAdapter
from apps.etl.errors import SchemaChangedError
from apps.etl.models import DataQualityLog, EtlRun, RawMops
from apps.etl.ratelimit import RateLimit, RateLimiter
from apps.etl.revenue_scan import (
    REVENUE_MISMATCH_CHECK,
    Figures,
    open_months,
    parse_revenue_page,
    revenue_rows,
)
from apps.market.calendar import TradingDays
from apps.market.models import MonthlyRevenue

TAIPEI = ZoneInfo("Asia/Taipei")
LIMIT = RateLimit(min_interval=6.0, max_retries=2, backoff_base=1.0, backoff_max=2.0, cooldown=5.0)
SCAN_DAY = date(2026, 11, 5)
OCTOBER_FILERS = {"1414", "1452", "1477", "1256", "1590", "5225", "2070", "4406", "4561"}
FINMIND_2026_09 = {
    "1101": (14321556000, 0.07626848357922, 0.05963671135746473),
    "2330": (511857399000, 0.5464861207105232, -0.00572631592589723),
    "1256": (561575000, 0.33804863985208367, -0.13451787602218046),
    "8069": (3613296000, 0.13537650846120863, 0.048504675663394003),
    "6488": (5128632000, -0.13126055194406472, 0.07645702059225967),
}
PAGES = {("sii", 0), ("sii", 1), ("otc", 0), ("otc", 1)}


@pytest.mark.parametrize(
    ("day", "days_after", "months"),
    [
        (date(2026, 11, 5), 20, ["2026-10"]),
        (date(2026, 12, 3), 20, ["2026-10", "2026-11"]),
        (date(2026, 12, 6), 20, ["2026-11"]),
        (date(2026, 12, 31), 10, []),
        (date(2026, 3, 7), 20, ["2026-01", "2026-02"]),
        (date(2026, 1, 2), 25, ["2025-11", "2025-12"]),
        (date(2026, 1, 2), 20, ["2025-12"]),
    ],
)
def test_a_month_is_open_from_the_next_month_until_days_past_its_deadline(
    day: date, days_after: int, months: list[str]
) -> None:
    assert open_months(day, days_after) == months


@pytest.fixture
def page(fixture_path: Callable[[str], Path]) -> Callable[[str, int, int], str]:
    def _read(market: str, month: int, origin: int) -> str:
        path = fixture_path(f"mops/revenue/t21sc03_{market}_115_{month}_{origin}.html")
        return path.read_bytes().decode("cp950")

    return _read


def test_a_domestic_page_gives_each_filer_in_thousands(
    page: Callable[[str, int, int], str],
) -> None:
    found = parse_revenue_page(page("sii", 9, 0), "sii", 0, "2026-09")
    assert set(found) == {"1101", "1102", "1623", "1702", "2330", "2882"}
    assert found["1101"] == (14321556, 13515534, 13306676)
    assert found["2882"] == (20405136, 29486864, 38783021)


def test_a_foreign_page_has_its_own_total(page: Callable[[str, int, int], str]) -> None:
    found = parse_revenue_page(page("sii", 9, 1), "sii", 1, "2026-09")
    assert found == {"1256": (561575, 648858, 419697), "1340": (6390, 7115, 2859)}
    with pytest.raises(SchemaChangedError, match="does not end with 全部國內上市公司合計"):
        parse_revenue_page(page("sii", 9, 1), "sii", 0, "2026-09")


def test_no_filers_yet_is_an_empty_answer(page: Callable[[str, int, int], str]) -> None:
    assert parse_revenue_page(page("otc", 10, 1), "otc", 1, "2026-10") == {}
    assert parse_revenue_page(page("sii", 11, 0), "sii", 0, "2026-11") == {}


@pytest.mark.parametrize(
    ("market", "period"), [("otc", "2026-09"), ("sii", "2026-08"), ("sii", "2025-09")]
)
def test_a_page_of_another_market_or_month_fails(
    page: Callable[[str, int, int], str], market: str, period: str
) -> None:
    with pytest.raises(SchemaChangedError, match="has the title"):
        parse_revenue_page(page("sii", 9, 0), market, 0, period)


@pytest.mark.parametrize(
    ("old", "new", "count", "message"),
    [
        ("單位：千元", "單位：元", -1, r"has the units \['元'\]"),
        ("單位：千元", "單位：元", 1, r"has the units \['元', '千元'\]"),
        ("<th class=tt>去年當月營收</th>", "<th class=tt>去年營收</th>", 1, "lacks 去年當月營收"),
        ("公司<br>代號", "代號", 1, "lists 1101 outside a revenue table"),
        ("14,321,556", "14,321.556", 1, "1101 has the amount"),
        ("<td align=center>-</td>", "", -1, "has 10 cells"),
    ],
)
def test_a_changed_layout_fails(
    page: Callable[[str, int, int], str], old: str, new: str, count: int, message: str
) -> None:
    text = page("sii", 9, 0)
    assert old in text
    with pytest.raises(SchemaChangedError, match=message):
        parse_revenue_page(text.replace(old, new, count), "sii", 0, "2026-09")


def test_a_company_row_in_a_table_without_a_header_fails(
    page: Callable[[str, int, int], str],
) -> None:
    closing = "</table></td></tr></table>"
    stray = "<table><tr><td>9999</td><td>x</td></tr></table>"
    text = page("sii", 9, 0).replace(closing, closing + stray, 1)
    with pytest.raises(SchemaChangedError, match="lists 9999 outside a revenue table"):
        parse_revenue_page(text, "sii", 0, "2026-09")


def test_a_page_cut_before_its_total_fails(page: Callable[[str, int, int], str]) -> None:
    text = page("sii", 9, 0)
    cut = text[: text.rindex("<table  width=100%")] + "<!--全部國內上市公司合計--></body></html>"
    with pytest.raises(SchemaChangedError, match="does not end with 全部國內上市公司合計"):
        parse_revenue_page(cut, "sii", 0, "2026-09")


def test_a_company_listed_twice_fails(page: Callable[[str, int, int], str]) -> None:
    text = page("sii", 9, 0)
    start = text.index("<tr align=right><td align=center>1101</td>")
    row = text[start : text.index("</tr>", start) + len("</tr>")]
    with pytest.raises(SchemaChangedError, match="lists 1101 twice"):
        parse_revenue_page(text.replace(row, row + row), "sii", 0, "2026-09")


def test_a_page_without_tables_or_the_empty_notice_fails(
    page: Callable[[str, int, int], str],
) -> None:
    text = page("sii", 11, 0).replace("查無資料", "請稍後再試")
    with pytest.raises(SchemaChangedError, match="has no revenue table"):
        parse_revenue_page(text, "sii", 0, "2026-11")


def test_revenue_and_growth_match_the_finmind_rows(page: Callable[[str, int, int], str]) -> None:
    companies: dict[str, Figures] = {}
    for market, origin in PAGES:
        companies |= parse_revenue_page(page(market, 9, origin), market, origin, "2026-09")
    rows = {row["stock_id"]: row for row in revenue_rows(companies, "2026-09")}
    for code, (revenue, yoy, mom) in FINMIND_2026_09.items():
        assert rows[code]["period"] == "2026-09"
        assert rows[code]["revenue"] == revenue
        assert rows[code]["yoy"] == pytest.approx(yoy, abs=1e-12)
        assert rows[code]["mom"] == pytest.approx(mom, abs=1e-12)


def test_growth_without_a_positive_earlier_month_is_empty() -> None:
    [row] = revenue_rows({"9999": (500, 0, -20)}, "2026-09")
    assert (row["revenue"], row["yoy"], row["mom"]) == (500_000, None, None)


def weekdays(start: date, end: date) -> TradingDays:
    span = (start + timedelta(days=offset) for offset in range((end - start).days + 1))
    return TradingDays([day for day in span if day.weekday() < 5], start, end)


@pytest.fixture
def adapter(fake_sleep: Callable[[float], None]) -> MopsAdapter:
    calendar = weekdays(date(2026, 1, 1), date(2026, 12, 31))
    return MopsAdapter(calendar, RateLimiter(LIMIT, sleep=fake_sleep))


@pytest.fixture
def body(fixture_path: Callable[[str], Path]) -> Callable[[str, int, int], bytes]:
    return lambda market, month, origin: fixture_path(
        f"mops/revenue/t21sc03_{market}_115_{month}_{origin}.html"
    ).read_bytes()


def serve(market: str, month: int, origin: int, content: bytes) -> None:
    name = f"t21sc03_115_{month}_{origin}.html"
    responses.get(f"{settings.MOPS_BASE_URL}/nas/t21/{market}/{name}", body=content)


def serve_october(body: Callable[[str, int, int], bytes]) -> None:
    for market, origin in PAGES:
        serve(market, 10, origin, body(market, 10, origin))


def scan_at(adapter: MopsAdapter, moment: str) -> EtlRun:
    found = datetime.fromisoformat(moment).replace(tzinfo=TAIPEI)
    with time_machine.travel(found, tick=False):
        return adapter.run(REVENUE_SCAN, found.date(), found.date())


def october() -> dict[str, MonthlyRevenue]:
    return {row.stock_id: row for row in MonthlyRevenue.objects.filter(period="2026-10")}


@pytest.mark.django_db
@responses.activate
def test_the_scan_inserts_each_filer_dated_by_the_session(
    adapter: MopsAdapter, body: Callable[[str, int, int], bytes]
) -> None:
    serve_october(body)
    run = scan_at(adapter, "2026-11-05 19:00:00")
    assert (run.status, run.rows) == (EtlRun.Status.SUCCESS, len(OCTOBER_FILERS))
    rows = october()
    assert set(rows) == OCTOBER_FILERS
    assert {row.announce_date for row in rows.values()} == {SCAN_DAY}
    assert rows["1477"].revenue == 2_448_949_000
    assert rows["1477"].yoy == pytest.approx(2448949 / 2128555 - 1)
    assert RawMops.objects.filter(dataset=REVENUE_SCAN).count() == len(PAGES)
    assert len(responses.calls) == len(PAGES)


@pytest.mark.django_db
@responses.activate
def test_rows_written_after_the_evening_check_count_for_the_next_session(
    adapter: MopsAdapter, body: Callable[[str, int, int], bytes]
) -> None:
    serve_october(body)
    with time_machine.travel(datetime(2026, 11, 5, 22, 40, tzinfo=TAIPEI), tick=False):
        frame = adapter.normalize(REVENUE_SCAN, adapter.fetch(REVENUE_SCAN, SCAN_DAY, SCAN_DAY))
    with time_machine.travel(datetime(2026, 11, 5, 22, 55, tzinfo=TAIPEI), tick=False):
        adapter.upsert(REVENUE_SCAN, frame)
    assert {row.announce_date for row in october().values()} == {date(2026, 11, 6)}


@pytest.mark.django_db
@responses.activate
def test_a_stored_month_keeps_its_figures_and_date_and_differences_are_logged(
    adapter: MopsAdapter, body: Callable[[str, int, int], bytes]
) -> None:
    MonthlyRevenue.objects.create(
        stock_id="1414", period="2026-10", revenue=1_000, announce_date=date(2026, 11, 3)
    )
    serve_october(body)
    run = scan_at(adapter, "2026-11-05 19:00:00")
    kept = october()["1414"]
    assert (kept.revenue, kept.yoy, kept.announce_date) == (1_000, None, date(2026, 11, 3))
    assert run.rows == len(OCTOBER_FILERS) - 1
    logged = DataQualityLog.objects.get(check_name=REVENUE_MISMATCH_CHECK)
    assert (logged.date, logged.severity) == (SCAN_DAY, "info")
    [entry] = logged.detail["revenues"]
    assert entry["stock_id"] == "1414"
    assert entry["fields"]["revenue"] == [1_000, 36_475_000]
    assert entry["fields"]["yoy"][0] is None
    scan_at(adapter, "2026-11-05 22:40:00")
    assert DataQualityLog.objects.filter(check_name=REVENUE_MISMATCH_CHECK).count() == 1


@pytest.mark.django_db
@responses.activate
def test_only_growth_beyond_rounding_is_a_difference(
    adapter: MopsAdapter, body: Callable[[str, int, int], bytes]
) -> None:
    for code, now, last_month, last_year, shift in (
        ("1452", 134085, 60605, 105501, 0.01),
        ("1477", 2448949, 2930592, 2128555, 1e-9),
    ):
        MonthlyRevenue.objects.create(
            stock_id=code,
            period="2026-10",
            revenue=now * 1000,
            yoy=now / last_year - 1 + shift,
            mom=now / last_month - 1,
            announce_date=date(2026, 11, 3),
        )
    serve_october(body)
    scan_at(adapter, "2026-11-05 19:00:00")
    [entry] = DataQualityLog.objects.get(check_name=REVENUE_MISMATCH_CHECK).detail["revenues"]
    assert (entry["stock_id"], list(entry["fields"])) == ("1452", ["yoy"])


@pytest.mark.django_db
@responses.activate
def test_matching_figures_leave_no_log(
    adapter: MopsAdapter, body: Callable[[str, int, int], bytes]
) -> None:
    serve_october(body)
    scan_at(adapter, "2026-11-05 19:00:00")
    scan_at(adapter, "2026-11-05 22:40:00")
    assert not DataQualityLog.objects.filter(check_name=REVENUE_MISMATCH_CHECK).exists()


@pytest.mark.django_db
@responses.activate
def test_a_market_without_filers_past_the_deadline_fails(
    adapter: MopsAdapter, body: Callable[[str, int, int], bytes]
) -> None:
    empty = body("otc", 10, 1)
    for market, origin in PAGES:
        serve(market, 10, origin, empty if market == "otc" else body(market, 10, origin))
    with pytest.raises(SchemaChangedError, match="otc 2026-10 came back empty"):
        scan_at(adapter, "2026-11-16 19:00:00")
    assert not MonthlyRevenue.objects.exists()
    assert RawMops.objects.filter(dataset=REVENUE_SCAN).count() == len(PAGES)


@pytest.mark.django_db
@responses.activate
def test_a_market_without_filers_before_the_deadline_is_fine(
    adapter: MopsAdapter, body: Callable[[str, int, int], bytes]
) -> None:
    empty = body("otc", 10, 1)
    for market, origin in PAGES:
        serve(market, 10, origin, empty if market == "otc" else body(market, 10, origin))
    run = scan_at(adapter, "2026-11-13 19:00:00")
    assert run.rows == 6


@pytest.mark.django_db
@responses.activate
def test_early_december_scans_october_and_november(
    adapter: MopsAdapter, body: Callable[[str, int, int], bytes]
) -> None:
    serve_october(body)
    labels = {"sii": "上市".encode("cp950"), "otc": "上櫃".encode("cp950")}
    for market, origin in PAGES:
        serve(market, 11, origin, body("sii", 11, 0).replace(labels["sii"], labels[market]))
    run = scan_at(adapter, "2026-12-03 19:00:00")
    assert len(responses.calls) == 2 * len(PAGES)
    assert run.rows == len(OCTOBER_FILERS)
    assert {row.announce_date for row in october().values()} == {date(2026, 12, 3)}


@pytest.mark.django_db
@responses.activate
def test_a_company_in_both_markets_fails(
    adapter: MopsAdapter, body: Callable[[str, int, int], bytes]
) -> None:
    for market, origin in PAGES:
        content = body(market, 10, origin)
        if (market, origin) == ("otc", 0):
            content = content.replace(b">2070<", b">1414<")
        serve(market, 10, origin, content)
    with pytest.raises(SchemaChangedError, match="lists a company twice"):
        scan_at(adapter, "2026-11-05 19:00:00")
    assert not MonthlyRevenue.objects.exists()
    assert RawMops.objects.filter(dataset=REVENUE_SCAN).count() == len(PAGES) - 1


@pytest.mark.django_db
@responses.activate
def test_a_broken_page_is_kept_as_raw_before_failing(
    adapter: MopsAdapter, body: Callable[[str, int, int], bytes]
) -> None:
    for market, origin in PAGES:
        content = body(market, 10, origin)
        if (market, origin) == ("otc", 0):
            content = content.replace("千元".encode("cp950"), "元".encode("cp950"))
        serve(market, 10, origin, content)
    with pytest.raises(SchemaChangedError, match="has the units"):
        scan_at(adapter, "2026-11-05 19:00:00")
    kept = RawMops.objects.filter(dataset=REVENUE_SCAN)
    assert sorted(row.request_params["market"] for row in kept) == ["otc", "sii", "sii"]


@pytest.mark.django_db
@responses.activate
def test_a_company_on_both_pages_of_a_market_fails(
    adapter: MopsAdapter, body: Callable[[str, int, int], bytes]
) -> None:
    for market, origin in PAGES:
        content = body(market, 10, origin)
        if (market, origin) == ("sii", 1):
            content = content.replace(b">1256<", b">1414<")
        serve(market, 10, origin, content)
    with pytest.raises(SchemaChangedError, match="lists a company twice"):
        scan_at(adapter, "2026-11-05 19:00:00")
    assert not MonthlyRevenue.objects.exists()
    assert RawMops.objects.filter(dataset=REVENUE_SCAN).count() == 2


@pytest.mark.django_db
@responses.activate
def test_a_page_differing_only_in_its_generation_time_is_stored_once(
    adapter: MopsAdapter, body: Callable[[str, int, int], bytes]
) -> None:
    for stamp in (b"<!--18:58:04-->", b"<!--22:01:17-->"):
        for market, origin in PAGES:
            content = re.sub(rb"<!--\d{2}:\d{2}:\d{2}-->", stamp, body(market, 10, origin))
            serve(market, 10, origin, content)
    scan_at(adapter, "2026-11-05 19:00:00")
    scan_at(adapter, "2026-11-05 22:40:00")
    assert len(responses.calls) == 2 * len(PAGES)
    assert RawMops.objects.filter(dataset=REVENUE_SCAN).count() == len(PAGES)


@pytest.mark.django_db
def test_the_scan_reads_only_today_and_the_whole_market(adapter: MopsAdapter) -> None:
    with time_machine.travel(datetime(2026, 11, 5, 19, tzinfo=TAIPEI), tick=False):
        with pytest.raises(ValueError, match="sees only today"):
            adapter.run(REVENUE_SCAN, date(2026, 11, 4), date(2026, 11, 4))
        with pytest.raises(ValueError, match="covers the whole market"):
            adapter.fetch(REVENUE_SCAN, SCAN_DAY, SCAN_DAY, "1414")
