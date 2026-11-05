from collections.abc import Callable
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import responses
import time_machine
from django.conf import settings
from responses import matchers

from apps.etl.adapters.mops import MopsAdapter
from apps.etl.errors import SchemaChangedError
from apps.etl.models import DataQualityLog, RawMops
from apps.etl.mops_history import (
    MISMATCH_CHECKS,
    REVENUE_HISTORY,
    STATEMENT_HISTORY,
    months_between,
    quarters_between,
)
from apps.etl.ratelimit import RateLimit, RateLimiter
from apps.etl.revenue_scan import parse_revenue_page
from apps.etl.statement_summary import BALANCE, CASH_FLOW, INCOME, parse_summary
from apps.market.calendar import TradingDays
from apps.market.models import FinancialStatement, FinType, MonthlyRevenue

TAIPEI = ZoneInfo("Asia/Taipei")
LIMIT = RateLimit(min_interval=6.0, max_retries=2, backoff_base=1.0, backoff_max=2.0, cooldown=5.0)
TODAY = datetime(2026, 11, 6, 10, 0, tzinfo=TAIPEI)
PAGES = (("sii", 0), ("sii", 1), ("otc", 0), ("otc", 1))
JANUARY_FILERS = {"1101", "1256", "1337", "1591", "1780", "2330", "4139", "4205", "4526"}
Q1_FILERS = {"1101", "2330", "2855", "2882", "3105", "4526", "8069"}
CONSOLIDATED = FinType.CONSOLIDATED.value


@pytest.mark.parametrize(
    ("start", "end", "months"),
    [
        (date(2013, 1, 1), date(2013, 3, 31), ["2013-01", "2013-02", "2013-03"]),
        (date(2013, 1, 2), date(2013, 3, 1), ["2013-02", "2013-03"]),
        (date(2019, 12, 1), date(2020, 1, 31), ["2019-12", "2020-01"]),
    ],
)
def test_months_between_start_on_their_first_day(start: date, end: date, months: list[str]) -> None:
    assert months_between(start, end) == months


@pytest.mark.parametrize(
    ("start", "end", "quarters"),
    [
        (date(2012, 1, 1), date(2013, 6, 30), ["2013-Q1", "2013-Q2"]),
        (date(2013, 4, 1), date(2013, 12, 30), ["2013-Q2", "2013-Q3"]),
        (date(2019, 12, 31), date(2020, 3, 31), ["2019-Q4", "2020-Q1"]),
    ],
)
def test_quarters_between_end_inside_the_span_and_start_with_ifrs(
    start: date, end: date, quarters: list[str]
) -> None:
    assert quarters_between(start, end) == quarters


@pytest.fixture
def fixture(fixture_path: Callable[[str], Path]) -> Callable[[str], bytes]:
    return lambda name: fixture_path(f"mops/history/{name}").read_bytes()


def test_the_first_ifrs_month_has_no_comparable_previous_month(
    fixture: Callable[[str], bytes],
) -> None:
    page = fixture("t21sc03_sii_102_1_0.html").decode("cp950")
    found = parse_revenue_page(page, "sii", 0, "2013-01")
    assert found["1101"] == (9134465, None, 7125036)


def test_old_securities_balance_sheets_say_total_with_another_word(
    fixture: Callable[[str], bytes],
) -> None:
    page = fixture("t163sb05_sii_102_04.html").decode("utf-8")
    assert "資產合計" in parse_summary(page, "sii", 4, BALANCE)["2855"]


def weekdays(start: date, end: date) -> TradingDays:
    span = (start + timedelta(days=offset) for offset in range((end - start).days + 1))
    return TradingDays([day for day in span if day.weekday() < 5], start, end)


@pytest.fixture
def adapter(fake_sleep: Callable[[float], None]) -> MopsAdapter:
    calendar = weekdays(date(2012, 1, 1), date(2026, 12, 31))
    return MopsAdapter(calendar, RateLimiter(LIMIT, sleep=fake_sleep))


def serve_january(fixture: Callable[[str], bytes]) -> None:
    for market, origin in PAGES:
        name = f"t21sc03_{market}_102_1_{origin}.html"
        responses.get(
            f"{settings.MOPS_BASE_URL}/nas/t21/{market}/t21sc03_102_1_{origin}.html",
            body=fixture(name),
        )


def serve_q1(fixture: Callable[[str], bytes], empty: str | None = None) -> None:
    for report in (INCOME, BALANCE, CASH_FLOW):
        for market in ("sii", "otc"):
            form = {
                "encodeURIComponent": "1",
                "step": "1",
                "firstin": "1",
                "off": "1",
                "isQuery": "Y",
                "TYPEK": market,
                "year": "102",
                "season": "01",
            }
            body = fixture(f"{report}_{market}_102_01.html")
            if (report, market) == (empty, "otc"):
                body = (
                    fixture(f"{INCOME}_otc_102_01.html")
                    .replace(b">3105<", b">x<")
                    .replace(b">8069<", b">y<")
                )
            responses.post(
                f"{settings.MOPS_BASE_URL}/mops/web/ajax_{report}",
                body=body,
                match=[matchers.urlencoded_params_matcher(form)],
            )


def history(adapter: MopsAdapter, dataset: str, start: date, end: date) -> int:
    with time_machine.travel(TODAY, tick=False):
        return adapter.run(dataset, start, end).rows


@pytest.mark.django_db
@responses.activate
def test_revenue_history_fills_missing_months_with_the_conservative_date(
    adapter: MopsAdapter, fixture: Callable[[str], bytes]
) -> None:
    MonthlyRevenue.objects.create(
        stock_id="2330", period="2013-01", revenue=47_000_000_000, announce_date=date(2013, 2, 18)
    )
    serve_january(fixture)
    rows = history(adapter, REVENUE_HISTORY, date(2013, 1, 1), date(2013, 1, 31))
    assert rows == len(JANUARY_FILERS) - 1
    stored = {row.stock_id: row for row in MonthlyRevenue.objects.filter(period="2013-01")}
    assert set(stored) == JANUARY_FILERS
    assert (stored["1101"].revenue, stored["1101"].mom) == (9_134_465_000, None)
    assert stored["1101"].announce_date == date(2013, 2, 12)
    assert stored["2330"].revenue == 47_000_000_000
    logged = DataQualityLog.objects.get(check_name=MISMATCH_CHECKS[REVENUE_HISTORY])
    assert logged.detail["span"] == ["2013-01-01", "2013-01-31"]
    assert logged.detail["rows"] == [
        {
            "stock_id": "2330",
            "period": "2013-01",
            "fields": {"revenue": [47_000_000_000, 47_438_687_000]},
        }
    ]
    assert RawMops.objects.filter(dataset=REVENUE_HISTORY).count() == len(PAGES)


@pytest.mark.django_db
@responses.activate
def test_a_second_run_over_the_same_span_adds_nothing(
    adapter: MopsAdapter, fixture: Callable[[str], bytes]
) -> None:
    MonthlyRevenue.objects.create(
        stock_id="2330", period="2013-01", revenue=47_000_000_000, announce_date=date(2013, 2, 18)
    )
    serve_january(fixture)
    history(adapter, REVENUE_HISTORY, date(2013, 1, 1), date(2013, 1, 31))
    assert history(adapter, REVENUE_HISTORY, date(2013, 1, 1), date(2013, 1, 31)) == 0
    assert DataQualityLog.objects.filter(check_name=MISMATCH_CHECKS[REVENUE_HISTORY]).count() == 1


@pytest.mark.django_db
@responses.activate
def test_statement_history_fills_missing_quarters_and_compares_stored_ones(
    adapter: MopsAdapter, fixture: Callable[[str], bytes]
) -> None:
    FinancialStatement.objects.create(
        stock_id="4526",
        period="2013-Q1",
        fin_type=CONSOLIDATED,
        eps=Decimal("0.1200"),
        revenue=1_452_830_000,
        gross_margin=0.2203168987424544,
        op_margin=0.00629116964820385,
        net_income=26_804_000,
        debt_ratio=0.51605126542187,
        operating_cf=-42_412_000,
        announce_date=date(2013, 5, 16),
    )
    FinancialStatement.objects.create(
        stock_id="2330",
        period="2013-Q1",
        fin_type=FinType.INDIVIDUAL.value,
        announce_date=date(2013, 5, 16),
    )
    serve_q1(fixture)
    rows = history(adapter, STATEMENT_HISTORY, date(2013, 1, 1), date(2013, 3, 31))
    assert len(responses.calls) == 6
    assert rows == len(Q1_FILERS) - 2
    stored = {
        (row.stock_id, row.fin_type): row
        for row in FinancialStatement.objects.filter(period="2013-Q1")
    }
    assert ("2330", CONSOLIDATED) not in stored
    kept = stored[("4526", CONSOLIDATED)]
    assert (kept.operating_cf, kept.announce_date) == (-42_412_000, date(2013, 5, 16))
    added = stored[("1101", CONSOLIDATED)]
    assert (added.eps, added.revenue, added.roe) == (Decimal("0.3800"), 24_114_047_000, None)
    assert added.announce_date == date(2013, 6, 3)
    logged = DataQualityLog.objects.get(check_name=MISMATCH_CHECKS[STATEMENT_HISTORY])
    assert logged.detail["rows"] == [
        {
            "stock_id": "4526",
            "period": "2013-Q1",
            "fields": {"operating_cf": [-42_412_000, -92_392_000]},
        }
    ]


@pytest.mark.django_db
@responses.activate
def test_a_filed_quarter_that_comes_back_empty_fails(
    adapter: MopsAdapter, fixture: Callable[[str], bytes]
) -> None:
    serve_q1(fixture, empty=CASH_FLOW)
    with pytest.raises(SchemaChangedError):
        history(adapter, STATEMENT_HISTORY, date(2013, 1, 1), date(2013, 3, 31))
    assert not FinancialStatement.objects.exists()


@pytest.mark.django_db
def test_history_takes_any_span_but_only_the_whole_market(adapter: MopsAdapter) -> None:
    with time_machine.travel(TODAY, tick=False), pytest.raises(ValueError, match="whole market"):
        adapter.fetch(REVENUE_HISTORY, date(2013, 1, 1), date(2013, 1, 31), "1101")
