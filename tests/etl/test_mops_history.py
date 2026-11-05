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
    unsettled,
)
from apps.etl.ratelimit import RateLimit, RateLimiter
from apps.etl.revenue_scan import parse_revenue_page
from apps.etl.statement_summary import (
    BALANCE,
    CASH_FLOW,
    INCOME,
    OPERATING_CF,
    parse_summary,
    statement_rows,
)
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
        (date(2012, 11, 1), date(2013, 1, 31), ["2013-01"]),
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


def test_totals_named_with_another_word_still_give_debt_ratio_and_roe() -> None:
    def quarter(net: float, equity: float) -> dict:
        return {
            INCOME: {"2855": {"本期淨利（淨損）": net, "基本每股盈餘（元）": net / 10}},
            BALANCE: {"2855": {"資產合計": 100.0, "負債合計": 60.0, "權益合計": equity}},
            CASH_FLOW: {"2855": {OPERATING_CF[0]: 1.0}},
        }

    (row,) = statement_rows(quarter(4.0, 40.0), quarter(2.0, 40.0), 2)
    assert (row["debt_ratio"], row["roe"]) == (0.6, 0.05)


@pytest.mark.parametrize(
    ("dataset", "start", "end", "today", "days_after", "pending"),
    [
        (REVENUE_HISTORY, date(2026, 9, 1), date(2026, 9, 30), date(2026, 11, 4), 20, ["2026-09"]),
        (REVENUE_HISTORY, date(2026, 9, 1), date(2026, 9, 30), date(2026, 11, 5), 20, []),
        (
            STATEMENT_HISTORY,
            date(2026, 4, 1),
            date(2026, 6, 30),
            date(2026, 10, 15),
            45,
            ["2026-Q2"],
        ),
        (STATEMENT_HISTORY, date(2026, 4, 1), date(2026, 6, 30), date(2026, 10, 16), 45, []),
    ],
)
def test_a_period_settles_the_day_after_the_scan_last_looks(
    dataset: str, start: date, end: date, today: date, days_after: int, pending: list[str]
) -> None:
    assert unsettled(dataset, start, end, today, days_after) == pending


def weekdays(start: date, end: date, closed: date) -> TradingDays:
    span = (start + timedelta(days=offset) for offset in range((end - start).days + 1))
    return TradingDays([day for day in span if day.weekday() < 5 and day != closed], start, end)


@pytest.fixture
def adapter(fake_sleep: Callable[[float], None]) -> MopsAdapter:
    calendar = weekdays(date(2012, 1, 1), date(2026, 12, 31), closed=date(2013, 2, 4))
    return MopsAdapter(calendar, RateLimiter(LIMIT, sleep=fake_sleep))


def serve_january(fixture: Callable[[str], bytes], unfiled: str | None = None) -> None:
    for market, origin in PAGES:
        name = f"t21sc03_{market}_102_1_{origin}.html"
        if market == unfiled:
            name = f"t21sc03_{market}_102_1_{origin}_unfiled.html"
        responses.get(
            f"{settings.MOPS_BASE_URL}/nas/t21/{market}/t21sc03_102_1_{origin}.html",
            body=fixture(name),
        )


def serve_q1(
    fixture: Callable[[str], bytes], swapped: dict[tuple[str, str], bytes] | None = None
) -> None:
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
            body = (swapped or {}).get((report, market)) or fixture(
                f"{report}_{market}_102_01.html"
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
    assert stored["1101"].announce_date == date(2013, 2, 13)
    assert stored["2330"].revenue == 47_000_000_000
    logged = DataQualityLog.objects.get(check_name=MISMATCH_CHECKS[REVENUE_HISTORY])
    assert (logged.date, logged.detail["span"]) == (date(2013, 1, 31), ["2013-01-01", "2013-01-31"])
    assert {"stock_id": "1101", "period": "2013-01"} in logged.detail["added"]
    assert len(logged.detail["added"]) == rows
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
    first = DataQualityLog.objects.get(check_name=MISMATCH_CHECKS[REVENUE_HISTORY]).detail
    with time_machine.travel(TODAY + timedelta(days=1), tick=False):
        assert adapter.run(REVENUE_HISTORY, date(2013, 1, 1), date(2013, 1, 31)).rows == 0
    logged = DataQualityLog.objects.get(check_name=MISMATCH_CHECKS[REVENUE_HISTORY])
    assert logged.detail["added"] == first["added"]
    assert len(logged.detail["added"]) == len(JANUARY_FILERS) - 1
    assert len(logged.detail["rows"]) == 1


@pytest.mark.django_db
@responses.activate
def test_a_long_span_goes_a_year_at_a_time(
    adapter: MopsAdapter, fixture: Callable[[str], bytes]
) -> None:
    serve_january(fixture)
    with time_machine.travel(TODAY, tick=False):
        run = adapter.run(REVENUE_HISTORY, date(2012, 12, 1), date(2013, 1, 31))
    assert run.checkpoint == {"last_date": "2013-01-31"}
    logged = DataQualityLog.objects.get(check_name=MISMATCH_CHECKS[REVENUE_HISTORY])
    assert logged.detail["span"] == ["2013-01-01", "2013-01-31"]
    assert len(responses.calls) == len(PAGES)


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
    assert logged.detail["types"] == [
        {"stock_id": "2330", "period": "2013-Q1", "stored": ["individual"], "found": CONSOLIDATED}
    ]
    assert logged.detail["incomplete"] == {}


@pytest.mark.django_db
@responses.activate
def test_companies_missing_from_some_summaries_are_logged_not_written(
    adapter: MopsAdapter, fixture: Callable[[str], bytes]
) -> None:
    page = fixture(f"{CASH_FLOW}_otc_102_01.html").replace(b">8069<", b">8070<")
    serve_q1(fixture, {(CASH_FLOW, "otc"): page})
    history(adapter, STATEMENT_HISTORY, date(2013, 1, 1), date(2013, 3, 31))
    assert not FinancialStatement.objects.filter(stock_id__in=["8069", "8070"]).exists()
    logged = DataQualityLog.objects.get(check_name=MISMATCH_CHECKS[STATEMENT_HISTORY])
    assert logged.detail["incomplete"] == {"2013-Q1": ["8069", "8070"]}


@pytest.mark.django_db
@responses.activate
def test_a_filed_quarter_that_comes_back_empty_fails(
    adapter: MopsAdapter, fixture: Callable[[str], bytes]
) -> None:
    serve_q1(fixture, {(CASH_FLOW, "otc"): fixture(f"{CASH_FLOW}_otc_102_01_unfiled.html")})
    with pytest.raises(SchemaChangedError, match="came back empty"):
        history(adapter, STATEMENT_HISTORY, date(2013, 1, 1), date(2013, 3, 31))
    assert not FinancialStatement.objects.exists()


@pytest.mark.django_db
@responses.activate
def test_a_past_month_that_comes_back_empty_fails(
    adapter: MopsAdapter, fixture: Callable[[str], bytes]
) -> None:
    serve_january(fixture, unfiled="otc")
    with pytest.raises(SchemaChangedError, match="otc 2013-01 came back empty"):
        history(adapter, REVENUE_HISTORY, date(2013, 1, 1), date(2013, 1, 31))
    assert not MonthlyRevenue.objects.exists()


@pytest.mark.django_db
@responses.activate
@pytest.mark.parametrize(
    ("dataset", "end", "today", "pending"),
    [
        (REVENUE_HISTORY, date(2026, 10, 31), TODAY, "2026-10"),
        (REVENUE_HISTORY, date(2026, 11, 6), TODAY, "2026-11"),
        (REVENUE_HISTORY, date(2026, 9, 30), datetime(2026, 10, 30, tzinfo=TAIPEI), "2026-09"),
        (STATEMENT_HISTORY, date(2026, 9, 30), TODAY, "2026-Q3"),
        (STATEMENT_HISTORY, date(2026, 6, 30), datetime(2026, 9, 28, tzinfo=TAIPEI), "2026-Q2"),
    ],
)
def test_history_leaves_periods_still_being_filed_to_the_daily_scan(
    adapter: MopsAdapter, dataset: str, end: date, today: datetime, pending: str
) -> None:
    with (
        time_machine.travel(today, tick=False),
        pytest.raises(ValueError, match=f"leaves .*{pending}.* to the daily scan"),
    ):
        adapter.run(dataset, date(2026, 1, 1), end)


@pytest.mark.django_db
@responses.activate
@pytest.mark.parametrize("dataset", [REVENUE_HISTORY, STATEMENT_HISTORY])
def test_a_span_before_2013_is_refused(adapter: MopsAdapter, dataset: str) -> None:
    with time_machine.travel(TODAY, tick=False), pytest.raises(ValueError, match="from 2013"):
        adapter.run(dataset, date(2012, 1, 1), date(2012, 12, 31))


@pytest.mark.django_db
@responses.activate
def test_history_takes_any_span_but_only_the_whole_market(adapter: MopsAdapter) -> None:
    with time_machine.travel(TODAY, tick=False), pytest.raises(ValueError, match="whole market"):
        adapter.fetch(REVENUE_HISTORY, date(2013, 1, 1), date(2013, 1, 31), "1101")
