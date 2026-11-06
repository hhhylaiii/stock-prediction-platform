from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
import responses
import time_machine
from django.conf import settings
from responses import matchers

from apps.etl.adapters.mops import MopsAdapter
from apps.etl.models import DataQualityLog, RawMops
from apps.etl.mops_history import MISMATCH_CHECKS, STATEMENT_VIEWER, insert_missing_history
from apps.etl.ratelimit import RateLimit, RateLimiter
from apps.etl.statement_scan import SCAN_COLUMNS
from apps.etl.viewer_history import KINDS, viewer_frame, viewer_periods
from apps.market.calendar import TradingDays
from apps.market.models import FinancialStatement, StockMaster

TAIPEI = ZoneInfo("Asia/Taipei")

NOT_FILED = "檔案不存在!"
COLUMNS = ["stock_id", "period", "fin_type", "revenue", "net_income", "operating_cf", "eps"]
YEAR = ["2013-Q1", "2013-Q2", "2013-Q3", "2013-Q4"]


def test_quarters_stop_at_the_last_one_ended_while_listed() -> None:
    periods = viewer_periods(
        date(2012, 1, 1), date(2026, 12, 31), date(2026, 11, 6), 45, date(2016, 1, 4)
    )
    assert (periods[0], periods[-1], len(periods)) == ("2013-Q1", "2015-Q4", 12)


@pytest.mark.parametrize(
    ("today", "periods"),
    [(date(2026, 10, 15), ["2026-Q1"]), (date(2026, 10, 16), ["2026-Q1", "2026-Q2"])],
)
def test_a_listed_stock_stops_at_the_quarters_the_scan_still_watches(
    today: date, periods: list[str]
) -> None:
    assert viewer_periods(date(2026, 1, 1), date(2026, 12, 31), today, 45, None) == periods


@pytest.fixture
def pages(fixture_path: Callable[[str], Path]) -> dict[str, str]:
    folder = "mops/viewer/t164sb01_6211_{}_{}_C.html"
    return {
        period: fixture_path(folder.format(period[:4], period[-1])).read_bytes().decode("cp950")
        for period in YEAR
    }


def served_pages(pages: dict[str, str]) -> tuple:
    return served(pages)


def served(pages: dict[str, str], answers: dict[tuple[str, str], str] | None = None) -> tuple:
    asked: list[tuple[str, str]] = []

    def page(period: str, kind: str) -> str:
        asked.append((period, kind))
        if answers is not None:
            return answers.get((period, kind), NOT_FILED)
        return pages[period] if kind == "C" else NOT_FILED

    return page, asked


def test_every_missing_quarter_is_its_year_to_date_less_the_one_before(
    pages: dict[str, str],
) -> None:
    page, asked = served(pages)
    frame = viewer_frame("6211", YEAR, set(), page, COLUMNS)
    rows = {row["period"]: row for row in frame.to_dict("records")}
    assert (rows["2013-Q2"]["revenue"], rows["2013-Q3"]["revenue"]) == (107_602_000, 96_580_000)
    q4 = rows["2013-Q4"]
    assert (q4["revenue"], q4["net_income"], q4["operating_cf"]) == (
        114_315_000,
        54_225_000,
        13_520_000,
    )
    assert {row["fin_type"] for row in rows.values()} == {"consolidated"}
    assert asked == [(period, "C") for period in YEAR]
    assert frame.attrs["absent"] == []


def test_stored_quarters_are_read_and_compared_too(pages: dict[str, str]) -> None:
    page, asked = served(pages)
    frame = viewer_frame("6211", YEAR, set(YEAR), page, COLUMNS)
    assert list(frame["period"]) == YEAR
    assert frame.set_index("period").loc["2013-Q3", "revenue"] == 96_580_000
    assert [period for period, _ in asked] == YEAR
    assert frame.attrs["absent"] == []


def test_a_stored_quarter_after_one_never_filed_keeps_only_its_balances(
    pages: dict[str, str],
) -> None:
    answers = {(p, "C"): pages[p] for p in ("2013-Q1", "2013-Q3", "2013-Q4")}
    page, _ = served(pages, answers)
    frame = viewer_frame("6211", YEAR, set(YEAR), page, [*COLUMNS, "debt_ratio"])
    q3 = frame.set_index("period").loc["2013-Q3"]
    assert (q3["revenue"], q3["operating_cf"]) == (None, None)
    assert q3["debt_ratio"] is not None
    assert frame.attrs["absent"] == []


def test_a_quarter_never_filed_is_noted_and_its_successor_has_no_flows(
    pages: dict[str, str],
) -> None:
    answers = {(p, "C"): pages[p] for p in ("2013-Q1", "2013-Q2", "2013-Q4")}
    page, asked = served(pages, answers)
    frame = viewer_frame("6211", YEAR, set(), page, COLUMNS)
    assert frame.attrs["absent"] == ["2013-Q3"]
    q4 = frame.set_index("period").loc["2013-Q4"]
    assert (q4["revenue"], q4["operating_cf"]) == (None, None)
    assert ("2013-Q3", "A") in asked


@pytest.mark.django_db
def test_figures_the_viewer_could_not_work_out_are_not_logged_as_differences(
    pages: dict[str, str],
) -> None:
    answers = {(p, "C"): pages[p] for p in ("2013-Q1", "2013-Q3", "2013-Q4")}
    page, _ = served(pages, answers)
    FinancialStatement.objects.create(
        stock_id="6211",
        period="2013-Q3",
        fin_type="consolidated",
        revenue=96_580_000,
        debt_ratio=0.99,
        announce_date=date(2013, 12, 3),
    )
    frame = viewer_frame("6211", YEAR, {"2013-Q3"}, page, SCAN_COLUMNS)
    span = (date(2013, 1, 1), date(2013, 12, 31))
    insert_missing_history(
        STATEMENT_VIEWER, frame, lambda period: date(2014, 6, 3), span, {"stock_id": "6211"}
    )
    logged = DataQualityLog.objects.get(check_name=MISMATCH_CHECKS[STATEMENT_VIEWER])
    (row,) = logged.detail["rows"]
    assert set(row["fields"]) == {"debt_ratio"}


def test_a_stored_quarter_the_viewer_lacks_is_listed_apart(pages: dict[str, str]) -> None:
    answers = {(p, "C"): pages[p] for p in ("2013-Q1", "2013-Q3", "2013-Q4")}
    page, _ = served(pages, answers)
    frame = viewer_frame("6211", YEAR, {"2013-Q2"}, page, COLUMNS)
    assert (frame.attrs["absent"], frame.attrs["stored_absent"]) == ([], ["2013-Q2"])


def test_an_individual_filer_is_asked_for_that_kind_first_afterwards(
    pages: dict[str, str],
) -> None:
    answers = {(p, "A"): pages[p] for p in YEAR}
    page, asked = served(pages, answers)
    viewer_frame("6211", YEAR, set(), page, COLUMNS)
    assert asked == [
        ("2013-Q1", "C"),
        ("2013-Q1", "A"),
        ("2013-Q2", "A"),
        ("2013-Q3", "A"),
        ("2013-Q4", "A"),
    ]


TODAY = datetime(2026, 11, 6, 10, 0, tzinfo=TAIPEI)
LIMIT = RateLimit(min_interval=6.0, max_retries=2, backoff_base=1.0, backoff_max=2.0, cooldown=5.0)
SPAN = (date(2013, 1, 1), date(2013, 12, 31))


@pytest.fixture
def adapter(fake_sleep: Callable[[float], None]) -> MopsAdapter:
    days = [date(2012, 1, 1) + timedelta(days=n) for n in range(15 * 366)]
    calendar = TradingDays([d for d in days if d.weekday() < 5], days[0], days[-1])
    return MopsAdapter(calendar, RateLimiter(LIMIT, sleep=fake_sleep))


def serve(pages: dict[str, str], unfiled: tuple[str, ...] = ()) -> None:
    for period in YEAR:
        for kind in KINDS:
            params = {"step": "1", "CO_ID": "6211", "SYEAR": period[:4], "SSEASON": period[-1]}
            body = pages[period] if kind == "C" and period not in unfiled else NOT_FILED
            responses.get(
                f"{settings.MOPS_BASE_URL}/server-java/t164sb01",
                body=body.encode("cp950"),
                match=[matchers.query_param_matcher({**params, "REPORT_ID": kind})],
            )


def backfill(adapter: MopsAdapter) -> int:
    with time_machine.travel(TODAY, tick=False):
        return adapter.backfill(STATEMENT_VIEWER, ["6211"], *SPAN).rows


@pytest.fixture
def delisted() -> None:
    StockMaster.objects.create(
        stock_id="6211",
        name="福登",
        market="TPEx",
        list_date=date(2000, 1, 1),
        delist_date=date(2016, 1, 4),
    )


@pytest.mark.django_db
@responses.activate
@pytest.mark.usefixtures("delisted")
def test_a_delisted_stock_gets_its_missing_quarters_at_the_conservative_date(
    adapter: MopsAdapter, pages: dict[str, str]
) -> None:
    FinancialStatement.objects.create(
        stock_id="6211",
        period="2013-Q2",
        fin_type="consolidated",
        revenue=107_602_000,
        operating_cf=1_000,
        announce_date=date(2013, 8, 15),
    )
    serve(pages)
    assert backfill(adapter) == 3
    stored = {row.period: row for row in FinancialStatement.objects.filter(stock_id="6211")}
    assert stored["2013-Q2"].operating_cf == 1_000
    assert (stored["2013-Q3"].revenue, stored["2013-Q3"].announce_date) == (
        96_580_000,
        date(2013, 12, 3),
    )
    assert RawMops.objects.filter(dataset=STATEMENT_VIEWER).count() == 4
    logged = DataQualityLog.objects.get(check_name=MISMATCH_CHECKS[STATEMENT_VIEWER])
    assert (logged.detail["stock_id"], logged.detail["span"]) == (
        "6211",
        ["2013-01-01", "2013-12-31"],
    )
    assert [entry["period"] for entry in logged.detail["added"]] == [
        "2013-Q1",
        "2013-Q3",
        "2013-Q4",
    ]
    (differing,) = logged.detail["rows"]
    assert differing["fields"]["operating_cf"] == [1_000, 22_201_000]


@pytest.mark.django_db
@responses.activate
@pytest.mark.usefixtures("delisted")
def test_a_quarter_the_viewer_lacks_is_noted_and_not_written(
    adapter: MopsAdapter, pages: dict[str, str]
) -> None:
    serve(pages, unfiled=("2013-Q4",))
    assert backfill(adapter) == 3
    logged = DataQualityLog.objects.get(check_name=MISMATCH_CHECKS[STATEMENT_VIEWER])
    assert logged.detail["absent"] == ["2013-Q4"]
    assert not FinancialStatement.objects.filter(period="2013-Q4").exists()


@pytest.mark.django_db
def test_the_viewer_goes_only_stock_by_stock(adapter: MopsAdapter) -> None:
    with (
        time_machine.travel(TODAY, tick=False),
        pytest.raises(ValueError, match=r"use manage\.py backfill"),
    ):
        adapter.run(STATEMENT_VIEWER, *SPAN)


@pytest.mark.django_db
def test_each_stock_keeps_its_own_log_for_the_same_span() -> None:
    for code in ("6211", "6212"):
        record = dict.fromkeys(SCAN_COLUMNS)
        frame = pd.DataFrame(
            [
                {
                    **record,
                    "stock_id": code,
                    "period": "2013-Q1",
                    "fin_type": "consolidated",
                    "revenue": 1,
                }
            ],
            columns=SCAN_COLUMNS,
            dtype=object,
        )
        insert_missing_history(
            STATEMENT_VIEWER,
            frame,
            lambda _: date(2013, 6, 3),
            SPAN,
            {"stock_id": code, "absent": []},
        )
    logs = DataQualityLog.objects.filter(check_name=MISMATCH_CHECKS[STATEMENT_VIEWER])
    assert sorted(log.detail["stock_id"] for log in logs) == ["6211", "6212"]


@pytest.mark.django_db
def test_a_rerun_that_reads_no_page_keeps_what_the_last_one_compared() -> None:
    FinancialStatement.objects.create(
        stock_id="6211",
        period="2013-Q1",
        fin_type="consolidated",
        revenue=5_000,
        announce_date=date(2013, 6, 3),
    )
    record = {**dict.fromkeys(SCAN_COLUMNS), "stock_id": "6211", "period": "2013-Q1"}
    found = {**record, "fin_type": "consolidated", "revenue": 1_000_000}

    def run(*rows: dict) -> list:
        frame = pd.DataFrame(list(rows), columns=SCAN_COLUMNS, dtype=object)
        notes = {"stock_id": "6211", "absent": []}
        insert_missing_history(STATEMENT_VIEWER, frame, lambda _: date(2013, 6, 3), SPAN, notes)
        logs = DataQualityLog.objects.filter(check_name=MISMATCH_CHECKS[STATEMENT_VIEWER])
        return [entry["period"] for log in logs for entry in log.detail["rows"]]

    assert run(found) == ["2013-Q1"]
    assert run() == ["2013-Q1"]
    assert run({**found, "revenue": 5_000}) == []


def next_year(page: str) -> str:
    """The 2013-Q1 page restated as if it were 2014-Q1, to pair it with a real 2013-Q4."""
    return (
        page.replace("2013年", "2014年")
        .replace("2012年", "2013年")
        .replace("102年第1季", "103年第1季")
    )


def test_a_first_quarter_takes_its_roe_against_the_last_year_end(pages: dict[str, str]) -> None:
    served = {"2013-Q4": pages["2013-Q4"], "2014-Q1": next_year(pages["2013-Q1"])}
    asked: list[str] = []

    def page(period: str, kind: str) -> str:
        asked.append(period)
        return served.get(period, NOT_FILED) if kind == "C" else NOT_FILED

    frame = viewer_frame("6211", ["2013-Q4", "2014-Q1"], {"2013-Q4"}, page, SCAN_COLUMNS)
    row = frame.set_index("period").loc["2014-Q1"].to_dict()
    assert row["roe"] == pytest.approx(48_155 / ((700_768 + 704_935) / 2))
    assert asked == ["2013-Q3", "2013-Q3", "2013-Q4", "2014-Q1"]


def test_a_quarter_without_revenue_is_noted(pages: dict[str, str]) -> None:
    served = {**pages, "2013-Q2": pages["2013-Q2"].replace("營業收入合計", "營業總收入")}
    page, _ = served_pages(served)
    frame = viewer_frame("6211", YEAR, set(), page, COLUMNS)
    assert frame.attrs["no_revenue"] == ["2013-Q2"]


@pytest.mark.django_db
@responses.activate
def test_reruns_up_to_any_day_share_one_log_ending_at_the_last_quarter(
    adapter: MopsAdapter, pages: dict[str, str]
) -> None:
    StockMaster.objects.create(
        stock_id="6211",
        name="福登",
        market="TPEx",
        list_date=date(2000, 1, 1),
        delist_date=date(2014, 1, 4),
    )
    serve(pages)
    for day in (TODAY, TODAY + timedelta(days=1)):
        with time_machine.travel(day, tick=False):
            adapter.backfill(STATEMENT_VIEWER, ["6211"], date(2013, 1, 1), day.date())
    (logged,) = DataQualityLog.objects.filter(check_name=MISMATCH_CHECKS[STATEMENT_VIEWER])
    assert logged.detail["span"] == ["2013-01-01", "2013-12-31"]
    assert len(logged.detail["added"]) == len(YEAR)


def test_a_stored_first_ifrs_quarter_is_compared_without_one_before(
    pages: dict[str, str],
) -> None:
    page, _ = served(pages)
    frame = viewer_frame("6211", YEAR, {"2013-Q1"}, page, COLUMNS)
    assert list(frame["period"]) == YEAR


@pytest.mark.django_db
@responses.activate
@pytest.mark.usefixtures("delisted")
def test_the_log_names_quarters_whose_revenue_was_not_recognised(
    adapter: MopsAdapter, pages: dict[str, str]
) -> None:
    serve({**pages, "2013-Q3": pages["2013-Q3"].replace("營業收入合計", "營業總收入")})
    backfill(adapter)
    logged = DataQualityLog.objects.get(check_name=MISMATCH_CHECKS[STATEMENT_VIEWER])
    assert logged.detail["no_revenue"] == ["2013-Q3"]
