from datetime import date

import pytest
from django.utils import timezone
from playwright.sync_api import Locator, Page, expect
from pytest_django.live_server_helper import LiveServer

from apps.etl import filings
from apps.etl.models import DataQualityLog
from tests.e2e.conftest import Seed

pytestmark = pytest.mark.e2e


def filed(check: str, stocks: list[str], session: date) -> None:
    """One row per stock through the filing logger."""
    entry = {"period": "2026Q2", "fin_type": "consolidated", "filed": session.isoformat()}
    seen = {(stock, "2026Q2") for stock in stocks}
    filings._log(check, seen, {stock: [entry] for stock in stocks})


def replaced(day: date, stock: str) -> None:
    DataQualityLog.objects.create(
        date=day, check_name="fi_holding_zero_replaced", severity="info", detail={"stock_id": stock}
    )


def logged(seed: Seed, check: str, severity: str, detail: dict[str, object]) -> None:
    DataQualityLog.objects.create(
        date=seed.last_session, check_name=check, severity=severity, detail=detail
    )


def correction(page: Page, name: str) -> Locator:
    return page.locator("#corrections .check", has_text=name)


def test_each_correction_check_is_one_line_with_its_stocks_rows_and_span(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    filed(filings.FILED_LATER_CHECK, [f"{1000 + n}" for n in range(25)], seeded.last_session)
    replaced(date(2010, 1, 4), "1101")
    replaced(date(2010, 1, 5), "1101")
    replaced(date(2026, 10, 15), "2330")

    page.goto(f"{live_server.url}/#models")

    lines = page.locator("#corrections .check")
    expect(lines).to_have_count(2)
    holding = correction(page, "外資持股 0 值覆寫")
    expect(holding.locator(".muted")).to_have_text("2 檔、3 列，2010-01-04 至 2026-10-15")
    expect(holding.locator("i")).not_to_have_class("warn")
    later = correction(page, "更正財報逾期申報")
    today = timezone.localdate().isoformat()
    expect(later.locator(".muted")).to_have_text(f"25 檔、25 列，{today}")
    expect(later.locator("i")).to_have_class("warn")


def test_a_correction_written_once_per_run_counts_its_rows_only(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    for day in (date(2026, 11, 2), date(2026, 11, 5)):
        DataQualityLog.objects.create(
            date=day,
            check_name="market_history_rebuilt",
            severity="info",
            detail={"changed": 3, "sample": [["2330", "TWSE", "TPEx"]]},
        )

    page.goto(f"{live_server.url}/#models")

    rebuilt = correction(page, "市場歷史表重建")
    expect(rebuilt.locator(".muted")).to_have_text("2 列，2026-11-02 至 2026-11-05")


def test_corrections_stay_out_of_the_day_and_the_strip(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    filed(filings.FILED_LATER_CHECK, ["1101", "2330"], seeded.last_session)
    DataQualityLog.objects.filter(check_name=filings.FILED_LATER_CHECK).update(
        date=seeded.last_session
    )

    page.goto(f"{live_server.url}/#models")

    expect(correction(page, "更正財報逾期申報")).to_have_count(1)
    expect(page.locator("#quality")).not_to_contain_text("更正財報逾期申報")
    expect(page.locator("#status")).to_contain_text("品質檢查通過（1 項警告）")


def test_no_correction_says_so(page: Page, live_server: LiveServer, seeded: Seed) -> None:
    page.goto(f"{live_server.url}/#models")

    expect(page.locator("#corrections")).to_contain_text("沒有資料更正紀錄。")


def test_a_failed_correction_request_leaves_the_daily_checks(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    page.route("**/api/data-quality/corrections", lambda route: route.abort())
    page.goto(f"{live_server.url}/#models")

    expect(page.locator("#corrections .error")).to_be_visible()
    expect(page.locator("#quality .check").first).to_be_visible()


def test_a_check_written_per_stock_on_the_day_reads_as_one_line(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    logged(seeded, "probe_per_stock", "info", {"stock_id": "2330"})
    for stock in ("2317", "2454"):
        logged(seeded, "probe_per_stock", "warning", {"stock_id": stock})

    page.goto(f"{live_server.url}/#models")

    probe = page.locator("#quality .check", has_text="probe_per_stock")
    expect(probe).to_have_count(1)
    expect(probe.locator(".muted")).to_have_text("3 檔")
    expect(probe.locator("i")).to_have_class("warn")


def test_the_strip_and_the_day_count_warned_checks_not_rows(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    DataQualityLog.objects.filter(check_name="row_count").delete()
    for table in ("daily_price", "daily_chip", "daily_valuation"):
        logged(seeded, "row_count", "warning", {"table": table, "count": 1, "previous": 1800})

    page.goto(f"{live_server.url}/#models")

    expect(page.locator("#status")).to_contain_text("品質檢查通過（2 項警告）")
    expect(page.locator("#quality-sub")).to_contain_text("檢查通過，但有 2 項警告。")
    expect(page.locator("#quality-days")).to_contain_text("通過（2 項警告）")
    expect(page.locator("#quality .check", has_text="各表當日列數")).to_have_count(3)


def test_checks_without_a_stock_keep_one_line_per_row_and_their_names(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    logged(seeded, "statement_source_mismatch", "info", {"statements": []})
    logged(seeded, "exchange_fill", "info", {"fill": "sbl"})

    page.goto(f"{live_server.url}/#models")

    rows = page.locator("#quality .check", has_text="各表當日列數")
    expect(rows).to_have_count(2)
    expect(rows.nth(0)).to_contain_text("daily_price 1,842 列")
    expect(page.locator("#quality")).to_contain_text("財報兩來源數字不同")
    expect(page.locator("#quality")).to_contain_text("交易所逐日表補缺")
