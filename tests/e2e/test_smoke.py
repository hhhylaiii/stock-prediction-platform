import pytest
from playwright.sync_api import Page, expect
from pytest_django.live_server_helper import LiveServer

from apps.etl.models import DataQualityLog, EtlRun
from tests.e2e.conftest import EXTRA_RUNS, QUALITY, XSS_NAME, Seed

pytestmark = pytest.mark.e2e


def test_stock_view_draws_the_header_and_60_sessions(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    page.goto(f"{live_server.url}/#stock-2330")

    expect(page.locator("#stock-name")).to_have_text("台積電")
    expect(page.locator("#stock-close")).to_have_text("1,138")
    expect(page.locator(".stockhead .chg")).to_have_text("+2.0（+0.18%）")
    expect(page.locator("#price-chart rect.candle")).to_have_count(59)

    page.get_by_role("button", name="收盤線").click()
    expect(page.locator("#price-chart path.close-line")).to_have_count(1)
    expect(page.locator("#chart-legend")).to_contain_text("收盤價")
    page.reload()
    expect(page.locator("#price-chart path.close-line")).to_have_count(1)


def test_watching_a_stock_is_remembered(page: Page, live_server: LiveServer, seeded: Seed) -> None:
    page.goto(f"{live_server.url}/#stock-2330")

    page.get_by_role("button", name="加入追蹤").click()
    expect(page.locator("#watch-btn")).to_have_text("已追蹤")
    page.goto(f"{live_server.url}/#today")
    page.reload()
    expect(page.locator("#watch-only")).to_have_text("只看追蹤（1）")


def test_models_view_lists_every_run_of_the_day_and_each_check(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    page.goto(f"{live_server.url}/#models")

    rows = page.locator("#etl .trow")
    expect(rows).to_have_count(3 + EXTRA_RUNS)
    expect(rows.nth(0)).to_contain_text("15:31")
    expect(rows.nth(2)).to_contain_text("失敗")
    expect(page.locator("#etl-sub")).to_contain_text(seeded.last_session.isoformat())
    checks = page.locator("#quality .check")
    expect(checks).to_have_count(len(QUALITY))
    for summary in (
        "daily_price 1,842 列，前一交易日 1,840 列",
        "daily_price：整列 缺 1 檔、close 缺 2 檔",
        "過期：SOX",
        "全部資料集與排程工作都已更新",
        "新增 1 檔",
    ):
        expect(checks.filter(has_text=summary)).to_have_count(1)
    expect(page.locator("#quality-sub")).to_contain_text("檢查通過，但有 1 項警告")
    expect(page.locator("#status")).to_contain_text("品質檢查通過（1 項警告）")


def test_models_view_renders_api_strings_as_text(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    EtlRun.objects.create(
        source="twse",
        dataset=XSS_NAME,
        start_date=seeded.last_session,
        end_date=seeded.last_session,
        status=EtlRun.Status.FAILED,
        error=XSS_NAME,
    )
    DataQualityLog.objects.create(
        date=seeded.last_session, check_name=XSS_NAME[:64], severity="warning", detail={}
    )
    DataQualityLog.objects.create(
        date=seeded.last_session,
        check_name="index_freshness",
        severity="warning",
        detail={"stale": {XSS_NAME: ""}},
    )

    page.goto(f"{live_server.url}/#models")

    expect(page.locator("#etl")).to_contain_text(XSS_NAME)
    expect(page.locator("#quality")).to_contain_text(XSS_NAME)
    expect(page.locator("#view-models img")).to_have_count(0)
    assert page.evaluate("window.__xss") is None


def test_today_view_says_predictions_are_pending(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    page.goto(live_server.url)

    expect(page.locator("#today-empty")).to_contain_text("預測資料尚未產出")
    page.get_by_role("button", name="5 日").click()
    expect(page.locator("#ret-head")).to_have_text("5 日預測報酬")
    expect(page.locator("#prob-head")).to_have_text("漲幅超過 5% 機率（僅 20 日）")
    expect(page.locator("[data-nav=today]")).to_have_attribute("aria-current", "page")


def test_stock_names_render_as_text(page: Page, live_server: LiveServer, seeded: Seed) -> None:
    page.goto(f"{live_server.url}/#stock-9999")

    expect(page.locator("#stock-name")).to_have_text(XSS_NAME)
    expect(page.locator("#view-stock .tags")).to_contain_text(XSS_NAME)
    expect(page.locator("#view-stock img")).to_have_count(0)
    assert page.evaluate("window.__xss") is None


def test_unfinished_checks_are_not_called_a_failure(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    DataQualityLog.objects.filter(date=seeded.last_session).exclude(
        check_name="stock_master"
    ).delete()

    page.goto(f"{live_server.url}/#models")

    expect(page.locator("#status")).to_contain_text("品質檢查尚未完成")
    expect(page.locator("#status")).not_to_contain_text("未通過")
    expect(page.locator("#quality-sub")).to_contain_text("還有 5 項尚未執行")
    expect(page.locator("#quality-days .trow").first).to_contain_text("尚未完成")


def test_a_blocking_dataset_fails_the_day_and_is_named(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    DataQualityLog.objects.filter(date=seeded.last_session, check_name="etl_runs").update(
        severity="error",
        detail={"missing": ["twse/daily_price", "twse/margin"], "blocking": ["twse/daily_price"]},
    )

    page.goto(f"{live_server.url}/#models")

    expect(page.locator("#status")).to_contain_text("品質檢查未通過")
    expect(page.locator("#quality")).to_contain_text("缺閘門資料集：twse/daily_price")
    expect(page.locator("#quality-days .trow").first).to_contain_text("未通過")


def test_a_non_json_gateway_error_is_shown(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    page.route(
        "**/api/stocks/2330",
        lambda route: route.fulfill(
            status=502, body="<html>bad gateway</html>", content_type="text/html"
        ),
    )

    page.goto(f"{live_server.url}/#stock-2330")

    expect(page.locator("#view-stock .error")).to_contain_text("HTTP 502")


def test_a_dropped_connection_is_shown(page: Page, live_server: LiveServer, seeded: Seed) -> None:
    page.route("**/api/data-quality*", lambda route: route.abort())

    page.goto(f"{live_server.url}/#models")

    expect(page.locator("#quality .error")).to_have_text("無法連線到伺服器。")
    expect(page.locator("#model-checks .error")).to_have_text("無法連線到伺服器。")
    expect(page.locator("#etl .trow")).to_have_count(3 + EXTRA_RUNS)


def test_failed_prices_keep_the_stock_header(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    page.route("**/api/stocks/2330/prices*", lambda route: route.abort())

    page.goto(f"{live_server.url}/#stock-2330")

    expect(page.locator("#stock-name")).to_have_text("台積電")
    expect(page.locator("#price-chart .error")).to_have_text("無法連線到伺服器。")


def test_tooltip_shows_volume_in_lots(page: Page, live_server: LiveServer, seeded: Seed) -> None:
    page.goto(f"{live_server.url}/#stock-2330")
    overlay = page.locator("#price-chart .overlay")
    box = overlay.bounding_box()
    assert box is not None

    overlay.hover(position={"x": box["width"] - 1, "y": box["height"] / 2})

    expect(page.locator("#price-chart .tip")).to_contain_text("成交量 1,000 張")
    expect(page.locator("#price-chart .tip")).to_contain_text("5 日均 1,134")
    expect(page.locator("#price-chart .tip")).not_to_contain_text("20 日均")


def test_a_bad_stock_hash_keeps_the_nav_on_the_last_good_stock(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    page.goto(f"{live_server.url}/#stock-2330")
    expect(page.locator("#stock-name")).to_have_text("台積電")

    page.goto(f"{live_server.url}/#stock-%3Cb%3E")

    expect(page.locator("#stock-missing")).to_have_text("找不到這檔股票（<b>）。")
    expect(page.locator("[data-nav=stock]")).to_have_attribute("href", "#stock-2330")


def test_unknown_stock_shows_the_api_error(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    page.goto(f"{live_server.url}/#stock-0000")

    expect(page.locator("#stock-missing")).to_have_text("找不到這檔股票（0000）。")
