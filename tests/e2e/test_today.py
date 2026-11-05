import json

import pytest
from playwright.sync_api import Page, expect
from pytest_django.live_server_helper import LiveServer

from tests.e2e.conftest import EXTRA_STOCKS, TPEX_EXTRAS, XSS_NAME, Seed

pytestmark = pytest.mark.e2e


def test_today_lists_the_ranking_with_prices_and_bands(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    page.goto(live_server.url)

    rows = page.locator("#rows .trow")
    expect(rows).to_have_count(20)
    first = rows.nth(0)
    for text in (
        "2330",
        "台積電",
        "上市",
        "1,138",
        "+4.0%",
        "1,184",
        "1,081 – 1,275",
        "60%",
        "昨 4",
    ):
        expect(first).to_contain_text(text)
    expect(first).to_have_attribute("href", "#stock-2330")
    expect(rows.nth(1).locator(".name")).to_have_text(XSS_NAME)
    expect(page.locator("#rows img")).to_have_count(0)
    expect(page.locator("#count")).to_have_text("第 1–20 名，共 25 檔")
    expect(page.locator("#today-meta")).to_contain_text("涵蓋 25 檔")
    expect(page.locator("#today-meta")).to_contain_text("LightGBM 2026-10")
    expect(page.locator("#status")).not_to_contain_text("預測尚未產出")
    assert page.evaluate("window.__xss") is None


def test_five_days_reads_the_five_day_models(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    page.goto(live_server.url)
    expect(page.locator("#rows .trow")).to_have_count(20)

    page.get_by_role("button", name="5 日").click()

    first = page.locator("#rows .trow").nth(0)
    expect(first).to_contain_text("9999")
    second = page.locator("#rows .trow").nth(1)
    expect(second).to_contain_text("+1.0%")
    expect(second.locator(".prob")).to_have_text("—")
    expect(page.locator("#ret-head")).to_have_text("5 日預測報酬")


def test_market_and_search_filter_on_the_server(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    page.goto(live_server.url)
    expect(page.locator("#rows .trow")).to_have_count(20)

    page.get_by_role("button", name="上櫃").click()

    expect(page.locator("#rows .trow")).to_have_count(len(TPEX_EXTRAS) + 1)
    expect(page.locator("#count")).to_have_text(f"符合條件 {len(TPEX_EXTRAS) + 1} 檔")

    page.get_by_role("button", name="全部").click()
    page.locator("#q").fill("zzz<b>")

    expect(page.locator("#rows .empty")).to_have_text(
        "沒有符合「zzz<b>」的股票。試試代號或公司名稱的一部分。"
    )


def test_pages_move_through_the_ranking(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    page.goto(live_server.url)
    expect(page.locator("#prev-page")).to_be_disabled()

    page.locator("#next-page").click()

    expect(page.locator("#rows .trow")).to_have_count(5)
    expect(page.locator("#rows .trow").nth(0).locator(".rank")).to_have_text("21")
    expect(page.locator("#count")).to_have_text("第 21–25 名，共 25 檔")
    expect(page.locator("#next-page")).to_be_disabled()
    expect(page.locator("#prev-page")).to_be_enabled()


def test_watch_only_shows_the_watched_stocks(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    watched = [EXTRA_STOCKS[4], "2330", "0000"]
    page.add_init_script(f"localStorage.setItem('twp-watch', {json.dumps(json.dumps(watched))})")
    page.goto(live_server.url)

    page.locator("#watch-only").click()

    rows = page.locator("#rows .trow")
    expect(rows).to_have_count(2)
    expect(rows.nth(0)).to_contain_text("2330")
    expect(rows.nth(1)).to_contain_text(EXTRA_STOCKS[4])
    expect(page.locator("#count")).to_have_text("追蹤中的 2 檔")


def test_the_status_strip_reads_the_status_endpoint(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    page.goto(live_server.url)

    status = page.locator("#status")
    expect(status).to_contain_text(f"{seeded.last_session.isoformat()} 收盤資料")
    expect(status).to_contain_text("品質檢查通過（1 項警告）")
    expect(status).to_contain_text("預測尚未產出")
