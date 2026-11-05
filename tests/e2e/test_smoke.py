import re

import pytest
from playwright.sync_api import Page, expect
from pytest_django.live_server_helper import LiveServer

from apps.etl.quality import CHECKS
from tests.e2e.conftest import XSS_NAME, Seed

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


def test_models_view_lists_the_day_s_runs_and_checks(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    page.goto(f"{live_server.url}/#models")

    rows = page.locator("#etl .trow")
    expect(rows).to_have_count(3)
    expect(rows.nth(0)).to_contain_text("15:31")
    expect(rows.nth(2)).to_contain_text("失敗")
    expect(page.locator("#etl-sub")).to_contain_text(seeded.last_session.isoformat())
    expect(page.locator("#quality .check")).to_have_count(len(CHECKS))
    expect(page.locator("#quality-sub")).to_contain_text("全部檢查通過")
    expect(page.locator("#status")).to_contain_text("品質檢查通過")


def test_today_view_says_predictions_are_pending(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    page.goto(live_server.url)

    expect(page.locator("#today-empty")).to_contain_text("預測資料尚未產出")
    page.get_by_role("button", name="5 日").click()
    expect(page.locator("#ret-head")).to_have_text("5 日預測報酬")
    expect(page.locator("[data-nav=today]")).to_have_attribute("aria-current", "page")


def test_stock_names_render_as_text(page: Page, live_server: LiveServer, seeded: Seed) -> None:
    page.goto(f"{live_server.url}/#stock-9999")

    expect(page.locator("#stock-name")).to_have_text(XSS_NAME)
    expect(page.locator("#view-stock img")).to_have_count(0)
    assert page.evaluate("window.__xss") is None


def test_unknown_stock_shows_the_api_error(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    page.goto(f"{live_server.url}/#stock-0000")

    expect(page.locator("#view-stock .error")).to_have_text(re.compile("找不到"))
