import json
from datetime import date

import pytest
from playwright.sync_api import Locator, Page, Route, expect
from pytest_django.live_server_helper import LiveServer

from apps.market.models import Market
from tests.api.conftest import add_stock
from tests.e2e.conftest import XSS_NAME, Seed

pytestmark = pytest.mark.e2e


@pytest.fixture
def listed(seeded: Seed) -> Seed:
    """Two more stocks to find: one on TPEx and one delisted."""
    add_stock(stock_id="6488", name="環球晶", market=Market.TPEX)
    add_stock(stock_id="2498", name="宏達電", delist_date=date(2020, 6, 1))
    return seeded


def hits(page: Page) -> Locator:
    return page.locator("#stock-hits a")


def asked_stocks(page: Page) -> list[str]:
    """URLs of the stock lookups the page sends from now on."""
    asked: list[str] = []
    page.on("request", lambda request: asked.append(request.url))
    return asked


def looked_up(asked: list[str]) -> list[str]:
    return [url for url in asked if "/api/stocks?" in url]


def test_a_code_typed_on_the_stock_page_goes_to_that_stock(
    page: Page, live_server: LiveServer, listed: Seed
) -> None:
    page.goto(f"{live_server.url}/#stock-2330")
    expect(page.locator("#stock-name")).to_have_text("台積電")

    page.locator("#stock-q").fill("6488")
    page.locator("#stock-q").press("Enter")

    expect(page).to_have_url(f"{live_server.url}/#stock-6488")
    expect(page.locator("#stock-name")).to_have_text("環球晶")


def test_a_name_lists_its_stocks_and_a_pick_goes_there(
    page: Page, live_server: LiveServer, listed: Seed
) -> None:
    page.goto(f"{live_server.url}/#stock-2330")

    page.locator("#stock-q").fill("宏達")

    expect(hits(page)).to_have_count(1)
    expect(hits(page).nth(0)).to_have_text("2498 宏達電 上市 已下市")
    hits(page).nth(0).click()
    expect(page.locator("#stock-name")).to_have_text("宏達電")


def test_enter_on_a_name_goes_to_its_first_hit(
    page: Page, live_server: LiveServer, listed: Seed
) -> None:
    add_stock(stock_id="7777", name="環球測試")
    page.goto(f"{live_server.url}/#stock-2330")

    page.locator("#stock-q").fill("環球")
    expect(hits(page)).to_have_count(2)
    expect(hits(page).nth(0)).to_contain_text("6488")
    page.locator("#stock-q").press("Enter")

    expect(page.locator("#stock-name")).to_have_text("環球晶")


def test_the_hits_show_only_while_typing_and_close_on_escape_or_leaving(
    page: Page, live_server: LiveServer, listed: Seed
) -> None:
    page.goto(f"{live_server.url}/#stock-2330")
    expect(page.locator("#stock-name")).to_have_text("台積電")
    expect(page.locator("#stock-hits")).to_be_hidden()

    page.locator("#stock-q").fill("環球")
    expect(hits(page)).to_have_count(1)
    page.locator("#stock-q").press("Escape")
    expect(page.locator("#stock-hits")).to_be_hidden()

    page.locator("#stock-q").fill("環球")
    expect(hits(page)).to_have_count(1)
    page.locator("#stock-name").click()
    expect(page.locator("#stock-hits")).to_be_hidden()
    page.locator("#watch-btn").click()
    expect(page.locator("#watch-btn")).to_have_text("已追蹤")

    page.locator("#stock-q").fill("台")
    expect(hits(page).first).to_be_visible()
    page.locator("#stock-q").fill("")
    expect(page.locator("#stock-hits")).to_be_hidden()


def test_picking_the_stock_already_open_closes_the_hits(
    page: Page, live_server: LiveServer, listed: Seed
) -> None:
    page.goto(f"{live_server.url}/#stock-2330")

    page.locator("#stock-q").fill("2330")
    hits(page).first.click()
    expect(page.locator("#stock-hits")).to_be_hidden()

    page.locator("#stock-q").fill("2330")
    expect(hits(page).first).to_be_visible()
    page.locator("#stock-q").press("Enter")
    expect(page.locator("#stock-hits")).to_be_hidden()
    expect(page).to_have_url(f"{live_server.url}/#stock-2330")


def test_a_name_from_the_data_is_escaped_in_the_hits(
    page: Page, live_server: LiveServer, listed: Seed
) -> None:
    page.goto(f"{live_server.url}/#stock-2330")

    page.locator("#stock-q").fill("9999")

    expect(hits(page)).to_contain_text(XSS_NAME)
    assert page.evaluate("window.__xss") is None


def test_nothing_found_says_so_and_stays(page: Page, live_server: LiveServer, listed: Seed) -> None:
    page.goto(f"{live_server.url}/#stock-2330")

    page.locator("#stock-q").fill("0000")
    page.locator("#stock-q").press("Enter")

    expect(page.locator("#stock-hits")).to_have_text("找不到這檔股票。")
    expect(page).to_have_url(f"{live_server.url}/#stock-2330")


def test_an_unknown_code_in_the_address_says_it_is_not_found(
    page: Page, live_server: LiveServer, listed: Seed
) -> None:
    page.goto(f"{live_server.url}/#stock-0000")

    expect(page.locator("#stock-missing")).to_have_text("找不到這檔股票（0000）。")
    expect(page.locator("#view-stock .error")).to_have_count(0)
    page.locator("#stock-q").fill("2330")
    page.locator("#stock-q").press("Enter")
    expect(page.locator("#stock-name")).to_have_text("台積電")


def test_a_failed_search_shows_its_error(page: Page, live_server: LiveServer, listed: Seed) -> None:
    def refuse(route: Route) -> None:
        error = {"code": "throttled", "message": "請求太頻繁。"}
        body = {"success": False, "data": None, "error": error, "meta": {}}
        route.fulfill(status=429, content_type="application/json", body=json.dumps(body))

    page.route("**/api/stocks?*", refuse)
    page.goto(f"{live_server.url}/#stock-2330")

    page.locator("#stock-q").fill("台積")

    expect(page.locator("#stock-hits .error")).to_have_text("請求太頻繁。")


def test_the_nav_remembers_the_last_stock_across_reloads(
    page: Page, live_server: LiveServer, listed: Seed
) -> None:
    page.goto(f"{live_server.url}/#stock-6488")
    expect(page.locator("#stock-name")).to_have_text("環球晶")

    page.goto(f"{live_server.url}/#today")
    page.reload()

    expect(page.locator("[data-nav=stock]")).to_have_attribute("href", "#stock-6488")
    page.locator("[data-nav=stock]").click()
    expect(page.locator("#stock-name")).to_have_text("環球晶")


def test_the_nav_starts_on_2330_without_a_record(
    page: Page, live_server: LiveServer, listed: Seed
) -> None:
    page.goto(f"{live_server.url}/#today")

    expect(page.locator("[data-nav=stock]")).to_have_attribute("href", "#stock-2330")


def test_an_exact_code_on_today_links_its_stock_page_without_predictions(
    page: Page, live_server: LiveServer, listed: Seed
) -> None:
    page.goto(f"{live_server.url}/#today")

    page.locator("#q").fill("6488")

    link = page.locator("#today-goto a")
    expect(link).to_have_text("查看 6488 環球晶 個股頁")
    expect(link).to_have_attribute("href", "#stock-6488")
    expect(page.locator("#rows .empty")).to_have_text(
        "今日預測裡沒有符合「6488」的股票。試試代號或公司名稱的一部分。"
    )
    link.click()
    expect(page.locator("#stock-name")).to_have_text("環球晶")


def test_a_partial_code_or_name_on_today_adds_no_link(
    page: Page, live_server: LiveServer, listed: Seed
) -> None:
    page.goto(f"{live_server.url}/#today")
    expect(page.locator("#rows .empty")).to_be_visible()
    asked = asked_stocks(page)

    with page.expect_response(lambda response: response.url.endswith("/api/stocks?q=648")) as got:
        page.locator("#q").fill("648")
    got.value.finished()
    page.evaluate("() => new Promise(done => setTimeout(done, 100))")
    expect(page.locator("#today-goto")).to_be_hidden()

    with page.expect_response(lambda response: "/api/predictions?" in response.url):
        page.locator("#q").fill("環球晶")
    assert len(looked_up(asked)) == 1
    expect(page.locator("#today-goto")).to_be_hidden()

    page.locator("#q").fill("6488")
    expect(page.locator("#today-goto a")).to_be_visible()
    page.locator("#q").fill("")
    expect(page.locator("#today-goto")).to_be_hidden()


def test_paging_or_switching_today_keeps_the_link_without_looking_it_up_again(
    page: Page, live_server: LiveServer, listed: Seed
) -> None:
    page.goto(f"{live_server.url}/#today")
    asked = asked_stocks(page)
    page.locator("#q").fill("6488")
    expect(page.locator("#today-goto a")).to_be_visible()

    with page.expect_response(lambda response: "horizon=5" in response.url):
        page.locator("[data-h='5']").click()

    expect(page.locator("#today-goto a")).to_have_text("查看 6488 環球晶 個股頁")
    assert len(looked_up(asked)) == 1


def test_a_delisted_stock_says_it_has_no_predictions(
    page: Page, live_server: LiveServer, listed: Seed
) -> None:
    page.goto(f"{live_server.url}/#stock-2498")

    expect(page.locator("#stock-seg button:disabled")).to_have_count(2)
    expect(page.locator("#stock-seg-hint")).to_have_text("已下市，沒有預測")
