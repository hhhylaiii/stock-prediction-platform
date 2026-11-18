import json
from datetime import date, timedelta

import pytest
from playwright.sync_api import Page, Route, expect
from pytest_django.live_server_helper import LiveServer

from apps.api.holding import BOOK_START
from apps.research.models import BookAction, BookStatus
from tests.e2e.conftest import Seed
from tests.holding_books import add_closes, add_fill, add_list

pytestmark = pytest.mark.e2e

LIST = {
    "1101": (BookAction.SELL, 0, 0.42),
    "1102": (BookAction.KEEP, 0, 0.95),
    "1103": (BookAction.KEEP, 0, 0.81),
    "2330": (BookAction.BUY, 1, 0.97),
    "1104": (BookAction.BACKUP, 2, 0.93),
    "1105": (BookAction.BACKUP, 3, 0.29),
}
REVIEW = (
    "樣本 {n} 個交易日；"
    "規則的評估在第 60 個訊號日（2027-02-18）的 label 結算後整理，現在的數字不能下結論。"
)
BASIS = "帳本在換手日含開盤成交與成本；等權 universe 與加權報酬指數不扣成本。"


def md(day: date) -> str:
    return f"{day.month}/{day.day}"


def after(day: date) -> date:
    following = day + timedelta(days=1)
    while following.weekday() >= 5:
        following += timedelta(days=1)
    return following


def three_fills(
    seed: Seed,
    skip_middle: bool = False,
    last_close: float | None = 20402.0,
    first_close: float | None = 20000.0,
) -> None:
    """Fills on the last three sessions, each signed the one before; the last has a stuck sell."""
    s = seed.sessions
    wanted = {s[-3]: first_close, s[-2]: 20200.0, s[-1]: last_close}
    closes = {day: close for day, close in wanted.items() if close is not None}
    add_closes(closes)
    add_fill(s[-4], s[-3], 0.99, universe=0.5, cash=0.01, held=3)
    if not skip_middle:
        add_fill(s[-3], s[-2], 1.0098, universe=0.01, cash=0.01, held=3)
    add_fill(s[-2], s[-1], 1.0197, universe=0.02, cash=0.004, held=3, stuck=("1101",))


def tracks_match_cells(page: Page) -> bool:
    """Whether the first row has one grid track per cell, so no cell wraps to a second line."""
    tracks, cells = page.locator("#rows .trow").first.evaluate(
        "e => [getComputedStyle(e).gridTemplateColumns.split(' ').length, e.children.length]"
    )
    return bool(tracks == cells)


def five_day(page: Page, live_server: LiveServer) -> None:
    page.goto(f"{live_server.url}/#today")
    page.locator('[data-h="5"]').click()
    expect(page.locator("#holding")).to_be_visible()
    expect(page.locator("#rows .trow").first).to_be_visible()


def test_the_five_day_page_leads_with_the_rule_list(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    last = predicted.last_session
    add_list(last, after(last), LIST)
    three_fills(predicted)

    five_day(page, live_server)

    block = page.locator("#holding")
    expect(block.locator("h2")).to_have_text("持有規則名單（H70/N30，研究紀錄）")
    expect(block.locator(".lead-line")).to_have_text(
        f"{md(last)} 收盤後的規則名單，照規則在 {md(after(last))} 開盤模擬成交。"
        "這是研究平台依事先寫死的規則記下的紀錄，不是投資建議。"
    )
    groups = block.locator(".group")
    expect(groups.locator("h3")).to_have_text(["賣出 1 檔", "續抱 2 檔", "買進 1 檔", "備位 2 檔"])
    expect(groups.nth(0).locator(".hline")).to_have_text(["1101 測試00 42%"])
    expect(groups.nth(1).locator(".hline")).to_have_text(["1102 測試01 95%", "1103 測試02 81%"])
    expect(groups.nth(2).locator(".hline")).to_have_text(["2330 台積電 97%"])
    expect(groups.nth(3).locator("details")).not_to_have_attribute("open", "")
    expect(groups.nth(3).locator(".hline")).to_have_text(["1104 測試03 93%", "1105 測試04 29%"])
    expect(block.locator("details.rule")).to_contain_text("30 個等額位置")
    expect(block.locator(".backfilled, .no-prediction, .lagging")).to_have_count(0)


def test_the_ledger_and_the_comparison_read_from_the_first_fill_s_close(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    s, last = predicted.sessions, predicted.last_session
    add_list(last, after(last), LIST)
    three_fills(predicted)

    five_day(page, live_server)

    block = page.locator("#holding")
    expect(block.locator(".ledger")).to_have_text(
        f"帳本：{md(BOOK_START)} 收盤時是現金 1.0，{md(s[-3])} 開盤第一次模擬成交。"
        f"到 {md(last)} 收盤淨值 1.0197（+1.97%），持有 3 檔，現金 0.4%。"
    )
    expect(block.locator(".compare")).to_have_text(
        "同期對照（從第一次成交那天的收盤起，那天三者都是 1.0）："
        "帳本 +3.00%、等權 universe +3.02%、加權報酬指數 +2.01%。" + BASIS
    )
    expect(block.locator(".first-day")).to_contain_text("兩者差一個隔夜")
    expect(block.locator(".review")).to_have_text(REVIEW.format(n=3))
    expect(block.locator(".index-missing, .ledger-invalid, .no-prediction-days")).to_have_count(0)


def test_one_fill_compares_nothing_yet(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    s, last = predicted.sessions, predicted.last_session
    add_list(last, after(last), LIST)
    add_closes({last: 20000.0})
    add_fill(s[-2], last, 0.9986, cash=0.004, held=30)

    five_day(page, live_server)

    block = page.locator("#holding")
    expect(block.locator(".compare")).to_contain_text("：帳本 —、等權 universe —、加權報酬指數 —。")
    expect(block.locator(".review")).to_have_text(REVIEW.format(n=1))


def test_a_missing_last_close_blanks_only_the_index(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    last = predicted.last_session
    add_list(last, after(last), LIST)
    three_fills(predicted, last_close=None)

    five_day(page, live_server)

    block = page.locator("#holding")
    expect(block.locator(".compare")).to_contain_text(
        "帳本 +3.00%、等權 universe +3.02%、加權報酬指數 —。"
    )
    expect(block.locator(".index-missing")).to_have_text(
        f"加權報酬指數 {md(last)} 的收盤還沒進來，那天先不顯示。"
    )


def test_a_missing_fill_hides_the_comparison_and_keeps_the_list(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    last = predicted.last_session
    add_list(last, after(last), LIST)
    three_fills(predicted, skip_middle=True)

    five_day(page, live_server)

    block = page.locator("#holding")
    expect(block.locator(".ledger-invalid")).to_have_text("帳本資料有誤，對照暫不顯示；名單照常。")
    expect(block.locator(".compare, .review")).to_have_count(0)
    expect(block.locator(".group h3")).to_have_count(4)
    expect(block.locator(".ledger")).to_have_text(
        f"帳本：{md(BOOK_START)} 收盤時是現金 1.0。"
        f"到 {md(last)} 收盤淨值 1.0197（+1.97%），持有 3 檔，現金 0.4%。"
    )


def test_days_without_predictions_are_disclosed_only_when_there_are_some(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    s, last = predicted.sessions, predicted.last_session
    add_list(s[-3], s[-2], status=BookStatus.NO_PREDICTION)
    add_list(last, after(last), LIST)
    three_fills(predicted)

    five_day(page, live_server)

    expect(page.locator("#holding .no-prediction-days")).to_have_text(
        "其中 1 天前一天沒有預測，那幾天等權 universe 依預先登錄記 0，不是實際漲跌；"
        "那幾天之後的帳本對等權 universe 都含這段。"
    )


def test_a_list_without_predictions_says_the_rule_stood_still(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    last = predicted.last_session
    add_list(last, after(last), status=BookStatus.NO_PREDICTION)

    five_day(page, live_server)

    block = page.locator("#holding")
    expect(block.locator(".no-prediction")).to_have_text(
        "這一天沒有預測，規則不賣也不買，持股照常估值。"
    )
    expect(block.locator(".group")).to_have_count(0)


def test_a_list_older_than_the_predictions_says_the_book_step_has_not_run(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    s, last = predicted.sessions, predicted.last_session
    add_list(s[-2], last, LIST)

    five_day(page, live_server)

    expect(page.locator("#holding .lagging")).to_have_text(
        f"名單是 {md(s[-2])} 的，比今日預測（{md(last)}）早：今天早上的帳本還沒寫入，"
        "請看模型頁「持有規則影子帳本」那一列。"
    )


def test_a_backfilled_list_says_so(page: Page, live_server: LiveServer, predicted: Seed) -> None:
    last = predicted.last_session
    add_list(last, after(last), LIST, backfilled=True)

    five_day(page, live_server)

    expect(page.locator("#holding .backfilled")).to_have_text(
        "這份名單是補記的（成交日 09:00 之後才寫入），算法相同。"
    )


def test_keeps_past_ten_fold_away(page: Page, live_server: LiveServer, predicted: Seed) -> None:
    last = predicted.last_session
    keeps = {f"{1101 + n}": (BookAction.KEEP, 0, 0.99 - n / 100) for n in range(12)}
    add_list(last, after(last), keeps)

    five_day(page, live_server)

    group = page.locator("#holding .group").nth(1)
    expect(group.locator("h3")).to_have_text("續抱 12 檔")
    expect(group.locator("> .hline")).to_have_count(10)
    expect(group.locator("details summary")).to_have_text("其餘 2 檔")


def test_the_rule_column_reads_the_list_and_the_fill_s_stuck_sells(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    last = predicted.last_session
    add_list(last, after(last), LIST)
    three_fills(predicted)

    five_day(page, live_server)

    rule = {
        code: page.locator(f'#rows .trow[href="#stock-{code}"] .rule')
        for code in ("9999", "2330", "1101", "1102", "1104")
    }
    expect(rule["9999"]).to_have_text("—")
    expect(rule["2330"]).to_have_text("買進")
    expect(rule["1101"]).to_have_text("賣出、賣不掉")
    expect(rule["1102"]).to_have_text("續抱")
    expect(rule["1104"]).to_have_text("備位")


def test_the_five_day_columns_put_the_rule_first_and_the_predictions_last(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    last = predicted.last_session
    add_list(last, after(last), LIST)

    five_day(page, live_server)

    heads = page.locator("#today-table .thead > div:visible")
    expect(heads).to_have_text(
        [
            "排名",
            "規則",
            "股票",
            "市場",
            "收盤",
            "80% 預測區間（僅 20 日）",
            "跑贏大盤 5 個百分點機率（僅 20 日）",
            "昨日排名",
            "5 日預測報酬",
            "預測價",
        ]
    )
    first = page.locator("#rows .trow").first
    expect(first.locator("> div").last).to_have_class("right secondary")


def test_twenty_days_has_no_list_and_says_why(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    last = predicted.last_session
    add_list(last, after(last), LIST)

    page.goto(f"{live_server.url}/#today")
    expect(page.locator("#rows .trow").first).to_be_visible()

    expect(page.locator("#holding")).to_be_hidden()
    expect(page.locator("#holding-note")).to_have_text("持有規則名單只用 5 日排名。")
    expect(page.locator("#rule-head")).to_be_hidden()
    expect(page.locator("#rows .rule")).to_have_count(0)


def test_no_list_yet_says_so(page: Page, live_server: LiveServer, predicted: Seed) -> None:
    five_day(page, live_server)

    expect(page.locator("#holding")).to_contain_text("持有規則名單還沒有寫入。")
    expect(page.locator("#holding .group")).to_have_count(0)


def test_an_unreadable_book_leaves_the_table_alone(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    def fail(route: Route) -> None:
        route.fulfill(status=500, content_type="application/json", body="{}")

    page.route("**/api/holding", fail)

    five_day(page, live_server)

    expect(page.locator("#holding")).to_contain_text("持有規則名單暫時讀不到。")
    expect(page.locator("#rows .trow")).to_have_count(20)
    expect(page.locator("#rows .rule").first).to_have_text("—")


def test_the_block_never_reads_as_advice(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    last = predicted.last_session
    add_list(last, after(last), LIST, backfilled=True)
    three_fills(predicted)

    five_day(page, live_server)

    text = page.locator("#holding").inner_text()
    assert "不是投資建議" in text
    for word in ("建議", "推薦", "買點", "應該"):
        assert word not in text.replace("不是投資建議", "")


def test_a_missing_first_close_blanks_the_whole_index(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    s, last = predicted.sessions, predicted.last_session
    add_list(last, after(last), LIST)
    three_fills(predicted, first_close=None)

    five_day(page, live_server)

    block = page.locator("#holding")
    expect(block.locator(".compare")).to_contain_text("、加權報酬指數 —。")
    expect(block.locator(".index-missing")).to_have_text(
        f"加權報酬指數在第一次成交那天（{md(s[-3])}）沒有收盤，這條對照先不顯示。"
    )


def test_a_percentile_just_under_the_keep_line_reads_below_it(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    last = predicted.last_session
    add_list(last, after(last), {"1101": (BookAction.SELL, 0, 0.6996)})

    five_day(page, live_server)

    expect(page.locator("#holding .group").nth(0).locator(".hline")).to_have_text(
        ["1101 測試00 69%"]
    )


def test_paging_keeps_the_block_and_reads_the_book_once(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    last = predicted.last_session
    add_list(last, after(last), LIST)
    three_fills(predicted)
    asked: list[str] = []
    page.on("request", lambda request: asked.append(request.url))

    five_day(page, live_server)
    page.locator("#holding details.rule summary").click()
    page.locator("#next-page").click()

    expect(page.locator("#rows .trow")).to_have_count(5)
    expect(page.locator("#holding details.rule")).to_have_attribute("open", "")
    assert sum("/api/holding" in url for url in asked) == 1


def test_the_five_day_table_fits_a_1440_wide_window_sorted_or_not(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    last = predicted.last_session
    add_list(last, after(last), LIST)
    page.set_viewport_size({"width": 1440, "height": 900})

    five_day(page, live_server)
    assert tracks_match_cells(page)
    unsorted, room = page.locator("#view-today .scroll").evaluate(
        "e => [e.scrollWidth, e.clientWidth]"
    )
    page.locator('#view-today .thead [data-order="close"]').click()

    expect(page.locator("#first-head")).to_have_text("位置")
    width, visible = page.locator("#view-today .scroll").evaluate(
        "e => [e.scrollWidth, e.clientWidth]"
    )
    assert unsorted <= room
    assert width <= visible
    assert tracks_match_cells(page)
    expect(page.locator('#view-today .thead [data-order="price_pred"]')).to_be_in_viewport()


def test_a_book_cached_before_the_predictions_moved_is_read_again(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    last = predicted.last_session
    add_list(last, after(last), LIST)
    asked: list[str] = []

    def stale_once(route: Route) -> None:
        asked.append(route.request.url)
        response = route.fetch()
        body = response.json()
        if len(asked) == 1:
            body["meta"]["predictions_date"] = "2000-01-03"
        route.fulfill(response=response, body=json.dumps(body))

    page.route("**/api/holding", stale_once)

    five_day(page, live_server)

    expect(page.locator("#holding .lead-line")).to_be_visible()
    expect(page.locator("#holding .lagging")).to_have_count(0)
    assert len(asked) == 2


def test_a_failed_prediction_table_keeps_the_list(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    last = predicted.last_session
    add_list(last, after(last), LIST)

    def fail(route: Route) -> None:
        route.fulfill(status=500, content_type="application/json", body="{}")

    page.goto(f"{live_server.url}/#today")
    expect(page.locator("#rows .trow").first).to_be_visible()
    page.route("**/api/predictions?horizon=5*", fail)
    page.locator('[data-h="5"]').click()

    expect(page.locator("#rows .error")).to_be_visible()
    expect(page.locator("#holding .lead-line")).to_be_visible()


def test_a_lagging_list_is_read_again_once_the_book_step_wrote(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    s, last = predicted.sessions, predicted.last_session
    add_list(s[-2], last, LIST)

    five_day(page, live_server)
    expect(page.locator("#holding .lagging")).to_be_visible()
    add_list(last, after(last), LIST)
    page.locator("#next-page").click()

    expect(page.locator("#holding .lagging")).to_have_count(0)
    expect(page.locator("#holding .lead-line")).to_contain_text(f"{md(last)} 收盤後的規則名單")
