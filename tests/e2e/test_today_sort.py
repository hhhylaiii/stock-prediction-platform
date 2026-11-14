import json
from typing import Any

import pytest
from playwright.sync_api import Page, expect
from pytest_django.live_server_helper import LiveServer

from apps.research.models import ModelRun, Prediction
from tests.api.conftest import add_predictions
from tests.e2e.conftest import EXTRA_STOCKS, TPEX_EXTRAS, Seed

pytestmark = pytest.mark.e2e

ORDER_NOTE_20 = (
    "名次是排名模型排的相對強弱；預測報酬與預測價來自另一個報酬模型，兩個模型各自估計，"
    "名次在前的股票預測報酬也可能是負的。"
    "報酬模型驗收的是預測誤差，沒有驗過排序能力，所以依預測報酬排出的順序不像名次那樣經過驗證。"
    "點表頭可以改依收盤、預測報酬、預測價、跑贏大盤機率或昨日排名排序。"
)
ORDER_NOTE_5 = (
    "名次是排名模型排的相對強弱；預測報酬與預測價來自另一個報酬模型，兩個模型各自估計，"
    "名次在前的股票預測報酬也可能是負的。"
    "報酬模型驗收的是預測誤差，沒有驗過排序能力，所以依預測報酬排出的順序不像名次那樣經過驗證。"
    "點表頭可以改依收盤、預測報酬或預測價排序。"
)


def extra_return(n: int) -> float:
    """Distinct small returns in an order unlike the ranking."""
    return round(0.001 * ((n * 7) % 23) - 0.01, 4)


@pytest.fixture
def returns(predicted: Seed) -> Seed:
    """Twenty-day returns and probabilities for every stock but 9999, five-day returns for all."""
    quantile, prob = ModelRun.objects.get(pk="quantile-h20"), ModelRun.objects.get(pk="prob-h20")
    reg5 = ModelRun.objects.get(pk="reg-h5")
    last = predicted.last_session
    add_predictions(
        quantile,
        last,
        20,
        {
            code: {"q10": -0.2, "q50": extra_return(n), "q90": 0.2, "ret_pred": extra_return(n)}
            for n, code in enumerate(EXTRA_STOCKS)
        },
    )
    add_predictions(
        prob,
        last,
        20,
        {code: {"prob_up_5pct": round(0.3 + n / 100, 2)} for n, code in enumerate(EXTRA_STOCKS)},
    )
    add_predictions(
        reg5,
        last,
        5,
        {code: {"ret_pred": -extra_return(n)} for n, code in enumerate(["9999", *EXTRA_STOCKS])},
    )
    return predicted


def api_codes(page: Page, live_server: LiveServer, **params: Any) -> list[str]:
    query = "&".join(f"{key}={value}" for key, value in {"page_size": 100, **params}.items())
    body = page.request.get(f"{live_server.url}/api/predictions?{query}").json()
    return [row["stock_id"] for row in body["data"]]


def shown_codes(page: Page) -> list[str]:
    return page.locator("#rows .trow .stockcell .muted").all_inner_texts()


def sort_button(page: Page, order: str) -> Any:
    return page.locator(f'#view-today .thead [data-order="{order}"]')


def test_the_return_header_orders_the_market_highest_first(
    page: Page, live_server: LiveServer, returns: Seed
) -> None:
    page.goto(live_server.url)
    expect(page.locator("#rows .trow")).to_have_count(20)
    expect(page.locator("#first-head")).to_have_text("排名")
    expect(page.locator("#model-rank-head")).to_be_hidden()
    requests: list[str] = []
    page.on("request", lambda request: requests.append(request.url))

    sort_button(page, "ret_pred").click()

    expect(page.locator("#first-head")).to_have_text("位置")
    expect(sort_button(page, "ret_pred")).to_have_attribute("data-dir", "desc")
    expect(sort_button(page, "ret_pred")).to_have_attribute("aria-pressed", "true")
    expect(page.locator("#model-rank-head")).to_be_visible()
    expected = api_codes(page, live_server, ordering="-ret_pred")
    expect(page.locator("#rows .trow .stockcell .muted")).to_have_text(expected[:20])
    rows = page.locator("#rows .trow")
    expect(rows.nth(0).locator(".pos")).to_have_text("1")
    expect(rows.nth(0).locator(".rank")).to_have_text("1")
    expect(rows.nth(1).locator(".pos")).to_have_text("2")
    expect(rows.nth(1).locator(".rank")).to_have_text(str(3 + EXTRA_STOCKS.index(expected[1])))
    expect(page.locator("#count")).to_have_text("第 1–20 列，共 25 檔")
    assert any("ordering=-ret_pred" in url and "page=1" in url for url in requests)


def test_a_second_click_reverses_and_blanks_stay_last(
    page: Page, live_server: LiveServer, returns: Seed
) -> None:
    page.goto(live_server.url)
    expect(page.locator("#rows .trow")).to_have_count(20)

    sort_button(page, "ret_pred").click()
    expect(sort_button(page, "ret_pred")).to_have_attribute("data-dir", "desc")
    sort_button(page, "ret_pred").click()

    expect(sort_button(page, "ret_pred")).to_have_attribute("data-dir", "asc")
    expected = api_codes(page, live_server, ordering="ret_pred")
    assert expected[-1] == "9999"
    expect(page.locator("#rows .trow .stockcell .muted")).to_have_text(expected[:20])
    page.locator("#next-page").click()
    expect(page.locator("#rows .trow .stockcell .muted")).to_have_text(expected[20:])
    expect(page.locator("#rows .trow").nth(0).locator(".pos")).to_have_text("21")
    expect(page.locator("#rows .trow").last).to_contain_text("—")
    expect(page.locator("#count")).to_have_text("第 21–25 列，共 25 檔")


def test_the_model_rank_header_brings_back_the_rank_order(
    page: Page, live_server: LiveServer, returns: Seed
) -> None:
    page.goto(live_server.url)
    expect(page.locator("#rows .trow")).to_have_count(20)
    sort_button(page, "close").click()
    expect(page.locator("#first-head")).to_have_text("位置")

    page.locator("#model-rank-head").click()

    expect(page.locator("#first-head")).to_have_text("排名")
    expect(page.locator("#model-rank-head")).to_be_hidden()
    expect(page.locator("#rows .trow .pos")).to_have_count(0)
    expect(page.locator("#rows .trow").nth(0).locator(".rank")).to_have_text("1")
    expect(page.locator("#view-today .thead [data-dir]")).to_have_count(0)
    expect(page.locator("#count")).to_have_text("第 1–20 名，共 25 檔")


def test_the_order_is_kept_through_the_market_filter_and_the_search(
    page: Page, live_server: LiveServer, returns: Seed
) -> None:
    page.goto(live_server.url)
    expect(page.locator("#rows .trow")).to_have_count(20)
    sort_button(page, "close").click()
    expect(page.locator("#count")).to_have_text("第 1–20 列，共 25 檔")
    page.locator("#next-page").click()
    expect(page.locator("#count")).to_have_text("第 21–25 列，共 25 檔")

    page.get_by_role("button", name="上櫃").click()

    expected = api_codes(page, live_server, ordering="-close", market="TPEx")
    assert len(expected) == len(TPEX_EXTRAS) + 1
    expect(page.locator("#rows .trow .stockcell .muted")).to_have_text(expected)
    expect(page.locator("#rows .trow").nth(0).locator(".pos")).to_have_text("1")

    page.get_by_role("button", name="全部").click()
    page.locator("#q").fill("11")

    expected = api_codes(page, live_server, ordering="-close", q="11")
    assert len(expected) == len(EXTRA_STOCKS)
    expect(page.locator("#rows .trow .stockcell .muted")).to_have_text(expected[:20])
    expect(page.locator("#count")).to_have_text(f"符合條件 {len(EXTRA_STOCKS)} 檔")
    expect(sort_button(page, "close")).to_have_attribute("data-dir", "desc")


def test_the_probability_and_previous_rank_headers_need_their_columns(
    page: Page, live_server: LiveServer, returns: Seed
) -> None:
    page.goto(live_server.url)
    expect(page.locator("#rows .trow")).to_have_count(20)
    expect(sort_button(page, "prob_up_5pct")).to_be_enabled()
    expect(sort_button(page, "prev_rank")).to_be_enabled()

    page.get_by_role("button", name="5 日", exact=True).click()

    expect(page.locator("#ret-head")).to_have_text("5 日預測報酬")
    expect(sort_button(page, "prob_up_5pct")).to_be_disabled()
    expect(sort_button(page, "prev_rank")).to_be_disabled()
    expect(sort_button(page, "ret_pred")).to_be_enabled()
    color = "e => getComputedStyle(e).color"
    assert sort_button(page, "prev_rank").evaluate(color) != sort_button(page, "close").evaluate(
        color
    )


def test_a_horizon_without_the_column_shows_the_rank_order_and_keeps_the_choice(
    page: Page, live_server: LiveServer, returns: Seed
) -> None:
    page.goto(live_server.url)
    expect(page.locator("#rows .trow")).to_have_count(20)
    sort_button(page, "prob_up_5pct").click()
    expect(sort_button(page, "prob_up_5pct")).to_have_attribute("data-dir", "desc")
    by_probability = api_codes(page, live_server, ordering="-prob_up_5pct")

    page.get_by_role("button", name="5 日", exact=True).click()

    expect(page.locator("#ret-head")).to_have_text("5 日預測報酬")
    expect(page.locator("#first-head")).to_have_text("排名")
    expect(page.locator("#view-today .thead [data-dir]")).to_have_count(0)
    expect(page.locator("#rows .trow .stockcell .muted")).to_have_text(
        api_codes(page, live_server, horizon=5)[:20]
    )

    page.get_by_role("button", name="20 日", exact=True).click()

    expect(sort_button(page, "prob_up_5pct")).to_have_attribute("data-dir", "desc")
    expect(page.locator("#rows .trow .stockcell .muted")).to_have_text(by_probability[:20])


@pytest.mark.parametrize(
    ("order", "clicks", "ordering"),
    [
        ("ret_pred", 1, "-ret_pred"),
        ("ret_pred", 2, "ret_pred"),
        ("close", 1, "-close"),
        ("close", 2, "close"),
        ("price_pred", 1, "-price_pred"),
        ("price_pred", 2, "price_pred"),
        ("prob_up_5pct", 1, "-prob_up_5pct"),
        ("prob_up_5pct", 2, "prob_up_5pct"),
        ("prev_rank", 1, "prev_rank"),
        ("prev_rank", 2, "-prev_rank"),
    ],
)
def test_watch_only_orders_the_watched_stocks_as_the_server_does(
    page: Page, live_server: LiveServer, returns: Seed, order: str, clicks: int, ordering: str
) -> None:
    watched = [*reversed(EXTRA_STOCKS), "9999", "2330"]
    page.add_init_script(f"localStorage.setItem('twp-watch', {json.dumps(json.dumps(watched))})")
    page.goto(live_server.url)
    page.locator("#watch-only").click()
    expect(page.locator("#count")).to_have_text(f"追蹤中的 {len(watched)} 檔")

    for _ in range(clicks):
        sort_button(page, order).click()

    expected = api_codes(page, live_server, ordering=ordering)
    assert len(expected) == len(watched)
    expect(page.locator("#rows .trow .stockcell .muted")).to_have_text(expected)
    expect(sort_button(page, order)).to_have_attribute("data-dir", "desc" if clicks == 1 else "asc")
    expect(page.locator("#rows .trow").nth(0).locator(".pos")).to_have_text("1")
    expect(page.locator("#rows .trow").last.locator(".pos")).to_have_text(str(len(watched)))


def test_the_note_says_the_rank_and_the_return_come_from_two_models(
    page: Page, live_server: LiveServer, returns: Seed
) -> None:
    page.goto(live_server.url)

    expect(page.locator("#order-note")).to_have_text(ORDER_NOTE_20)

    page.get_by_role("button", name="5 日", exact=True).click()

    expect(page.locator("#order-note")).to_have_text(ORDER_NOTE_5)


def test_no_return_model_leaves_out_the_note_and_the_return_headers(
    page: Page, live_server: LiveServer, returns: Seed
) -> None:
    Prediction.objects.filter(run_id="quantile-h20").delete()
    page.goto(live_server.url)
    expect(page.locator("#rows .trow")).to_have_count(20)

    expect(page.locator("#order-note")).to_be_hidden()
    expect(sort_button(page, "ret_pred")).to_be_disabled()
    expect(sort_button(page, "price_pred")).to_be_disabled()
    expect(sort_button(page, "close")).to_be_enabled()


def test_the_sorted_table_fits_a_1440_wide_window(
    page: Page, live_server: LiveServer, returns: Seed
) -> None:
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto(live_server.url)
    expect(page.locator("#rows .trow")).to_have_count(20)

    sort_button(page, "prev_rank").click()

    expect(page.locator("#first-head")).to_have_text("位置")
    width, visible = page.locator("#view-today .scroll").evaluate(
        "e => [e.scrollWidth, e.clientWidth]"
    )
    assert width <= visible
    expect(sort_button(page, "prev_rank")).to_be_in_viewport()
