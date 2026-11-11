import json
from datetime import timedelta

import pytest
from django.utils import timezone
from playwright.sync_api import Page, Route, expect
from pytest_django.live_server_helper import LiveServer

from apps.etl.models import EtlRun
from apps.market.calendar import TradingDays
from apps.market.models import TradingCalendar
from apps.research.models import FeatureDaily, ModelMonitor, ModelRun, Prediction
from tests.api.conftest import add_predictions
from tests.e2e.conftest import (
    BAND_NOTE,
    EXTRA_STOCKS,
    PROB_HEAD,
    PROB_TIP,
    TPEX_EXTRAS,
    XSS_NAME,
    Seed,
    at,
)

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


def test_today_says_what_the_probability_and_the_band_mean(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    page.goto(live_server.url)
    expect(page.locator("#rows .trow")).to_have_count(20)

    expect(page.locator("#view-today .lead")).to_contain_text(
        "20 日的 80% 價格區間與報酬跑贏大盤（加權股價報酬指數，含息）5 個百分點以上的機率"
    )
    head = page.locator("#prob-head")
    expect(head).to_have_text(PROB_HEAD)
    expect(head).to_have_attribute("title", PROB_TIP)
    expect(page.locator("#band-note")).to_have_text(BAND_NOTE)
    expect(page.locator("#band-note")).to_be_visible()


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
    expect(page.locator("#band-head")).to_have_text("80% 預測區間（僅 20 日）")
    expect(page.locator("#rows .band")).to_have_count(0)


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
        "今日預測裡沒有符合「zzz<b>」的股票。試試代號或公司名稱的一部分。"
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


def test_a_double_click_moves_one_page(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    page.goto(live_server.url)
    expect(page.locator("#rows .trow")).to_have_count(20)

    page.locator("#next-page").dblclick()

    expect(page.locator("#count")).to_have_text("第 21–25 名，共 25 檔")
    expect(page.locator("#rows .error")).to_have_count(0)


def test_rows_outside_the_universe_are_flagged_and_counted_apart(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    Prediction.objects.filter(run_id="rank-h20", stock_id=EXTRA_STOCKS[-1]).update(
        in_universe=False, rank=None, pct=None
    )
    page.goto(live_server.url)
    expect(page.locator("#count")).to_have_text("第 1–20 列，共 25 檔，其中 24 檔在訓練範圍內")

    page.locator("#next-page").click()

    last = page.locator("#rows .trow").last
    expect(last).to_contain_text("訓練範圍外")
    expect(last.locator(".rank")).to_have_text("—")
    expect(page.locator("#count")).to_have_text("第 21–25 列，共 25 檔，其中 24 檔在訓練範圍內")


def test_degraded_models_monitoring_and_band_sources_are_shown(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    ModelRun.objects.filter(pk="quantile-h20").update(status="degraded")
    ModelRun.objects.filter(pk__in=["rank-h20", "quantile-h20"]).update(
        train_end=TradingDays.from_db().shift(predicted.sessions[-5], -20)
    )
    EtlRun.objects.filter(status="success", start_date=predicted.last_session).update(
        finished_at=at(predicted.last_session, 23, 4)
    )
    ModelMonitor.objects.bulk_create(
        ModelMonitor(run_id="rank-h20", date=day, horizon=20, ic=0.05, n_stocks=25)
        for day in predicted.sessions[-5:]
    )
    ModelMonitor.objects.bulk_create(
        ModelMonitor(run_id="quantile-h20", date=day, horizon=20, coverage=0.8, n_stocks=1)
        for day in predicted.sessions[-5:]
    )
    Prediction.objects.filter(run_id="vol-h20", stock_id=EXTRA_STOCKS[0]).delete()
    FeatureDaily.objects.create(
        stock_id=EXTRA_STOCKS[0], date=predicted.last_session, in_universe=True, tech_vol_20d=0.3
    )
    page.goto(live_server.url)

    expect(page.locator("#rows .trow").nth(1)).to_contain_text("（波動率區間）")
    expect(page.locator(f"a[href='#stock-{EXTRA_STOCKS[0]}']")).to_contain_text(
        "（近 20 日波動估計）"
    )
    meta = page.locator("#today-meta")
    expect(meta).to_contain_text("rank IC 0.050")
    expect(meta).to_contain_text("80% 區間的實際覆蓋率 80%")
    expect(meta).to_contain_text("1 個模型降級中，結果僅供參考，區間改用波動率區間")
    status = page.locator("#status")
    expect(status).to_contain_text("1 個模型降級中")
    expect(status).to_contain_text("23:04 更新")


def test_a_failed_ranking_request_is_shown(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    page.route("**/api/predictions?**", lambda route: route.abort())
    page.goto(live_server.url)

    expect(page.locator("#rows .error")).to_have_text("無法連線到伺服器。")
    expect(page.locator("#today-meta")).to_have_text("")


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


def test_watch_only_follows_the_horizon_and_the_search(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    ModelRun.objects.filter(pk="rank-h5").update(model_type="ridge", version="2026-09")
    page.add_init_script("localStorage.setItem('twp-watch', JSON.stringify(['2330']))")
    page.goto(live_server.url)
    page.locator("#watch-only").click()
    expect(page.locator("#count")).to_have_text("追蹤中的 1 檔")
    expect(page.locator("#today-meta")).to_contain_text("LightGBM 2026-10")
    requests: list[str] = []
    page.on("request", lambda request: requests.append(request.url))

    page.get_by_role("button", name="5 日").click()

    expect(page.locator("#today-meta")).to_contain_text("Ridge 2026-09")
    page.locator("#q").fill("zzz")
    expect(page.locator("#rows .empty")).to_have_text(
        "今日預測裡沒有符合「zzz」的股票。試試代號或公司名稱的一部分。"
    )
    assert not [url for url in requests if "/api/stocks/" in url]


def test_watch_only_shows_a_failed_request(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    page.add_init_script("localStorage.setItem('twp-watch', JSON.stringify(['2330']))")
    page.route("**/api/stocks/2330/predictions", lambda route: route.abort())
    page.goto(live_server.url)

    page.locator("#watch-only").click()

    expect(page.locator("#rows .error")).to_have_text("無法連線到伺服器。")


def test_the_status_strip_reads_the_status_endpoint(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    page.goto(live_server.url)

    status = page.locator("#status")
    expect(status).to_contain_text(f"{seeded.last_session.isoformat()} 收盤資料")
    expect(status).to_contain_text("品質檢查通過（1 項警告）")
    expect(status).to_contain_text("預測尚未產出")


def test_watch_only_shares_the_requests_still_in_flight(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    held: list[Route] = []
    page.add_init_script("localStorage.setItem('twp-watch', JSON.stringify(['2330']))")
    page.route("**/api/stocks/2330/predictions", lambda route: held.append(route))
    page.goto(live_server.url)
    expect(page.locator("#rows .trow")).to_have_count(20)

    page.locator("#watch-only").click()
    page.get_by_role("button", name="5 日").click()
    page.get_by_role("button", name="20 日").click()
    page.wait_for_timeout(300)

    assert len(held) == 1
    held[0].continue_()
    expect(page.locator("#count")).to_have_text("追蹤中的 1 檔")


def test_coming_back_to_today_refreshes_the_watched_stocks(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    page.add_init_script("localStorage.setItem('twp-watch', JSON.stringify(['2330']))")
    page.goto(live_server.url)
    expect(page.locator("#rows .trow")).to_have_count(20)
    page.locator("#watch-only").click()
    expect(page.locator("#count")).to_have_text("追蹤中的 1 檔")
    expect(page.locator("#rows .trow .rank")).to_have_text("1")

    page.locator("[data-nav=models]").click()
    Prediction.objects.filter(
        run_id="rank-h20", stock_id="2330", date=predicted.last_session
    ).update(rank=7)
    page.locator("[data-nav=today]").click()

    expect(page.locator("#rows .trow .rank")).to_have_text("7")


def test_a_stock_missing_from_the_master_shows_dashes(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    add_predictions(
        ModelRun.objects.get(pk="rank-h20"),
        predicted.last_session,
        20,
        {"7777": {"score": -1.0, "rank": 26, "pct": 0.0}},
    )
    page.goto(live_server.url)

    page.locator("#next-page").click()

    row = page.locator("a[href='#stock-7777']")
    expect(row.locator(".name")).to_have_text("—")
    expect(row.locator(".market")).to_have_text("—")


def test_the_status_strip_says_when_no_prices_exist(
    page: Page, live_server: LiveServer, transactional_db: None
) -> None:
    today = timezone.localdate()
    TradingCalendar.objects.bulk_create(
        TradingCalendar(date=today + timedelta(days=n), is_trading_day=True) for n in range(-10, 10)
    )
    page.goto(live_server.url)

    expect(page.locator("#status")).to_contain_text("尚無收盤資料")


def test_a_band_degraded_after_the_shown_day_is_not_called_replaced(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    later = predicted.last_session + timedelta(days=1)
    ModelRun.objects.filter(pk="quantile-h20").update(
        status="degraded", params={"degraded_on": later.isoformat()}
    )

    page.goto(live_server.url)

    expect(page.locator("#rows .trow").first).not_to_contain_text("（波動率區間）")
    meta = page.locator("#today-meta")
    expect(meta).to_contain_text("1 個模型降級中，結果僅供參考")
    expect(meta).not_to_contain_text("區間改用波動率區間")


def test_watch_only_names_the_replaced_band(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    ModelRun.objects.filter(pk="quantile-h20").update(
        status="degraded", params={"degraded_on": predicted.sessions[0].isoformat()}
    )
    page.add_init_script("localStorage.setItem('twp-watch', JSON.stringify(['2330']))")
    page.goto(live_server.url)

    page.locator("#watch-only").click()

    expect(page.locator("#count")).to_have_text("追蹤中的 1 檔")
    expect(page.locator("#rows .trow")).to_have_count(1)
    expect(page.locator("#rows .trow").first).to_contain_text("（波動率區間）")
    expect(page.locator("#today-meta")).to_contain_text("區間改用波動率區間")
