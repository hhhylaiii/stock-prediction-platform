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
from tests.api.conftest import FORCED_REASON, add_predictions, forced_params
from tests.e2e.conftest import (
    BAND_NOTE,
    EXTRA_STOCKS,
    MODEL_REQUIRES,
    ONLINE,
    PROB_HEAD,
    PROB_TIP,
    RELEASE_REASON,
    TPEX_EXTRAS,
    XSS_NAME,
    Seed,
    at,
    released_on,
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
    expect(page.locator("#today-meta")).not_to_contain_text("放行")
    assert page.evaluate("window.__xss") is None


def test_today_names_the_models_that_went_live_on_a_released_holdout(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    ModelRun.objects.filter(pk="rank-h20").update(params=released_on("未達標"))
    ModelRun.objects.filter(pk__in=["quantile-h20", "vol-h20", "prob-h20"]).update(
        params=released_on("明顯退化")
    )

    page.goto(live_server.url)

    meta = page.locator("#today-meta")
    expect(meta).to_contain_text(
        "排名 20 日模型的 holdout 判定未達標，經使用者放行上線；"
        "報酬區間 20 日、波動率 20 日、跑贏大盤機率 20 日模型的 holdout 判定明顯退化，"
        "經使用者放行上線。原因見模型頁。"
    )
    expect(meta).not_to_contain_text(RELEASE_REASON)

    page.get_by_role("button", name="5 日").click()

    expect(page.locator("#rows .trow").nth(0)).to_contain_text("9999")
    expect(meta).to_contain_text("涵蓋 25 檔")
    expect(meta).not_to_contain_text("放行")


FORCED_GATES_TEXT = "與現役模型的預測相關、影子模型最近 6 個月 OOS"


def test_today_names_the_models_swapped_in_by_force_after_the_released_ones(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    ModelRun.objects.filter(pk="rank-h20").update(
        params={**released_on("未達標"), **forced_params("rank-h20")}
    )
    ModelRun.objects.filter(pk="prob-h20").update(
        params=forced_params("prob-h20", "classification")
    )

    page.goto(live_server.url)

    meta = page.locator("#today-meta")
    expect(meta).to_contain_text(
        "排名 20 日模型的 holdout 判定未達標，經使用者放行上線；"
        f"排名 20 日、跑贏大盤機率 20 日模型的換模門檻沒過（{FORCED_GATES_TEXT}），"
        "經使用者強制換上。原因見模型頁。"
    )
    expect(meta).not_to_contain_text(FORCED_REASON)


def test_five_days_names_only_the_released_models_it_shows(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    ModelRun.objects.filter(pk__in=["rank-h5", "reg-h5"]).update(params=released_on("未達標"))
    ModelRun.objects.filter(pk="vol-h20").update(params=released_on("明顯退化"))
    page.goto(live_server.url)
    expect(page.locator("#rows .trow")).to_have_count(20)

    page.get_by_role("button", name="5 日").click()

    meta = page.locator("#today-meta")
    expect(meta).to_contain_text(
        "排名 5 日、絕對報酬 5 日模型的 holdout 判定未達標，經使用者放行上線。原因見模型頁。"
    )
    expect(meta).not_to_contain_text("波動率")


def test_watch_only_names_the_released_models_too(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    ModelRun.objects.filter(pk="rank-h20").update(params=released_on("未達標"))
    page.add_init_script("localStorage.setItem('twp-watch', JSON.stringify(['2330']))")
    page.goto(live_server.url)

    page.locator("#watch-only").click()

    expect(page.locator("#count")).to_have_text("追蹤中的 1 檔")
    expect(page.locator("#today-meta")).to_contain_text(
        "排名 20 日模型的 holdout 判定未達標，經使用者放行上線。原因見模型頁。"
    )


def test_today_says_what_the_probability_and_the_band_mean(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    page.goto(live_server.url)
    expect(page.locator("#rows .trow")).to_have_count(20)

    expect(page.locator("#view-today .lead")).to_contain_text(
        "20 日的 80% 價格區間與報酬跑贏大盤（加權股價報酬指數，含息）超過 5 個百分點的機率"
    )
    head = page.locator("#prob-head")
    expect(head).to_have_text(PROB_HEAD)
    expect(head).to_have_attribute("title", PROB_TIP)
    expect(page.locator("#band-note")).to_have_text(BAND_NOTE)
    expect(page.locator("#band-note")).to_be_visible()
    expect(page.locator("#model-note")).to_be_hidden()


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
    expect(page.locator("#band-note")).to_be_hidden()
    expect(page.locator("#model-note")).to_be_hidden()


def test_no_predictions_yet_says_what_puts_a_model_online(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    page.goto(live_server.url)

    expect(page.locator("#today-empty")).to_have_text(
        f"預測資料尚未產出。排名{MODEL_REQUIRES}；上線後，這裡會列出全市場的排名。"
    )
    expect(page.locator("#band-note")).to_be_hidden()
    expect(page.locator("#model-note")).to_be_hidden()


def test_a_day_without_the_band_and_odds_models_says_which_columns_wait(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    Prediction.objects.filter(run_id__in=["quantile-h20", "prob-h20"]).delete()

    page.goto(live_server.url)

    rows = page.locator("#rows .trow")
    expect(rows).to_have_count(20)
    expect(rows.nth(0)).to_contain_text("（波動率區間）")
    expect(rows.nth(0).locator(".prob")).to_have_text("—")
    expect(page.locator("#model-note")).to_have_text(
        "這一期沒有報酬區間、跑贏大盤機率模型的預測：預測報酬、預測價、跑贏大盤機率留空，"
        f"價格區間改用波動估計。{ONLINE}"
    )
    expect(page.locator("#band-note")).to_be_hidden()

    page.get_by_role("button", name="5 日").click()

    expect(page.locator("#rows .trow").nth(0)).to_contain_text("9999")
    expect(page.locator("#model-note")).to_be_hidden()


def test_five_days_without_the_return_model_says_so(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    Prediction.objects.filter(run_id="reg-h5").delete()
    page.goto(live_server.url)
    expect(page.locator("#rows .trow")).to_have_count(20)

    page.get_by_role("button", name="5 日").click()

    expect(page.locator("#model-note")).to_have_text(
        f"這一期沒有絕對報酬模型的預測：預測報酬、預測價留空。{ONLINE}"
    )

    page.get_by_role("button", name="20 日").click()

    expect(page.locator("#band-note")).to_be_visible()
    expect(page.locator("#model-note")).to_be_hidden()


def test_a_request_that_fails_after_a_good_one_clears_both_notes(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    Prediction.objects.filter(run_id="prob-h20").delete()
    page.goto(live_server.url)
    expect(page.locator("#band-note")).to_be_visible()
    expect(page.locator("#model-note")).to_be_visible()
    page.route("**/api/predictions?**", lambda route: route.abort())

    page.get_by_role("button", name="5 日").click()

    expect(page.locator("#rows .error")).to_have_text("無法連線到伺服器。")
    expect(page.locator("#band-note")).to_be_hidden()
    expect(page.locator("#model-note")).to_be_hidden()


def test_a_search_without_a_match_drops_the_missing_model_note(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    Prediction.objects.filter(run_id="prob-h20").delete()
    page.goto(live_server.url)
    expect(page.locator("#model-note")).to_be_visible()

    page.locator("#q").fill("zzz")

    expect(page.locator("#rows .empty")).to_contain_text("今日預測裡沒有符合「zzz」的股票")
    expect(page.locator("#model-note")).to_be_hidden()


def test_a_day_missing_only_the_odds_model_keeps_the_band_note(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    Prediction.objects.filter(run_id="prob-h20").delete()

    page.goto(live_server.url)

    expect(page.locator("#model-note")).to_have_text(
        f"這一期沒有跑贏大盤機率模型的預測：跑贏大盤機率留空。{ONLINE}"
    )
    expect(page.locator("#band-note")).to_be_visible()


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
    expect(page.locator("#band-note")).to_be_hidden()
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
    expect(page.locator("#band-note")).to_be_hidden()
    expect(page.locator("#model-note")).to_be_hidden()


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
    expect(page.locator("#band-note")).to_be_visible()


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
