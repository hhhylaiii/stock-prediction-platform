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
    BASIS_TODAY,
    EXTRA_STOCKS,
    FALLBACK_NOTE,
    MODEL_REQUIRES,
    ONLINE,
    PROB_HEAD,
    PROB_TIP,
    RELEASE_REASON,
    TPEX_EXTRAS,
    XSS_NAME,
    Seed,
    at,
    no_ranking_text,
    rank_rows,
    released_on,
    take_ranking_offline,
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
    expect(page.locator("#basis-note")).to_be_hidden()
    expect(page.locator("#fallback-note")).to_be_hidden()
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
    expect(page.locator("#fallback-note")).to_be_hidden()

    page.get_by_role("button", name="5 日").click()

    expect(page.locator("#rows .trow").nth(0)).to_contain_text("9999")
    expect(page.locator("#model-note")).to_be_hidden()


def test_a_day_on_the_fallback_band_says_how_often_it_held_in_the_holdout(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    Prediction.objects.filter(run_id__in=["quantile-h20", "vol-h20", "prob-h20", "reg-h5"]).delete()
    FeatureDaily.objects.create(
        stock_id="2330", date=predicted.last_session, in_universe=True, tech_vol_20d=0.22
    )

    page.goto(live_server.url)

    rows = page.locator("#rows .trow")
    expect(rows).to_have_count(20)
    expect(rows.nth(0)).to_contain_text("（近 20 日波動估計）")
    expect(page.locator("#fallback-note")).to_have_text(FALLBACK_NOTE)
    expect(page.locator("#fallback-note")).to_be_visible()
    expect(page.locator("#band-note")).to_be_hidden()
    expect(page.locator("#model-note")).to_be_visible()
    expect(page.locator("#band-note + #fallback-note + #basis-note + #model-note")).to_have_count(1)
    expect(page.locator("#basis-note")).to_be_visible()

    page.get_by_role("button", name="5 日").click()

    expect(page.locator("#rows .trow").nth(0)).to_contain_text("9999")
    expect(page.locator("#model-note")).to_contain_text("這一期沒有絕對報酬模型的預測")
    expect(page.locator("#fallback-note")).to_be_hidden()
    expect(page.locator("#basis-note")).to_be_hidden()

    page.get_by_role("button", name="20 日").click()

    expect(page.locator("#fallback-note")).to_have_text(FALLBACK_NOTE)
    expect(page.locator("#fallback-note")).to_be_visible()

    page.locator("#q").fill("zzz")

    expect(page.locator("#rows .empty")).to_contain_text("今日預測裡沒有符合「zzz」的股票")
    expect(page.locator("#fallback-note")).to_be_hidden()


def test_a_fallback_band_no_row_draws_is_not_explained(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    Prediction.objects.filter(run_id__in=["quantile-h20", "vol-h20", "prob-h20"]).delete()

    page.goto(live_server.url)

    expect(page.locator("#rows .trow")).to_have_count(20)
    expect(page.locator("#model-note")).to_be_visible()
    expect(page.locator("#fallback-note")).to_be_hidden()


def test_one_stock_on_the_fallback_band_while_the_band_model_serves_is_not_explained(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    Prediction.objects.filter(run_id="vol-h20").delete()
    FeatureDaily.objects.create(
        stock_id=EXTRA_STOCKS[0], date=predicted.last_session, in_universe=True, tech_vol_20d=0.3
    )

    page.goto(live_server.url)

    expect(page.locator(f"a[href='#stock-{EXTRA_STOCKS[0]}']")).to_contain_text(
        "（近 20 日波動估計）"
    )
    expect(page.locator("#band-note")).to_be_visible()
    expect(page.locator("#fallback-note")).to_be_hidden()


def test_one_stock_on_the_fallback_band_while_the_volatility_model_serves_is_not_explained(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    Prediction.objects.filter(run_id__in=["quantile-h20", "prob-h20"]).delete()
    Prediction.objects.filter(run_id="vol-h20", stock_id=EXTRA_STOCKS[0]).delete()
    FeatureDaily.objects.create(
        stock_id=EXTRA_STOCKS[0], date=predicted.last_session, in_universe=True, tech_vol_20d=0.3
    )

    page.goto(live_server.url)

    expect(page.locator("#rows .trow").nth(0)).to_contain_text("（波動率區間）")
    expect(page.locator(f"a[href='#stock-{EXTRA_STOCKS[0]}']")).to_contain_text(
        "（近 20 日波動估計）"
    )
    expect(page.locator("#fallback-note")).to_be_hidden()


def test_a_failed_request_clears_the_fallback_note(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    Prediction.objects.filter(run_id__in=["quantile-h20", "vol-h20", "prob-h20"]).delete()
    FeatureDaily.objects.create(
        stock_id="2330", date=predicted.last_session, in_universe=True, tech_vol_20d=0.22
    )
    page.goto(live_server.url)
    expect(page.locator("#fallback-note")).to_be_visible()
    page.route("**/api/predictions?**", lambda route: route.abort())

    page.get_by_role("button", name="上市").click()

    expect(page.locator("#rows .error")).to_have_text("無法連線到伺服器。")
    expect(page.locator("#fallback-note")).to_be_hidden()


def test_watch_only_explains_the_fallback_band_too(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    Prediction.objects.filter(run_id__in=["quantile-h20", "vol-h20", "prob-h20"]).delete()
    FeatureDaily.objects.create(
        stock_id="2330", date=predicted.last_session, in_universe=True, tech_vol_20d=0.22
    )
    page.add_init_script("localStorage.setItem('twp-watch', JSON.stringify(['2330']))")
    page.goto(live_server.url)

    page.locator("#watch-only").click()

    expect(page.locator("#count")).to_have_text("追蹤中的 1 檔")
    expect(page.locator("#rows .trow").first).to_contain_text("（近 20 日波動估計）")
    expect(page.locator("#fallback-note")).to_have_text(FALLBACK_NOTE)
    expect(page.locator("#fallback-note")).to_be_visible()


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


def test_a_day_without_the_volatility_model_says_nothing_on_the_list(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    Prediction.objects.filter(run_id="vol-h20").delete()

    page.goto(live_server.url)

    expect(page.locator("#rows .trow")).to_have_count(20)
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
    expect(status.locator("span", has_text="尚未產出")).to_have_text("預測尚未產出")
    expect(status).not_to_contain_text("排名未上線")


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


def test_the_prices_are_said_to_include_dividends_on_both_horizons(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    page.goto(live_server.url)

    expect(page.locator("#rows .trow")).to_have_count(20)
    note = page.locator("#basis-note")
    expect(note).to_have_text(BASIS_TODAY)
    expect(note).to_be_visible()
    expect(page.locator("#fallback-note + #basis-note + #model-note")).to_have_count(1)

    page.get_by_role("button", name="5 日").click()

    expect(page.locator("#rows .trow").nth(0)).to_contain_text("9999")
    expect(page.locator("#rows .trow").nth(1)).to_contain_text("1,149")
    expect(page.locator("#rows .band")).to_have_count(0)
    expect(note).to_be_visible()


def test_a_band_without_a_predicted_price_still_gets_the_basis_note(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    Prediction.objects.filter(run_id__in=["quantile-h20", "prob-h20", "reg-h5"]).delete()

    page.goto(live_server.url)

    rows = page.locator("#rows .trow")
    expect(rows.nth(0)).to_contain_text("（波動率區間）")
    expect(page.locator("#model-note")).to_contain_text("預測報酬、預測價")
    expect(page.locator("#basis-note")).to_be_visible()

    page.get_by_role("button", name="5 日").click()

    expect(rows.nth(0)).to_contain_text("9999")
    expect(page.locator("#model-note")).to_contain_text("這一期沒有絕對報酬模型的預測")
    expect(page.locator("#basis-note")).to_be_hidden()


def test_a_page_without_a_price_or_a_band_does_not_explain_the_basis(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    Prediction.objects.filter(run_id__in=["quantile-h20", "vol-h20", "prob-h20"]).delete()

    page.goto(live_server.url)

    expect(page.locator("#rows .trow")).to_have_count(20)
    expect(page.locator("#rows .band")).to_have_count(0)
    expect(page.locator("#basis-note")).to_be_hidden()


def test_the_basis_note_goes_when_the_rows_do(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    page.goto(live_server.url)
    expect(page.locator("#basis-note")).to_be_visible()

    page.locator("#q").fill("zzz")

    expect(page.locator("#rows .empty")).to_contain_text("今日預測裡沒有符合「zzz」的股票")
    expect(page.locator("#basis-note")).to_be_hidden()

    page.locator("#q").fill("")

    expect(page.locator("#rows .trow")).to_have_count(20)
    expect(page.locator("#basis-note")).to_be_visible()
    page.route("**/api/predictions?**", lambda route: route.abort())

    page.get_by_role("button", name="上市").click()

    expect(page.locator("#rows .error")).to_have_text("無法連線到伺服器。")
    expect(page.locator("#basis-note")).to_be_hidden()


def test_watch_only_explains_the_basis_too(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    page.add_init_script("localStorage.setItem('twp-watch', JSON.stringify(['2330']))")
    page.goto(live_server.url)

    page.locator("#watch-only").click()

    expect(page.locator("#count")).to_have_text("追蹤中的 1 檔")
    expect(page.locator("#basis-note")).to_have_text(BASIS_TODAY)
    expect(page.locator("#basis-note")).to_be_visible()


def test_the_page_opens_on_five_days_when_only_that_ranking_is_live(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    take_ranking_offline(20)
    asked: list[str] = []
    page.on("request", lambda request: asked.append(request.url))
    page.clock.install()

    page.goto(live_server.url)

    rows = page.locator("#rows .trow")
    expect(rows).to_have_count(20)
    expect(rows.nth(1)).to_contain_text("2330")
    page.clock.fast_forward(5000)
    assert page.evaluate("document.querySelectorAll('#rows .trow').length") == 20
    tables = [url for url in asked if "/api/predictions?" in url]
    assert len(tables) == 1
    assert "horizon=5" in tables[0]
    expect(page.locator("[data-h='5']")).to_have_attribute("aria-pressed", "true")
    expect(page.locator("[data-h='20']")).to_have_attribute("aria-pressed", "false")
    expect(page.locator("#ret-head")).to_have_text("5 日預測報酬")
    expect(page.locator("#today-empty")).to_have_count(0)
    waiting = page.locator("#status span", has_text="排名未上線")
    expect(waiting).to_have_text("20 日排名未上線")
    expect(page.locator("#status")).not_to_contain_text("預測尚未產出")


def test_the_twenty_day_page_says_no_ranking_is_live_while_five_days_are(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    take_ranking_offline(20)
    page.goto(live_server.url)
    expect(page.locator("#rows .trow")).to_have_count(20)

    page.locator("[data-h='20']").click()

    expect(page.locator("#today-empty")).to_have_text(no_ranking_text(20, 5))
    expect(page.locator("#today-empty a")).to_have_attribute("href", "#models")
    expect(page.locator("[data-h='20']")).to_have_attribute("aria-pressed", "true")
    expect(page.locator("#rows .trow")).to_have_count(0)


def test_a_picked_horizon_is_kept_when_the_page_is_left_and_opened_again(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    take_ranking_offline(20)
    page.goto(live_server.url)
    expect(page.locator("#rows .trow")).to_have_count(20)
    page.locator("[data-h='20']").click()
    expect(page.locator("#today-empty")).to_be_visible()

    page.locator("[data-nav='models']").click()
    expect(page.locator("#view-models")).to_be_visible()
    page.locator("[data-nav='today']").click()

    expect(page.locator("#today-empty")).to_have_text(no_ranking_text(20, 5))
    expect(page.locator("[data-h='20']")).to_have_attribute("aria-pressed", "true")


def test_the_five_day_page_says_so_when_only_twenty_days_are_live(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    take_ranking_offline(5)
    page.goto(live_server.url)
    expect(page.locator("#rows .trow")).to_have_count(20)
    expect(page.locator("[data-h='20']")).to_have_attribute("aria-pressed", "true")

    page.locator("[data-h='5']").click()

    expect(page.locator("#today-empty")).to_have_text(no_ranking_text(5, 20))
    expect(page.locator("#status span", has_text="排名未上線")).to_have_text("5 日排名未上線")


def test_the_header_names_the_horizon_whose_predictions_are_behind(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    Prediction.objects.filter(run_id="rank-h20", date=predicted.last_session).delete()

    page.goto(live_server.url)

    behind = page.locator("#status span", has_text="尚未產出")
    expect(behind).to_have_count(1)
    expect(behind).to_have_text("20 日預測尚未產出")
    expect(page.locator("#status")).not_to_contain_text("排名未上線")
    expect(page.locator("[data-h='20']")).to_have_attribute("aria-pressed", "true")
    expect(page.locator("#rows .trow")).to_have_count(4)


def test_the_header_counts_a_degraded_model_without_a_twenty_day_ranking(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    take_ranking_offline(20)
    ModelRun.objects.filter(pk="reg-h5").update(status="degraded")

    page.goto(live_server.url)

    expect(page.locator("#status")).to_contain_text("1 個模型降級中")


def test_today_opens_on_twenty_days_with_the_old_words_when_the_status_request_fails(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    take_ranking_offline(20)
    page.route("**/api/status", lambda route: route.abort())
    page.clock.install()
    page.clock.pause_at(timezone.now() + timedelta(seconds=5))

    page.goto(live_server.url)

    expect(page.locator("#today-empty")).to_have_text(
        f"預測資料尚未產出。排名{MODEL_REQUIRES}；上線後，這裡會列出全市場的排名。"
    )
    expect(page.locator("[data-h='20']")).to_have_attribute("aria-pressed", "true")
    expect(page.locator("#status")).to_contain_text("無法連線到伺服器。")


def test_the_header_keeps_one_chip_when_no_horizon_has_caught_up(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    last, previous = predicted.last_session, predicted.sessions[-2]
    Prediction.objects.filter(run_id="rank-h20", date=last).delete()
    Prediction.objects.filter(run_id="rank-h5", date=last).update(date=previous)

    page.goto(live_server.url)

    waiting = page.locator("#status span", has_text="尚未產出")
    expect(waiting).to_have_count(1)
    expect(waiting).to_have_text("預測尚未產出")
    expect(page.locator("#status")).not_to_contain_text("排名未上線")


def test_today_loads_while_the_status_request_hangs_and_follows_it_when_it_answers(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    take_ranking_offline(20)
    held: list[Route] = []
    asked: list[str] = []
    page.route("**/api/status", lambda route: held.append(route))
    page.on("request", lambda request: asked.append(request.url))
    page.clock.install()
    page.clock.pause_at(timezone.now() + timedelta(seconds=5))

    page.goto(live_server.url)

    assert page.locator("[data-h='20']").get_attribute("aria-pressed") == "true"
    assert page.locator("[data-mkt='all']").get_attribute("aria-pressed") == "true"
    page.clock.fast_forward(1400)
    assert page.locator("#rows").text_content() == "載入中…"
    assert not [url for url in asked if "/api/predictions?" in url]
    page.clock.fast_forward(200)
    expect(page.locator("#today-empty")).to_have_text(
        f"預測資料尚未產出。排名{MODEL_REQUIRES}；上線後，這裡會列出全市場的排名。"
    )
    expect(page.locator("[data-h='20']")).to_have_attribute("aria-pressed", "true")

    held[0].continue_()

    expect(page.locator("#rows .trow")).to_have_count(20)
    expect(page.locator("[data-h='5']")).to_have_attribute("aria-pressed", "true")
    expect(page.locator("#status")).to_contain_text("20 日排名未上線")


def test_a_reader_who_picked_a_horizon_does_not_wait_for_the_status(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    page.goto(live_server.url)
    expect(page.locator("#rows .trow")).to_have_count(20)
    page.locator("[data-h='5']").click()
    expect(page.locator("#rows .trow").nth(0)).to_contain_text("9999")
    page.locator("[data-nav='models']").click()
    expect(page.locator("#view-models")).to_be_visible()
    page.clock.install()
    page.clock.pause_at(timezone.now() + timedelta(seconds=5))
    held: list[Route] = []
    page.route("**/api/status", lambda route: held.append(route))

    with page.expect_request(lambda request: "/api/predictions?" in request.url) as table:
        page.locator("[data-nav='today']").click()

    assert "horizon=5" in table.value.url
    assert len(held) == 1


def test_falling_back_to_twenty_days_starts_from_its_first_page(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    take_ranking_offline(20)
    page.goto(live_server.url)
    expect(page.locator("[data-h='5']")).to_have_attribute("aria-pressed", "true")
    page.locator("#next-page").click()
    expect(page.locator("#count")).to_have_text("第 21–25 名，共 25 檔")
    ModelRun.objects.filter(pk="rank-h20").update(status="active")
    add_predictions(
        ModelRun.objects.get(pk="rank-h20"),
        predicted.last_session,
        20,
        rank_rows(["2330", "9999", *EXTRA_STOCKS]),
    )

    page.locator("[data-nav='models']").click()
    expect(page.locator("#view-models")).to_be_visible()
    page.locator("[data-nav='today']").click()

    expect(page.locator("[data-h='20']")).to_have_attribute("aria-pressed", "true")
    expect(page.locator("#count")).to_have_text("第 1–20 名，共 25 檔")


def test_a_later_status_failure_keeps_what_the_page_last_knew(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    take_ranking_offline(20)
    page.goto(live_server.url)
    expect(page.locator("[data-h='5']")).to_have_attribute("aria-pressed", "true")
    page.locator("[data-nav='models']").click()
    expect(page.locator("#view-models")).to_be_visible()
    page.route("**/api/status", lambda route: route.abort())

    page.locator("[data-nav='today']").click()

    expect(page.locator("#status")).to_contain_text("無法連線到伺服器。")
    expect(page.locator("#rows .trow")).to_have_count(20)
    expect(page.locator("[data-h='5']")).to_have_attribute("aria-pressed", "true")

    page.locator("[data-h='20']").click()

    expect(page.locator("#today-empty")).to_have_text(no_ranking_text(20, 5))


def test_a_stock_s_late_predictions_do_not_move_today_s_horizon(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    Prediction.objects.filter(run_id="rank-h20", stock_id="9999").delete()
    held: list[Route] = []
    page.route("**/api/stocks/9999/predictions", lambda route: held.append(route))
    page.clock.install()
    page.goto(f"{live_server.url}/#stock-9999")
    expect(page.locator(".stockhead")).to_be_visible()
    page.locator("[data-nav='today']").click()
    expect(page.locator("#rows .trow")).to_have_count(20)
    page.clock.fast_forward(5000)

    held[0].continue_()

    expect(page.locator("#forecasts .trow")).to_have_count(2)
    assert page.locator("[data-h='20']").get_attribute("aria-pressed") == "true"
    assert page.locator("#ret-head").text_content() == "20 日預測報酬"


def test_a_status_answer_for_a_page_the_reader_left_does_not_load_today(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    take_ranking_offline(20)
    held: list[Route] = []
    asked: list[str] = []
    page.route("**/api/status", lambda route: held.append(route))
    page.on("request", lambda request: asked.append(request.url))
    page.clock.install()
    page.clock.pause_at(timezone.now() + timedelta(seconds=5))
    page.goto(live_server.url)
    page.goto(f"{live_server.url}/#stock-2330")
    expect(page.locator(".stockhead .kpi")).to_contain_text("5 日預測排名")

    held[0].continue_()

    expect(page.locator("#status")).to_contain_text("20 日排名未上線")
    page.clock.fast_forward(5000)
    assert page.locator("[data-nav='stock']").get_attribute("aria-current") == "page"
    assert not [url for url in asked if "/api/predictions?" in url]


def test_a_click_while_the_status_is_awaited_loads_the_table_once(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    held: list[Route] = []
    asked: list[str] = []
    page.route("**/api/status", lambda route: held.append(route))
    page.on("request", lambda request: asked.append(request.url))
    page.clock.install()
    page.clock.pause_at(timezone.now() + timedelta(seconds=5))
    page.goto(live_server.url)
    page.locator("[data-h='5']").click()
    expect(page.locator("#rows .trow").nth(0)).to_contain_text("9999")

    held[0].continue_()

    expect(page.locator("#status")).to_contain_text("收盤資料")
    page.clock.fast_forward(5000)
    assert page.evaluate("document.querySelectorAll('#rows .trow').length") == 20
    tables = [url for url in asked if "/api/predictions?" in url]
    assert len(tables) == 1
    assert "horizon=5" in tables[0]


def test_the_empty_page_follows_a_status_that_answers_after_the_reader_picked(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    take_ranking_offline(20)
    held: list[Route] = []
    asked: list[str] = []
    page.route("**/api/status", lambda route: held.append(route))
    page.on("request", lambda request: asked.append(request.url))
    page.clock.install()
    page.clock.pause_at(timezone.now() + timedelta(seconds=5))
    page.goto(live_server.url)
    page.locator("[data-h='20']").click()
    expect(page.locator("#today-empty")).to_have_text(
        f"預測資料尚未產出。排名{MODEL_REQUIRES}；上線後，這裡會列出全市場的排名。"
    )

    held[0].continue_()

    expect(page.locator("#today-empty")).to_have_text(no_ranking_text(20, 5))
    page.clock.fast_forward(5000)
    assert page.locator("[data-h='20']").get_attribute("aria-pressed") == "true"
    assert len([url for url in asked if "/api/predictions?" in url]) == 1


def test_an_unpicked_page_returns_to_twenty_days_when_no_ranking_is_live_any_more(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    take_ranking_offline(20)
    page.goto(live_server.url)
    expect(page.locator("[data-h='5']")).to_have_attribute("aria-pressed", "true")
    take_ranking_offline(5)
    page.locator("[data-nav='models']").click()
    expect(page.locator("#view-models")).to_be_visible()

    page.locator("[data-nav='today']").click()

    expect(page.locator("[data-h='20']")).to_have_attribute("aria-pressed", "true")
    expect(page.locator("#today-empty")).to_have_text(
        f"預測資料尚未產出。排名{MODEL_REQUIRES}；上線後，這裡會列出全市場的排名。"
    )
