import pytest
from playwright.sync_api import Page, Route, expect
from pytest_django.live_server_helper import LiveServer

from apps.market.calendar import TradingDays
from apps.market.models import DailyPrice
from apps.research.models import (
    REST_FEATURE,
    FeatureDaily,
    LabelDaily,
    ModelRun,
    Prediction,
    PredictionContrib,
)
from tests.api.conftest import add_model_run, add_predictions
from tests.e2e.conftest import (
    BAND_NOTE,
    PROB_HEAD,
    PROB_TIP,
    RELEASE_REASON,
    Seed,
    released_on,
)

pytestmark = pytest.mark.e2e


def add_contribs(
    seed: Seed, top: list[tuple[str, float]], rest: tuple[float, int], horizon: int = 20
) -> None:
    rows = [
        PredictionContrib(
            run_id=f"rank-h{horizon}",
            stock_id="2330",
            date=seed.last_session,
            horizon=horizon,
            feature=feature,
            position=position,
            contrib=value,
        )
        for position, (feature, value) in enumerate(top, start=1)
    ]
    rows.append(
        PredictionContrib(
            run_id=f"rank-h{horizon}",
            stock_id="2330",
            date=seed.last_session,
            horizon=horizon,
            feature=REST_FEATURE,
            position=len(top) + 1,
            contrib=rest[0],
            n_features=rest[1],
        )
    )
    PredictionContrib.objects.bulk_create(rows)


@pytest.fixture
def explained(predicted: Seed) -> Seed:
    """Contributions, raw values and the realised volatility of 2330 on the last session."""
    add_contribs(
        predicted,
        [
            ("tech_ret_60d", 0.031),
            ("chip_fi_net_20d", -0.024),
            ("tech_future_<b>", 0.002),
            ("fund_dividend_yield", 0.0015),
            ("tech_vol_20d", -0.001),
        ],
        (0.004, 72),
    )
    FeatureDaily.objects.create(
        stock_id="2330",
        date=predicted.last_session,
        in_universe=True,
        tech_ret_60d=0.12,
        tech_vol_20d=0.22,
        tech_vol_60d=0.25,
        fund_dividend_yield=0.0345,
    )
    return predicted


@pytest.fixture
def matured(predicted: Seed) -> Seed:
    """Two matured 20-day predictions of 2330, one inside its band and one above it."""
    sessions = predicted.sessions
    trained = TradingDays.from_db().shift(sessions[-61], -20)
    ModelRun.objects.filter(pk="quantile-h20").update(train_end=trained)
    quantile = ModelRun.objects.get(pk="quantile-h20")
    band = {"q10": -0.05, "q50": 0.02, "q90": 0.12, "ret_pred": 0.02}
    for day, actual in ((sessions[-61], 0.2), (sessions[-41], 0.03)):
        add_predictions(quantile, day, 20, {"2330": band})
        LabelDaily.objects.create(
            stock_id="2330", date=day, horizon=20, fwd_ret=actual, fwd_ret_open=actual
        )
    return predicted


def test_the_header_ranks_the_stock_and_the_table_lists_both_horizons(
    page: Page, live_server: LiveServer, explained: Seed
) -> None:
    page.goto(f"{live_server.url}/#stock-2330")

    expect(page.locator(".stockhead .kpi")).to_contain_text("20 日預測排名")
    expect(page.locator(".stockhead .kpi .v")).to_have_text("第 1 名 / 25")
    rows = page.locator("#forecasts .trow")
    expect(rows).to_have_count(2)
    for text in ("5 日", "+1.0%", "1,149"):
        expect(rows.nth(0)).to_contain_text(text)
    expect(rows.nth(0).locator(".band-text")).to_have_text("—")
    expect(rows.nth(0).locator(".prob")).to_have_text("—")
    for text in ("20 日", "+4.0%", "1,184", "1,081 – 1,275", "60%"):
        expect(rows.nth(1)).to_contain_text(text)
    pair = page.locator("#forecasts .pair")
    expect(pair).to_contain_text("30.0%")
    expect(pair).to_contain_text("過去 60 日實際為 25.0%")
    expect(pair).to_contain_text("LightGBM 2026-10")
    expect(page.locator("#forecasts .sub")).to_have_text(
        "排名、報酬與區間機率各由自己的模型產出；排名模型多用了幾個籌碼與除息日特徵，其餘特徵相同。"
        f"價格以 {explained.last_session.isoformat()} 收盤 1,138 換算。"
    )
    head = page.locator("#forecasts .thead .prob-head")
    expect(head).to_have_text(PROB_HEAD)
    expect(head).to_have_attribute("title", PROB_TIP)
    expect(page.locator("#forecasts .band-note")).to_have_text(BAND_NOTE)
    expect(page.locator("#forecasts .released-note")).to_have_count(0)


def test_the_forecasts_name_the_models_that_went_live_on_a_released_holdout(
    page: Page, live_server: LiveServer, explained: Seed
) -> None:
    ModelRun.objects.filter(pk__in=["rank-h5", "rank-h20"]).update(params=released_on("未達標"))
    ModelRun.objects.filter(pk__in=["quantile-h20", "vol-h20"]).update(
        params=released_on("明顯退化")
    )

    page.goto(f"{live_server.url}/#stock-2330")

    note = page.locator("#forecasts .released-note")
    expect(note).to_have_text(
        "排名 20 日、排名 5 日模型的 holdout 判定未達標，經使用者放行上線；"
        "報酬區間 20 日、波動率 20 日模型的 holdout 判定明顯退化，經使用者放行上線。原因在模型頁"
    )
    expect(note.locator("a")).to_have_attribute("href", "#models")
    expect(page.locator("#forecasts")).not_to_contain_text(RELEASE_REASON)


def test_a_model_serving_both_horizons_on_different_days_is_named_once(
    page: Page, live_server: LiveServer, explained: Seed
) -> None:
    last, previous = explained.last_session, explained.sessions[-2]
    Prediction.objects.filter(run_id="rank-h20", date=last).delete()
    add_predictions(
        add_model_run("vol-old", "volatility"), previous, 20, {"2330": {"vol_pred": 0.3}}
    )
    ModelRun.objects.filter(pk__in=["vol-h20", "vol-old"]).update(params=released_on("明顯退化"))

    page.goto(f"{live_server.url}/#stock-2330")

    expect(page.locator("#forecasts .released-note")).to_have_text(
        "波動率 20 日模型的 holdout 判定明顯退化，經使用者放行上線。原因在模型頁"
    )


def test_a_volatility_band_leaves_out_the_quantile_coverage_note(
    page: Page, live_server: LiveServer, explained: Seed
) -> None:
    ModelRun.objects.filter(pk="quantile-h20").update(
        status="degraded", params={"degraded_on": explained.sessions[0].isoformat()}
    )
    page.goto(f"{live_server.url}/#stock-2330")

    rows = page.locator("#forecasts .trow")
    expect(rows.nth(1).locator(".band-text")).to_contain_text("（波動率區間）")
    expect(page.locator("#forecasts .note")).to_have_text("有模型降級中，結果僅供參考。")
    expect(page.locator("#forecasts .band-note")).to_have_count(0)


def test_the_horizon_switch_reranks_and_redraws(
    page: Page, live_server: LiveServer, explained: Seed
) -> None:
    add_contribs(explained, [("tech_rsi_14", 0.01)], (0.0, 76), horizon=5)
    page.goto(f"{live_server.url}/#stock-2330")
    expect(page.locator("#price-chart path.fan-band")).to_have_count(1)
    expect(page.locator("#chart-title")).to_have_text("過去 60 個交易日，與未來 20 日的預測")
    expect(page.locator("#stock-seg button:disabled")).to_have_count(0)
    expect(page.locator("#stock-seg-hint")).to_have_text("")
    expect(page.locator("#chart-legend")).to_contain_text("80% 預測區間（內插）")
    expect(page.locator("#chart-legend")).to_contain_text("預測中位數（內插）")

    page.locator("#stock-seg [data-sh='5']").click()

    expect(page.locator(".stockhead .kpi .v")).to_have_text("第 2 名 / 25")
    expect(page.locator("#chart-title")).to_have_text("過去 60 個交易日，與未來 5 日的預測")
    expect(page.locator("#price-chart path.fan-median")).to_have_count(1)
    expect(page.locator("#price-chart path.fan-band")).to_have_count(0)
    expect(page.locator("#chart-legend")).not_to_contain_text("80% 預測區間")
    expect(page.locator("#ret-head")).to_have_text("5 日預測報酬")
    expect(page.locator("#features .feat")).to_have_count(1)
    expect(page.locator("#features .feat")).to_contain_text("RSI（14）")


def test_contributions_carry_chinese_names_and_raw_values(
    page: Page, live_server: LiveServer, explained: Seed
) -> None:
    page.goto(f"{live_server.url}/#stock-2330")

    feats = page.locator("#features .feat")
    expect(feats).to_have_count(5)
    expect(feats.nth(0)).to_contain_text("60 日報酬")
    expect(feats.nth(0)).to_contain_text("+12.0%")
    expect(feats.nth(0).locator(".contrib")).to_have_text("+0.031")
    expect(feats.nth(1).locator(".contrib")).to_have_text("−0.024")
    expect(feats.nth(2)).to_contain_text("tech_future_<b>")
    expect(page.locator("#features b")).to_have_count(0)
    expect(feats.nth(3)).to_contain_text("殖利率")
    expect(feats.nth(3).locator(".value")).to_have_text("3.45%")
    expect(feats.nth(4).locator(".value")).to_have_text("22.0%")
    expect(page.locator("#features .note")).to_have_text("共 77 個特徵，其餘 72 個合計 +0.004。")
    expect(page.locator("#features .recompute")).to_have_text(
        "特徵值取自目前的特徵表，每晚重算近期資料後可能更新，和預測當時不一定相同；貢獻值是預測當時算的。"
    )


def test_history_counts_the_returns_inside_the_band(
    page: Page, live_server: LiveServer, matured: Seed
) -> None:
    page.goto(f"{live_server.url}/#stock-2330")

    rows = page.locator("#history .trow")
    expect(rows).to_have_count(2)
    expect(rows.nth(0)).to_contain_text(matured.sessions[-41].isoformat())
    expect(rows.nth(0)).to_contain_text("落在區間內")
    expect(rows.nth(1)).to_contain_text("+20.0%")
    expect(rows.nth(1)).to_contain_text("超出區間")
    expect(page.locator("#history .sechead")).to_contain_text(
        "最近 2 次有區間的預測中，實際報酬有 1 次落在 80% 區間內"
    )


def test_a_stock_without_predictions_says_so(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    page.goto(f"{live_server.url}/#stock-2330")

    expect(page.locator(".stockhead .kpi .v")).to_contain_text("尚未產出")
    expect(page.locator("#forecasts")).to_contain_text("預測資料尚未產出。")
    expect(page.locator("#features")).to_contain_text("預測資料尚未產出。")
    expect(page.locator("#history")).to_contain_text("還沒有到期的預測。")
    expect(page.locator("#chart-title")).to_have_text("過去 60 個交易日")
    expect(page.locator("#price-chart path.fan-median")).to_have_count(0)
    expect(page.locator("#stock-seg button:disabled")).to_have_count(2)
    expect(page.locator("#stock-seg [data-sh='5']")).to_have_css("color", "rgb(107, 106, 100)")
    expect(page.locator("#stock-seg-hint")).to_have_text("預測產出後可切換 5 日／20 日")


def test_a_horizon_without_a_prediction_cannot_be_picked(
    page: Page, live_server: LiveServer, explained: Seed
) -> None:
    Prediction.objects.filter(stock_id="2330", horizon=5).delete()
    page.goto(f"{live_server.url}/#stock-2330")

    expect(page.locator("#stock-seg [data-sh='5']")).to_be_disabled()
    expect(page.locator("#stock-seg [data-sh='20']")).to_be_enabled()
    expect(page.locator("#stock-seg-hint")).to_have_text("5 日預測尚未產出")
    expect(page.locator("#chart-title")).to_have_text("過去 60 個交易日，與未來 20 日的預測")


def test_a_failed_contribution_request_keeps_the_rest(
    page: Page, live_server: LiveServer, explained: Seed
) -> None:
    page.route("**/api/stocks/2330/features*", lambda route: route.abort())

    page.goto(f"{live_server.url}/#stock-2330")

    expect(page.locator("#features .error")).to_have_text("無法連線到伺服器。")
    expect(page.locator("#forecasts .trow")).to_have_count(2)


def test_the_rank_shows_when_the_stock_is_outside_the_universe(
    page: Page, live_server: LiveServer, explained: Seed
) -> None:
    Prediction.objects.filter(
        run_id="rank-h20", stock_id="2330", date=explained.last_session
    ).update(in_universe=False, rank=None, pct=None)

    page.goto(f"{live_server.url}/#stock-2330")

    expect(page.locator(".stockhead .kpi .v")).to_contain_text("訓練範圍外")


def test_a_failed_forecast_request_is_shown(
    page: Page, live_server: LiveServer, explained: Seed
) -> None:
    page.route("**/api/stocks/2330/predictions", lambda route: route.abort())

    page.goto(f"{live_server.url}/#stock-2330")

    expect(page.locator(".stockhead .kpi .v")).to_contain_text("讀取失敗")
    expect(page.locator("#forecasts .error")).to_have_text("無法連線到伺服器。")
    expect(page.locator("#features .feat")).to_have_count(5)


def one_row_pages(route: Route) -> None:
    route.continue_(url=route.request.url.replace("page_size=8", "page_size=1"))


def test_history_expands_to_every_prediction(
    page: Page, live_server: LiveServer, matured: Seed
) -> None:
    page.route("**/api/stocks/2330/history?*", one_row_pages)
    page.goto(f"{live_server.url}/#stock-2330")
    expect(page.locator("#history .trow")).to_have_count(1)

    page.get_by_role("button", name="顯示全部 2 次預測").click()

    expect(page.locator("#history .trow")).to_have_count(2)


def test_a_failed_expansion_keeps_the_rows(
    page: Page, live_server: LiveServer, matured: Seed
) -> None:
    def history(route: Route) -> None:
        if "page_size=100" in route.request.url:
            route.abort()
        else:
            one_row_pages(route)

    page.route("**/api/stocks/2330/history?*", history)
    page.goto(f"{live_server.url}/#stock-2330")

    page.get_by_role("button", name="顯示全部 2 次預測").click()

    expect(page.locator("#history .error")).to_have_text("無法連線到伺服器。")
    expect(page.locator("#history .trow")).to_have_count(1)
    expect(page.get_by_role("button", name="顯示全部 2 次預測")).to_be_enabled()


GAP_AT_FAN_START = """() => {
  const coords = d => d.split(/[ML]/).filter(Boolean).map(p => p.trim().split(' ').map(Number));
  const chart = document.querySelector('#price-chart');
  const start = coords(chart.querySelector('path.fan-median').getAttribute('d'))[0];
  const line = coords(chart.querySelector('path.close-line').getAttribute('d'));
  const bar = line.find(p => Math.abs(p[0] - start[0]) < 0.01);
  return bar ? Math.abs(bar[1] - start[1]) : null;
}"""


def test_the_fan_starts_on_the_adjusted_bar_of_its_prediction_day(
    page: Page, live_server: LiveServer, explained: Seed
) -> None:
    day = explained.sessions[-2]
    DailyPrice.objects.filter(stock_id="2330", date__lte=day).update(adj_factor=0.95)

    def earlier(route: Route) -> None:
        response = route.fetch()
        body = response.json()
        for entry in body["data"]["horizons"]:
            if entry["prediction"]:
                entry["date"] = day.isoformat()
                entry["prediction"]["close_date"] = day.isoformat()
                entry["prediction"]["close"] = explained.closes[-2]
        route.fulfill(response=response, json=body)

    page.route("**/api/stocks/2330/predictions", earlier)
    page.goto(f"{live_server.url}/#stock-2330")
    page.get_by_role("button", name="收盤線").click()
    expect(page.locator("#price-chart path.fan-median")).to_have_count(1)

    gap = page.evaluate(GAP_AT_FAN_START)
    assert gap is not None
    assert gap < 0.5


def test_the_forecast_area_has_its_own_tooltip(
    page: Page, live_server: LiveServer, explained: Seed
) -> None:
    page.goto(f"{live_server.url}/#stock-2330")
    expect(page.locator("#price-chart path.fan-median")).to_have_count(1)
    overlay = page.locator("#price-chart .overlay")
    box = overlay.bounding_box()
    assert box is not None

    overlay.hover(position={"x": box["width"] - 1, "y": box["height"] / 2})

    tip = page.locator("#price-chart .tip")
    expect(tip).to_contain_text("+20 個交易日")
    expect(tip).to_contain_text("預測中位數 1,184")
    expect(tip).to_contain_text("80% 區間 1,081 – 1,275")
    expect(page.locator("#price-chart svg")).to_have_attribute(
        "aria-label", "台積電過去 60 個交易日K 線與成交量，與未來 20 日的預測"
    )


def test_switching_the_horizon_here_sends_today_back_to_page_one(
    page: Page, live_server: LiveServer, explained: Seed
) -> None:
    page.goto(live_server.url)
    page.locator("#next-page").click()
    expect(page.locator("#count")).to_have_text("第 21–25 名，共 25 檔")

    page.goto(f"{live_server.url}/#stock-2330")
    page.locator("#stock-seg [data-sh='5']").click()
    page.locator("[data-nav=today]").click()

    expect(page.locator("#count")).to_have_text("第 1–20 名，共 25 檔")


LAST_POINTS = """() => {
  const coords = d => d.split(/[ML]/).filter(Boolean).map(p => p.trim().split(' ').map(Number));
  const chart = document.querySelector('#price-chart');
  const start = coords(chart.querySelector('path.fan-median').getAttribute('d'))[0];
  const line = coords(chart.querySelector('path.close-line').getAttribute('d'));
  return { start: start, last: line[line.length - 1] };
}"""


def test_a_halted_prediction_day_starts_the_fan_on_that_day(
    page: Page, live_server: LiveServer, explained: Seed
) -> None:
    def halted(route: Route) -> None:
        response = route.fetch()
        body = response.json()
        for key in ("open", "high", "low", "close", "volume", "turnover"):
            body["data"][-1][key] = None
        route.fulfill(response=response, json=body)

    def earlier_close(route: Route) -> None:
        response = route.fetch()
        body = response.json()
        for entry in body["data"]["horizons"]:
            if entry["prediction"]:
                entry["prediction"]["close_date"] = explained.sessions[-2].isoformat()
        route.fulfill(response=response, json=body)

    page.route("**/api/stocks/2330/prices*", halted)
    page.route("**/api/stocks/2330/predictions", earlier_close)
    page.goto(f"{live_server.url}/#stock-2330")
    page.get_by_role("button", name="收盤線").click()
    expect(page.locator("#price-chart path.fan-median")).to_have_count(1)

    points = page.evaluate(LAST_POINTS)
    assert points["start"][0] > points["last"][0]
    assert abs(points["start"][1] - points["last"][1]) < 0.5


BAND_HOLDS_MEDIAN = """() => {
  const coords = d => d.replace(' Z', '').split(/[ML]/).filter(Boolean)
    .map(p => p.trim().split(' ').map(Number));
  const chart = document.querySelector('#price-chart');
  const median = coords(chart.querySelector('path.fan-median').getAttribute('d'));
  const band = coords(chart.querySelector('path.fan-band').getAttribute('d'));
  const top = band.slice(0, median.length), bottom = band.slice(median.length).reverse();
  return median.every((p, t) => top[t][1] <= p[1] + 1e-6 && p[1] <= bottom[t][1] + 1e-6);
}"""


def test_the_band_holds_the_median_when_both_quantiles_are_up(
    page: Page, live_server: LiveServer, explained: Seed
) -> None:
    Prediction.objects.filter(run_id="quantile-h20", stock_id="2330").update(
        q10=0.02, q50=0.06, q90=0.12, ret_pred=0.06
    )

    page.goto(f"{live_server.url}/#stock-2330")
    expect(page.locator("#price-chart path.fan-band")).to_have_count(1)

    assert page.evaluate(BAND_HOLDS_MEDIAN) is True


def test_every_unit_reads_as_its_kind(page: Page, live_server: LiveServer, explained: Seed) -> None:
    FeatureDaily.objects.filter(stock_id="2330", date=explained.last_session).update(
        liq_amihud_20d=3.2e-12, tech_limit_up_cnt_20d=2, tech_attention_t=1, fund_gm_yoy_chg=0.012
    )
    add_contribs(
        explained,
        [
            ("liq_amihud_20d", 0.004),
            ("tech_limit_up_cnt_20d", 0.003),
            ("tech_attention_t", -0.002),
            ("fund_gm_yoy_chg", 0.001),
        ],
        (0.0, 73),
        horizon=5,
    )
    page.goto(f"{live_server.url}/#stock-2330")

    page.locator("#stock-seg [data-sh='5']").click()

    values = page.locator("#features .feat .value")
    expect(values).to_have_text(["3.20×10⁻¹²", "2", "是", "+1.20 個百分點"])


def test_a_ranked_stock_without_contributions_says_so(
    page: Page, live_server: LiveServer, explained: Seed
) -> None:
    page.goto(f"{live_server.url}/#stock-9999")

    expect(page.locator("#features")).to_contain_text("這一期沒有這檔股票的特徵貢獻。")


def test_the_header_and_forecasts_do_not_wait_for_the_history(
    page: Page, live_server: LiveServer, explained: Seed
) -> None:
    held: list[Route] = []
    page.route("**/api/stocks/2330/history?*", lambda route: held.append(route))
    page.goto(f"{live_server.url}/#stock-2330")

    expect(page.locator(".stockhead .kpi .v")).to_have_text("第 1 名 / 25")
    expect(page.locator("#forecasts .trow")).to_have_count(2)
    expect(page.locator("#history .placeholder")).to_have_text("載入中…")
    assert len(held) == 1
    held[0].continue_()
    expect(page.locator("#history")).to_contain_text("還沒有到期的預測。")


def test_forecasts_without_a_fan_keep_the_hover(
    page: Page, live_server: LiveServer, explained: Seed
) -> None:
    held: list[Route] = []
    page.route("**/api/stocks/9999/predictions", lambda route: held.append(route))
    page.goto(f"{live_server.url}/#stock-9999")
    overlay = page.locator("#price-chart .overlay")
    expect(overlay).to_have_count(1)
    box = overlay.bounding_box()
    assert box is not None
    overlay.hover(position={"x": box["width"] - 1, "y": box["height"] / 2})
    expect(page.locator("#price-chart .tip")).to_be_visible()

    held[0].continue_()
    expect(page.locator(".stockhead .kpi .v")).to_have_text("第 2 名 / 25")

    expect(page.locator("#price-chart .tip")).to_be_visible()


def test_each_horizon_names_its_own_close_when_they_differ(
    page: Page, live_server: LiveServer, explained: Seed
) -> None:
    def older_five(route: Route) -> None:
        response = route.fetch()
        body = response.json()
        for entry in body["data"]["horizons"]:
            if entry["horizon"] == 5:
                entry["prediction"]["close_date"] = explained.sessions[-2].isoformat()
                entry["prediction"]["close"] = explained.closes[-2]
        route.fulfill(response=response, json=body)

    page.route("**/api/stocks/2330/predictions", older_five)
    page.goto(f"{live_server.url}/#stock-2330")

    expect(page.locator("#forecasts .sub")).to_contain_text(
        f"20 日以 {explained.last_session.isoformat()} 收盤 1,138、"
        f"5 日以 {explained.sessions[-2].isoformat()} 收盤 1,136 換算"
    )
