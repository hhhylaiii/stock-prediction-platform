import pytest
from playwright.sync_api import Page, expect
from pytest_django.live_server_helper import LiveServer

from apps.research.models import (
    REST_FEATURE,
    FeatureDaily,
    LabelDaily,
    ModelRun,
    PredictionContrib,
)
from tests.api.conftest import add_predictions
from tests.e2e.conftest import Seed

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
        [("tech_ret_60d", 0.031), ("chip_fi_net_20d", -0.024), ("tech_future_<b>", 0.002)],
        (0.004, 72),
    )
    FeatureDaily.objects.create(
        stock_id="2330",
        date=predicted.last_session,
        in_universe=True,
        tech_ret_60d=0.12,
        tech_vol_20d=0.22,
        tech_vol_60d=0.25,
    )
    return predicted


@pytest.fixture
def matured(predicted: Seed) -> Seed:
    """Two matured 20-day predictions of 2330, one inside its band and one above it."""
    sessions = predicted.sessions
    ModelRun.objects.filter(pk="quantile-h20").update(train_end=sessions[0])
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
    expect(page.locator("#forecasts .sub")).to_contain_text(
        f"以 {explained.last_session.isoformat()} 收盤 1,138 換算"
    )


def test_the_horizon_switch_reranks_and_redraws(
    page: Page, live_server: LiveServer, explained: Seed
) -> None:
    add_contribs(explained, [("tech_rsi_14", 0.01)], (0.0, 76), horizon=5)
    page.goto(f"{live_server.url}/#stock-2330")
    expect(page.locator("#price-chart path.fan-band")).to_have_count(1)
    expect(page.locator("#chart-legend")).to_contain_text("80% 預測區間（內插）")
    expect(page.locator("#chart-legend")).to_contain_text("預測中位數（內插）")

    page.locator("#stock-seg [data-sh='5']").click()

    expect(page.locator(".stockhead .kpi .v")).to_have_text("第 2 名 / 25")
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
    expect(feats).to_have_count(3)
    expect(feats.nth(0)).to_contain_text("60 日報酬")
    expect(feats.nth(0)).to_contain_text("+12.0%")
    expect(feats.nth(0).locator(".contrib")).to_have_text("+0.031")
    expect(feats.nth(1).locator(".contrib")).to_have_text("−0.024")
    expect(feats.nth(2)).to_contain_text("tech_future_<b>")
    expect(page.locator("#features b")).to_have_count(0)
    expect(page.locator("#features .note")).to_have_text("共 75 個特徵，其餘 72 個合計 +0.004。")


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
    expect(page.locator("#price-chart path.fan-median")).to_have_count(0)


def test_a_failed_contribution_request_keeps_the_rest(
    page: Page, live_server: LiveServer, explained: Seed
) -> None:
    page.route("**/api/stocks/2330/features*", lambda route: route.abort())

    page.goto(f"{live_server.url}/#stock-2330")

    expect(page.locator("#features .error")).to_have_text("無法連線到伺服器。")
    expect(page.locator("#forecasts .trow")).to_have_count(2)
