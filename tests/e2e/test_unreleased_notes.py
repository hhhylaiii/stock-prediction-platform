import re
from datetime import date

import pytest
from playwright.sync_api import Page, expect
from pytest_django.live_server_helper import LiveServer

from apps.etl.models import DataQualityLog
from apps.research.models import (
    FeatureDaily,
    HoldoutRelease,
    HoldoutRun,
    ModelRun,
    ModelStatus,
    Prediction,
)
from tests.e2e.conftest import ONLINE, Seed, no_ranking_text, take_ranking_offline

pytestmark = [pytest.mark.e2e, pytest.mark.django_db(transaction=True)]

INTERVAL_KEY = "interval-holdout-20261112"
INTERVAL_RUNS = ["quantile-h20", "vol-h20", "prob-h20"]
UNRELEASED_MODELS = (
    "報酬區間、波動率與跑贏大盤機率（20 日）目前沒有上線中的模型："
    "這三個模型屬於「區間與機率」那一組，"
    "holdout（2025 年起，到 2026-11-11 已到期的預測，只跑一次）的絕對門檻都達標"
    "（20 日區間的總覆蓋率 75.0%，門檻 70% 到 90%），"
    "但逐摺平均的覆蓋率 70.9% 低於研究期最差的連續 8 摺 73.4%，判定「明顯退化」；"
    "使用者 2026-11-12 決定先不放行（之後可以再放行），所以沒有建正式模型。"
)
UNRELEASED_EMPTY = (
    "20 日的預測報酬、預測價、模型的 80% 區間、預測波動與跑贏大盤機率來自「區間與機率」那一組模型，"
    "這組也沒有上線：holdout 判定「明顯退化」，使用者 2026-11-12 決定先不放行。"
)
UNRELEASED_WHY = (
    "這幾個模型屬於「區間與機率」那一組，holdout 判定「明顯退化」，"
    "使用者 2026-11-12 決定先不放行；原因列在模型頁。"
)
UNRELEASED_STOCK = f"20 日排名目前沒有上線中的模型，這一列留空；{UNRELEASED_EMPTY}原因列在模型頁。"


def claim(run_key: str = INTERVAL_KEY) -> HoldoutRun:
    return HoldoutRun.objects.create(
        scope="interval", attempt=1, run_key=run_key, last_date=date(2026, 11, 11)
    )


def hold_interval_back(run_key: str = INTERVAL_KEY) -> HoldoutRun:
    """The interval scope was judged, never released, and none of its runs serve."""
    ModelRun.objects.filter(pk__in=INTERVAL_RUNS).update(status=ModelStatus.CANDIDATE)
    return claim(run_key)


@pytest.fixture
def valued(predicted: Seed) -> Seed:
    """2330's trailing volatility, so its 20-day band can fall back to it."""
    FeatureDaily.objects.create(
        stock_id="2330", date=predicted.last_session, in_universe=True, tech_vol_20d=0.22
    )
    return predicted


def settled(page: Page) -> None:
    """No section still loading, so the test ends with no request in flight."""
    expect(page.locator("#status, section.view:not([hidden])").get_by_text("載入中")).to_have_count(
        0
    )


def open_twenty(page: Page, live_server: LiveServer) -> None:
    page.goto(live_server.url)
    page.locator("#view-today [data-h='20']").click()


def test_the_models_page_says_why_the_interval_models_are_not_live(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    hold_interval_back()

    page.goto(f"{live_server.url}/#models")

    expect(page.locator("#model-table .unreleased-note")).to_have_text(UNRELEASED_MODELS)
    expect(page.locator("#model-table .scroll + .unreleased-note")).to_have_count(1)
    settled(page)


def test_the_interval_note_follows_the_held_ranking(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    take_ranking_offline(20)
    gate = {"name": "shadow_oos", "reason": "未達門檻", "value": -0.013, "threshold": "rank IC > 0"}
    held = {"objective": "rank", "horizon": 20, "run_id": "rank-n20", "gates": [gate]}
    DataQualityLog.objects.create(
        date=predicted.last_session,
        check_name="model_promotion",
        severity="warning",
        detail={"scope": "rank", "run_key": "rank-2026-11", "held_slots": [held]},
    )
    hold_interval_back()

    page.goto(f"{live_server.url}/#models")

    expect(page.locator("#model-table .held-note + .unreleased-note")).to_have_count(1)
    settled(page)


def test_today_twenty_without_a_ranking_adds_the_interval_models(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    take_ranking_offline(20)
    hold_interval_back()

    open_twenty(page, live_server)

    expect(page.locator("#today-empty")).to_have_text(no_ranking_text(20, 5) + UNRELEASED_EMPTY)
    page.locator("#view-today [data-h='5']").click()
    expect(page.locator("#rows .trow").first).to_be_visible()
    expect(page.locator("#view-today")).not_to_contain_text("區間與機率")
    settled(page)


def test_today_twenty_with_a_ranking_gives_the_interval_reason(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    hold_interval_back()

    open_twenty(page, live_server)

    expect(page.locator("#model-note")).to_have_text(
        "這一期沒有報酬區間、跑贏大盤機率模型的預測：預測報酬、預測價、跑贏大盤機率留空，"
        f"價格區間改用波動估計。{UNRELEASED_WHY}"
    )
    settled(page)


def test_watch_only_keeps_the_interval_reason(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    hold_interval_back()
    page.add_init_script("localStorage.setItem('twp-watch', JSON.stringify(['2330']))")

    open_twenty(page, live_server)
    page.locator("#watch-only").click()

    expect(page.locator("#count")).to_have_text("追蹤中的 1 檔")
    expect(page.locator("#model-note")).to_have_text(
        "這一期沒有報酬區間、跑贏大盤機率模型的預測：預測報酬、預測價、跑贏大盤機率留空，"
        f"價格區間改用波動估計。{UNRELEASED_WHY}"
    )
    settled(page)


def test_the_stock_page_names_a_horizon_without_a_ranking_model(
    page: Page, live_server: LiveServer, valued: Seed
) -> None:
    take_ranking_offline(20)
    hold_interval_back()

    page.goto(f"{live_server.url}/#stock-2330")

    expect(page.locator("#stock-seg-hint")).to_have_text("20 日排名目前沒有上線中的模型")
    note = page.locator("#forecasts .unreleased-note")
    expect(note).to_have_text(UNRELEASED_STOCK)
    expect(note.locator("a")).to_have_text("模型頁")
    expect(note.locator("a")).to_have_attribute("href", "#models")
    expect(page.locator("#forecasts .model-note")).to_have_count(0)
    expect(page.locator("#stock-seg [data-sh='20']")).to_be_disabled()
    expect(page.locator("#stock-seg [data-sh='5']")).to_have_attribute("aria-pressed", "true")
    expect(page.locator(".stockhead .kpi")).to_contain_text("5 日預測排名")
    expect(page.locator(".stockhead .kpi .v")).to_have_text(re.compile(r"^第 \d+ 名 / \d+$"))
    settled(page)


def test_the_stock_page_gives_the_interval_reason_once_twenty_ranks(
    page: Page, live_server: LiveServer, valued: Seed
) -> None:
    hold_interval_back()

    page.goto(f"{live_server.url}/#stock-2330")

    note = page.locator("#forecasts .model-note")
    expect(note).to_have_text(
        "20 日沒有報酬區間、波動率、跑贏大盤機率模型的預測："
        "預測報酬、預測價、預測 20 日波動、跑贏大盤機率留空，"
        f"價格區間改用波動估計。{UNRELEASED_WHY}"
    )
    expect(note.locator("a")).to_have_attribute("href", "#models")
    expect(page.locator("#forecasts .unreleased-note")).to_have_count(0)
    settled(page)


@pytest.mark.parametrize("case", ["released", "serving", "another_key"])
def test_the_interval_note_needs_an_unreleased_claim_and_nothing_serving(
    page: Page, live_server: LiveServer, valued: Seed, case: str
) -> None:
    take_ranking_offline(20)
    if case == "released":
        holdout = hold_interval_back()
        HoldoutRelease.objects.create(
            holdout=holdout, status="明顯退化", report_sha256="ab" * 32, reason="kept"
        )
    elif case == "serving":
        claim()
    else:
        hold_interval_back("interval-holdout-20261201")

    page.goto(f"{live_server.url}/#models")
    expect(page.locator("#model-table .trow").first).to_be_visible()
    expect(page.locator("#model-table .unreleased-note")).to_have_count(0)
    open_twenty(page, live_server)
    expect(page.locator("#today-empty")).to_be_visible()
    expect(page.locator("#today-empty")).not_to_contain_text("區間與機率")
    page.goto(f"{live_server.url}/#stock-2330")
    expect(page.locator("#stock-seg-hint")).to_have_text("20 日排名目前沒有上線中的模型")
    expect(page.locator("#forecasts .unreleased-note")).to_have_count(0)
    settled(page)


def test_a_ranking_model_without_this_stock_s_row_is_still_pending(
    page: Page, live_server: LiveServer, valued: Seed
) -> None:
    hold_interval_back()
    Prediction.objects.filter(run_id="rank-h20", stock_id="2330").delete()

    page.goto(f"{live_server.url}/#stock-2330")

    expect(page.locator("#stock-seg-hint")).to_have_text("20 日預測尚未產出")
    expect(page.locator("#forecasts .unreleased-note")).to_have_count(0)
    settled(page)


def test_five_day_notes_keep_the_general_reason(
    page: Page, live_server: LiveServer, valued: Seed
) -> None:
    ModelRun.objects.filter(pk="reg-h5").update(status=ModelStatus.CANDIDATE)
    hold_interval_back()

    page.goto(live_server.url)
    page.locator("#view-today [data-h='5']").click()
    expect(page.locator("#model-note")).to_have_text(
        f"這一期沒有絕對報酬模型的預測：預測報酬、預測價留空。{ONLINE}"
    )
    page.goto(f"{live_server.url}/#stock-2330")
    expect(page.locator("#forecasts .model-note")).to_have_text(
        "5 日沒有絕對報酬模型的預測：預測報酬、預測價留空。"
        "20 日沒有報酬區間、波動率、跑贏大盤機率模型的預測："
        "預測報酬、預測價、預測 20 日波動、跑贏大盤機率留空，"
        f"價格區間改用波動估計。{ONLINE}"
    )
    settled(page)


def test_an_empty_five_day_page_does_not_name_the_twenty_day_models(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    take_ranking_offline(5)
    hold_interval_back()

    page.goto(live_server.url)
    page.locator("#view-today [data-h='5']").click()

    expect(page.locator("#today-empty")).to_have_text(no_ranking_text(5, 20))
    settled(page)


def test_the_hint_names_a_five_day_horizon_without_a_ranking_model(
    page: Page, live_server: LiveServer, valued: Seed
) -> None:
    take_ranking_offline(5)

    page.goto(f"{live_server.url}/#stock-2330")

    expect(page.locator("#stock-seg-hint")).to_have_text("5 日排名目前沒有上線中的模型")
    settled(page)


def test_with_no_ranking_at_all_the_twenty_day_page_still_names_the_interval_models(
    page: Page, live_server: LiveServer, predicted: Seed
) -> None:
    take_ranking_offline(5)
    take_ranking_offline(20)
    hold_interval_back()

    page.goto(live_server.url)

    expect(page.locator("#today-empty")).to_have_text(
        "預測資料尚未產出。排名模型要在 holdout 判定達標之後才會上線，"
        "判定未達標或明顯退化時要經使用者放行；上線後，這裡會列出全市場的排名。" + UNRELEASED_EMPTY
    )
    settled(page)
