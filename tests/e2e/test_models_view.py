import json
from datetime import date, timedelta
from statistics import fmean, stdev

import pytest
from playwright.sync_api import Page, Route, expect
from pytest_django.live_server_helper import LiveServer

from apps.api.serving import HOLDOUT_RELEASE
from apps.market.calendar import TradingDays
from apps.research.models import BacktestMetric, ModelMonitor, ModelRun
from tests.e2e.conftest import Seed

pytestmark = pytest.mark.e2e

RESEARCH = (date(2014, 1, 2), date(2024, 12, 3))
RANK_ICS = [0.04 + 0.002 * n for n in range(10)]
XSS_VERSION = "<img src=x onerror=1>"


def metric(
    model_type: str,
    name: str,
    value: float,
    grain: str = "total",
    period: tuple[date, date] = RESEARCH,
) -> BacktestMetric:
    return BacktestMetric(
        run_key="rank-serving",
        model_type=model_type,
        objective="rank",
        horizon=20,
        metric=name,
        grain=grain,
        period_start=period[0],
        period_end=period[1],
        segment="",
        value=value,
    )


def month_starts(count: int) -> list[date]:
    year, month, starts = 2022, 11, []
    for _ in range(count):
        starts.append(date(year, month, 1))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return starts


MONTHS = month_starts(26)


def month_rows() -> list[BacktestMetric]:
    """26 months; two negative ones and a 2-day last month that is negative too."""
    rows = []
    for n, start in enumerate(MONTHS):
        end = (MONTHS[n + 1] if n + 1 < len(MONTHS) else date(2025, 1, 1)) - timedelta(days=1)
        last = start == MONTHS[-1]
        model = -0.01 if last else 0.04 + (-0.07 if n in (5, 9) else 0.0)
        rows += [
            metric("lightgbm", "rank_ic", model, "month", (start, end)),
            metric("ridge", "rank_ic", 0.02, "month", (start, end)),
            metric("lightgbm", "rank_ic_days", 2.0 if last else 21.0, "month", (start, end)),
        ]
    return rows


@pytest.fixture
def scored(predicted: Seed) -> Seed:
    """Monitoring for every serving run, research totals and 26 months of rank IC."""
    days = predicted.sessions[-10:]
    ModelRun.objects.update(train_end=TradingDays.from_db().shift(days[0], -20))
    monitors = {
        "rank-h5": {"ic": 0.03},
        "reg-h5": {"mae": 0.058, "direction_acc": 0.54},
        "quantile-h20": {"coverage": 0.79, "width": 0.131},
        "vol-h20": {"mae": 0.041},
        "prob-h20": {"auc": 0.58, "brier": 0.231},
    }
    ModelMonitor.objects.bulk_create(
        ModelMonitor(
            run_id=run_id, date=day, horizon=int(run_id[-2:].strip("h")), n_stocks=25, **values
        )
        for run_id, values in monitors.items()
        for day in days
    )
    ModelMonitor.objects.bulk_create(
        ModelMonitor(run_id="rank-h20", date=day, horizon=20, n_stocks=25, ic=ic)
        for day, ic in zip(days, RANK_ICS, strict=True)
    )
    BacktestMetric.objects.bulk_create(
        [
            metric("lightgbm", "rank_ic", 0.041),
            metric("lightgbm", "rank_icir", 0.62),
            metric("ridge", "rank_ic", 0.027),
            metric("ridge", "rank_icir", 0.41),
            *month_rows(),
        ]
    )
    return predicted


def test_the_serving_models_are_listed_with_their_scores(
    page: Page, live_server: LiveServer, scored: Seed
) -> None:
    ModelRun.objects.filter(pk="prob-h20").update(version=XSS_VERSION, status="degraded")
    ModelRun.objects.filter(pk="reg-h5").update(model_type=XSS_VERSION)

    page.goto(f"{live_server.url}/#models")

    rows = page.locator("#model-table .trow")
    expect(rows).to_have_count(6)
    icir = fmean(RANK_ICS) / stdev(RANK_ICS)
    trained = ModelRun.objects.get(pk="rank-h20").train_end
    for text in ("排名", "20 日", "LightGBM", "2026-10", f"2023-10 到 {trained:%Y-%m}"):
        expect(rows.nth(0)).to_contain_text(text)
    expect(rows.nth(0)).to_contain_text(f"rank IC 0.049，ICIR {icir:.2f}（近 10 個評估日）")
    expect(rows.nth(0)).to_contain_text("研究期 rank IC 0.041，ICIR 0.62；Ridge 0.027，ICIR 0.41")
    expect(rows.nth(1)).to_contain_text("5 日")
    expect(rows.nth(1)).to_contain_text("研究期成績尚未產出")
    expect(rows.nth(2).locator("div").last).to_have_text("—")
    expect(rows.nth(2)).to_contain_text(XSS_VERSION)
    expect(rows.nth(3)).to_contain_text("報酬區間")
    expect(rows.nth(2)).to_contain_text("MAE 5.8%，方向準確率 54.0%")
    expect(rows.nth(3)).to_contain_text("覆蓋率 79.0%，平均寬度 13.1 個百分點")
    expect(rows.nth(4)).to_contain_text("MAE 4.1 個百分點（年化）")
    expect(rows.nth(5)).to_contain_text("AUC 0.580，Brier 0.231")
    expect(rows.nth(5)).to_contain_text("降級中")
    expect(rows.nth(5).locator(".version")).to_have_text(XSS_VERSION)
    expect(page.locator("#model-table img")).to_have_count(0)
    rules = page.locator("#model-rules")
    expect(rules).to_contain_text("近 6 個月")
    expect(rules).to_contain_text("換版後接著累積")
    expect(rules).to_contain_text("報酬區間模型 20 日覆蓋率不在 70%–90%")
    expect(rules).to_contain_text("跑贏大盤機率模型 20 日 Brier 高於 baseline")
    expect(rules).to_contain_text("降級不會自動恢復，要人工 restore")
    expect(rules).to_contain_text("換模後的寬限期與 restore 後的重設期只告警、不降級")
    expect(rules).to_contain_text(
        "重設期以目標與期間為單位，約 60 + h 個交易日（20 日約 80 天、5 日約 65 天）"
    )


def test_a_degraded_model_says_since_when_and_whose_rows_judged_it(
    page: Page, live_server: LiveServer, scored: Seed
) -> None:
    window = {"own_rows": 20, "days": 60, "mostly_predecessors": True}
    ModelRun.objects.filter(pk="prob-h20").update(
        status="degraded", params={"degraded_on": "2026-10-20", "degraded_window": window}
    )
    ModelRun.objects.filter(pk="quantile-h20").update(
        status="degraded", params={"degraded_on": "2026-10-21"}
    )

    page.goto(f"{live_server.url}/#models")

    rows = page.locator("#model-table .trow")
    expect(rows.nth(5)).to_contain_text("降級中")
    expect(rows.nth(5).locator(".degraded")).to_have_text(
        "2026-10-20 起；窗口 20／60 列為本 run（33%），多為前任 run 的列"
    )
    expect(rows.nth(3).locator(".degraded")).to_have_text("2026-10-21 起")
    expect(rows.nth(0).locator(".degraded")).to_have_count(0)


RELEASE_REASON = f"差距在 seed 間標準差內，先上線觀察 {XSS_VERSION}"
RELEASE = {
    "status": "未達標",
    "failed": ["排名 20 日 IC"],
    "flagged": ["排名 5 日 IC"],
    "reason": RELEASE_REASON,
    "released_at": "2026-11-12T10:03:00.123456+08:00",
    "report": "/srv/host/reports/rank-holdout-20261111.md",
    "sha256": "ab" * 32,
}
NEXT = "xpath=following-sibling::*[1]"


def test_a_model_built_on_a_released_holdout_says_the_verdict_and_the_reason(
    page: Page, live_server: LiveServer, scored: Seed
) -> None:
    flagged = {**RELEASE, "status": "明顯退化", "failed": []}
    ModelRun.objects.filter(pk="rank-h20").update(params={"production": {HOLDOUT_RELEASE: RELEASE}})
    ModelRun.objects.filter(pk="prob-h20").update(params={"production": {HOLDOUT_RELEASE: flagged}})

    page.goto(f"{live_server.url}/#models")

    notes = page.locator("#model-table .released")
    expect(notes).to_have_count(2)
    expect(notes.nth(0).locator("div")).to_have_text(
        [
            "holdout 判定「未達標」，未達的門檻：排名 20 日 IC；退化旗標：排名 5 日 IC",
            f"2026-11-12 經使用者放行：{RELEASE_REASON}",
        ]
    )
    expect(notes.nth(1).locator("div").first).to_have_text(
        "holdout 判定「明顯退化」，退化旗標：排名 5 日 IC"
    )
    rows = page.locator("#model-table .trow")
    expect(rows).to_have_count(6)
    expect(rows.nth(0).locator(NEXT)).to_have_class("released")
    expect(rows.nth(1).locator(NEXT)).to_have_class("trow model-cols")
    expect(rows.nth(5).locator(NEXT)).to_have_class("released")
    expect(page.locator("#model-table")).not_to_contain_text("/srv/host")
    expect(page.locator("#model-table img")).to_have_count(0)
    expect(page.locator("#model-rules")).to_contain_text(
        "正式模型只建在 holdout 判定達標的設計上；判定未達標或明顯退化時，要經使用者放行才建，"
        "判定與放行原因列在上表受影響的模型下方。"
    )


def test_adjacent_models_on_one_release_share_a_single_note(
    page: Page, live_server: LiveServer, scored: Seed
) -> None:
    shared = ["rank-h20", "quantile-h20", "vol-h20", "prob-h20"]
    ModelRun.objects.filter(pk__in=shared).update(params={"production": {HOLDOUT_RELEASE: RELEASE}})

    page.goto(f"{live_server.url}/#models")

    notes = page.locator("#model-table .released")
    expect(notes).to_have_count(2)
    verdict = "holdout 判定「未達標」，未達的門檻：排名 20 日 IC；退化旗標：排名 5 日 IC"
    expect(notes.nth(0).locator("div").first).to_have_text(verdict)
    expect(notes.nth(1).locator("div")).to_have_text(
        [f"以上 3 個模型：{verdict}", f"2026-11-12 經使用者放行：{RELEASE_REASON}"]
    )
    rows = page.locator("#model-table .trow")
    expect(rows.nth(0).locator(NEXT)).to_have_class("released")
    expect(rows.nth(3).locator(NEXT)).to_have_class("trow model-cols")
    expect(rows.nth(4).locator(NEXT)).to_have_class("trow model-cols has-release")
    expect(rows.nth(5).locator(NEXT)).to_have_class("released")


def test_adjacent_models_on_different_releases_keep_their_own_notes(
    page: Page, live_server: LiveServer, scored: Seed
) -> None:
    other = {**RELEASE, "reason": "另一筆放行"}
    ModelRun.objects.filter(pk="rank-h20").update(params={"production": {HOLDOUT_RELEASE: RELEASE}})
    ModelRun.objects.filter(pk="rank-h5").update(params={"production": {HOLDOUT_RELEASE: other}})

    page.goto(f"{live_server.url}/#models")

    notes = page.locator("#model-table .released")
    expect(notes).to_have_count(2)
    expect(notes.nth(0).locator("div").last).to_have_text(
        f"2026-11-12 經使用者放行：{RELEASE_REASON}"
    )
    expect(notes.nth(1).locator("div").last).to_have_text("2026-11-12 經使用者放行：另一筆放行")
    expect(page.locator("#model-table")).not_to_contain_text("以上")
    rows = page.locator("#model-table .trow")
    expect(rows.nth(0).locator(NEXT)).to_have_class("released")
    expect(rows.nth(1).locator(NEXT)).to_have_class("released")


BARS_INSIDE_PLOT = """() => {
  const plot = document.querySelector('#ic-chart .overlay').getBBox();
  return [...document.querySelectorAll('#ic-chart rect.bar-model, #ic-chart rect.bar-baseline')]
    .map(bar => bar.getBBox())
    .every(box => box.y >= plot.y - 0.5 && box.y + box.height <= plot.y + plot.height + 0.5);
}"""


def test_the_monthly_chart_draws_24_months_and_marks_the_short_one(
    page: Page, live_server: LiveServer, scored: Seed
) -> None:
    page.goto(f"{live_server.url}/#models")

    chart = page.locator("#ic-chart")
    expect(chart.locator("rect.bar-model")).to_have_count(24)
    expect(chart.locator("rect.bar-baseline")).to_have_count(24)
    expect(chart.locator("rect.bar-model.partial")).to_have_count(1)
    assert page.evaluate(BARS_INSIDE_PLOT) is True
    sub = page.locator("#ic-sub")
    expect(sub).to_contain_text("LightGBM 有 21 個月高於 Ridge")
    expect(sub).to_contain_text("負值月份 2 個")
    expect(sub).to_contain_text("天數少於 10 天的月份不計")
    expect(page.locator("#ic-legend")).to_contain_text("LightGBM")
    expect(page.locator("#ic-legend")).to_contain_text("Ridge")


def test_the_chart_tooltip_names_the_month_and_both_values(
    page: Page, live_server: LiveServer, scored: Seed
) -> None:
    page.goto(f"{live_server.url}/#models")
    overlay = page.locator("#ic-chart .overlay")
    expect(overlay).to_have_count(1)
    box = overlay.bounding_box()
    assert box is not None

    overlay.hover(position={"x": box["width"] - 1, "y": box["height"] / 2})

    tip = page.locator("#ic-chart .tip")
    expect(tip).to_contain_text("2024 年 12 月")
    expect(tip).to_contain_text("LightGBM −0.010，Ridge 0.020")
    expect(tip).to_contain_text("只有 2 天")


def test_nothing_served_or_researched_yet(
    page: Page, live_server: LiveServer, seeded: Seed
) -> None:
    page.goto(f"{live_server.url}/#models")

    expect(page.locator("#model-table")).to_contain_text("模型尚未上線。")
    expect(page.locator("#ic-chart")).to_contain_text("回測結果尚未產出。")


def test_failed_model_requests_are_shown(page: Page, live_server: LiveServer, scored: Seed) -> None:
    page.route("**/api/models**", lambda route: route.abort())

    page.goto(f"{live_server.url}/#models")

    expect(page.locator("#model-table .error")).to_have_text("無法連線到伺服器。")
    expect(page.locator("#ic-chart .error")).to_have_text("無法連線到伺服器。")


def test_a_slot_without_matured_days_says_so(
    page: Page, live_server: LiveServer, scored: Seed
) -> None:
    ModelMonitor.objects.filter(run_id="vol-h20").delete()

    page.goto(f"{live_server.url}/#models")

    expect(page.locator("#model-table .trow").nth(4)).to_contain_text("尚無到期的評估日")


def test_month_labels_and_model_names_are_escaped(
    page: Page, live_server: LiveServer, scored: Seed
) -> None:
    name = '"><img src=x onerror=window.__x=1>'

    def tampered(route: Route) -> None:
        response = route.fetch()
        body = response.json()
        body["data"][0]["month"] = "2023-<img src=x onerror=window.__x=1>"
        body["meta"]["model_type"] = name
        route.fulfill(response=response, body=json.dumps(body))

    page.route("**/api/models/ic-monthly?*", tampered)
    page.goto(f"{live_server.url}/#models")

    expect(page.locator("#ic-chart rect.bar-model")).to_have_count(24)
    expect(page.locator("#ic-legend")).to_contain_text(name)
    expect(page.locator("#ic-chart img, #ic-legend img")).to_have_count(0)
    assert name in (page.locator("#ic-chart svg").get_attribute("aria-label") or "")
    assert page.evaluate("window.__x") is None
