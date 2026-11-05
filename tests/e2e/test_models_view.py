from datetime import date, timedelta

import pytest
from playwright.sync_api import Page, expect
from pytest_django.live_server_helper import LiveServer

from apps.research.models import BacktestMetric, ModelMonitor, ModelRun
from tests.e2e.conftest import Seed

pytestmark = pytest.mark.e2e

RESEARCH = (date(2014, 1, 2), date(2024, 12, 3))
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
    monitors = {
        "rank-h20": {"ic": 0.05},
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

    page.goto(f"{live_server.url}/#models")

    rows = page.locator("#model-table .trow")
    expect(rows).to_have_count(6)
    for text in ("排名", "20 日", "LightGBM", "2026-10", "2023-10 到 2026-09", "rank IC 0.050"):
        expect(rows.nth(0)).to_contain_text(text)
    expect(rows.nth(0)).to_contain_text("研究期 rank IC 0.041，ICIR 0.62；Ridge 0.027，ICIR 0.41")
    expect(rows.nth(1)).to_contain_text("5 日")
    expect(rows.nth(2)).to_contain_text("MAE 5.8%，方向準確率 54.0%")
    expect(rows.nth(3)).to_contain_text("覆蓋率 79.0%，平均寬度 13.1 個百分點")
    expect(rows.nth(4)).to_contain_text("MAE 4.1 個百分點（年化）")
    expect(rows.nth(5)).to_contain_text("AUC 0.580，Brier 0.231")
    expect(rows.nth(5)).to_contain_text("降級中")
    expect(rows.nth(5).locator(".version")).to_have_text(XSS_VERSION)
    expect(page.locator("#model-table img")).to_have_count(0)
    expect(rows.nth(0)).to_contain_text("近 10 個評估日")
    expect(page.locator("#model-rules")).to_contain_text("覆蓋率不在 70%–90%")


def test_the_monthly_chart_draws_24_months_and_marks_the_short_one(
    page: Page, live_server: LiveServer, scored: Seed
) -> None:
    page.goto(f"{live_server.url}/#models")

    chart = page.locator("#ic-chart")
    expect(chart.locator("rect.bar-model")).to_have_count(24)
    expect(chart.locator("rect.bar-baseline")).to_have_count(24)
    expect(chart.locator("rect.bar-model.partial")).to_have_count(1)
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
