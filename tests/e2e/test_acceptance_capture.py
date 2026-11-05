import os
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from django.core.cache import cache
from django.core.management import call_command
from playwright.sync_api import Page, expect
from pytest_django.live_server_helper import LiveServer

from apps.etl.models import DataQualityLog, EtlRun
from apps.market.models import DailyPrice, Market, StockMaster
from apps.research.features.registry import FEATURES
from apps.research.models import (
    REST_FEATURE,
    BacktestMetric,
    FeatureDaily,
    LabelDaily,
    ModelMonitor,
    ModelRun,
    Prediction,
    PredictionContrib,
)
from tests.e2e.conftest import Seed
from tests.e2e.test_models_view import metric, month_rows

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.environ.get("ACCEPTANCE_OUT"), reason="set ACCEPTANCE_OUT to capture the report"
    ),
]

STOCKS = 1842
RANKS = ("rank-h20", "rank-h5")
RAW_2330 = {
    "tech_ret_1d": 0.004,
    "tech_ret_5d": -0.012,
    "tech_ret_20d": 0.035,
    "tech_ret_60d": 0.12,
    "tech_mom_250_20": 0.28,
    "tech_ma_gap_5d": 0.006,
    "tech_ma_gap_20d": 0.018,
    "tech_ma_gap_60d": 0.07,
    "tech_vol_60d": 0.28,
    "tech_volume_ratio_5_20": 1.12,
    "tech_turnover_rate_5d": 0.0042,
}
HORIZONS = {
    "rank-h20": 20,
    "rank-h5": 5,
    "quantile-h20": 20,
    "prob-h20": 20,
    "vol-h20": 20,
    "reg-h5": 5,
}
MONITORS: dict[str, dict[str, float]] = {
    "rank-h20": {"ic": 0.048},
    "rank-h5": {"ic": 0.031},
    "reg-h5": {"mae": 0.058, "direction_acc": 0.54},
    "quantile-h20": {"coverage": 0.79, "width": 0.131},
    "vol-h20": {"mae": 0.041},
    "prob-h20": {"auc": 0.58, "brier": 0.231},
}


def fields(run_id: str, n: int) -> dict[str, Any]:
    """Prediction values of the n-th stock for one serving run."""
    ret = 0.06 - 0.12 * n / STOCKS
    return {
        "rank-h20": {"score": 1 - n / STOCKS, "rank": n + 1, "pct": 1 - n / STOCKS},
        "rank-h5": {"score": 1 - n / STOCKS, "rank": n + 1, "pct": 1 - n / STOCKS},
        "quantile-h20": {"q10": ret - 0.09, "q50": ret, "q90": ret + 0.08, "ret_pred": ret},
        "prob-h20": {"prob_up_5pct": 0.55 - 0.3 * n / STOCKS},
        "vol-h20": {"vol_pred": 0.25 + 0.2 * n / STOCKS},
        "reg-h5": {"ret_pred": ret / 3},
    }[run_id]


def add_universe(last: date) -> list[str]:
    """Fill the market to 1,842 stocks with one bar and a full set of predictions each."""
    codes = [
        f"{3000 + n}"
        for n in range(STOCKS - Prediction.objects.filter(run_id="rank-h20", date=last).count())
    ]
    StockMaster.objects.bulk_create(
        StockMaster(
            stock_id=code,
            name=f"壓測{n:04d}",
            market=Market.TPEX if n % 3 == 0 else Market.TWSE,
            industry="其他業",
        )
        for n, code in enumerate(codes)
    )
    DailyPrice.objects.bulk_create(
        DailyPrice(
            stock_id=code,
            date=last,
            open=50,
            high=52,
            low=48,
            close=50 + n % 400,
            volume=1_000_000,
            turnover=50_000_000,
        )
        for n, code in enumerate(codes)
    )
    offset = STOCKS - len(codes)
    every = list(StockMaster.objects.order_by("stock_id").values_list("stock_id", flat=True))
    ranked = [(run_id, offset + n, code) for run_id in RANKS for n, code in enumerate(codes)]
    others = [
        (run_id, n, code)
        for run_id in HORIZONS
        if run_id not in RANKS
        for n, code in enumerate(every)
    ]
    Prediction.objects.bulk_create(
        [
            Prediction(
                run_id=run_id,
                stock_id=code,
                date=last,
                horizon=HORIZONS[run_id],
                in_universe=True,
                **fields(run_id, n),
            )
            for run_id, n, code in ranked + others
        ],
        ignore_conflicts=True,
    )
    FeatureDaily.objects.bulk_create(
        [
            FeatureDaily(stock_id=code, date=last, in_universe=True, tech_vol_20d=0.3)
            for code in every
        ],
        ignore_conflicts=True,
    )
    return codes


def add_depth(sessions: tuple[date, ...]) -> None:
    """Every stock predicted by every run on each earlier session, as months of serving would."""
    codes = list(
        StockMaster.objects.exclude(stock_id="2330")
        .order_by("stock_id")
        .values_list("stock_id", flat=True)
    )
    Prediction.objects.bulk_create(
        (
            Prediction(
                run_id=run_id,
                stock_id=code,
                date=day,
                horizon=horizon,
                in_universe=True,
                **fields(run_id, n),
            )
            for day in sessions[:-1]
            for run_id, horizon in HORIZONS.items()
            for n, code in enumerate(codes)
        ),
        batch_size=20_000,
        ignore_conflicts=True,
    )


def add_track_record(sessions: tuple[date, ...]) -> None:
    """Daily predictions of 2330 over the window, their labels and every run's monitoring."""
    Prediction.objects.bulk_create(
        [
            Prediction(
                run_id=run_id,
                stock_id="2330",
                date=day,
                horizon=horizon,
                in_universe=True,
                **fields(run_id, n % 40),
            )
            for run_id, horizon in HORIZONS.items()
            for n, day in enumerate(sessions[:-1])
        ],
        ignore_conflicts=True,
    )
    LabelDaily.objects.bulk_create(
        LabelDaily(
            stock_id="2330",
            date=day,
            horizon=horizon,
            fwd_ret=0.05 * ((n % 7) - 3) / 3,
            fwd_ret_open=0.05 * ((n % 7) - 3) / 3,
            fwd_rank_pct=(n % 10 + 0.5) / 10,
            fwd_vol_20d=0.28,
            fwd_up_5pct=n % 3 == 0,
        )
        for horizon in (5, 20)
        for n, day in enumerate(sessions[:-horizon])
    )
    ModelMonitor.objects.bulk_create(
        ModelMonitor(run_id=run_id, date=day, horizon=HORIZONS[run_id], n_stocks=STOCKS, **values)
        for run_id, values in MONITORS.items()
        for day in sessions[:-1]
    )


def add_research_and_checks(last: date) -> None:
    BacktestMetric.objects.bulk_create(
        [
            metric("lightgbm", "rank_ic", 0.041),
            metric("lightgbm", "rank_icir", 0.62),
            metric("ridge", "rank_ic", 0.027),
            metric("ridge", "rank_icir", 0.41),
            *month_rows(),
        ]
    )
    names = [feature.name for feature in FEATURES][:12]
    rows = [
        (run_id, position, name, (0.02 - 0.003 * position) * (1 if position % 2 else -1), 1)
        for run_id in ("rank-h20", "rank-h5")
        for position, name in enumerate(names, start=1)
    ]
    rest = [
        (run_id, len(names) + 1, REST_FEATURE, 0.004, len(FEATURES) - len(names))
        for run_id in ("rank-h20", "rank-h5")
    ]
    PredictionContrib.objects.bulk_create(
        PredictionContrib(
            run_id=run_id,
            stock_id="2330",
            date=last,
            horizon=HORIZONS[run_id],
            feature=name,
            position=position,
            contrib=value,
            n_features=count,
        )
        for run_id, position, name, value, count in rows + rest
    )
    clean = {"share": 0.004, "flagged": 7, "n_stocks": STOCKS, "stocks": ["3001"]}
    DataQualityLog.objects.bulk_create(
        [
            DataQualityLog(
                date=last, check_name="output_width_ratio", severity="warning", detail=clean
            ),
            DataQualityLog(
                date=last,
                check_name="output_odds_vs_q50",
                severity="info",
                detail={**clean, "flagged": 0, "share": 0.0},
            ),
        ]
    )


@pytest.fixture
def bench(predicted: Seed) -> Seed:
    """The e2e seed widened to a full market, with a track record and research results."""
    first = predicted.sessions[0] - timedelta(days=1)
    ModelRun.objects.update(train_end=first)
    EtlRun.objects.filter(dataset__startswith="extra").delete()
    StockMaster.objects.filter(stock_id="9999").update(name="測試上櫃", industry="其他業")
    add_universe(predicted.last_session)
    add_track_record(predicted.sessions)
    add_depth(predicted.sessions)
    add_research_and_checks(predicted.last_session)
    FeatureDaily.objects.filter(stock_id="2330", date=predicted.last_session).update(**RAW_2330)
    return predicted


def test_capture_timings_and_views_on_the_seed(
    page: Page, live_server: LiveServer, bench: Seed, tmp_path: Path
) -> None:
    out_dir = Path(os.environ.get("ACCEPTANCE_OUT", tmp_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = StringIO()

    call_command("measure_api", "--repeat", "20", stdout=out)
    cache.clear()

    (out_dir / "seed-timings.md").write_text(out.getvalue())
    rows = out.getvalue().splitlines()[2:]
    assert len(rows) == 11
    assert all(row.split(" | ")[1] == "200" for row in rows)
    assert all(float(row.split(" | ")[3]) < 500 for row in rows)

    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto(f"{live_server.url}/#today")
    expect(page.locator("#rows .trow")).to_have_count(20)
    page.screenshot(path=out_dir / "today.png", full_page=True)

    page.goto(f"{live_server.url}/#stock-2330")
    expect(page.locator("#price-chart path.fan-median")).to_have_count(1)
    expect(page.locator("#history .trow").first).to_be_visible()
    page.screenshot(path=out_dir / "stock.png", full_page=True)

    page.goto(f"{live_server.url}/#models")
    expect(page.locator("#ic-chart rect.bar-model")).to_have_count(24)
    expect(page.locator("#model-table .trow")).to_have_count(6)
    page.screenshot(path=out_dir / "models.png", full_page=True)
