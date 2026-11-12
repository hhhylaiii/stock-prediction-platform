from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any

import pytest
from django.core.cache import cache
from django.utils import timezone

from apps.etl.models import DataQualityLog, EtlRun
from apps.market.models import Market, TradingCalendar
from tests.api.conftest import add_bars, add_model_run, add_predictions, add_stock

XSS_NAME = '<img src=x onerror="window.__xss=1">'
PROB_HEAD = "跑贏大盤 5 個百分點機率"
PROB_TIP = "20 日報酬跑贏大盤（加權股價報酬指數，含息）超過 5 個百分點的機率"
BAND_NOTE = (
    "20 日價格區間是 80% 區間。設計定案後只跑一次的 holdout"
    "（2025 年起，到 2026-11-11 已到期的預測）裡，"
    "實際報酬落在區間內的比例是 75.0%；7 個完整的季度在 59% 到 84% 之間，"
    "偏低的是 2025 年第 1 季 59.4%、2026 年第 2 季 69.6%，"
    "最後一段只有 2026 年 10 月 1 日至 12 日的 7 個交易日（樣本很少）、40.4%。"
    "定案前的研究期 2016–2024 年是 77.7%，大盤急跌的季度曾明顯偏低："
    "2020 年第 1 季 49%、2022 年第 2 季 66%、第 3 季 68%。"
)
RELEASE_REASON = "差距在 seed 間標準差內，先上線觀察"
BAR_COUNT = 70
GAP_INDEX = 64
EXTRA_RUNS = 101
POPULATION = {"missing": 0, "population": 1842, "markets": {}}
QUALITY: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("row_count", "info", {"table": "daily_price", "count": 1842, "previous": 1840}),
    ("row_count", "info", {"table": "margin", "count": 1201, "previous": 1199}),
    ("big_move", "info", {"table": "daily_price", "count": 3, "stocks": {"1101": 0.12}}),
    (
        "missing_values",
        "info",
        {
            "table": "daily_price",
            "columns": {"row": {**POPULATION, "missing": 1}, "close": {**POPULATION, "missing": 2}},
        },
    ),
    ("trading_calendar", "info", {"rows": {}}),
    ("index_freshness", "warning", {"stale": {"SOX": "2026-10-28"}}),
    ("etl_runs", "info", {"missing": [], "blocking": []}),
    (
        "stock_master",
        "info",
        {"added": ["6999"], "moved": [], "relisted": [], "delisted": [], "vanished": []},
    ),
)


@dataclass(frozen=True)
class Seed:
    last_session: date
    closes: tuple[float | None, ...]
    sessions: tuple[date, ...] = ()


EXTRA_STOCKS = tuple(f"{1101 + n}" for n in range(23))
TPEX_EXTRAS = frozenset(EXTRA_STOCKS[::3])


@pytest.fixture(scope="session", autouse=True)
def allow_orm_beside_playwright() -> Iterator[None]:
    """Playwright's sync API runs an event loop in the test thread, which Django would refuse."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("DJANGO_ALLOW_ASYNC_UNSAFE", "true")
        yield


@pytest.fixture(autouse=True)
def fresh_throttle_cache() -> Iterator[None]:
    cache.clear()
    yield
    cache.clear()


def at(day: date, hour: int, minute: int) -> datetime:
    return timezone.make_aware(datetime.combine(day, time(hour, minute)))


@pytest.fixture
def seeded(transactional_db: None) -> Seed:
    """A calendar around today, two stocks, one day of ETL runs and a passing quality check."""
    today = timezone.localdate()
    days = [today + timedelta(days=n) for n in range(-200, 41)]
    TradingCalendar.objects.bulk_create(
        TradingCalendar(date=day, is_trading_day=day.weekday() < 5) for day in days
    )
    sessions = [day for day in days if day.weekday() < 5 and day <= today][-BAR_COUNT:]
    last = sessions[-1]
    closes = tuple(1000.0 + 2 * n for n in range(BAR_COUNT))
    add_stock()
    add_bars(
        "2330",
        [d for n, d in enumerate(sessions) if n != GAP_INDEX],
        [c for n, c in enumerate(closes) if n != GAP_INDEX],
    )
    add_stock(stock_id="9999", name=XSS_NAME, industry=XSS_NAME, market=Market.TPEX)
    add_bars("9999", sessions[-3:], [10.0, 10.5, 11.0])
    for source, dataset, hour, minute, status in (
        ("twse", "daily_price", 15, 31, EtlRun.Status.SUCCESS),
        ("tpex", "daily_price", 15, 33, EtlRun.Status.SUCCESS),
        ("twse", "margin", 21, 31, EtlRun.Status.FAILED),
    ):
        EtlRun.objects.create(
            source=source,
            dataset=dataset,
            start_date=last,
            end_date=last,
            status=status,
            rows=1047,
            started_at=at(last, hour, minute),
        )
    EtlRun.objects.create(
        source="finmind",
        dataset="TaiwanStockPrice",
        start_date=date(2010, 1, 4),
        end_date=last,
        status=EtlRun.Status.SUCCESS,
        started_at=at(last, 9, 0),
    )
    EtlRun.objects.bulk_create(
        EtlRun(
            source="tpex",
            dataset=f"extra{n:03d}",
            start_date=last,
            end_date=last,
            status=EtlRun.Status.SUCCESS,
            rows=n,
            started_at=at(last, 22, 0) + timedelta(seconds=n),
        )
        for n in range(EXTRA_RUNS)
    )
    DataQualityLog.objects.bulk_create(
        DataQualityLog(date=last, check_name=check, severity=severity, detail=detail)
        for check, severity, detail in QUALITY
    )
    return Seed(last_session=last, closes=closes, sessions=tuple(sessions))


def released_on(status: str) -> dict[str, Any]:
    """Production params of a run built on a holdout the user released with that verdict."""
    release = {
        "status": status,
        "failed": ["排名 20 日 IC"] if status == "未達標" else [],
        "flagged": ["排名 5 日 IC"],
        "reason": RELEASE_REASON,
        "released_at": "2026-11-12T10:03:00.123456+08:00",
        "report": "/srv/host/reports/holdout-20261111.md",
        "sha256": "ab" * 32,
    }
    return {"production": {"role": "model", "holdout_release": release}}


def rank_rows(order: list[str]) -> dict[str, dict[str, Any]]:
    total = len(order)
    return {
        code: {"score": float(total - n), "rank": n + 1, "pct": (total - n) / total}
        for n, code in enumerate(order)
    }


@pytest.fixture
def predicted(seeded: Seed) -> Seed:
    """Serving runs of every role on the last session, 25 ranked stocks, one previous day."""
    last, previous = seeded.last_session, seeded.sessions[-2]
    for n, code in enumerate(EXTRA_STOCKS):
        market = Market.TPEX if code in TPEX_EXTRAS else Market.TWSE
        add_stock(stock_id=code, name=f"測試{n:02d}", market=market)
        add_bars(code, [last], [50.0 + n])
    order = ["2330", "9999", *EXTRA_STOCKS]
    rank20, rank5 = add_model_run("rank-h20", "rank"), add_model_run("rank-h5", "rank")
    add_predictions(rank20, last, 20, rank_rows(order))
    add_predictions(rank20, previous, 20, rank_rows(["1101", "1102", "1103", "2330"]))
    add_predictions(rank5, last, 5, rank_rows(["9999", "2330", *EXTRA_STOCKS]))
    add_predictions(
        add_model_run("quantile-h20", "quantile"),
        last,
        20,
        {"2330": {"q10": -0.05, "q50": 0.04, "q90": 0.12, "ret_pred": 0.04}},
    )
    add_predictions(
        add_model_run("prob-h20", "classification"), last, 20, {"2330": {"prob_up_5pct": 0.6}}
    )
    add_predictions(
        add_model_run("vol-h20", "volatility"),
        last,
        20,
        {code: {"vol_pred": 0.3} for code in order},
    )
    add_predictions(add_model_run("reg-h5", "regression"), last, 5, {"2330": {"ret_pred": 0.01}})
    return seeded
