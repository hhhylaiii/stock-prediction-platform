from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import pytest
from django.core.cache import cache

from apps.market.models import DailyPrice, Market, StockMaster, TradingCalendar
from apps.research.gates import Gate, SlotGates
from apps.research.models import ModelRun, Prediction
from apps.research.monitor import PROMOTED_ON
from apps.research.predict import Slot
from apps.research.promotion import PROMOTION, _failed

TAIPEI = ZoneInfo("Asia/Taipei")
CALENDAR_START = date(2026, 7, 1)
EARLY_CALENDAR_START = date(2026, 1, 1)
CALENDAR_END = date(2026, 11, 30)
HOLIDAYS = frozenset({date(2026, 9, 25), date(2026, 10, 9)})
OCT_28, OCT_29, OCT_30 = date(2026, 10, 28), date(2026, 10, 29), date(2026, 10, 30)
NOV_2 = date(2026, 11, 2)


def is_session(day: date) -> bool:
    return day.weekday() < 5 and day not in HOLIDAYS


def calendar_days(start: date = CALENDAR_START, end: date = CALENDAR_END) -> list[date]:
    return [start + timedelta(days=n) for n in range((end - start).days + 1)]


SESSIONS = [day for day in calendar_days() if is_session(day)]


def sessions_until(last: date) -> list[date]:
    return [day for day in SESSIONS if day <= last]


def trained_until(last_in_sample: date, horizon: int = 20) -> date:
    """The train_end that keeps last_in_sample in sample and puts the next session out of it."""
    return SESSIONS[SESSIONS.index(last_in_sample) + 1 - horizon]


def taipei(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=TAIPEI)


class JsonBody(Protocol):
    def json(self) -> Any: ...


def envelope_of(response: JsonBody) -> dict[str, Any]:
    """Decoded body after checking the four envelope keys are present."""
    body: dict[str, Any] = response.json()
    assert set(body) == {"success", "data", "error", "meta"}
    return body


@pytest.fixture(autouse=True)
def fresh_throttle_cache() -> Iterator[None]:
    cache.clear()
    yield
    cache.clear()


@pytest.fixture
def calendar_start() -> date:
    """First stored calendar day; a module that trains runs before July starts it earlier."""
    return CALENDAR_START


@pytest.fixture
def calendar_in_db(db: None, calendar_start: date) -> list[date]:
    """Weekdays from calendar_start to November 2026 minus two holidays, in trading_calendar."""
    days = calendar_days(calendar_start)
    TradingCalendar.objects.bulk_create(
        TradingCalendar(date=day, is_trading_day=is_session(day)) for day in days
    )
    return [day for day in days if is_session(day)]


def add_stock(
    stock_id: str = "2330",
    name: str = "台積電",
    market: str = Market.TWSE,
    industry: str = "半導體業",
    list_date: date | None = date(1994, 9, 5),
    delist_date: date | None = None,
) -> StockMaster:
    return StockMaster.objects.create(
        stock_id=stock_id,
        name=name,
        market=market,
        industry=industry,
        list_date=list_date,
        delist_date=delist_date,
    )


def add_bars(
    stock_id: str,
    days: Sequence[date],
    closes: Sequence[float | None],
    factors: Sequence[float] | None = None,
) -> list[DailyPrice]:
    """One bar per day around its close; a None close is a day without trades."""
    rows = []
    for index, (day, close) in enumerate(zip(days, closes, strict=True)):
        price = None if close is None else Decimal(str(close))
        rows.append(
            DailyPrice(
                stock_id=stock_id,
                date=day,
                open=None if price is None else price - 1,
                high=None if price is None else price + 2,
                low=None if price is None else price - 2,
                close=price,
                volume=0 if price is None else 1_000_000 + index,
                turnover=0 if price is None else int(price * 1_000_000),
                adj_factor=1.0 if factors is None else factors[index],
            )
        )
    return DailyPrice.objects.bulk_create(rows)


def add_model_run(
    run_id: str,
    objective: str,
    status: str = "active",
    model_type: str = "lightgbm",
    version: str = "2026-10",
    created_at: datetime | None = None,
    train_end: date = date(2026, 9, 30),
) -> ModelRun:
    return ModelRun.objects.create(
        run_id=run_id,
        run_key=f"{objective}-serving",
        model_type=model_type,
        objective=objective,
        version=version,
        train_start=date(2023, 10, 1),
        train_end=train_end,
        feature_hash="f" * 64,
        status=status,
        created_at=created_at or taipei(2026, 10, 1),
    )


RELEASE_REASON = "kept for the model page"
RELEASE_REPORT = "/srv/host/reports/rank-holdout.md"


def release_params(status: object) -> dict[str, Any]:
    """Production params as build_run records a release, with the given verdict in it."""
    release = {
        "status": status,
        "failed": ["排名 20 日 IC"],
        "flagged": [],
        "reason": RELEASE_REASON,
        "released_at": "2026-11-12T10:03:00+08:00",
        "report": RELEASE_REPORT,
        "sha256": "ab" * 32,
    }
    return {"production": {"role": "model", "holdout_release": release}}


FORCED_REASON = "kept with the forced swap"
FORCED_GATES = (
    Gate("agreement", False, 0.412, ">= 0.5", 1842, ""),
    Gate("gain_overlap", True, 8.0, "前 10 名重疊 >= 6", 10, ""),
    Gate("shadow_oos", False, -0.004, "> 0", 120, "no shadow rows"),
)


def forced_params(
    run_id: str, objective: str = "rank", horizon: int = 20, reason: str = FORCED_REASON
) -> dict[str, Any]:
    """Params as promote --force leaves them on a run whose own gates failed."""
    checked = SlotGates(Slot(objective, horizon), run_id, "serving-old", list(FORCED_GATES))
    note = {"forced": True, "reason": reason, "failed": _failed([checked])}
    return {PROMOTED_ON: "2026-10-30", PROMOTION: note}


def add_predictions(
    run: ModelRun, day: date, horizon: int, rows: dict[str, dict[str, Any]]
) -> list[Prediction]:
    """One prediction row per stock; in_universe defaults to True."""
    return Prediction.objects.bulk_create(
        Prediction(
            run=run,
            stock_id=stock_id,
            date=day,
            horizon=horizon,
            **{"in_universe": True, **fields},
        )
        for stock_id, fields in rows.items()
    )


@dataclass(frozen=True)
class Runs:
    rank20: ModelRun
    quantile: ModelRun
    vol: ModelRun
    prob: ModelRun
    rank5: ModelRun
    reg5: ModelRun


def seed_market() -> None:
    add_stock()
    add_stock(stock_id="2317", name="鴻海", industry="其他電子業")
    add_stock(stock_id="6488", name="環球晶", market=Market.TPEX)
    add_stock(stock_id="8299", name="群聯", market=Market.TPEX)
    add_bars("2330", [OCT_29, OCT_30], [990.0, 1000.0])
    add_bars("2317", [OCT_29, OCT_30], [198.0, 200.0])
    add_bars("6488", [OCT_29, OCT_30], [490.0, 500.0])
    add_bars("8299", [OCT_29, OCT_30], [800.0, None])


@pytest.fixture
def runs(db: None) -> Runs:
    seed_market()
    seeded = Runs(
        rank20=add_model_run("rank-h20", "rank"),
        quantile=add_model_run("quantile-h20", "quantile"),
        vol=add_model_run("vol-h20", "volatility"),
        prob=add_model_run("prob-h20", "classification"),
        rank5=add_model_run("rank-h5", "rank"),
        reg5=add_model_run("reg-h5", "regression"),
    )
    add_predictions(
        seeded.rank20,
        OCT_30,
        20,
        {
            "2330": {"score": 0.9, "rank": 1, "pct": 0.99},
            "6488": {"score": 0.5, "rank": 2, "pct": 0.5},
            "2317": {"score": 0.1, "rank": 3, "pct": 0.01},
            "8299": {"score": 0.3, "in_universe": False},
        },
    )
    add_predictions(
        seeded.rank20,
        OCT_29,
        20,
        {
            "6488": {"score": 0.8, "rank": 1, "pct": 0.99},
            "2330": {"score": 0.4, "rank": 2, "pct": 0.5},
            "2317": {"score": 0.1, "rank": 3, "pct": 0.01},
        },
    )
    add_predictions(
        seeded.quantile,
        OCT_30,
        20,
        {
            "2330": {"q10": -0.05, "q50": 0.04, "q90": 0.12, "ret_pred": 0.04},
            "2317": {"q10": -0.1, "q50": -0.02, "q90": 0.05, "ret_pred": -0.02},
            "8299": {"q10": -0.2, "q50": 0.0, "q90": 0.2, "ret_pred": 0.0, "in_universe": False},
        },
    )
    add_predictions(
        seeded.vol,
        OCT_30,
        20,
        {code: {"vol_pred": 0.3} for code in ("2330", "2317", "6488", "8299")},
    )
    add_predictions(
        seeded.prob,
        OCT_30,
        20,
        {
            "2330": {"prob_up_5pct": 0.6},
            "2317": {"prob_up_5pct": 0.2},
            "6488": {"prob_up_5pct": 0.4},
        },
    )
    add_predictions(
        seeded.rank5,
        OCT_30,
        5,
        {
            "2317": {"score": 0.7, "rank": 1, "pct": 0.99},
            "2330": {"score": 0.2, "rank": 2, "pct": 0.5},
            "6488": {"score": 0.1, "rank": 3, "pct": 0.01},
        },
    )
    add_predictions(
        seeded.reg5,
        OCT_30,
        5,
        {"2317": {"ret_pred": 0.02}, "2330": {"ret_pred": 0.01}, "6488": {"ret_pred": -0.01}},
    )
    return seeded
