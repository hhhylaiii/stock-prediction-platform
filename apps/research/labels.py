"""Forward labels per stock, day and horizon from the research panel (SPEC §6 Label 定義)."""

import math
from datetime import date

import pandas as pd
from django.db import transaction
from django.db.models import Max

from apps.etl.upsert import upsert
from apps.market.calendar import OutsideCalendarError, TradingDays
from apps.market.models import DailyPrice
from apps.research.models import LabelDaily
from apps.research.panel import require_sorted
from apps.research.universe import load_universe_panel

HORIZONS = (5, 20)
VOL_DAYS = 20
UP_THRESHOLD = 0.05
UP_TOLERANCE = 1e-9
ANNUAL_ROOT = math.sqrt(252)
LABEL_COLUMNS = [
    "stock_id",
    "date",
    "horizon",
    "fwd_ret",
    "fwd_ret_open",
    "fwd_rank_pct",
    "fwd_vol_20d",
    "fwd_up_5pct",
    "untradable_t1",
]
LABEL_FIELDS = [name for name in LABEL_COLUMNS if name not in ("stock_id", "date", "horizon")]


def _ahead(values: pd.Series, keys: list[pd.Series], steps: int, ended: pd.Series) -> pd.Series:
    """Value steps rows later in the listing; past its end, its last value if the listing ended."""
    grouped = values.groupby(keys)
    beyond = grouped.cumcount() + steps >= grouped.transform("size")
    return grouped.shift(-steps).where(~beyond, grouped.transform("last").where(ended))


def _flags(values: pd.Series, known: pd.Series) -> pd.Series:
    return values.astype(bool).astype(object).where(known, None)


def label_panel(panel: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Labels per row; ended listings settle at their last close, ranks wait for a settled day."""
    require_sorted(panel)
    keys = [panel["stock_id"], panel["segment"]]
    ended = panel["date"].groupby(keys).transform("max") < panel["date"].max()
    close = panel["adj_close"].groupby(keys).ffill()
    entry = panel["adj_open"].groupby(keys).shift(-1)
    has_next = panel["date"].groupby(keys).shift(-1).notna()
    blocked = panel["limit_up_open"].groupby(keys).shift(-1).eq(True)
    untradable = _flags((has_next & (entry.isna() | blocked)) | ~has_next, has_next | ended)
    vol = _forward_volatility(close, keys, ended).where(panel["adj_close"].notna())
    days = pd.Index(panel["date"].unique()).sort_values()
    position = pd.Series(days.get_indexer(panel["date"]), index=panel.index)
    parts = []
    for horizon in horizons:
        exit_close = _ahead(close, keys, horizon, ended)
        open_return = exit_close / entry - 1
        settled = position + horizon < len(days)
        eligible = panel["in_universe"].astype(bool) & has_next & ~(entry.isna() | blocked)
        eligible &= settled
        rank = open_return.where(eligible).groupby(panel["date"]).rank(pct=True)
        counted = horizon == VOL_DAYS
        parts.append(
            pd.DataFrame(
                {
                    "stock_id": panel["stock_id"],
                    "date": panel["date"],
                    "horizon": horizon,
                    "fwd_ret": exit_close / panel["adj_close"] - 1,
                    "fwd_ret_open": open_return,
                    "fwd_rank_pct": rank,
                    "fwd_vol_20d": vol if counted else math.nan,
                    "fwd_up_5pct": _flags(
                        open_return > UP_THRESHOLD + UP_TOLERANCE, open_return.notna() & counted
                    ),
                    "untradable_t1": untradable,
                }
            )
        )
    labels = pd.concat(parts, ignore_index=True)
    return labels.sort_values(["stock_id", "date", "horizon"], ignore_index=True)[LABEL_COLUMNS]


def _forward_volatility(close: pd.Series, keys: list[pd.Series], ended: pd.Series) -> pd.Series:
    """Annualized std of the next VOL_DAYS daily returns; days after a listing ends are flat."""
    step = (close / close.groupby(keys).shift(1) - 1).fillna(0.0)
    first = step.groupby(keys).cumsum()
    second = (step * step).groupby(keys).cumsum()
    total = _ahead(first, keys, VOL_DAYS, ended) - first
    squares = _ahead(second, keys, VOL_DAYS, ended) - second
    variance = ((squares - total * total / VOL_DAYS) / (VOL_DAYS - 1)).clip(lower=0.0)
    return (variance.pow(0.5) * ANNUAL_ROOT).where(close.notna())


def latest_price_date() -> date | None:
    """Last day daily_price holds; labels settle on prices up to it."""
    last: date | None = DailyPrice.objects.aggregate(last=Max("date"))["last"]
    return last


def build_labels(calendar: TradingDays, start: date, end: date) -> pd.DataFrame:
    """Labels of the trading days in [start, end], settled on prices through the data end."""
    data_end = latest_price_date()
    if data_end is None:
        return pd.DataFrame(columns=LABEL_COLUMNS)
    low, high = max(start, calendar.start), min(end, data_end, calendar.end)
    days = calendar.between(low, high) if low <= high else ()
    if not days:
        return pd.DataFrame(columns=LABEL_COLUMNS)
    try:
        reach = min(calendar.shift(days[-1], max(HORIZONS)), data_end)
    except OutsideCalendarError:
        reach = min(calendar.end, data_end)
    labels = label_panel(load_universe_panel(calendar, days[0], reach))
    inside = (labels["date"] >= days[0]) & (labels["date"] <= days[-1])
    return labels[inside].reset_index(drop=True)


def write_labels(labels: pd.DataFrame, start: date, end: date) -> int:
    """Replace the label_daily rows dated in [start, end], so rows no longer built go away."""
    if not labels["date"].between(start, end).all():
        raise ValueError(f"labels fall outside {start}..{end}")
    keys = ["stock_id", "date", "horizon"]
    values = [column for column in LABEL_COLUMNS if column not in keys]
    with transaction.atomic():
        LabelDaily.objects.filter(date__range=(start, end)).delete()
        return upsert(LabelDaily, labels, unique_fields=keys, update_fields=values)
