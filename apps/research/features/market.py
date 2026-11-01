"""Market-level features of SPEC §6; every stock of a day shares the same values."""

from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np
import pandas as pd

from apps.market.calendar import OutsideCalendarError, TradingDays
from apps.market.models import DailyPrice, IndexDaily, IndexId, MacroIndicator
from apps.research.features.registry import Group, feature_names

TAIEX_DAYS = (5, 20)
SOX_DAYS = 5
USDTWD_DAYS = 20
ADR_SHARES = 5
TSMC = "2330"
BUSINESS_SIGNAL_ID = "business_signal"
US_STALE = pd.Timedelta(7, unit="D")
SIGNAL_STALE = pd.Timedelta(62, unit="D")
MARKET_WARMUP = 30
US_LOOKBACK = timedelta(days=60)
SIGNAL_LOOKBACK = timedelta(days=120)
SERIES = (IndexId.TAIEX, IndexId.SOX, IndexId.TSM_ADR, IndexId.USDTWD)


def _closes(index_closes: pd.DataFrame, index_id: str) -> pd.Series:
    rows = index_closes[index_closes["index_id"] == index_id].sort_values("date")
    return pd.Series(rows["close"].astype("float64").to_numpy(), index=pd.Index(rows["date"]))


def _asof(values: pd.Series, days: Sequence[date], stale: pd.Timedelta) -> np.ndarray:
    """Latest value dated on or before each day, NaN when older than stale."""
    left = pd.DataFrame({"on": pd.to_datetime(pd.Series(list(days), dtype="object"))})
    if values.empty:
        return np.full(len(left), np.nan)
    right = pd.DataFrame(
        {"on": pd.to_datetime(pd.Series(values.index, dtype="object")), "value": values.to_numpy()}
    )
    merged = pd.merge_asof(left, right, on="on", tolerance=stale)
    return merged["value"].to_numpy(dtype=np.float64)


def _signal(signal: pd.DataFrame, days: Sequence[date]) -> np.ndarray:
    ordered = signal.sort_values(["announce_date", "period"])
    values = pd.Series(
        ordered["value"].astype("float64").to_numpy(), index=ordered["announce_date"]
    )
    return _asof(values, days, SIGNAL_STALE)


def market_features(
    days: Sequence[date],
    index_closes: pd.DataFrame,
    tsmc_close: pd.Series,
    signal: pd.DataFrame,
) -> pd.DataFrame:
    """Market features per Taipei trading day; US series count their own trading days."""
    grid = pd.Index(list(days))
    taiex = _closes(index_closes, IndexId.TAIEX).reindex(grid)
    sox = _closes(index_closes, IndexId.SOX)
    usd = _closes(index_closes, IndexId.USDTWD)
    adr = _asof(_closes(index_closes, IndexId.TSM_ADR), days, US_STALE)
    rate = _asof(usd, days, US_STALE)
    local = tsmc_close.astype("float64").reindex(grid).to_numpy()
    columns = {
        **{f"mkt_taiex_ret_{n}d": (taiex / taiex.shift(n) - 1).to_numpy() for n in TAIEX_DAYS},
        "mkt_sox_ret_5d": _asof(sox / sox.shift(SOX_DAYS) - 1, days, US_STALE),
        "mkt_tsm_adr_gap": adr * rate / ADR_SHARES / local - 1,
        "mkt_usdtwd_chg_20d": _asof(usd / usd.shift(USDTWD_DAYS) - 1, days, US_STALE),
        "mkt_business_signal": _signal(signal, days),
    }
    names = feature_names(Group.MARKET)
    return pd.DataFrame({"date": list(days), **{name: columns[name] for name in names}})


def attach_market_features(panel: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Stock, date and the market features of each row's day."""
    names = feature_names(Group.MARKET)
    values = market.set_index("date")[names].reindex(panel["date"]).to_numpy(dtype=np.float64)
    columns = {name: values[:, k] for k, name in enumerate(names)}
    return pd.DataFrame({"stock_id": panel["stock_id"], "date": panel["date"], **columns})


def load_market_features(calendar: TradingDays, start: date, end: date) -> pd.DataFrame:
    """Market features of the trading days in [start, end], read from the database."""
    days = calendar.between(max(start, calendar.start), min(end, calendar.end))
    if not days:
        return pd.DataFrame(columns=["date", *feature_names(Group.MARKET)])
    try:
        warm = calendar.shift(days[0], -MARKET_WARMUP)
    except OutsideCalendarError:
        warm = calendar.start
    grid = calendar.between(warm, days[-1])
    closes = IndexDaily.objects.filter(
        index_id__in=[series.value for series in SERIES],
        date__range=(warm - US_LOOKBACK, days[-1]),
    ).values_list("index_id", "date", "close")
    tsmc = DailyPrice.objects.filter(stock_id=TSMC, date__range=(days[0], days[-1]))
    signal = MacroIndicator.objects.filter(
        indicator_id=BUSINESS_SIGNAL_ID,
        announce_date__range=(days[0] - SIGNAL_LOOKBACK, days[-1]),
    ).values_list("period", "value", "announce_date")
    features = market_features(
        grid,
        pd.DataFrame(list(closes), columns=["index_id", "date", "close"]),
        pd.Series(dict(tsmc.values_list("date", "close")), dtype="float64"),
        pd.DataFrame(list(signal), columns=["period", "value", "announce_date"]),
    )
    return features[features["date"] >= days[0]].reset_index(drop=True)
