"""Training universe flags per stock and day (SPEC §6 處理步驟 1)."""

from datetime import date

import pandas as pd

from apps.market.calendar import TradingDays
from apps.market.models import DispositionEvent, DispositionType

MIN_AGE = 60
TURNOVER_WINDOW = 20
MIN_MEDIAN_TURNOVER = 10_000_000
EXCLUSIONS = ("halted", "no_trade", "new_listing", "disposition", "full_delivery", "illiquid")
EVENT_FLAGS = {
    DispositionType.ATTENTION: "attention",
    DispositionType.DISPOSITION: "disposition",
    DispositionType.FULL_DELIVERY: "full_delivery",
}
EVENT_COLUMNS = ["stock_id", "type", "start_date", "end_date"]


def load_disposition_events(start: date, end: date) -> pd.DataFrame:
    """Attention, disposition and full-delivery periods that overlap [start, end]."""
    rows = (
        DispositionEvent.objects.filter(start_date__lte=end)
        .exclude(end_date__lt=start)
        .values_list(*EVENT_COLUMNS)
    )
    return pd.DataFrame(list(rows), columns=EVENT_COLUMNS)


def event_days(events: pd.DataFrame, calendar: TradingDays, last_day: date) -> pd.DataFrame:
    """One row per trading day each event covers; open-ended events run through last_day."""
    rows: list[tuple[str, date, str]] = []
    for stock_id, kind, start, end in events[EVENT_COLUMNS].itertuples(index=False):
        low = max(start, calendar.start)
        high = min(last_day if end is None or pd.isna(end) else end, last_day, calendar.end)
        if low <= high:
            rows.extend((stock_id, day, kind) for day in calendar.between(low, high))
    return pd.DataFrame(rows, columns=["stock_id", "date", "type"])


def _median_turnover(panel: pd.DataFrame) -> pd.Series:
    traded = panel["turnover"].where(panel["has_bar"], 0.0).fillna(0.0)
    keys = [panel["stock_id"], panel["segment"]]
    return traded.groupby(keys).transform(
        lambda series: series.rolling(TURNOVER_WINDOW, min_periods=TURNOVER_WINDOW).median()
    )


def apply_universe(
    panel: pd.DataFrame, events: pd.DataFrame, calendar: TradingDays
) -> pd.DataFrame:
    """New panel with exclusion flags and in_universe; rows must be sorted by stock and date."""
    keys = pd.MultiIndex.from_frame(panel[["stock_id", "date"]])
    covered = event_days(events, calendar, max(panel["date"], default=calendar.start))
    flags = {
        column: keys.isin(
            pd.MultiIndex.from_frame(covered.loc[covered["type"] == kind, ["stock_id", "date"]])
        )
        for kind, column in EVENT_FLAGS.items()
    }
    median = _median_turnover(panel)
    result = panel.assign(
        halted=~panel["has_bar"].astype(bool),
        no_trade=panel["has_bar"].astype(bool) & panel["close"].isna(),
        new_listing=panel["age"] < MIN_AGE,
        illiquid=~(median >= MIN_MEDIAN_TURNOVER),
        turnover_med_20d=median,
        **flags,
    )
    return result.assign(in_universe=~result[list(EXCLUSIONS)].any(axis=1))
