"""Each stock's market on each day, from the market history (SPEC §6)."""

import logging
from collections.abc import Sequence
from typing import Literal

import pandas as pd

from apps.market.models import MarketHistory

HISTORY_COLUMNS = ["stock_id", "start_date", "end_date", "market"]

logger = logging.getLogger(__name__)


def load_market_history(stock_ids: Sequence[str] | None = None) -> pd.DataFrame:
    """Market spans of the given stocks, or of every stock."""
    spans = MarketHistory.objects.all()
    if stock_ids is not None:
        spans = spans.filter(stock_id__in=list(stock_ids))
    return pd.DataFrame(list(spans.values_list(*HISTORY_COLUMNS)), columns=HISTORY_COLUMNS)


def _nearest(
    days: pd.DataFrame, spans: pd.DataFrame, direction: Literal["backward", "forward"]
) -> pd.DataFrame:
    return pd.merge_asof(
        days, spans, left_on="day", right_on="start", by="stock_id", direction=direction
    )


def markets_on(rows: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Each row's market on its day and whether a span holds the day (listed).

    Spans include both ends and a missing end_date is open. An unlisted day takes the latest
    earlier span's market, else the first later span's, else the row's current market.
    """
    days = pd.DataFrame(
        {"stock_id": rows["stock_id"], "day": pd.to_datetime(rows["date"]), "row": rows.index}
    ).sort_values("day")
    spans = pd.DataFrame(
        {
            "stock_id": history["stock_id"].astype(rows["stock_id"].dtype),
            "start": pd.to_datetime(history["start_date"]),
            "end": pd.to_datetime(history["end_date"]),
            "span_market": history["market"].astype("object"),
        }
    ).sort_values("start")
    earlier, later = _nearest(days, spans, "backward"), _nearest(days, spans, "forward")
    held = earlier["start"].notna() & (earlier["end"].isna() | earlier["day"].le(earlier["end"]))
    market = earlier["span_market"].where(earlier["span_market"].notna(), later["span_market"])
    if history.empty and len(rows):
        logger.warning(
            "market history has no span for any of %d stocks; none of their rows is listed",
            rows["stock_id"].nunique(),
        )
    found = pd.DataFrame(
        {"market": market.to_numpy(), "listed": held.to_numpy()}, index=earlier["row"]
    ).reindex(rows.index)
    return found.assign(
        market=found["market"].where(found["market"].notna(), rows["market"]),
        listed=found["listed"].astype(bool),
    )
