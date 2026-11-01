"""FinMind REST client for the one-off history backfill (SPEC §3, §5)."""

import logging
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta
from datetime import time as clock_time
from typing import Any

import pandas as pd
import requests
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.db import connection
from django.utils import timezone

from apps.etl.adapters.base import BaseAdapter
from apps.etl.adapters.finmind_datasets import (
    chip_rows,
    dividend_events,
    price_bars,
    share_counts,
    valuations,
)
from apps.etl.adapters.finmind_fundamentals import monthly_revenues, statements
from apps.etl.client import SourceClient
from apps.etl.errors import SchemaChangedError, SourceRequestError
from apps.etl.models import RawFinmind
from apps.etl.ratelimit import RateLimiter, ThrottledError
from apps.etl.upsert import insert_missing, upsert
from apps.market.adjust import refresh_adj_factor, refresh_where_later_events
from apps.market.calendar import TradingDays
from apps.market.models import (
    DailyChip,
    DailyPrice,
    DailyValuation,
    DividendEvent,
    FinancialStatement,
    MonthlyRevenue,
    ShareChangeReason,
    SharesOutstanding,
    StockMaster,
)

logger = logging.getLogger(__name__)
DATA_PATH = "/api/v4/data"
HOURLY_QUOTA_WITH_TOKEN = 600
HOURLY_QUOTA_ANONYMOUS = 300
QUOTA_EXHAUSTED = 402
MAINTENANCE_WEEKDAY = 6
MAINTENANCE_END = clock_time(7, 0)


def maintenance_wait(moment: datetime) -> float:
    """Seconds until FinMind's Sunday 00:00–07:00 Taipei maintenance ends, else 0."""
    local = timezone.localtime(moment)
    if local.weekday() != MAINTENANCE_WEEKDAY or local.time() >= MAINTENANCE_END:
        return 0.0
    reopen = local.replace(hour=MAINTENANCE_END.hour, minute=0, second=0, microsecond=0)
    return (reopen - local) / timedelta(seconds=1)


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


class FinmindClient(SourceClient):
    source = "finmind"
    label = "FinMind"
    raw_model = RawFinmind
    throttle_statuses = frozenset({QUOTA_EXHAUSTED, 429})

    def __init__(
        self,
        limiter: RateLimiter,
        base_url: str,
        timeout: float,
        token: str,
        session: requests.Session | None = None,
        now: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        quota = HOURLY_QUOTA_WITH_TOKEN if token else HOURLY_QUOTA_ANONYMOUS
        if limiter.limit.min_interval * quota < 3600:
            raise ImproperlyConfigured(
                f"FinMind allows {quota} requests per hour; "
                f"min_interval must be at least {3600 / quota:g} seconds"
            )
        super().__init__(limiter, base_url, timeout, session)
        self.token = token
        self._now = now if now is not None else timezone.now
        self._sleep = sleep if sleep is not None else _sleep

    @classmethod
    def build(
        cls, limiter: RateLimiter, session: requests.Session | None = None
    ) -> "FinmindClient":
        return cls(
            limiter,
            str(settings.FINMIND_BASE_URL),
            float(settings.FINMIND_TIMEOUT),
            str(settings.FINMIND_API_TOKEN),
            session,
        )

    def data(self, dataset: str, finmind_dataset: str, **params: str) -> list[dict[str, Any]]:
        """Rows of one FinMind dataset; the whole response is kept in raw_finmind."""
        query = {"dataset": finmind_dataset, **params}
        payload = self.get_json(DATA_PATH, query)
        self.save_raw(dataset, query, payload)
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise SchemaChangedError(f"FinMind {finmind_dataset} data is not a list")
        return rows

    def _send(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Every attempt, retries included, first waits out the Sunday maintenance window."""
        wait = maintenance_wait(self._now())
        if wait > 0:
            logger.info("FinMind maintenance until 07:00 Taipei; waiting %.0f s", wait)
            self._sleep(wait)
        return super()._send(method, path, **kwargs)

    def headers(self) -> dict[str, str]:
        headers = super().headers()
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def check_payload(self, payload: dict[str, Any]) -> None:
        status = payload.get("status")
        if status == QUOTA_EXHAUSTED:
            raise ThrottledError(f"FinMind quota exhausted: {payload.get('msg')}")
        if status != 200:
            raise SourceRequestError(f"FinMind answered {status}: {payload.get('msg')}")


FINMIND_DATASETS = {
    "daily_price": ("TaiwanStockPrice",),
    "dividend_event": (
        "TaiwanStockDividend",
        "TaiwanStockDividendResult",
        "TaiwanStockCapitalReductionReferencePrice",
        "TaiwanStockSplitPrice",
    ),
    "daily_valuation": ("TaiwanStockPER",),
    "shares_outstanding": ("TaiwanStockShareholding",),
    "daily_chip": (
        "TaiwanStockInstitutionalInvestorsBuySell",
        "TaiwanStockMarginPurchaseShortSale",
        "TaiwanDailyShortSaleBalances",
        "TaiwanStockShareholding",
    ),
    "monthly_revenue": ("TaiwanStockMonthRevenue",),
    "financial_statement": (
        "TaiwanStockFinancialStatements",
        "TaiwanStockBalanceSheet",
        "TaiwanStockCashFlowsStatement",
    ),
}
RAW_NAMES = {
    "TaiwanStockDividend": "dividend_policy",
    "TaiwanStockDividendResult": "dividend_result",
    "TaiwanStockCapitalReductionReferencePrice": "capital_reduction",
    "TaiwanStockSplitPrice": "split",
    "TaiwanStockInstitutionalInvestorsBuySell": "institutional",
    "TaiwanStockMarginPurchaseShortSale": "margin",
    "TaiwanDailyShortSaleBalances": "short_sale_balance",
    "TaiwanStockShareholding": "shareholding",
    "TaiwanStockFinancialStatements": "income_statement",
    "TaiwanStockBalanceSheet": "balance_sheet",
    "TaiwanStockCashFlowsStatement": "cash_flow",
}
WHOLE_MARKET = frozenset({"TaiwanStockSplitPrice"})
TRADING_DATE_COLUMNS = {
    "daily_price": "date",
    "daily_valuation": "date",
    "shares_outstanding": "effective_date",
    "daily_chip": "date",
}
RECORD_DATE_LOOKAHEAD = timedelta(days=30)
LISTED_ONLY = frozenset({"daily_price", "daily_valuation", "daily_chip"})
LOOKBACK = {
    "monthly_revenue": timedelta(days=366),
    "financial_statement": timedelta(days=366),
}
EVENT_FIELDS = ("cash_dividend", "stock_dividend")
FROM_COLUMN = "_finmind_dataset"
DROP_REPEATED_COUNTS_SQL = """
DELETE FROM shares_outstanding s
USING (
    SELECT effective_date, shares,
           LAG(shares) OVER (ORDER BY effective_date) AS previous
    FROM shares_outstanding
    WHERE stock_id = %(stock)s
) ordered
WHERE s.stock_id = %(stock)s
  AND s.effective_date = ordered.effective_date
  AND ordered.shares = ordered.previous
  AND s.reason = %(unknown)s
"""


class FinmindAdapter(BaseAdapter):
    """Per-stock history backfill; FinMind is secondary for prices, so it never overwrites them."""

    source = "finmind"
    raw_model = RawFinmind
    per_stock = True
    datasets = frozenset(FINMIND_DATASETS)

    def __init__(
        self,
        calendar: TradingDays,
        limiter: RateLimiter,
        session: requests.Session | None = None,
    ) -> None:
        super().__init__(calendar, limiter)
        self.client = FinmindClient.build(limiter, session)
        self._whole_market: dict[tuple[str, date, date], list[dict[str, Any]]] = {}

    def fetch(
        self, dataset: str, start: date, end: date, stock_id: str | None = None
    ) -> pd.DataFrame:
        self.require_dataset(dataset)
        if stock_id is None:
            raise ValueError("FinMind datasets are fetched one stock at a time")
        rows: list[dict[str, Any]] = []
        for finmind_dataset in FINMIND_DATASETS[dataset]:
            if finmind_dataset in WHOLE_MARKET:
                fetched = self._market_rows(finmind_dataset, start, end, stock_id)
            else:
                last = (
                    end + RECORD_DATE_LOOKAHEAD if finmind_dataset == "TaiwanStockDividend" else end
                )
                fetched = self.client.data(
                    RAW_NAMES.get(finmind_dataset, dataset),
                    finmind_dataset,
                    data_id=stock_id,
                    start_date=(start - LOOKBACK.get(dataset, timedelta(0))).isoformat(),
                    end_date=last.isoformat(),
                )
            rows.extend({**row, FROM_COLUMN: finmind_dataset} for row in fetched)
        frame = pd.DataFrame(rows, dtype=object)
        frame.attrs.update(stock_id=stock_id, start=start, end=end)
        return frame

    def normalize(self, dataset: str, raw_df: pd.DataFrame) -> pd.DataFrame:
        stock_id = str(raw_df.attrs["stock_id"])
        parts = {name: _rows_from(raw_df, name) for name in FINMIND_DATASETS[dataset]}
        if dataset == "dividend_event":
            frame = dividend_events(
                parts["TaiwanStockDividend"],
                parts["TaiwanStockDividendResult"],
                stock_id,
                start=raw_df.attrs["start"],
                end=raw_df.attrs["end"],
                reductions=parts["TaiwanStockCapitalReductionReferencePrice"],
                splits=parts["TaiwanStockSplitPrice"],
            )
        elif dataset == "monthly_revenue":
            frame = monthly_revenues(
                parts["TaiwanStockMonthRevenue"],
                stock_id,
                self.calendar,
                raw_df.attrs["start"],
                raw_df.attrs["end"],
            )
        elif dataset == "financial_statement":
            frame = statements(
                parts["TaiwanStockFinancialStatements"],
                parts["TaiwanStockBalanceSheet"],
                parts["TaiwanStockCashFlowsStatement"],
                stock_id,
                self.calendar,
                raw_df.attrs["start"],
                raw_df.attrs["end"],
            )
        elif dataset == "daily_chip":
            frame = chip_rows(
                parts["TaiwanStockInstitutionalInvestorsBuySell"],
                parts["TaiwanStockMarginPurchaseShortSale"],
                parts["TaiwanDailyShortSaleBalances"],
                parts["TaiwanStockShareholding"],
                stock_id,
            )
        else:
            normalizer = {
                "daily_price": price_bars,
                "daily_valuation": valuations,
                "shares_outstanding": share_counts,
            }[dataset]
            frame = normalizer(parts[FINMIND_DATASETS[dataset][0]], stock_id)
        if dataset in TRADING_DATE_COLUMNS:
            self._require_trading_days(frame[TRADING_DATE_COLUMNS[dataset]], dataset, stock_id)
        return frame

    def upsert(self, dataset: str, frame: pd.DataFrame) -> int:
        if not frame.empty and dataset in LISTED_ONLY:
            frame = _listed_only(frame)
        if frame.empty:
            return 0
        stock_id = str(frame["stock_id"].iloc[0])
        if dataset == "daily_price":
            rows = insert_missing(DailyPrice, frame, ("stock_id", "date"))
            refresh_where_later_events([stock_id], frame["date"].min())
            return rows
        if dataset == "daily_valuation":
            return insert_missing(DailyValuation, frame, ("stock_id", "date"))
        if dataset == "daily_chip":
            return insert_missing(DailyChip, frame, ("stock_id", "date"))
        if dataset == "monthly_revenue":
            return insert_missing(MonthlyRevenue, frame, ("stock_id", "period"))
        if dataset == "financial_statement":
            keys = ("stock_id", "period", "fin_type")
            return insert_missing(FinancialStatement, frame, keys)
        if dataset == "shares_outstanding":
            rows = insert_missing(SharesOutstanding, frame, ("stock_id", "effective_date"))
            with connection.cursor() as cursor:
                cursor.execute(
                    DROP_REPEATED_COUNTS_SQL,
                    {"stock": stock_id, "unknown": ShareChangeReason.UNKNOWN},
                )
            return rows
        with_factor = frame["adj_factor"].notna()
        rows = upsert(
            DividendEvent,
            frame[with_factor],
            ("stock_id", "ex_date"),
            (*EVENT_FIELDS, "adj_factor"),
        )
        rows += upsert(DividendEvent, frame[~with_factor], ("stock_id", "ex_date"), EVENT_FIELDS)
        refresh_adj_factor(stock_id)
        return rows

    def _market_rows(
        self, finmind_dataset: str, start: date, end: date, stock_id: str
    ) -> list[dict[str, Any]]:
        """Small whole-market lists are fetched once per run and filtered per stock."""
        key = (finmind_dataset, start, end)
        if key not in self._whole_market:
            self._whole_market[key] = self.client.data(
                RAW_NAMES[finmind_dataset],
                finmind_dataset,
                start_date=start.isoformat(),
                end_date=end.isoformat(),
            )
        rows = self._whole_market[key]
        if any("stock_id" not in row for row in rows):
            raise SchemaChangedError(f"FinMind {finmind_dataset} rows lost key stock_id")
        return [row for row in rows if str(row["stock_id"]).strip() == stock_id]

    def _require_trading_days(self, days: pd.Series, dataset: str, stock_id: str) -> None:
        """SPEC §5 rejects a batch dated off the calendar; ex-dates may fall on closures."""
        for day in sorted(set(days)):
            if not self.calendar.is_trading_day(day):
                raise SchemaChangedError(
                    f"FinMind {dataset} for {stock_id}: {day} is not a trading day"
                )


def _listed_only(frame: pd.DataFrame) -> pd.DataFrame:
    """Market rows before stock_master.list_date are emerging-market trades (SPEC §8.6)."""
    stock_id = str(frame["stock_id"].iloc[0])
    listed = StockMaster.objects.filter(stock_id=stock_id).values_list("list_date", flat=True)
    first = listed.first()
    return frame if first is None else frame[frame["date"] >= first]


def _rows_from(raw_df: pd.DataFrame, finmind_dataset: str) -> list[dict[str, Any]]:
    """Rows of one FinMind dataset; keys a row never had stay absent so key checks still fail."""
    if raw_df.empty:
        return []
    picked = raw_df.loc[raw_df[FROM_COLUMN] == finmind_dataset].drop(columns=FROM_COLUMN)
    return [
        {str(key): value for key, value in row.items() if not _is_missing(value)}
        for row in picked.to_dict("records")
    ]


def _is_missing(value: Any) -> bool:
    return isinstance(value, float) and value != value
