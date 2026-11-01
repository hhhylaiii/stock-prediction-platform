"""TDCC weekly shareholding distribution (集保股權分散表) into weekly_holding (SPEC §3, §4)."""

import csv
import io
import math
from collections import defaultdict
from collections.abc import Hashable, Iterator
from datetime import date, timedelta
from typing import NamedTuple

import pandas as pd
import requests
from django.conf import settings

from apps.etl.adapters.base import BaseAdapter
from apps.etl.adapters.exchange_tables import is_foreign_currency_etf
from apps.etl.client import SourceClient
from apps.etl.errors import SchemaChangedError, SourceNotReadyError, SourceRequestError
from apps.etl.exchange_rows import stock_code
from apps.etl.models import RawTdcc
from apps.etl.ratelimit import RateLimiter
from apps.etl.upsert import upsert
from apps.market.calendar import TradingDays
from apps.market.models import WeeklyHolding

TDCC_MIN_INTERVAL = 1.0
HOLDING_PATH = "/getOD.ashx"
HOLDING_FILE = {"id": "1-5"}
WEEKLY_HOLDING = "weekly_holding"
FIELDS = ["資料日期", "證券代號", "持股分級", "人數", "股數", "占集保庫存數比例%"]
COLUMNS = ["data_date", "stock_id", "level", "holders", "shares", "pct"]
SIZE_LEVELS = range(1, 16)
ADJUSTMENT = 16
TOTAL = 17
MAX_PERCENT = 100.0
PUBLISH_DELAY = timedelta(days=1)


class Level(NamedTuple):
    data_date: date
    stock_id: str
    level: int
    holders: int
    shares: int
    pct: float


class TdccClient(SourceClient):
    source = "tdcc"
    label = "TDCC"
    raw_model = RawTdcc
    min_interval = TDCC_MIN_INTERVAL

    @classmethod
    def build(cls, limiter: RateLimiter, session: requests.Session | None = None) -> "TdccClient":
        return cls(limiter, str(settings.TDCC_BASE_URL), float(settings.TDCC_TIMEOUT), session)

    def holding_csv(self) -> str:
        body = self.get_content(f"{self.base_url}{HOLDING_PATH}", HOLDING_FILE)
        try:
            return body.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise SchemaChangedError("TDCC shareholding file is not UTF-8") from None


class TdccAdapter(BaseAdapter):
    source = "tdcc"
    raw_model = RawTdcc
    datasets = frozenset({WEEKLY_HOLDING})

    def __init__(
        self,
        calendar: TradingDays,
        limiter: RateLimiter,
        session: requests.Session | None = None,
    ) -> None:
        super().__init__(calendar, limiter)
        self.client = TdccClient.build(limiter, session)

    def units(self, dataset: str, start: date, end: date) -> Iterator[tuple[date, date]]:
        """One request for the whole range: the file only ever holds the latest week."""
        days = self.trading_days(start, end)
        yield days[0], days[-1]

    def fetch(
        self, dataset: str, start: date, end: date, stock_id: str | None = None
    ) -> pd.DataFrame:
        self.require_dataset(dataset)
        text = self.client.holding_csv()
        rows = list(csv.reader(io.StringIO(text)))
        if len(rows) < 2:
            raise SourceNotReadyError("TDCC shareholding file has no rows")
        self.save_raw(dataset, HOLDING_FILE, {"text": text})
        if [name.strip() for name in rows[0]] != FIELDS:
            raise SchemaChangedError(f"TDCC shareholding columns are {rows[0]}")
        if any(len(row) != len(FIELDS) for row in rows[1:]):
            raise SchemaChangedError("TDCC shareholding rows do not match its columns")
        frame = pd.DataFrame(rows[1:], columns=COLUMNS, dtype=object)
        self._check_week(frame, start, end)
        return frame

    def normalize(self, dataset: str, raw_df: pd.DataFrame) -> pd.DataFrame:
        self.require_dataset(dataset)
        records = [_level(row) for row in raw_df.to_dict("records")]
        by_code: dict[str, list[Level]] = defaultdict(list)
        for record in records:
            by_code[record.stock_id].append(record)
        for code, levels in by_code.items():
            _check_levels(code, levels)
        kept = [record for record in records if not is_foreign_currency_etf(record.stock_id)]
        frame = pd.DataFrame(kept, columns=COLUMNS)
        frame["publish_date"] = frame["data_date"] + PUBLISH_DELAY
        return frame

    def upsert(self, dataset: str, frame: pd.DataFrame) -> int:
        self.require_dataset(dataset)
        return upsert(
            WeeklyHolding,
            frame,
            unique_fields=("stock_id", "data_date", "level"),
            update_fields=("holders", "shares", "pct", "publish_date"),
        )

    def _check_week(self, frame: pd.DataFrame, start: date, end: date) -> None:
        texts = set(frame["data_date"].str.strip())
        if len(texts) != 1:
            raise SchemaChangedError(f"TDCC shareholding file has data dates {sorted(texts)}")
        week = _day(texts.pop())
        if week < start:
            raise SourceNotReadyError(f"TDCC still serves {week}; {start}..{end} is not out yet")
        if week > end:
            raise SourceRequestError(
                f"TDCC only serves the latest week ({week}); {start}..{end} is gone"
            )
        if not self.calendar.is_trading_day(week):
            raise SchemaChangedError(f"TDCC data date {week} is not a trading day")


def _day(text: str) -> date:
    try:
        if len(text) != len("YYYYMMDD") or not text.isdigit():
            raise ValueError(text)
        return date(int(text[:4]), int(text[4:6]), int(text[6:]))
    except ValueError:
        raise SchemaChangedError(f"unexpected TDCC data date {text!r}") from None


def _level(row: dict[Hashable, object]) -> Level:
    level = _count(row["level"])
    if not 1 <= level <= TOTAL:
        raise SchemaChangedError(f"TDCC holding levels run 1..{TOTAL}, not {level}")
    return Level(
        data_date=_day(str(row["data_date"]).strip()),
        stock_id=stock_code(row["stock_id"]),
        level=level,
        holders=_count(row["holders"]),
        shares=_count(row["shares"]),
        pct=_percent(row["pct"]),
    )


def _count(text: object) -> int:
    cleaned = str(text).strip()
    if not cleaned.isdigit():
        raise SchemaChangedError(f"unexpected TDCC count {text!r}")
    return int(cleaned)


def _percent(text: object) -> float:
    try:
        number = float(str(text).strip())
    except ValueError:
        raise SchemaChangedError(f"unexpected TDCC percent {text!r}") from None
    if not math.isfinite(number) or not 0 <= number <= MAX_PERCENT:
        raise SchemaChangedError(f"unexpected TDCC percent {text!r}")
    return number


def _check_levels(code: str, records: list[Level]) -> None:
    """Holders of levels 1–15 make the total; their shares less the level-16 adjustment do too."""
    levels = {record.level: record for record in records}
    if len(records) != TOTAL or len(levels) != TOTAL:
        raise SchemaChangedError(f"TDCC {code} has levels {sorted(levels)}, not 1..{TOTAL}")
    holders = sum(levels[level].holders for level in SIZE_LEVELS)
    shares = sum(levels[level].shares for level in SIZE_LEVELS) - levels[ADJUSTMENT].shares
    total = levels[TOTAL]
    if (holders, shares) != (total.holders, total.shares):
        raise SchemaChangedError(
            f"TDCC {code} levels add up to {holders} holders and {shares} shares, "
            f"not the total {total.holders} and {total.shares}"
        )
