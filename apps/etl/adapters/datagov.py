"""data.gov.tw business-cycle signal (景氣對策信號) into macro_indicator (SPEC §3, §6)."""

import csv
import io
import math
import zipfile
from collections.abc import Iterator
from datetime import date
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests
from django.conf import settings
from django.utils import timezone

from apps.etl.adapters.base import BaseAdapter
from apps.etl.client import SourceClient
from apps.etl.errors import SchemaChangedError, SourceNotReadyError
from apps.etl.models import RawDatagov
from apps.etl.ratelimit import RateLimiter
from apps.etl.upsert import upsert
from apps.market.calendar import CalendarError, TradingDays
from apps.market.models import MacroIndicator

DATAGOV_MIN_INTERVAL = 1.0
BUSINESS_SIGNAL = "business_signal"
SIGNAL_DATASET_ID = "6099"
META_PATH = "/api/v2/rest/dataset/"
ARCHIVE_FORMAT = "ZIP"
SIGNAL_FILE = "景氣指標與燈號.csv"
PERIOD_FIELD = "Date"
SCORE_FIELD = "景氣對策信號綜合分數"
NOT_SCORED = "-"
SCORE_RANGE = (9.0, 45.0)
GOVERNMENT_HOST = ".gov.tw"
MAX_FILE_BYTES = 10_000_000
ANNOUNCE_LAG_MONTHS = 2
COLUMNS = ["indicator_id", "period", "value", "announce_date"]


class DatagovClient(SourceClient):
    source = "datagov"
    label = "data.gov.tw"
    raw_model = RawDatagov
    min_interval = DATAGOV_MIN_INTERVAL

    @classmethod
    def build(
        cls, limiter: RateLimiter, session: requests.Session | None = None
    ) -> "DatagovClient":
        return cls(
            limiter, str(settings.DATAGOV_BASE_URL), float(settings.DATAGOV_TIMEOUT), session
        )

    def archive_link(self, dataset_id: str) -> str:
        """The dataset's current ZIP link; the file path changes whenever it is re-uploaded."""
        meta = self.get_json(f"{META_PATH}{dataset_id}", {})
        result = meta.get("result") if meta.get("success") is True else None
        resources = result.get("distribution") if isinstance(result, dict) else None
        links = [
            resource.get("resourceDownloadUrl")
            for resource in resources or []
            if isinstance(resource, dict) and resource.get("resourceFormat") == ARCHIVE_FORMAT
        ]
        if len(links) != 1 or not isinstance(links[0], str):
            raise SchemaChangedError(f"data.gov.tw dataset {dataset_id} lists ZIP links {links}")
        url = urlparse(links[0])
        if url.scheme != "https" or not (url.hostname or "").endswith(GOVERNMENT_HOST):
            raise SchemaChangedError(
                f"data.gov.tw download link {links[0]!r} is not a gov.tw https"
            )
        return links[0]

    def archive_text(self, url: str, member: str) -> str:
        body = self.get_content(url, {})
        try:
            archive = zipfile.ZipFile(io.BytesIO(body))
        except zipfile.BadZipFile:
            raise SchemaChangedError(f"data.gov.tw {url} is not a ZIP archive") from None
        try:
            info = archive.getinfo(member)
        except KeyError:
            raise SchemaChangedError(f"data.gov.tw archive lost {member}") from None
        if info.file_size > MAX_FILE_BYTES:
            raise SchemaChangedError(f"data.gov.tw {member} is {info.file_size} bytes")
        try:
            return archive.read(member).decode("utf-8-sig")
        except UnicodeDecodeError:
            raise SchemaChangedError(f"data.gov.tw {member} is not UTF-8") from None


class DatagovAdapter(BaseAdapter):
    source = "datagov"
    raw_model = RawDatagov
    datasets = frozenset({BUSINESS_SIGNAL})

    def __init__(
        self,
        calendar: TradingDays,
        limiter: RateLimiter,
        session: requests.Session | None = None,
    ) -> None:
        super().__init__(calendar, limiter)
        self.client = DatagovClient.build(limiter, session)

    def units(self, dataset: str, start: date, end: date) -> Iterator[tuple[date, date]]:
        """One request on any date up to today: the file carries the whole monthly history."""
        today = timezone.localdate()
        if end > today:
            raise CalendarError(f"{end} is after today ({today}); nothing has been published")
        yield start, end

    def fetch(
        self, dataset: str, start: date, end: date, stock_id: str | None = None
    ) -> pd.DataFrame:
        self.require_dataset(dataset)
        url = self.client.archive_link(SIGNAL_DATASET_ID)
        text = self.client.archive_text(url, SIGNAL_FILE)
        self.save_raw(
            dataset,
            {"dataset_id": SIGNAL_DATASET_ID},
            {"url": url, "file": SIGNAL_FILE, "text": text},
        )
        rows = list(csv.DictReader(io.StringIO(text)))
        if not rows or not {PERIOD_FIELD, SCORE_FIELD} <= set(rows[0]):
            raise SchemaChangedError(f"data.gov.tw {SIGNAL_FILE} columns lack {SCORE_FIELD}")
        frame = pd.DataFrame(
            [(row[PERIOD_FIELD], row[SCORE_FIELD]) for row in rows],
            columns=["period", "score"],
            dtype=object,
        )
        due = _period(_shift_months(end.replace(day=1), -ANNOUNCE_LAG_MONTHS))
        latest = max(_period(_month(value)) for value in frame["period"])
        if latest < due:
            raise SourceNotReadyError(f"data.gov.tw {SIGNAL_FILE} ends at {latest}; {due} is due")
        return frame

    def normalize(self, dataset: str, raw_df: pd.DataFrame) -> pd.DataFrame:
        self.require_dataset(dataset)
        rows = []
        for text, score in zip(raw_df["period"], raw_df["score"], strict=True):
            month = _month(text)
            if str(score).strip() == NOT_SCORED:
                continue
            rows.append(
                {
                    "indicator_id": BUSINESS_SIGNAL,
                    "period": _period(month),
                    "value": _score(score),
                    "announce_date": _shift_months(month, ANNOUNCE_LAG_MONTHS),
                }
            )
        frame = pd.DataFrame(rows, columns=COLUMNS)
        repeated = frame["period"][frame["period"].duplicated()].tolist()
        if repeated:
            raise SchemaChangedError(f"data.gov.tw {SIGNAL_FILE} repeats periods {repeated}")
        return frame

    def upsert(self, dataset: str, frame: pd.DataFrame) -> int:
        self.require_dataset(dataset)
        return upsert(
            MacroIndicator,
            frame,
            unique_fields=("indicator_id", "period"),
            update_fields=("value", "announce_date"),
        )


def _month(text: Any) -> date:
    cleaned = str(text).strip()
    try:
        if len(cleaned) != len("YYYYMM") or not cleaned.isdigit():
            raise ValueError(cleaned)
        return date(int(cleaned[:4]), int(cleaned[4:]), 1)
    except ValueError:
        raise SchemaChangedError(f"unexpected data.gov.tw period {text!r}") from None


def _period(month: date) -> str:
    return f"{month:%Y-%m}"


def _shift_months(month: date, months: int) -> date:
    index = month.year * 12 + month.month - 1 + months
    return date(index // 12, index % 12 + 1, 1)


def _score(text: Any) -> float:
    try:
        number = float(str(text).strip())
    except ValueError:
        raise SchemaChangedError(f"unexpected business signal score {text!r}") from None
    low, high = SCORE_RANGE
    if not math.isfinite(number) or not low <= number <= high:
        raise SchemaChangedError(f"unexpected business signal score {text!r}")
    return number
