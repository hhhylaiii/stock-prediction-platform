from datetime import date
from typing import ClassVar

import pandas as pd
import pytest

from apps.etl.adapters.base import BaseAdapter
from apps.etl.models import EtlRun, RawRecord, RawTwse
from apps.etl.ratelimit import RateLimiter
from apps.etl.raw import payload_hash
from apps.etl.upsert import upsert
from apps.market.calendar import CalendarError, NotATradingDayError, TradingDays
from apps.market.models import DailyPrice


class EchoAdapter(BaseAdapter):
    """Fetches a canned payload per day so the base workflow can be tested without HTTP."""

    source = "echo"
    raw_model: ClassVar[type[RawRecord]] = RawTwse
    datasets = frozenset({"daily_price"})

    def __init__(
        self,
        calendar: TradingDays,
        limiter: RateLimiter,
        fail_on: date | None = None,
        failure: type[BaseException] = RuntimeError,
    ):
        super().__init__(calendar, limiter)
        self.fail_on = fail_on
        self.failure = failure

    def fetch(
        self, dataset: str, start: date, end: date, stock_id: str | None = None
    ) -> pd.DataFrame:
        if start == self.fail_on:
            raise self.failure(f"boom on {start}")
        payload = {"day": start.isoformat(), "rows": [["2330", "1,000"]]}
        self.save_raw(dataset, {"date": start.isoformat()}, payload)
        return pd.DataFrame(payload["rows"], columns=["code", "shares"]).assign(date=start)

    def normalize(self, dataset: str, raw_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "stock_id": raw_df["code"],
                "date": raw_df["date"],
                "volume": raw_df["shares"].str.replace(",", "").astype(int),
                "turnover": 0,
            }
        )

    def upsert(self, dataset: str, frame: pd.DataFrame) -> int:
        return upsert(DailyPrice, frame, ("stock_id", "date"), ("volume", "turnover"))


@pytest.fixture
def adapter(october_2026_in_db: TradingDays, twse_limiter: RateLimiter) -> EchoAdapter:
    return EchoAdapter(october_2026_in_db, twse_limiter)


def test_payload_hash_ignores_key_order() -> None:
    assert payload_hash({"a": 1, "b": [1, 2]}) == payload_hash({"b": [1, 2], "a": 1})
    assert payload_hash({"a": 1}) != payload_hash({"a": 2})


@pytest.mark.django_db
def test_run_writes_raw_standard_and_a_successful_etl_run(adapter: EchoAdapter) -> None:
    run = adapter.run("daily_price", date(2026, 10, 27), date(2026, 10, 28))
    assert run.status == EtlRun.Status.SUCCESS
    assert run.rows == 2
    assert run.checkpoint == {"last_date": "2026-10-28"}
    assert run.finished_at is not None
    assert RawTwse.objects.filter(dataset="daily_price").count() == 2
    assert DailyPrice.objects.filter(stock_id="2330").count() == 2


@pytest.mark.django_db
def test_rerun_is_idempotent_for_raw_and_standard(adapter: EchoAdapter) -> None:
    adapter.run("daily_price", date(2026, 10, 28), date(2026, 10, 28))
    adapter.run("daily_price", date(2026, 10, 28), date(2026, 10, 28))
    assert RawTwse.objects.count() == 1
    assert DailyPrice.objects.count() == 1
    assert EtlRun.objects.count() == 2


@pytest.mark.django_db
def test_failure_marks_the_run_failed_and_keeps_the_last_checkpoint(
    october_2026_in_db: TradingDays, twse_limiter: RateLimiter
) -> None:
    adapter = EchoAdapter(october_2026_in_db, twse_limiter, fail_on=date(2026, 10, 28))
    with pytest.raises(RuntimeError, match="boom"):
        adapter.run("daily_price", date(2026, 10, 27), date(2026, 10, 29))
    run = EtlRun.objects.get()
    assert run.status == EtlRun.Status.FAILED
    assert run.error == "RuntimeError: boom on 2026-10-28"
    assert run.checkpoint == {"last_date": "2026-10-27"}
    assert run.rows == 1


@pytest.mark.django_db
def test_interrupt_still_marks_the_run_failed(
    october_2026_in_db: TradingDays, twse_limiter: RateLimiter
) -> None:
    adapter = EchoAdapter(
        october_2026_in_db, twse_limiter, fail_on=date(2026, 10, 28), failure=KeyboardInterrupt
    )
    with pytest.raises(KeyboardInterrupt):
        adapter.run("daily_price", date(2026, 10, 27), date(2026, 10, 29))
    run = EtlRun.objects.get()
    assert run.status == EtlRun.Status.FAILED
    assert run.finished_at is not None
    assert run.checkpoint == {"last_date": "2026-10-27"}


@pytest.mark.django_db
def test_single_non_trading_day_is_rejected_without_a_run(adapter: EchoAdapter) -> None:
    with pytest.raises(NotATradingDayError):
        adapter.run("daily_price", date(2026, 10, 23), date(2026, 10, 23))
    assert EtlRun.objects.count() == 0


@pytest.mark.django_db
def test_range_without_trading_days_is_rejected(adapter: EchoAdapter) -> None:
    with pytest.raises(CalendarError):
        adapter.run("daily_price", date(2026, 10, 24), date(2026, 10, 26))


@pytest.mark.django_db
def test_range_skips_closed_days(adapter: EchoAdapter) -> None:
    run = adapter.run("daily_price", date(2026, 10, 22), date(2026, 10, 27))
    assert run.rows == 2
    assert sorted(DailyPrice.objects.values_list("date", flat=True)) == [
        date(2026, 10, 22),
        date(2026, 10, 27),
    ]


@pytest.mark.django_db
def test_unknown_dataset_is_rejected(adapter: EchoAdapter) -> None:
    with pytest.raises(ValueError, match="no dataset"):
        adapter.run("monthly_revenue", date(2026, 10, 28), date(2026, 10, 28))
