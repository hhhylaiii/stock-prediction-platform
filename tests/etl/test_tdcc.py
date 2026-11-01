from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import responses
from django.core.management import call_command
from responses import matchers

from apps.etl.adapters.tdcc import TdccAdapter
from apps.etl.errors import SchemaChangedError, SourceNotReadyError, SourceRequestError
from apps.etl.models import EtlRun, RawTdcc
from apps.etl.ratelimit import RateLimiter, RetryableError
from apps.market.calendar import NotATradingDayError, TradingDays
from apps.market.models import WeeklyHolding

URL = "https://opendata.tdcc.com.tw/getOD.ashx"
FIXTURE = "tdcc/tdcc_1-5_20261030.csv"
DAY = date(2026, 10, 30)
DATASET = "weekly_holding"


@pytest.fixture
def adapter(october_2026: TradingDays, twse_limiter: RateLimiter) -> TdccAdapter:
    return TdccAdapter(october_2026, twse_limiter)


@pytest.fixture
def csv_text(fixture_path: Callable[[str], Path]) -> str:
    return fixture_path(FIXTURE).read_bytes().decode("utf-8-sig")


def _serve(text: str) -> None:
    responses.get(
        URL,
        body=b"\xef\xbb\xbf" + text.encode("utf-8"),
        content_type="text/csv; charset=UTF-8",
        match=[matchers.query_param_matcher({"id": "1-5"})],
    )


def _frame(adapter: TdccAdapter, text: str, start: date = DAY, end: date = DAY) -> pd.DataFrame:
    _serve(text)
    return adapter.normalize(DATASET, adapter.fetch(DATASET, start, end))


def _levels(frame: pd.DataFrame, code: str) -> dict[int, tuple[int, int, float]]:
    rows = frame[frame["stock_id"] == code].to_dict("records")
    return {row["level"]: (row["holders"], row["shares"], row["pct"]) for row in rows}


def _edit(text: str, code: str, level: int, column: int, value: str) -> str:
    lines = text.split("\r\n")
    for index, line in enumerate(lines):
        cells = line.split(",")
        if len(cells) == 6 and cells[1].strip() == code and cells[2] == str(level):
            cells[column] = value
            lines[index] = ",".join(cells)
    return "\r\n".join(lines)


@responses.activate
@pytest.mark.django_db
def test_fetch_keeps_the_csv_text_as_raw(adapter: TdccAdapter, csv_text: str) -> None:
    _frame(adapter, csv_text)
    raw = RawTdcc.objects.get()
    assert raw.dataset == DATASET
    assert raw.request_params == {"id": "1-5"}
    assert raw.payload == {"text": csv_text}


@responses.activate
@pytest.mark.django_db
def test_every_code_keeps_seventeen_levels_as_published(
    adapter: TdccAdapter, csv_text: str
) -> None:
    frame = _frame(adapter, csv_text)
    levels = _levels(frame, "2330")
    assert sorted(levels) == list(range(1, 18))
    assert levels[1] == (2610784, 306416985, 1.18)
    assert levels[15] == (1478, 21883958871, 84.38)
    assert levels[16] == (1, 1000, 0.0)
    assert levels[17] == (3185409, 25932370067, 100.0)
    assert set(frame["data_date"]) == {DAY}
    assert set(frame["publish_date"]) == {date(2026, 10, 31)}


@responses.activate
@pytest.mark.django_db
def test_codes_are_trimmed_and_foreign_currency_lines_dropped(
    adapter: TdccAdapter, csv_text: str
) -> None:
    frame = _frame(adapter, csv_text)
    assert set(frame["stock_id"]) == {"1238", "2330", "6488", "00631L"}
    assert len(frame) == 4 * 17


@responses.activate
@pytest.mark.django_db
def test_upsert_writes_each_level_once_and_reruns_add_nothing(
    adapter: TdccAdapter, csv_text: str
) -> None:
    adapter.upsert(DATASET, _frame(adapter, csv_text))
    adapter.upsert(DATASET, _frame(adapter, csv_text))
    assert WeeklyHolding.objects.count() == 4 * 17
    stored = WeeklyHolding.objects.get(stock_id="00631L", data_date=DAY, level=16)
    assert (stored.holders, stored.shares, stored.pct) == (1, 1500000, 0.01)


@responses.activate
@pytest.mark.django_db
def test_a_week_inside_the_requested_range_is_accepted(adapter: TdccAdapter, csv_text: str) -> None:
    frame = _frame(adapter, csv_text, date(2026, 10, 26), DAY)
    assert set(frame["data_date"]) == {DAY}


@responses.activate
@pytest.mark.django_db
def test_an_older_week_means_not_published_yet_and_keeps_raw(
    adapter: TdccAdapter, csv_text: str
) -> None:
    _serve(csv_text.replace("20261030,", "20261023,"))
    with pytest.raises(SourceNotReadyError, match="still serves 2026-10-23"):
        adapter.fetch(DATASET, DAY, DAY)
    assert RawTdcc.objects.count() == 1


@responses.activate
@pytest.mark.django_db
def test_a_newer_week_means_the_requested_one_is_gone(adapter: TdccAdapter, csv_text: str) -> None:
    _serve(csv_text)
    with pytest.raises(SourceRequestError, match="only serves the latest week"):
        adapter.fetch(DATASET, date(2026, 10, 23), date(2026, 10, 23))


@responses.activate
@pytest.mark.django_db
def test_a_header_only_file_is_not_published_and_not_kept(adapter: TdccAdapter) -> None:
    _serve("資料日期,證券代號,持股分級,人數,股數,占集保庫存數比例%\r\n")
    with pytest.raises(SourceNotReadyError, match="no rows"):
        adapter.fetch(DATASET, DAY, DAY)
    assert RawTdcc.objects.count() == 0


@responses.activate
@pytest.mark.django_db
def test_a_renamed_column_is_a_schema_change_and_keeps_raw(
    adapter: TdccAdapter, csv_text: str
) -> None:
    _serve(csv_text.replace("占集保庫存數比例%", "比例"))
    with pytest.raises(SchemaChangedError, match="columns"):
        adapter.fetch(DATASET, DAY, DAY)
    assert RawTdcc.objects.count() == 1


@responses.activate
@pytest.mark.django_db
def test_two_data_dates_in_one_file_is_a_schema_change(adapter: TdccAdapter, csv_text: str) -> None:
    _serve(csv_text.replace("20261030,2330", "20261029,2330"))
    with pytest.raises(SchemaChangedError, match="data dates"):
        adapter.fetch(DATASET, DAY, DAY)


@responses.activate
@pytest.mark.django_db
def test_a_data_date_off_the_calendar_is_a_schema_change(
    adapter: TdccAdapter, csv_text: str
) -> None:
    _serve(csv_text.replace("20261030,", "20261031,"))
    with pytest.raises(SchemaChangedError, match="not a trading day"):
        adapter.fetch(DATASET, date(2026, 10, 26), date(2026, 10, 31))


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda text: _edit(text, "6488", 3, 2, "18"), "levels"),
        (lambda text: _edit(text, "6488", 3, 2, "4"), "levels"),
        (lambda text: _edit(text, "6488", 3, 3, '"1,234"'), "count"),
        (lambda text: _edit(text, "6488", 3, 3, "-1"), "count"),
        (lambda text: _edit(text, "6488", 3, 5, "abc"), "percent"),
        (lambda text: _edit(text, "6488", 3, 5, "100.01"), "percent"),
        (lambda text: _edit(text, "6488", 3, 1, "64 88"), "code"),
    ],
    ids=["level-18", "level-repeated", "comma", "negative", "percent-word", "percent-over", "code"],
)
def test_bad_values_are_schema_changes(
    adapter: TdccAdapter, csv_text: str, change: Callable[[str], str], message: str
) -> None:
    with pytest.raises(SchemaChangedError, match=message):
        _frame(adapter, change(csv_text))


@responses.activate
@pytest.mark.django_db
def test_a_missing_level_is_a_schema_change(adapter: TdccAdapter, csv_text: str) -> None:
    lines = csv_text.split("\r\n")
    text = "\r\n".join(line for line in lines if not line.startswith("20261030,6488  ,5,"))
    with pytest.raises(SchemaChangedError, match="levels"):
        _frame(adapter, text)


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    ("level", "column"),
    [(4, 3), (4, 4), (16, 4), (17, 3)],
    ids=["holders", "shares", "adjustment", "total-holders"],
)
def test_levels_that_do_not_add_up_to_the_total_are_a_schema_change(
    adapter: TdccAdapter, csv_text: str, level: int, column: int
) -> None:
    line = next(
        row.split(",")
        for row in csv_text.split("\r\n")
        if row.startswith("20261030,2330  ,") and row.split(",")[2] == str(level)
    )
    text = _edit(csv_text, "2330", level, column, str(int(line[column]) + 1))
    with pytest.raises(SchemaChangedError, match="add up"):
        _frame(adapter, text)


@responses.activate
def test_non_trading_day_is_rejected_before_any_request(adapter: TdccAdapter) -> None:
    with pytest.raises(NotATradingDayError):
        list(adapter.units(DATASET, date(2026, 10, 31), date(2026, 10, 31)))
    assert len(responses.calls) == 0


@responses.activate
@pytest.mark.usefixtures("october_2026_in_db")
def test_fetch_command_writes_the_week(csv_text: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)
    _serve(csv_text)
    call_command("fetch", source="tdcc", dataset=DATASET, date="2026-10-30")
    run = EtlRun.objects.get()
    assert (run.status, run.rows, run.checkpoint) == ("success", 68, {"last_date": "2026-10-30"})
    assert WeeklyHolding.objects.filter(data_date=DAY).count() == 68


@responses.activate
@pytest.mark.django_db
def test_a_refused_download_is_a_request_error(adapter: TdccAdapter) -> None:
    responses.get(URL, status=404)
    with pytest.raises(SourceRequestError, match="404"):
        adapter.fetch(DATASET, DAY, DAY)


@responses.activate
@pytest.mark.django_db
def test_server_errors_are_retried_then_raised(adapter: TdccAdapter) -> None:
    responses.get(URL, status=503)
    with pytest.raises(RetryableError, match="503"):
        adapter.fetch(DATASET, DAY, DAY)
    assert len(responses.calls) == 3


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda text: text.replace("20261030,6488  ,3,", "20261030,6488  ,3,1,"), "rows"),
        (lambda text: text.replace("20261030,", "2026103,"), "data date"),
    ],
    ids=["extra-cell", "short-date"],
)
def test_malformed_files_are_schema_changes(
    adapter: TdccAdapter, csv_text: str, change: Callable[[str], str], message: str
) -> None:
    _serve(change(csv_text))
    with pytest.raises(SchemaChangedError, match=message):
        adapter.fetch(DATASET, DAY, DAY)


@responses.activate
@pytest.mark.django_db
def test_a_file_that_is_not_utf8_is_a_schema_change(adapter: TdccAdapter) -> None:
    responses.get(URL, body="資料日期".encode("big5"))
    with pytest.raises(SchemaChangedError, match="UTF-8"):
        adapter.fetch(DATASET, DAY, DAY)
