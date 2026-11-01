import copy
import io
import zipfile
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import responses
from django.core.management import call_command

from apps.etl.adapters.datagov import DatagovAdapter
from apps.etl.errors import SchemaChangedError, SourceNotReadyError
from apps.etl.models import EtlRun, RawDatagov
from apps.etl.ratelimit import RateLimiter
from apps.market.calendar import CalendarError, TradingDays
from apps.market.models import MacroIndicator

META_URL = "https://data.gov.tw/api/v2/rest/dataset/6099"
META = "datagov/dataset_6099.json"
CSV = "datagov/business_indicators_202609.csv"
MEMBER = "景氣指標與燈號.csv"
DATASET = "business_signal"
RUN_DAY = date(2026, 11, 1)


@pytest.fixture
def adapter(october_2026: TradingDays, twse_limiter: RateLimiter) -> DatagovAdapter:
    return DatagovAdapter(october_2026, twse_limiter)


@pytest.fixture
def meta(load_fixture: Callable[[str], Any]) -> dict[str, Any]:
    return copy.deepcopy(load_fixture(META))


@pytest.fixture
def csv_text(fixture_path: Callable[[str], Path]) -> str:
    return fixture_path(CSV).read_bytes().decode("utf-8-sig")


def _download_url(meta: dict[str, Any]) -> str:
    url: str = meta["result"]["distribution"][0]["resourceDownloadUrl"]
    return url


def _zip(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, text in files.items():
            archive.writestr(name, b"\xef\xbb\xbf" + text.encode("utf-8"))
    return buffer.getvalue()


def _serve(meta: dict[str, Any], body: bytes) -> None:
    responses.get(META_URL, json=meta)
    responses.get(_download_url(meta), body=body, content_type="application/zip")


def _frame(
    adapter: DatagovAdapter, meta: dict[str, Any], text: str, end: date = RUN_DAY
) -> pd.DataFrame:
    _serve(meta, _zip({"manifest.csv": "name\n", MEMBER: text}))
    return adapter.normalize(DATASET, adapter.fetch(DATASET, end, end))


def _values(frame: pd.DataFrame) -> dict[str, tuple[float, date]]:
    return {row["period"]: (row["value"], row["announce_date"]) for row in frame.to_dict("records")}


@responses.activate
@pytest.mark.django_db
def test_fetch_reads_the_download_link_then_keeps_the_csv_as_raw(
    adapter: DatagovAdapter, meta: dict[str, Any], csv_text: str
) -> None:
    _frame(adapter, meta, csv_text)
    assert responses.calls[0].request.url == META_URL
    raw = RawDatagov.objects.get()
    assert raw.dataset == DATASET
    assert raw.request_params == {"dataset_id": "6099"}
    assert raw.payload == {"url": _download_url(meta), "file": MEMBER, "text": csv_text}


@responses.activate
@pytest.mark.django_db
def test_scores_become_monthly_values_announced_mid_month_two_months_on(
    adapter: DatagovAdapter, meta: dict[str, Any], csv_text: str
) -> None:
    frame = _frame(adapter, meta, csv_text)
    assert set(frame["indicator_id"]) == {"business_signal"}
    assert _values(frame) == {
        "1984-01": (42.0, date(1984, 3, 15)),
        "1984-02": (39.0, date(1984, 4, 15)),
        "2026-07": (41.0, date(2026, 9, 15)),
        "2026-08": (40.0, date(2026, 10, 15)),
        "2026-09": (41.0, date(2026, 11, 15)),
    }


@responses.activate
@pytest.mark.django_db
def test_rerun_overwrites_a_revised_score_and_keeps_its_announce_date(
    adapter: DatagovAdapter, meta: dict[str, Any], csv_text: str
) -> None:
    adapter.upsert(DATASET, _frame(adapter, meta, csv_text))
    revised = csv_text.replace(
        "105.1973421,141.5689863,104.9316227,40,", "105.1973421,141.5689863,104.9316227,39,"
    )
    adapter.upsert(DATASET, _frame(adapter, meta, revised))
    assert MacroIndicator.objects.count() == 5
    stored = MacroIndicator.objects.get(indicator_id="business_signal", period="2026-08")
    assert (stored.value, stored.announce_date) == (39.0, date(2026, 10, 15))


@responses.activate
@pytest.mark.django_db
def test_a_missing_expected_month_means_not_published_yet_and_keeps_raw(
    adapter: DatagovAdapter, meta: dict[str, Any], csv_text: str
) -> None:
    with pytest.raises(SourceNotReadyError, match="2026-10"):
        _frame(adapter, meta, csv_text, end=date(2026, 12, 15))
    assert RawDatagov.objects.count() == 1


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize("end", [date(2026, 12, 14), date(2026, 11, 15)])
def test_a_run_needs_only_months_already_due(
    adapter: DatagovAdapter, meta: dict[str, Any], csv_text: str, end: date
) -> None:
    frame = _frame(adapter, meta, csv_text, end=end)
    assert "2026-09" in set(frame["period"])


@responses.activate
@pytest.mark.django_db
def test_a_file_without_any_score_means_not_published_yet(
    adapter: DatagovAdapter, meta: dict[str, Any], csv_text: str
) -> None:
    header, *rows = csv_text.splitlines(keepends=True)
    unscored = [row for row in rows if row.rstrip().endswith(",-,-")]
    with pytest.raises(SourceNotReadyError, match="none"):
        _frame(adapter, meta, header + "".join(unscored))


@responses.activate
@pytest.mark.django_db
def test_a_due_month_without_a_score_means_not_published_yet(
    adapter: DatagovAdapter, meta: dict[str, Any], csv_text: str
) -> None:
    text = csv_text.replace("105.9287826,41,紅", "105.9287826,-,-")
    with pytest.raises(SourceNotReadyError, match="2026-09"):
        _frame(adapter, meta, text, end=date(2026, 11, 15))


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    "url",
    [
        "http://ws.ndc.gov.tw/Download.ashx?u=x",
        "https://evil.example.com/Download.ashx?u=x",
        "https://ndc.gov.tw.example.com/x.zip",
    ],
)
def test_a_download_link_off_government_https_is_refused(
    adapter: DatagovAdapter, meta: dict[str, Any], url: str
) -> None:
    meta["result"]["distribution"][0]["resourceDownloadUrl"] = url
    responses.get(META_URL, json=meta)
    with pytest.raises(SchemaChangedError, match="download link"):
        adapter.fetch(DATASET, RUN_DAY, RUN_DAY)
    assert len(responses.calls) == 1


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    "change",
    [
        lambda m: m["result"]["distribution"][0].__setitem__("resourceFormat", "CSV"),
        lambda m: m["result"].__setitem__("distribution", []),
        lambda m: m.__setitem__("success", False),
    ],
    ids=["no-zip", "no-distribution", "failed"],
)
def test_metadata_without_one_zip_is_a_schema_change(
    adapter: DatagovAdapter, meta: dict[str, Any], change: Callable[[dict[str, Any]], None]
) -> None:
    change(meta)
    responses.get(META_URL, json=meta)
    with pytest.raises(SchemaChangedError, match="6099"):
        adapter.fetch(DATASET, RUN_DAY, RUN_DAY)


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"not a zip", "ZIP"),
        (_zip({"manifest.csv": "name\n"}), MEMBER),
    ],
    ids=["not-zip", "no-member"],
)
def test_an_unusable_archive_is_a_schema_change(
    adapter: DatagovAdapter, meta: dict[str, Any], body: bytes, message: str
) -> None:
    _serve(meta, body)
    with pytest.raises(SchemaChangedError, match=message):
        adapter.fetch(DATASET, RUN_DAY, RUN_DAY)


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda text: text.replace("景氣對策信號綜合分數", "綜合分數"), "columns"),
        (lambda text: text.replace(",40,紅", ",四十,紅"), "score"),
        (lambda text: text.replace(",40,紅", ",46,紅"), "score"),
        (lambda text: text.replace(",40,紅", ",nan,紅"), "score"),
        (lambda text: text.replace("202608,", "2026-08,"), "period"),
        (lambda text: text.replace("202608,", "202613,"), "period"),
        (lambda text: text.replace("202608,", "202607,"), "repeats"),
    ],
    ids=["renamed", "word", "over", "nan", "dashed", "month-13", "repeated"],
)
def test_bad_rows_are_schema_changes_and_keep_raw(
    adapter: DatagovAdapter,
    meta: dict[str, Any],
    csv_text: str,
    change: Callable[[str], str],
    message: str,
) -> None:
    with pytest.raises(SchemaChangedError, match=message):
        _frame(adapter, meta, change(csv_text))
    assert RawDatagov.objects.count() == 1


@responses.activate
def test_an_end_after_today_is_rejected_before_any_request(adapter: DatagovAdapter) -> None:
    with pytest.raises(CalendarError):
        list(adapter.units(DATASET, date(2999, 1, 1), date(2999, 1, 1)))
    assert len(responses.calls) == 0


@responses.activate
@pytest.mark.usefixtures("october_2026_in_db")
def test_fetch_command_runs_on_a_non_trading_first_of_month(
    meta: dict[str, Any], csv_text: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)
    _serve(meta, _zip({MEMBER: csv_text}))
    call_command("fetch", source="datagov", dataset=DATASET, date="2026-11-01")
    run = EtlRun.objects.get()
    assert (run.status, run.rows, run.checkpoint) == ("success", 5, {"last_date": "2026-11-01"})


@responses.activate
@pytest.mark.django_db
def test_an_oversized_member_is_a_schema_change(
    adapter: DatagovAdapter, meta: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apps.etl.adapters.datagov.MAX_FILE_BYTES", 10)
    _serve(meta, _zip({MEMBER: "Date,景氣對策信號綜合分數\n"}))
    with pytest.raises(SchemaChangedError, match="bytes"):
        adapter.fetch(DATASET, RUN_DAY, RUN_DAY)


@responses.activate
@pytest.mark.django_db
def test_a_member_that_is_not_utf8_is_a_schema_change(
    adapter: DatagovAdapter, meta: dict[str, Any]
) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(MEMBER, "景氣".encode("big5"))
    _serve(meta, buffer.getvalue())
    with pytest.raises(SchemaChangedError, match="UTF-8"):
        adapter.fetch(DATASET, RUN_DAY, RUN_DAY)
