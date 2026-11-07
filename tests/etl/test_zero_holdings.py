import copy
import json
from collections.abc import Callable, Mapping
from datetime import date, timedelta
from io import StringIO
from typing import Any

import pytest
from django.core.management import call_command
from django.utils import timezone

from apps.etl.models import DataQualityLog, RawTpex, RawTwse
from apps.etl.zero_holdings import REPLACED_CHECK, replace_zero_holdings
from apps.market.models import DailyChip, StockMaster

pytestmark = pytest.mark.django_db

DAYS = [date(2026, 9, 1) + timedelta(days=offset) for offset in range(25)]
NO_PAGE = DAYS[10]
TWSE_PAGE = "twse/foreign_holding/mi_qfiis_20261028.json"
TPEX_PAGE = "tpex/foreign_holding/qfii_20261028.json"
TWSE_HOLDING = 7
TPEX_HOLDING = 7


def _twse_page(
    load_fixture: Callable[[str], Any], day: date, values: Mapping[str, float], version: int = 0
) -> int:
    payload = copy.deepcopy(load_fixture(TWSE_PAGE))
    payload["date"] = f"{day:%Y%m%d}"
    listed = {row[0] for row in payload["data"]}
    for code in sorted(set(values) - listed):
        payload["data"].append([code, code, *payload["data"][2][2:]])
    payload["total"] = len(payload["data"])
    for row in payload["data"]:
        row[TWSE_HOLDING] = values.get(row[0], row[TWSE_HOLDING])
    params = {"date": f"{day:%Y%m%d}", "selectType": "ALLBUT0999", "response": "json"}
    raw = RawTwse.objects.create(
        dataset="foreign_holding",
        request_params=params,
        payload=payload,
        source_hash=f"{day}/{version}",
    )
    return raw.pk


def _tpex_page(load_fixture: Callable[[str], Any], day: date, values: Mapping[str, float]) -> int:
    payload = copy.deepcopy(load_fixture(TPEX_PAGE))
    payload["date"] = f"{day:%Y%m%d}"
    table = payload["tables"][0]
    table["date"] = f"{day.year - 1911}/{day:%m/%d}"
    for row in table["data"]:
        if row[1] in values:
            row[TPEX_HOLDING] = f"{values[row[1]]}%"
    params = {"date": f"{day:%Y/%m/%d}", "response": "json"}
    raw = RawTpex.objects.create(
        dataset="foreign_holding", request_params=params, payload=payload, source_hash=str(day)
    )
    return raw.pk


def _stored(code: str, values: Mapping[date, float]) -> None:
    DailyChip.objects.bulk_create(
        DailyChip(stock_id=code, date=day, fi_holding_pct=value) for day, value in values.items()
    )


def _holdings(code: str) -> dict[date, float | None]:
    return dict(DailyChip.objects.filter(stock_id=code).values_list("date", "fi_holding_pct"))


@pytest.fixture
def market(load_fixture: Callable[[str], Any]) -> dict[date, int]:
    """Stored FinMind zeros of five codes against TWSE pages of every day but one."""
    _stored("1101", dict.fromkeys(DAYS, 0.0))
    _stored("2330", dict.fromkeys(DAYS, 0.0))
    _stored("0050", dict.fromkeys(DAYS, 0.0))
    _stored("2317", {day: 0.0 if day == DAYS[11] else 5.0 for day in DAYS if day != NO_PAGE})
    _stored("1213", {day: 0.0 if day == DAYS[15] else 1.5 for day in DAYS})
    pages = {}
    for index, day in enumerate(DAYS):
        if day == NO_PAGE:
            continue
        values = {
            "1101": 0.0 if index == 3 else 13.36,
            "2330": 70.0 if index < 5 else 0.0,
            "2317": 4.9,
            "1213": 0.0 if index == 15 else 1.5,
            "0050": 2.97,
        }
        pages[day] = _twse_page(load_fixture, day, values)
    return pages


def test_a_wrong_run_and_a_short_zero_are_replaced_and_a_mixed_run_is_left(
    market: dict[date, int],
) -> None:
    report = replace_zero_holdings()
    whole = _holdings("1101")
    assert {day for day, value in whole.items() if value == 0} == {DAYS[3], NO_PAGE}
    assert {value for value in whole.values() if value} == {13.36}
    assert set(_holdings("2330").values()) == {0.0}
    assert set(_holdings("0050").values()) == {0.0}
    assert _holdings("2317")[DAYS[11]] == pytest.approx(4.9)
    assert _holdings("1213")[DAYS[15]] == 0.0
    assert report["replaced"] == {
        "twse/whole_run": {"stocks": 1, "rows": 23},
        "twse/short_run": {"stocks": 1, "rows": 1},
    }
    assert report["left"] == {
        "mixed_run": {"stocks": 1, "rows": 5},
        "thin_run": {},
        "unchecked": {"no_page": 2, "unreadable": 0, "not_listed": 0},
    }
    short = DataQualityLog.objects.get(check_name=REPLACED_CHECK, detail__stock_id="2317")
    assert (short.date, short.severity) == (DAYS[11], DataQualityLog.Severity.INFO)
    assert short.detail == {
        "stock_id": "2317",
        "kind": "short_run",
        "old": 0.0,
        "new": 4.9,
        "source": "twse",
        "raw_id": market[DAYS[11]],
    }
    assert DataQualityLog.objects.filter(check_name=REPLACED_CHECK).count() == 24


def test_a_second_run_replaces_and_logs_nothing_more(market: dict[date, int]) -> None:
    replace_zero_holdings()
    before = sorted(DailyChip.objects.values_list("stock_id", "date", "fi_holding_pct"))
    report = replace_zero_holdings()
    assert report["replaced"] == {}
    assert DataQualityLog.objects.filter(check_name=REPLACED_CHECK).count() == 24
    assert sorted(DailyChip.objects.values_list("stock_id", "date", "fi_holding_pct")) == before


def test_a_dry_run_reports_without_writing(market: dict[date, int]) -> None:
    report = replace_zero_holdings(dry_run=True)
    assert report["replaced"]["twse/whole_run"] == {"stocks": 1, "rows": 23}
    assert set(_holdings("1101").values()) == {0.0}
    assert not DataQualityLog.objects.filter(check_name=REPLACED_CHECK).exists()


def test_a_tpex_zero_is_read_from_the_tpex_page(load_fixture: Callable[[str], Any]) -> None:
    days = DAYS[:3]
    _stored("3293", {days[0]: 35.0, days[1]: 0.0, days[2]: 35.2})
    raw_ids = [_tpex_page(load_fixture, day, {"3293": 35.1}) for day in days]
    report = replace_zero_holdings()
    assert _holdings("3293")[days[1]] == pytest.approx(35.1)
    assert report["replaced"] == {"tpex/short_run": {"stocks": 1, "rows": 1}}
    log = DataQualityLog.objects.get(check_name=REPLACED_CHECK)
    assert (log.detail["source"], log.detail["raw_id"]) == ("tpex", raw_ids[1])


def test_the_command_prints_the_report_and_honours_dry_run(market: dict[date, int]) -> None:
    out = StringIO()
    call_command("replace_zero_holdings", "--dry-run", stdout=out)
    printed = json.loads(out.getvalue())
    assert printed["dry_run"] is True
    assert printed["replaced"]["twse/short_run"] == {"stocks": 1, "rows": 1}
    assert set(_holdings("1101").values()) == {0.0}


def test_a_row_set_back_to_zero_is_replaced_again_without_a_second_log(
    market: dict[date, int],
) -> None:
    replace_zero_holdings()
    DailyChip.objects.filter(stock_id="2317", date=DAYS[11]).update(fi_holding_pct=0.0)
    replace_zero_holdings()
    assert _holdings("2317")[DAYS[11]] == pytest.approx(4.9)
    assert (
        DataQualityLog.objects.filter(check_name=REPLACED_CHECK, detail__stock_id="2317").count()
        == 1
    )


def test_an_unreadable_page_counts_as_unchecked(
    load_fixture: Callable[[str], Any], caplog: pytest.LogCaptureFixture
) -> None:
    days = DAYS[:3]
    _stored("2317", {days[0]: 5.0, days[1]: 0.0, days[2]: 5.0})
    raw_id = _twse_page(load_fixture, days[1], {"2317": 4.9})
    RawTwse.objects.filter(pk=raw_id).update(payload={"stat": "OK", "date": f"{days[1]:%Y%m%d}"})
    report = replace_zero_holdings()
    assert report["replaced"] == {}
    assert report["left"]["unchecked"] == {"no_page": 0, "unreadable": 1, "not_listed": 0}
    assert _holdings("2317")[days[1]] == 0.0
    assert "unreadable" in caplog.text


def test_depositary_receipts_are_left_alone(load_fixture: Callable[[str], Any]) -> None:
    days = DAYS[:3]
    StockMaster.objects.create(stock_id="2317", name="2317", market="TWSE", industry="存託憑證")
    for code in ("9105", "2317"):
        _stored(code, {days[0]: 3.0, days[1]: 0.0, days[2]: 3.0})
    for day in days:
        _twse_page(load_fixture, day, {"9105": 3.0, "2317": 3.0})
    assert replace_zero_holdings()["replaced"] == {}
    assert _holdings("9105")[days[1]] == 0.0
    assert _holdings("2317")[days[1]] == 0.0


def test_the_run_length_and_the_two_ninety_percent_bounds(
    load_fixture: Callable[[str], Any],
) -> None:
    days = DAYS[:20]
    above = {
        "1301": set(range(10)),
        "1303": set(range(10)),
        "1326": set(range(18)),
        "1402": set(range(17)),
        "2002": set(range(20)),
        "2105": set(range(20)),
        "2201": set(range(8)),
    }
    listed = {"2002": set(range(18)), "2105": set(range(17)), "2201": set(range(10))}
    _stored("1301", dict.fromkeys(days[:19], 0.0))
    for code in ("1303", "1326", "1402", "2002", "2105", "2201"):
        _stored(code, dict.fromkeys(days, 0.0))
    for index, day in enumerate(days):
        values = {
            code: 6.0 if index in hits else 0.0
            for code, hits in above.items()
            if index in listed.get(code, set(range(20)))
        }
        _twse_page(load_fixture, day, values)
        _tpex_page(load_fixture, day, {})
    report = replace_zero_holdings()
    assert report["replaced"] == {
        "twse/short_run": {"stocks": 1, "rows": 10},
        "twse/whole_run": {"stocks": 2, "rows": 36},
    }
    assert report["left"] == {
        "mixed_run": {"stocks": 2, "rows": 27},
        "thin_run": {"stocks": 2, "rows": 25},
        "unchecked": {"no_page": 0, "unreadable": 0, "not_listed": 15},
    }
    for code in ("1303", "1402", "2105", "2201"):
        assert set(_holdings(code).values()) == {0.0}


def test_the_latest_page_of_a_day_is_read(load_fixture: Callable[[str], Any]) -> None:
    days = DAYS[:3]
    _stored("2317", {days[0]: 5.0, days[1]: 0.0, days[2]: 5.0})
    newer = _twse_page(load_fixture, days[1], {"2317": 4.9})
    older = _twse_page(load_fixture, days[1], {"2317": 3.0}, version=1)
    RawTwse.objects.filter(pk=older).update(fetched_at=timezone.now() - timedelta(hours=4))
    replace_zero_holdings()
    assert _holdings("2317")[days[1]] == pytest.approx(4.9)
    log = DataQualityLog.objects.get(check_name=REPLACED_CHECK)
    assert log.detail["raw_id"] == newer
