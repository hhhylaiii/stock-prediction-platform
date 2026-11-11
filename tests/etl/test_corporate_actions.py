from datetime import date
from decimal import Decimal
from typing import Any

import pandas as pd
import pytest

from apps.etl.adapters.corporate_actions import (
    MISMATCH_CHECK,
    Action,
    action_frame,
    events_from,
    price_factor,
    stock_value,
    write_events,
)
from apps.etl.adapters.finmind_datasets import EVENT_COLUMNS
from apps.etl.errors import SchemaChangedError
from apps.etl.models import DataQualityLog
from apps.etl.roc import parse_roc_day
from apps.market.adjust import ADJ_CHECK
from apps.market.models import DailyPrice, DividendEvent, StockMaster

EX = date(2026, 10, 27)
EVE = date(2026, 10, 26)
LED = date(2026, 11, 3)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("115/03/17", date(2026, 3, 17)),
        ("115年03月17日", date(2026, 3, 17)),
        ("1150825", date(2026, 8, 25)),
        ("990104", date(2010, 1, 4)),
        (" 99/01/04 ", date(2010, 1, 4)),
    ],
)
def test_parse_roc_day_reads_every_form_the_exchanges_print(text: str, expected: date) -> None:
    assert parse_roc_day(text) == expected


@pytest.mark.parametrize("text", ["", "115-13-01", "2026/03/17", "115年3月", "11508"])
def test_parse_roc_day_refuses_anything_else(text: str) -> None:
    with pytest.raises(SchemaChangedError):
        parse_roc_day(text)


def test_price_factor_is_reference_over_close_and_blank_without_prices() -> None:
    assert price_factor(Decimal("30.04"), Decimal("44.40"), "x") == pytest.approx(30.04 / 44.4)
    assert price_factor(Decimal("10"), None, "x") is None
    assert price_factor(Decimal("0"), Decimal("12"), "x") is None


def test_stock_value_counts_shares_per_thousand_at_a_ten_dollar_par() -> None:
    assert stock_value(Decimal("50")) == Decimal("0.5")
    assert stock_value(Decimal("0")) == Decimal("0")


def test_events_from_multiplies_the_factors_of_one_day_and_adds_the_amounts() -> None:
    frame = action_frame(
        [
            Action("1101", EX, Decimal("1.5"), Decimal("0"), 0.9, "ex_rights"),
            Action("1101", EX, Decimal("0"), Decimal("0"), 0.5, "capital_reduction"),
            Action("1101", EX, Decimal("0.25"), Decimal("0.3"), 1.0, "ex_rights_detail"),
            Action("2330", EX, Decimal("0"), Decimal("0"), None, "par_value_change"),
        ]
    )
    events = events_from(frame).set_index("stock_id")
    assert events.loc["1101", "adj_factor"] == pytest.approx(0.45)
    assert events.loc["1101", "cash_dividend"] == Decimal("1.75")
    assert events.loc["1101", "stock_dividend"] == Decimal("0.3")
    assert events.loc["2330", "adj_factor"] is None
    assert events["announce_date"].isna().all()


def test_events_from_leaves_the_factor_blank_when_any_table_of_the_day_has_none() -> None:
    frame = action_frame(
        [
            Action("1101", EX, Decimal("1.5"), Decimal("0"), 0.9, "ex_rights"),
            Action("1101", EX, Decimal("0"), Decimal("0"), None, "capital_reduction"),
        ]
    )
    events = events_from(frame).set_index("stock_id")
    assert events.loc["1101", "adj_factor"] is None
    assert events.loc["1101", "cash_dividend"] == Decimal("1.5")


def test_events_from_an_empty_frame_has_the_event_columns() -> None:
    events = events_from(action_frame([]))
    assert events.empty
    assert list(events.columns) == EVENT_COLUMNS


def _listed(*codes: str) -> None:
    StockMaster.objects.bulk_create(
        StockMaster(stock_id=code, name=code, market="TWSE") for code in codes
    )


def _price(code: str, day: date) -> None:
    DailyPrice.objects.create(
        stock_id=code, date=day, open=10, high=10, low=10, close=10, volume=1, turnover=10
    )


def _events(*actions: Action) -> pd.DataFrame:
    return events_from(action_frame(actions))


@pytest.mark.django_db
def test_write_events_adds_listed_codes_and_readjusts_earlier_prices() -> None:
    _listed("1101")
    _price("1101", EVE)
    _price("1101", EX)
    frame = _events(
        Action("1101", EX, Decimal("1.2"), Decimal("0"), 0.88, "ex_rights"),
        Action("9999", EX, Decimal("1"), Decimal("0"), 0.9, "ex_rights"),
    )
    assert write_events(frame, "twse", EVE) == 1
    event = DividendEvent.objects.get()
    assert (event.stock_id, event.ex_date, event.announce_date) == ("1101", EX, None)
    assert event.adj_factor == pytest.approx(0.88)
    factors = dict(DailyPrice.objects.values_list("date", "adj_factor"))
    assert factors == {EVE: pytest.approx(0.88), EX: 1.0}
    note = DataQualityLog.objects.get(check_name=ADJ_CHECK)
    assert note.detail == {"stocks": {"1101": EX.isoformat()}}


@pytest.mark.django_db
def test_write_events_fills_a_blank_factor_and_logs_but_keeps_different_figures() -> None:
    _listed("1101", "2330")
    _price("1101", EVE)
    _price("1101", EX)
    DividendEvent.objects.create(
        stock_id="1101",
        ex_date=EX,
        cash_dividend=Decimal("1.2"),
        announce_date=date(2026, 10, 20),
    )
    DividendEvent.objects.create(
        stock_id="2330", ex_date=EX, cash_dividend=Decimal("70"), adj_factor=0.99
    )
    frame = _events(
        Action("1101", EX, Decimal("1.2"), Decimal("0"), 0.88, "ex_rights"),
        Action("2330", EX, Decimal("70.5"), Decimal("0"), 0.98, "ex_rights"),
    )
    assert write_events(frame, "twse", EVE) == 1
    kept = {row.stock_id: row for row in DividendEvent.objects.all()}
    assert kept["1101"].adj_factor == pytest.approx(0.88)
    assert kept["1101"].announce_date == date(2026, 10, 20)
    factors = dict(DailyPrice.objects.values_list("date", "adj_factor"))
    assert factors == {EVE: pytest.approx(0.88), EX: 1.0}
    assert kept["2330"].adj_factor == pytest.approx(0.99)
    assert kept["2330"].cash_dividend == Decimal("70")
    log = DataQualityLog.objects.get(check_name=MISMATCH_CHECK)
    assert log.date == EVE
    assert log.severity == DataQualityLog.Severity.INFO
    assert log.detail == {
        "source": "twse",
        "events": [
            {
                "stock_id": "2330",
                "ex_date": "2026-10-27",
                "fields": {"cash_dividend": ["70", "70.5"], "adj_factor": [0.99, 0.98]},
            }
        ],
    }


@pytest.mark.django_db
def test_write_events_again_writes_nothing_and_replaces_its_own_log_only() -> None:
    _listed("2330")
    DividendEvent.objects.create(stock_id="2330", ex_date=EX, adj_factor=0.99)
    DataQualityLog.objects.create(
        date=EVE,
        check_name=MISMATCH_CHECK,
        severity=DataQualityLog.Severity.INFO,
        detail={"source": "tpex", "events": []},
    )
    frame = _events(Action("2330", EX, Decimal("0"), Decimal("0"), 0.98, "ex_rights"))
    write_events(frame, "twse", EVE)
    assert write_events(frame, "twse", EVE) == 0
    sources = sorted(
        log.detail["source"] for log in DataQualityLog.objects.filter(check_name=MISMATCH_CHECK)
    )
    assert sources == ["tpex", "twse"]


@pytest.mark.django_db
def test_write_events_removes_its_log_once_the_figures_agree() -> None:
    _listed("2330")
    DividendEvent.objects.create(stock_id="2330", ex_date=EX, adj_factor=0.99)
    write_events(
        _events(Action("2330", EX, Decimal("0"), Decimal("0"), 0.98, "ex_rights")),
        "twse",
        EVE,
    )
    write_events(
        _events(Action("2330", EX, Decimal("0"), Decimal("0"), 0.99, "ex_rights")),
        "twse",
        EVE,
    )
    assert not DataQualityLog.objects.filter(check_name=MISMATCH_CHECK).exists()


def _stored_from_a_notice() -> None:
    _listed("2330", "6655")
    DividendEvent.objects.create(
        stock_id="6655", ex_date=LED, cash_dividend=Decimal("2.6"), announce_date=EVE
    )
    DividendEvent.objects.create(
        stock_id="2330", ex_date=LED, cash_dividend=Decimal("70"), adj_factor=0.99
    )


@pytest.mark.django_db
def test_from_the_day_the_exchanges_lead_the_results_table_settles_the_figures() -> None:
    _stored_from_a_notice()
    frame = _events(
        Action("6655", LED, Decimal("2.65"), Decimal("0.5"), 0.97, "ex_rights"),
        Action("2330", LED, Decimal("70.5"), Decimal("0"), 0.98, "ex_rights"),
    )

    assert write_events(frame, "twse", date(2026, 11, 2)) == 2

    kept = {row.stock_id: row for row in DividendEvent.objects.all()}
    assert (kept["6655"].cash_dividend, kept["6655"].stock_dividend) == (
        Decimal("2.65"),
        Decimal("0.5"),
    )
    assert kept["6655"].adj_factor == pytest.approx(0.97)
    assert kept["6655"].announce_date == EVE
    assert kept["2330"].cash_dividend == Decimal("70.5")
    assert kept["2330"].adj_factor == pytest.approx(0.98)
    log = DataQualityLog.objects.get(check_name=MISMATCH_CHECK)
    assert (log.date, log.severity) == (date(2026, 11, 2), DataQualityLog.Severity.INFO)
    assert log.detail == {
        "source": "twse",
        "events": [
            {
                "stock_id": "2330",
                "ex_date": "2026-11-03",
                "fields": {"cash_dividend": ["70", "70.5"], "adj_factor": [0.99, 0.98]},
                "overwritten": True,
            },
            {
                "stock_id": "6655",
                "ex_date": "2026-11-03",
                "fields": {"cash_dividend": ["2.6", "2.65"], "stock_dividend": ["0", "0.5"]},
                "overwritten": True,
            },
        ],
    }


@pytest.mark.django_db
def test_a_settled_event_keeps_its_overwrite_on_record_when_the_day_runs_again() -> None:
    _stored_from_a_notice()
    frame = _events(Action("6655", LED, Decimal("2.65"), Decimal("0"), 0.97, "ex_rights"))
    write_events(frame, "twse", date(2026, 11, 2))

    assert write_events(frame, "twse", date(2026, 11, 2)) == 0

    log = DataQualityLog.objects.get(check_name=MISMATCH_CHECK)
    assert [(event["stock_id"], event["overwritten"]) for event in log.detail["events"]] == [
        ("6655", True)
    ]


@pytest.mark.django_db
def test_a_results_row_without_a_factor_settles_the_amounts_and_keeps_the_stored_one() -> None:
    _stored_from_a_notice()
    frame = _events(Action("2330", LED, Decimal("70.5"), Decimal("0"), None, "ex_rights"))

    assert write_events(frame, "twse", date(2026, 11, 2)) == 1

    event = DividendEvent.objects.get(stock_id="2330")
    assert event.cash_dividend == Decimal("70.5")
    assert event.adj_factor == pytest.approx(0.99)


@pytest.mark.parametrize(("leads_from", "cash"), [("2026-11-03", "70.5"), ("2026-11-04", "70")])
@pytest.mark.django_db
def test_the_first_day_the_exchanges_lead_is_a_setting(
    settings: Any, leads_from: str, cash: str
) -> None:
    settings.DIVIDEND_EXCHANGE_PRIMARY_FROM = leads_from
    _stored_from_a_notice()
    frame = _events(Action("2330", LED, Decimal("70.5"), Decimal("0"), 0.98, "ex_rights"))

    write_events(frame, "twse", date(2026, 11, 2))

    assert DividendEvent.objects.get(stock_id="2330").cash_dividend == Decimal(cash)


@pytest.mark.django_db
def test_with_settling_switched_off_the_results_table_only_logs_as_before(settings: Any) -> None:
    settings.DIVIDEND_RESULTS_SETTLE = False
    _stored_from_a_notice()
    frame = _events(
        Action("6655", LED, Decimal("2.65"), Decimal("0"), 0.97, "ex_rights"),
        Action("2330", LED, Decimal("70.5"), Decimal("0"), 0.98, "ex_rights"),
    )

    assert write_events(frame, "twse", date(2026, 11, 2)) == 1

    kept = {row.stock_id: row for row in DividendEvent.objects.all()}
    assert kept["6655"].cash_dividend == Decimal("2.6")
    assert kept["6655"].adj_factor == pytest.approx(0.97)
    assert kept["2330"].cash_dividend == Decimal("70")
    assert kept["2330"].adj_factor == pytest.approx(0.99)
    events = DataQualityLog.objects.get(check_name=MISMATCH_CHECK).detail["events"]
    assert [(event["stock_id"], "overwritten" in event) for event in events] == [
        ("2330", False),
        ("6655", False),
    ]
