from datetime import date
from decimal import Decimal

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
from apps.market.models import DailyPrice, DividendEvent, StockMaster

EX = date(2026, 11, 3)


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
            Action("2330", EX, Decimal("0"), Decimal("0"), None, "par_value_change"),
        ]
    )
    events = events_from(frame).set_index("stock_id")
    assert events.loc["1101", "adj_factor"] == pytest.approx(0.45)
    assert events.loc["1101", "cash_dividend"] == Decimal("1.5")
    assert events.loc["2330", "adj_factor"] is None
    assert events["announce_date"].isna().all()


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
    _price("1101", date(2026, 11, 2))
    _price("1101", EX)
    frame = _events(
        Action("1101", EX, Decimal("1.2"), Decimal("0"), 0.88, "ex_rights"),
        Action("9999", EX, Decimal("1"), Decimal("0"), 0.9, "ex_rights"),
    )
    assert write_events(frame, "twse", date(2026, 11, 2)) == 1
    event = DividendEvent.objects.get()
    assert (event.stock_id, event.ex_date, event.announce_date) == ("1101", EX, None)
    assert event.adj_factor == pytest.approx(0.88)
    factors = dict(DailyPrice.objects.values_list("date", "adj_factor"))
    assert factors == {date(2026, 11, 2): pytest.approx(0.88), EX: 1.0}


@pytest.mark.django_db
def test_write_events_fills_a_blank_factor_and_logs_but_keeps_different_figures() -> None:
    _listed("1101", "2330")
    _price("1101", date(2026, 11, 2))
    _price("1101", EX)
    DividendEvent.objects.create(
        stock_id="1101",
        ex_date=EX,
        cash_dividend=Decimal("1.2"),
        announce_date=date(2026, 10, 20),
    )
    DividendEvent.objects.create(
        stock_id="2330", ex_date=EX, cash_dividend=Decimal("5"), adj_factor=0.99
    )
    frame = _events(
        Action("1101", EX, Decimal("1.2"), Decimal("0"), 0.88, "ex_rights"),
        Action("2330", EX, Decimal("5.5"), Decimal("0"), 0.98, "ex_rights"),
    )
    assert write_events(frame, "twse", date(2026, 11, 2)) == 1
    kept = {row.stock_id: row for row in DividendEvent.objects.all()}
    assert kept["1101"].adj_factor == pytest.approx(0.88)
    assert kept["1101"].announce_date == date(2026, 10, 20)
    factors = dict(DailyPrice.objects.values_list("date", "adj_factor"))
    assert factors == {date(2026, 11, 2): pytest.approx(0.88), EX: 1.0}
    assert kept["2330"].adj_factor == pytest.approx(0.99)
    assert kept["2330"].cash_dividend == Decimal("5")
    log = DataQualityLog.objects.get(check_name=MISMATCH_CHECK)
    assert log.date == date(2026, 11, 2)
    assert log.severity == DataQualityLog.Severity.INFO
    assert log.detail == {
        "source": "twse",
        "events": [
            {
                "stock_id": "2330",
                "ex_date": "2026-11-03",
                "fields": {"cash_dividend": ["5", "5.5"], "adj_factor": [0.99, 0.98]},
            }
        ],
    }


@pytest.mark.django_db
def test_write_events_again_writes_nothing_and_replaces_its_own_log_only() -> None:
    _listed("2330")
    DividendEvent.objects.create(stock_id="2330", ex_date=EX, adj_factor=0.99)
    DataQualityLog.objects.create(
        date=date(2026, 11, 2),
        check_name=MISMATCH_CHECK,
        severity=DataQualityLog.Severity.INFO,
        detail={"source": "tpex", "events": []},
    )
    frame = _events(Action("2330", EX, Decimal("0"), Decimal("0"), 0.98, "ex_rights"))
    write_events(frame, "twse", date(2026, 11, 2))
    assert write_events(frame, "twse", date(2026, 11, 2)) == 0
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
        date(2026, 11, 2),
    )
    write_events(
        _events(Action("2330", EX, Decimal("0"), Decimal("0"), 0.99, "ex_rights")),
        "twse",
        date(2026, 11, 2),
    )
    assert not DataQualityLog.objects.filter(check_name=MISMATCH_CHECK).exists()
