from datetime import date, timedelta
from decimal import Decimal

import pytest

from apps.etl.models import DataQualityLog
from apps.market import adjust
from apps.market.adjust import (
    ADJ_CHECK,
    note_adj_change,
    refresh_adj_factor,
    refresh_where_later_events,
)
from apps.market.models import DailyPrice, DividendEvent

pytestmark = pytest.mark.django_db
MONDAY = date(2026, 10, 19)
DAYS = [MONDAY + timedelta(days=n) for n in range(5)]


def prices(stock_id: str = "2330", days: list[date] = DAYS) -> None:
    DailyPrice.objects.bulk_create(
        DailyPrice(stock_id=stock_id, date=day, close=Decimal("100"), volume=1, turnover=100)
        for day in days
    )


def event(ex_date: date, factor: float, stock_id: str = "2330") -> None:
    DividendEvent.objects.create(stock_id=stock_id, ex_date=ex_date, adj_factor=factor)


def notes() -> list[DataQualityLog]:
    return list(DataQualityLog.objects.filter(check_name=ADJ_CHECK).order_by("pk"))


def test_an_event_inside_the_history_notes_its_ex_date() -> None:
    prices()
    refresh_adj_factor("2330")
    event(DAYS[2], 0.9)

    assert refresh_adj_factor("2330") == 2

    [note] = notes()
    assert note.severity == DataQualityLog.Severity.INFO
    assert note.detail == {"stocks": {"2330": DAYS[2].isoformat()}}


def test_an_event_after_the_last_price_rescales_every_row_alike_and_notes_nothing() -> None:
    prices()
    event(DAYS[-1] + timedelta(days=3), 0.9)

    assert refresh_adj_factor("2330") == len(DAYS)

    assert notes() == []


def test_two_events_changed_together_note_the_earlier_ex_date() -> None:
    prices()
    event(DAYS[1], 0.9)
    event(DAYS[3], 0.8)

    refresh_adj_factor("2330")

    assert notes()[0].detail == {"stocks": {"2330": DAYS[1].isoformat()}}


def test_an_unchanged_refresh_notes_nothing() -> None:
    prices()
    event(DAYS[2], 0.9)
    refresh_adj_factor("2330")
    DataQualityLog.objects.all().delete()

    assert refresh_adj_factor("2330") == 0

    assert notes() == []


def test_new_price_rows_take_their_factor_without_a_note() -> None:
    prices(days=DAYS[:4])
    event(DAYS[-1] + timedelta(days=3), 0.9)
    refresh_adj_factor("2330")
    prices(days=DAYS[4:])

    assert refresh_where_later_events(["2330"], DAYS[4]) == 1

    assert DailyPrice.objects.get(stock_id="2330", date=DAYS[4]).adj_factor == pytest.approx(0.9)
    assert notes() == []


def test_notes_of_a_day_share_one_row_keeping_each_stock_s_earliest_session() -> None:
    note_adj_change("2330", DAYS[3])
    note_adj_change("2317", DAYS[4])
    note_adj_change("2330", DAYS[1])
    note_adj_change("2330", DAYS[2])

    [note] = notes()
    assert note.detail == {"stocks": {"2330": DAYS[1].isoformat(), "2317": DAYS[4].isoformat()}}


def test_past_the_stock_limit_a_note_keeps_only_the_earliest_session_and_a_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adjust, "ADJ_STOCK_LIMIT", 2)
    note_adj_change("2330", DAYS[3])
    note_adj_change("2317", DAYS[2])
    note_adj_change("2454", DAYS[4])
    note_adj_change("2603", DAYS[0])

    [note] = notes()
    assert note.detail == {"earliest": DAYS[0].isoformat(), "count": 4}


def test_a_rebuilt_note_is_closed_and_a_later_change_opens_another() -> None:
    note_adj_change("2330", DAYS[3])
    DataQualityLog.objects.update(detail={"stocks": {"2330": DAYS[3].isoformat()}, "rebuilt": "x"})

    note_adj_change("2317", DAYS[4])

    assert [note.detail for note in notes()] == [
        {"stocks": {"2330": DAYS[3].isoformat()}, "rebuilt": "x"},
        {"stocks": {"2317": DAYS[4].isoformat()}},
    ]
