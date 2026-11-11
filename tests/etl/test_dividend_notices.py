from datetime import date
from decimal import Decimal

import pytest

from apps.etl.adapters.exchange_notices import Notice
from apps.etl.dividend_notices import NOTICE_CHECK, apply_notices
from apps.etl.models import DataQualityLog, DividendNotice
from apps.market.models import DividendEvent, StockMaster

pytestmark = pytest.mark.django_db

MON, TUE, WED, THU = (date(2026, 11, day) for day in (9, 10, 11, 12))
EX = date(2026, 11, 24)
ZERO = Decimal(0)


def _listed(*codes: str) -> None:
    StockMaster.objects.bulk_create(
        StockMaster(stock_id=code, name=code, market="TWSE") for code in codes
    )


def _cash(code: str, amount: str = "2.6", ex_date: date = EX) -> Notice:
    return Notice(code, ex_date, Decimal(amount), ZERO)


def _many(count: int) -> list[Notice]:
    return [_cash(f"{1101 + index}") for index in range(count)]


def _event(code: str, ex_date: date = EX) -> DividendEvent:
    return DividendEvent.objects.get(stock_id=code, ex_date=ex_date)


def _log(session: date, source: str = "twse") -> DataQualityLog:
    return DataQualityLog.objects.get(check_name=NOTICE_CHECK, date=session, detail__source=source)


def test_a_new_event_is_dated_by_the_session_that_first_saw_it() -> None:
    _listed("6655")

    assert apply_notices("twse", [_cash("6655")], TUE) == 1
    apply_notices("twse", [_cash("6655")], WED)

    event = _event("6655")
    assert (event.cash_dividend, event.stock_dividend) == (Decimal("2.6"), ZERO)
    assert (event.adj_factor, event.announce_date) == (None, TUE)
    sighting = DividendNotice.objects.get(source="twse", stock_id="6655", ex_date=EX)
    assert (sighting.first_seen, sighting.last_seen, sighting.created_event) == (TUE, WED, True)


@pytest.mark.parametrize("ex_date", [TUE, MON])
def test_an_ex_date_that_has_come_is_left_alone(ex_date: date) -> None:
    _listed("6655", "2344")

    apply_notices("twse", [_cash("6655", ex_date=ex_date), _cash("2344")], TUE)

    assert list(DividendEvent.objects.values_list("stock_id", flat=True)) == ["2344"]
    assert not DividendNotice.objects.filter(stock_id="6655").exists()


def test_a_code_outside_the_master_is_seen_but_gets_no_event() -> None:
    _listed("2344")

    assert apply_notices("twse", [_cash("00940"), _cash("2344")], TUE) == 1

    assert list(DividendEvent.objects.values_list("stock_id", flat=True)) == ["2344"]
    assert DividendNotice.objects.filter(stock_id="00940", created_event=False).exists()


def test_a_code_that_joins_the_master_later_keeps_its_first_session() -> None:
    apply_notices("twse", [_cash("7799")], TUE)
    _listed("7799")

    apply_notices("twse", [_cash("7799")], WED)

    assert _event("7799").announce_date == TUE


def test_a_stored_event_only_has_its_blank_date_filled() -> None:
    _listed("6655", "2344")
    DividendEvent.objects.create(
        stock_id="6655", ex_date=EX, cash_dividend=Decimal("2.5"), adj_factor=0.97
    )
    DividendEvent.objects.create(
        stock_id="2344", ex_date=EX, cash_dividend=Decimal("2"), announce_date=MON
    )

    assert apply_notices("twse", [_cash("6655"), _cash("2344", "3")], TUE) == 1

    assert (_event("6655").announce_date, _event("6655").cash_dividend) == (TUE, Decimal("2.5"))
    assert (_event("2344").announce_date, _event("2344").cash_dividend) == (MON, Decimal("2"))
    assert _log(TUE).detail["dated"] == 1


def test_an_amount_announced_later_reaches_the_event_this_table_made() -> None:
    _listed("0051")
    apply_notices("twse", [_cash("0051", "0")], TUE)

    assert apply_notices("twse", [_cash("0051", "1.2")], WED) == 1

    event = _event("0051")
    assert (event.cash_dividend, event.announce_date) == (Decimal("1.2"), TUE)
    assert _log(WED).detail["revised"] == 1


def test_a_settled_event_no_longer_follows_the_notice() -> None:
    _listed("0051")
    apply_notices("twse", [_cash("0051", "1.2")], TUE)
    DividendEvent.objects.filter(stock_id="0051").update(adj_factor=0.99)

    assert apply_notices("twse", [_cash("0051", "1.5")], WED) == 0

    assert _event("0051").cash_dividend == Decimal("1.2")


def test_reading_the_same_list_again_changes_nothing() -> None:
    _listed("6655")
    apply_notices("twse", [_cash("6655")], TUE)

    assert apply_notices("twse", [_cash("6655")], TUE) == 0

    assert DividendEvent.objects.count() == 1
    assert DataQualityLog.objects.filter(check_name=NOTICE_CHECK).count() == 1


def test_an_event_missing_once_stays() -> None:
    _listed("6655", "2344")
    apply_notices("twse", [_cash("6655"), _cash("2344")], MON)

    apply_notices("twse", [_cash("2344")], TUE)

    assert _event("6655").announce_date == MON
    log = _log(TUE)
    assert (log.severity, log.detail["gone"], log.detail["removed"]) == (
        "info",
        ["6655@2026-11-24"],
        [],
    )


def test_an_event_two_sound_lists_in_a_row_lack_is_removed_and_logged() -> None:
    _listed("6655", "2344")
    apply_notices("twse", [_cash("6655"), _cash("2344")], MON)
    apply_notices("twse", [_cash("2344")], TUE)

    assert apply_notices("twse", [_cash("2344")], WED) == 1

    assert list(DividendEvent.objects.values_list("stock_id", flat=True)) == ["2344"]
    log = _log(WED)
    assert log.severity == "warning"
    assert log.detail["removed"] == [
        {
            "stock_id": "6655",
            "ex_date": "2026-11-24",
            "cash_dividend": "2.60000000",
            "stock_dividend": "0.00000000",
            "announce_date": "2026-11-09",
            "first_seen": "2026-11-09",
            "last_seen": "2026-11-09",
            "removed_on": "2026-11-11",
        }
    ]


def test_a_removal_stays_in_the_log_when_the_list_is_read_again() -> None:
    _listed("6655", "2344")
    apply_notices("twse", [_cash("6655"), _cash("2344")], MON)
    apply_notices("twse", [_cash("2344")], TUE)
    apply_notices("twse", [_cash("2344")], WED)

    assert apply_notices("twse", [_cash("2344")], WED) == 0

    assert [row["stock_id"] for row in _log(WED).detail["removed"]] == ["6655"]


def test_a_moved_ex_date_is_dated_by_the_day_it_moved() -> None:
    _listed("6655", "2344")
    moved = date(2026, 11, 25)
    apply_notices("twse", [_cash("6655"), _cash("2344")], MON)
    apply_notices("twse", [_cash("6655", ex_date=moved), _cash("2344")], TUE)

    apply_notices("twse", [_cash("6655", ex_date=moved), _cash("2344")], WED)

    assert dict(
        DividendEvent.objects.filter(stock_id="6655").values_list("ex_date", "announce_date")
    ) == {moved: TUE}


def test_a_removed_event_that_returns_keeps_its_first_session() -> None:
    _listed("6655", "2344")
    apply_notices("twse", [_cash("6655"), _cash("2344")], MON)
    apply_notices("twse", [_cash("2344")], TUE)
    apply_notices("twse", [_cash("2344")], WED)

    apply_notices("twse", [_cash("6655"), _cash("2344")], THU)

    assert _event("6655").announce_date == MON


def test_a_list_that_lost_too_many_events_removes_nothing_and_warns() -> None:
    notices = _many(12)
    _listed(*(notice.stock_id for notice in notices))
    apply_notices("twse", notices, MON)

    apply_notices("twse", notices[:6], TUE)
    apply_notices("twse", notices[:6], WED)

    assert DividendEvent.objects.count() == 12
    log = _log(TUE)
    assert (log.severity, log.detail["sound"], len(log.detail["gone"])) == ("warning", False, 6)
    assert _log(WED).detail["removed"] == []


def test_a_broken_list_removes_nothing_even_after_a_sound_one() -> None:
    notices = _many(12)
    _listed("6655", *(notice.stock_id for notice in notices))
    apply_notices("twse", [_cash("6655"), *notices], MON)
    apply_notices("twse", notices, TUE)

    apply_notices("twse", notices[:6], WED)

    assert _event("6655").announce_date == MON
    assert (_log(WED).detail["sound"], _log(WED).detail["removed"]) == (False, [])


def test_the_list_after_a_broken_one_must_be_sound_twice_before_anything_goes() -> None:
    notices = _many(12)
    _listed(*(notice.stock_id for notice in notices))
    apply_notices("twse", notices, MON)
    apply_notices("twse", notices[:6], TUE)
    apply_notices("twse", notices[:6], WED)

    apply_notices("twse", notices[:6], THU)

    assert DividendEvent.objects.count() == 6
    assert len(_log(THU).detail["removed"]) == 6


def test_up_to_five_events_may_leave_a_short_list_at_once() -> None:
    notices = _many(7)
    _listed(*(notice.stock_id for notice in notices))
    apply_notices("twse", notices, MON)

    apply_notices("twse", notices[:2], TUE)
    apply_notices("twse", notices[:2], WED)

    assert DividendEvent.objects.count() == 2
    assert _log(TUE).detail["sound"] is True


def test_a_list_with_nothing_to_come_is_not_sound() -> None:
    _listed("6655")
    apply_notices("twse", [_cash("6655")], MON)

    apply_notices("twse", [_cash("6655", ex_date=MON)], TUE)
    apply_notices("twse", [_cash("6655", ex_date=MON)], WED)

    assert _event("6655").announce_date == MON
    assert _log(WED).detail["sound"] is False


def test_only_events_this_table_made_and_left_unsettled_are_removed() -> None:
    _listed("6655", "2344", "1707")
    DividendEvent.objects.create(stock_id="6655", ex_date=EX, cash_dividend=Decimal("2.5"))
    apply_notices("twse", [_cash("6655"), _cash("2344"), _cash("1707")], MON)
    DividendEvent.objects.filter(stock_id="2344").update(adj_factor=0.98)

    apply_notices("twse", [_cash("1707")], TUE)
    apply_notices("twse", [_cash("1707")], WED)

    assert sorted(DividendEvent.objects.values_list("stock_id", flat=True)) == [
        "1707",
        "2344",
        "6655",
    ]
    assert _log(WED).detail["removed"] == []


def test_an_ex_date_arriving_before_two_lists_agree_is_reported_unsettled() -> None:
    _listed("6655", "2344")
    apply_notices("twse", [_cash("6655", ex_date=WED), _cash("2344")], MON)
    apply_notices("twse", [_cash("2344")], TUE)

    apply_notices("twse", [_cash("2344")], WED)

    assert _event("6655", WED).adj_factor is None
    log = _log(WED)
    assert (log.severity, log.detail["unsettled"]) == ("warning", ["6655@2026-11-11"])


def test_an_event_the_other_exchange_now_lists_is_not_removed() -> None:
    _listed("6655", "2344")
    apply_notices("twse", [_cash("6655"), _cash("2344")], MON)
    for session in (TUE, WED):
        apply_notices("tpex", [_cash("6655")], session)
        apply_notices("twse", [_cash("2344")], session)

    assert _event("6655").announce_date == MON
    assert _log(WED).detail["removed"] == []


def test_each_exchange_keeps_its_own_lists() -> None:
    _listed("6655", "6488")
    apply_notices("twse", [_cash("6655")], MON)
    apply_notices("tpex", [_cash("6488")], MON)

    apply_notices("twse", [_cash("6655")], TUE)
    apply_notices("twse", [_cash("6655")], WED)

    assert DividendEvent.objects.count() == 2
    assert _log(MON, "tpex").detail["listed"] == 1
