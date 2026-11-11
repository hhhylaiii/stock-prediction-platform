from datetime import date
from decimal import Decimal

import pytest

from apps.etl.adapters.corporate_actions import Action, action_frame, events_from, write_events
from apps.etl.adapters.exchange_notices import Notice
from apps.etl.dividend_notices import MAX_GONE, NOTICE_CHECK, apply_notices
from apps.etl.models import DataQualityLog, DividendNotice
from apps.market.adjust import ADJ_CHECK, refresh_adj_factor
from apps.market.models import DailyPrice, DividendEvent, StockMaster

pytestmark = pytest.mark.django_db

MON, TUE, WED, THU, FRI = (date(2026, 11, day) for day in (9, 10, 11, 12, 13))
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


def _written(source: str, notices: list[Notice], session: date) -> int:
    return apply_notices(source, notices, session).rows


def test_a_new_event_is_dated_by_the_session_that_first_saw_it() -> None:
    _listed("6655")

    assert _written("twse", [_cash("6655")], TUE) == 1
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

    assert _written("twse", [_cash("00940"), _cash("2344")], TUE) == 1

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

    outcome = apply_notices("twse", [_cash("6655"), _cash("2344", "3")], TUE)

    assert (_event("6655").announce_date, _event("6655").cash_dividend) == (TUE, Decimal("2.5"))
    assert (_event("2344").announce_date, _event("2344").cash_dividend) == (MON, Decimal("2"))
    assert (outcome.rows, _log(TUE).detail["dated"]) == (1, 1)
    assert [row["stock_id"] for row in outcome.dated] == ["6655"]


def test_an_amount_announced_later_reaches_the_event_this_table_made() -> None:
    _listed("0051")
    apply_notices("twse", [_cash("0051", "0")], TUE)

    outcome = apply_notices("twse", [_cash("0051", "1.2")], WED)

    event = _event("0051")
    assert (event.cash_dividend, event.announce_date) == (Decimal("1.2"), TUE)
    assert (outcome.rows, _log(WED).detail["revised"]) == (1, 1)
    assert outcome.revised[0]["was"] == ["0.00000000", "0.00000000"]


def test_a_settled_event_no_longer_follows_the_notice() -> None:
    _listed("0051")
    apply_notices("twse", [_cash("0051", "1.2")], TUE)
    DividendEvent.objects.filter(stock_id="0051").update(adj_factor=0.99)

    assert _written("twse", [_cash("0051", "1.5")], WED) == 0

    assert _event("0051").cash_dividend == Decimal("1.2")


def test_reading_the_same_list_again_changes_nothing() -> None:
    _listed("6655")
    apply_notices("twse", [_cash("6655")], TUE)

    assert _written("twse", [_cash("6655")], TUE) == 0

    assert DividendEvent.objects.count() == 1
    assert DataQualityLog.objects.filter(check_name=NOTICE_CHECK).count() == 1


def test_an_older_list_read_later_only_widens_what_was_seen() -> None:
    _listed("6655")
    apply_notices("twse", [_cash("6655")], WED)

    apply_notices("twse", [_cash("6655")], MON)

    sighting = DividendNotice.objects.get(stock_id="6655")
    assert (sighting.first_seen, sighting.last_seen) == (MON, WED)


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

    assert _written("twse", [_cash("2344")], WED) == 1

    assert list(DividendEvent.objects.values_list("stock_id", flat=True)) == ["2344"]
    assert DividendNotice.objects.get(stock_id="6655").withdrawn == WED
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


def test_a_removal_is_logged_once_however_often_the_list_is_read() -> None:
    _listed("6655", "2344")
    apply_notices("twse", [_cash("6655"), _cash("2344")], MON)
    apply_notices("twse", [_cash("2344")], TUE)
    apply_notices("twse", [_cash("2344")], WED)

    assert _written("twse", [_cash("2344")], WED) == 0

    assert [row["stock_id"] for row in _log(WED).detail["removed"]] == ["6655"]


def test_an_event_put_back_by_hand_is_not_removed_again() -> None:
    _listed("6655", "2344")
    apply_notices("twse", [_cash("6655"), _cash("2344")], MON)
    apply_notices("twse", [_cash("2344")], TUE)
    apply_notices("twse", [_cash("2344")], WED)
    DividendEvent.objects.create(
        stock_id="6655", ex_date=EX, cash_dividend=Decimal("2.6"), announce_date=MON
    )

    for session in (THU, FRI):
        assert _written("twse", [_cash("2344")], session) == 0

    assert _event("6655").announce_date == MON
    assert (_log(FRI).severity, _log(FRI).detail["gone"]) == ("info", [])


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
    assert DividendNotice.objects.get(stock_id="6655").withdrawn is None


def test_a_removed_event_listed_again_the_same_session_comes_back() -> None:
    _listed("6655", "2344")
    apply_notices("twse", [_cash("6655"), _cash("2344")], MON)
    apply_notices("twse", [_cash("2344")], TUE)
    apply_notices("twse", [_cash("2344")], WED)

    apply_notices("twse", [_cash("6655"), _cash("2344")], WED)

    assert _event("6655").announce_date == MON
    assert DividendNotice.objects.get(stock_id="6655").withdrawn is None


def test_an_old_list_read_again_does_not_miss_events_first_listed_after_it() -> None:
    later = _many(4)
    _listed("6655", *(notice.stock_id for notice in later))
    apply_notices("twse", [_cash("6655")], MON)
    apply_notices("twse", [_cash("6655"), *later], TUE)

    assert apply_notices("twse", [_cash("6655")], MON, WED).rows == 0

    assert (_log(MON).detail["sound"], _log(MON).detail["gone"]) == (True, [])


def test_older_lists_read_again_bring_no_removed_event_back() -> None:
    _listed("6655", "2344")
    lists = [(MON, [_cash("6655"), _cash("2344")]), (TUE, [_cash("2344")]), (WED, [_cash("2344")])]
    for session, notices in lists:
        apply_notices("twse", notices, session)

    rows = [apply_notices("twse", notices, session, THU).rows for session, notices in lists]

    assert rows == [0, 0, 0]
    assert list(DividendEvent.objects.values_list("stock_id", flat=True)) == ["2344"]
    assert len(_log(WED).detail["removed"]) == 1


def test_an_old_list_read_again_makes_no_event_whose_ex_date_has_come() -> None:
    _listed("6655", "2344")
    apply_notices("twse", [_cash("6655", ex_date=WED), _cash("2344")], MON)
    DividendEvent.objects.filter(stock_id="6655").delete()

    assert apply_notices("twse", [_cash("6655", ex_date=WED), _cash("2344")], MON, THU).rows == 0

    assert list(DividendEvent.objects.values_list("stock_id", flat=True)) == ["2344"]


@pytest.mark.parametrize(("lost", "sound", "left"), [(MAX_GONE, True, 7 - MAX_GONE), (4, False, 7)])
def test_no_more_than_three_events_may_leave_a_list_at_once(
    lost: int, sound: bool, left: int
) -> None:
    notices = _many(7)
    _listed(*(notice.stock_id for notice in notices))
    apply_notices("twse", notices, MON)

    apply_notices("twse", notices[lost:], TUE)
    apply_notices("twse", notices[lost:], WED)

    assert DividendEvent.objects.count() == left
    assert (_log(TUE).detail["sound"], len(_log(TUE).detail["gone"])) == (sound, lost)
    assert _log(TUE).severity == ("info" if sound else "warning")


def test_a_list_cut_short_night_after_night_never_removes_anything() -> None:
    notices = _many(12)
    _listed(*(notice.stock_id for notice in notices))
    apply_notices("twse", notices, MON)

    for session in (TUE, WED, THU, FRI):
        apply_notices("twse", notices[:6], session)

    assert DividendEvent.objects.count() == 12
    assert [_log(day).detail["sound"] for day in (TUE, WED, THU, FRI)] == [False] * 4
    assert len(_log(FRI).detail["gone"]) == 6


def test_a_broken_list_removes_nothing_even_after_a_sound_one() -> None:
    notices = _many(12)
    _listed("6655", *(notice.stock_id for notice in notices))
    apply_notices("twse", [_cash("6655"), *notices], MON)
    apply_notices("twse", notices, TUE)

    apply_notices("twse", notices[:6], WED)

    assert _event("6655").announce_date == MON
    assert (_log(WED).detail["sound"], _log(WED).detail["removed"]) == (False, [])


def test_after_a_broken_list_two_sound_ones_are_needed_again() -> None:
    notices = _many(12)
    _listed("6655", *(notice.stock_id for notice in notices))
    apply_notices("twse", [_cash("6655"), *notices], MON)
    apply_notices("twse", notices[:6], TUE)

    apply_notices("twse", notices, WED)
    assert (_log(WED).detail["sound"], _event("6655").announce_date) == (True, MON)
    apply_notices("twse", notices, THU)

    assert not DividendEvent.objects.filter(stock_id="6655").exists()
    assert DividendEvent.objects.count() == 12


def test_events_whose_ex_date_arrives_are_not_missed_from_the_list() -> None:
    arriving = [_cash(f"{2201 + index}", ex_date=WED) for index in range(5)]
    later = [_cash("6655"), _cash("2344")]
    _listed(*(notice.stock_id for notice in [*arriving, *later]))
    apply_notices("twse", [*arriving, *later], MON)
    apply_notices("twse", [*arriving, *later], TUE)

    apply_notices("twse", later, WED)

    assert (_log(WED).detail["sound"], _log(WED).detail["gone"]) == (True, [])


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


@pytest.mark.parametrize(
    ("session", "unsettled"),
    [(date(2026, 11, 17), ["6655@2026-11-10"]), (date(2026, 11, 18), [])],
)
def test_an_unsettled_event_is_reported_for_a_week(session: date, unsettled: list[str]) -> None:
    _listed("6655", "2344")
    apply_notices("twse", [_cash("6655", ex_date=TUE), _cash("2344")], MON)

    apply_notices("twse", [_cash("2344")], session)

    assert _log(session).detail["unsettled"] == unsettled


def test_a_list_read_after_the_cutoff_does_not_call_tomorrows_ex_date_unsettled() -> None:
    _listed("6655", "2344")
    apply_notices("twse", [_cash("6655", ex_date=WED), _cash("2344")], MON)

    apply_notices("twse", [_cash("2344")], WED, TUE)

    assert _log(WED).detail["unsettled"] == []


def test_an_event_the_other_exchange_now_lists_is_not_removed() -> None:
    _listed("6655", "2344")
    apply_notices("twse", [_cash("6655"), _cash("2344")], MON)
    for session in (TUE, WED):
        apply_notices("tpex", [_cash("6655")], session)
        apply_notices("twse", [_cash("2344")], session)

    assert _event("6655").announce_date == MON
    assert _log(WED).detail["removed"] == []


def test_an_event_the_other_exchange_listed_only_before_is_removed() -> None:
    _listed("6655", "2344")
    apply_notices("twse", [_cash("6655"), _cash("2344")], MON)
    apply_notices("tpex", [_cash("6655")], MON)
    apply_notices("twse", [_cash("2344")], TUE)

    apply_notices("twse", [_cash("2344")], WED)

    assert not DividendEvent.objects.filter(stock_id="6655").exists()


def test_an_event_that_returns_on_the_other_exchange_keeps_its_first_session() -> None:
    _listed("6655", "2344")
    apply_notices("twse", [_cash("6655"), _cash("2344")], MON)
    apply_notices("twse", [_cash("2344")], TUE)
    apply_notices("twse", [_cash("2344")], WED)

    apply_notices("tpex", [_cash("6655")], THU)

    assert _event("6655").announce_date == MON


def test_each_exchange_keeps_its_own_lists() -> None:
    _listed("6655", "6488")
    apply_notices("twse", [_cash("6655")], MON)
    apply_notices("tpex", [_cash("6488")], MON)

    apply_notices("twse", [_cash("6655")], TUE)
    apply_notices("twse", [_cash("6655")], WED)

    assert DividendEvent.objects.count() == 2
    assert _log(MON, "tpex").detail["listed"] == 1


def test_a_preview_alerts_nobody(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[object] = []
    monkeypatch.setattr("apps.etl.dividend_notices.dispatch", sent.append)
    _listed("6655")

    apply_notices("twse", [_cash("6655")], TUE, quiet=True)
    assert sent == []
    apply_notices("twse", [_cash("6655")], WED)

    assert len(sent) == 1


def _prices(code: str, *days: date) -> None:
    DailyPrice.objects.bulk_create(
        DailyPrice(
            stock_id=code, date=day, open=50, high=50, low=50, close=50, volume=1, turnover=50
        )
        for day in days
    )


def _factors(code: str) -> dict[date, float]:
    return dict(DailyPrice.objects.filter(stock_id=code).values_list("date", "adj_factor"))


def test_an_event_from_a_notice_leaves_every_adjusted_price_as_it_was() -> None:
    _listed("6655")
    _prices("6655", MON, TUE, WED)
    before = _factors("6655")

    apply_notices("twse", [_cash("6655")], TUE)

    assert _event("6655").adj_factor is None
    assert refresh_adj_factor("6655") == 0
    assert _factors("6655") == before == {MON: 1.0, TUE: 1.0, WED: 1.0}
    assert not DataQualityLog.objects.filter(check_name=ADJ_CHECK).exists()


@pytest.mark.parametrize("noticed", ["2.6", "2.5", "0"])
def test_the_results_table_ends_the_same_with_or_without_a_notice_first(noticed: str) -> None:
    _listed("6655", "2344")
    for code in ("6655", "2344"):
        _prices(code, MON, TUE, EX)
    apply_notices("twse", [_cash("6655", noticed)], TUE)
    results = events_from(
        action_frame(
            Action(code, EX, Decimal("2.6"), Decimal("0.5"), 0.95, "ex_rights")
            for code in ("6655", "2344")
        )
    )

    write_events(results, "twse", EX)

    first, alone = _event("6655"), _event("2344")
    assert (first.cash_dividend, first.stock_dividend, first.adj_factor) == (
        alone.cash_dividend,
        alone.stock_dividend,
        alone.adj_factor,
    )
    assert _factors("6655") == _factors("2344")
    assert _factors("6655") == {MON: pytest.approx(0.95), TUE: pytest.approx(0.95), EX: 1.0}
    assert (first.announce_date, alone.announce_date) == (TUE, None)
