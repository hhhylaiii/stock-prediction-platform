import re
import threading
from dataclasses import replace
from datetime import date

import pytest
from django.db import IntegrityError, connection, transaction

from apps.market.calendar import TradingDays
from apps.research.holdout import (
    HoldoutError,
    HoldoutTicket,
    authorize_folds,
    claim_holdout,
    holdout_claimed,
    latest_ticket,
    require_key,
)
from apps.research.models import HoldoutRun
from apps.research.walkforward import Fold, WalkForwardConfig, generate_folds

LAST = date(2026, 10, 30)


@pytest.fixture(scope="module")
def folds(long_calendar: TradingDays) -> list[Fold]:
    return generate_folds(long_calendar, LAST)


@pytest.mark.django_db
def test_first_claim_records_the_scope_and_returns_a_ticket() -> None:
    ticket = claim_holdout("rank", "rank-v1", LAST)

    assert ticket == HoldoutTicket("rank", "rank-v1", 1, LAST)
    row = HoldoutRun.objects.get()
    assert (row.scope, row.attempt, row.run_key, row.reason) == ("rank", 1, "rank-v1", "")
    assert str(row) == "rank #1 rank-v1"
    assert holdout_claimed("rank")
    assert not holdout_claimed("interval")


@pytest.mark.django_db
def test_second_claim_without_force_is_refused() -> None:
    claim_holdout("rank", "rank-v1", LAST)

    with pytest.raises(HoldoutError, match="already"):
        claim_holdout("rank", "rank-v2", LAST)
    assert HoldoutRun.objects.count() == 1


@pytest.mark.django_db
@pytest.mark.parametrize("reason", ["", "   "])
def test_forced_rerun_needs_a_reason(reason: str) -> None:
    claim_holdout("rank", "rank-v1", LAST)

    with pytest.raises(HoldoutError, match="needs a reason"):
        claim_holdout("rank", "rank-v2", LAST, force=True, reason=reason)
    assert HoldoutRun.objects.count() == 1


@pytest.mark.django_db
def test_a_reason_without_force_is_still_refused() -> None:
    claim_holdout("rank", "rank-v1", LAST)

    with pytest.raises(HoldoutError, match="already"):
        claim_holdout("rank", "rank-v2", LAST, reason="new idea")
    assert HoldoutRun.objects.count() == 1


@pytest.mark.django_db
def test_attempts_count_up_within_each_scope() -> None:
    claim_holdout("rank", "rank-v1", LAST)
    claim_holdout("interval", "interval-v1", LAST)

    attempts = [
        claim_holdout("rank", "rank-v2", LAST, force=True, reason="label fix").attempt,
        claim_holdout("rank", "rank-v3", LAST, force=True, reason="cost fix").attempt,
        claim_holdout("interval", "interval-v2", LAST, force=True, reason="label fix").attempt,
    ]

    assert attempts == [2, 3, 2]


@pytest.mark.django_db
def test_forced_rerun_with_a_reason_is_the_next_attempt() -> None:
    claim_holdout("rank", "rank-v1", LAST)

    ticket = claim_holdout("rank", "rank-v2", LAST, force=True, reason="  label bug fixed in 3.2  ")

    assert ticket == HoldoutTicket("rank", "rank-v2", 2, LAST)
    assert HoldoutRun.objects.get(attempt=2).reason == "label bug fixed in 3.2"


@pytest.mark.django_db
def test_scopes_are_claimed_independently() -> None:
    claim_holdout("rank", "rank-v1", LAST)

    assert claim_holdout("interval", "interval-v1", LAST).attempt == 1


@pytest.mark.django_db
def test_database_refuses_attempt_zero() -> None:
    with pytest.raises(IntegrityError), transaction.atomic():
        HoldoutRun.objects.create(scope="rank", attempt=0, run_key="rank-v1", last_date=LAST)


@pytest.mark.django_db(transaction=True)
def test_claim_inside_an_outer_transaction_is_refused() -> None:
    with pytest.raises(RuntimeError), transaction.atomic():
        claim_holdout("rank", "rank-v1", LAST)
    assert not HoldoutRun.objects.exists()


@pytest.mark.django_db
def test_database_refuses_a_rerun_without_a_reason() -> None:
    HoldoutRun.objects.create(scope="rank", attempt=1, run_key="rank-v1", last_date=LAST)

    with pytest.raises(IntegrityError), transaction.atomic():
        HoldoutRun.objects.create(
            scope="rank", attempt=2, run_key="rank-v2", reason="  ", last_date=LAST
        )


@pytest.mark.django_db(transaction=True)
def test_two_first_claims_at_once_let_only_one_through() -> None:
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def claim(run_key: str) -> None:
        barrier.wait()
        try:
            claim_holdout("rank", run_key, LAST)
            outcomes.append("claimed")
        except HoldoutError:
            outcomes.append("refused")
        finally:
            connection.close()

    threads = [threading.Thread(target=claim, args=(f"rank-v{n}",)) for n in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["claimed", "refused"]
    assert HoldoutRun.objects.count() == 1


def test_folds_without_holdout_need_no_ticket(folds: list[Fold]) -> None:
    authorize_folds([fold for fold in folds if not fold.holdout], "rank-v1", None)


def test_holdout_folds_without_a_ticket_are_refused(folds: list[Fold]) -> None:
    holdout = [fold.index for fold in folds if fold.holdout]

    with pytest.raises(HoldoutError, match=re.escape(f"folds {holdout}")):
        authorize_folds(folds, "rank-v1", None)


@pytest.mark.django_db
def test_holdout_folds_run_with_a_recorded_ticket(folds: list[Fold]) -> None:
    ticket = claim_holdout("rank", "rank-v1", LAST)

    authorize_folds(folds, "rank-v1", ticket)


@pytest.mark.django_db
def test_ticket_for_another_run_key_is_refused(folds: list[Fold]) -> None:
    ticket = claim_holdout("rank", "rank-v1", LAST)

    with pytest.raises(HoldoutError, match="rank-v2"):
        authorize_folds(folds, "rank-v2", ticket)


@pytest.mark.django_db
def test_ticket_of_another_scope_is_refused(folds: list[Fold]) -> None:
    claim_holdout("rank", "shared-v1", LAST)

    with pytest.raises(HoldoutError, match="holdout_run"):
        authorize_folds(folds, "shared-v1", HoldoutTicket("interval", "shared-v1", 1, LAST))


@pytest.mark.django_db
def test_ticket_with_a_later_last_date_than_recorded_is_refused(folds: list[Fold]) -> None:
    ticket = claim_holdout("rank", "rank-v1", date(2025, 6, 30))

    with pytest.raises(HoldoutError, match="holdout_run"):
        authorize_folds(folds, "rank-v1", replace(ticket, last_date=LAST))


@pytest.mark.django_db
def test_superseded_ticket_is_refused(folds: list[Fold]) -> None:
    first = claim_holdout("rank", "rank-v1", LAST)
    claim_holdout("rank", "rank-v2", LAST, force=True, reason="label fix")

    with pytest.raises(HoldoutError, match="superseded"):
        authorize_folds(folds, "rank-v1", first)


def test_folds_after_the_spec_holdout_start_stay_holdout(long_calendar: TradingDays) -> None:
    moved = generate_folds(long_calendar, LAST, WalkForwardConfig(holdout_start=date(2099, 1, 1)))
    late = [fold for fold in moved if fold.test_end >= date(2025, 1, 1)]

    assert late
    assert not any(fold.holdout for fold in late)
    with pytest.raises(HoldoutError, match="are holdout"):
        authorize_folds(late, "rank-v1", None)


@pytest.mark.django_db
def test_ticket_without_a_recorded_claim_is_refused(folds: list[Fold]) -> None:
    claim_holdout("rank", "rank-v1", LAST)

    with pytest.raises(HoldoutError, match="holdout_run"):
        authorize_folds(folds, "rank-v1", HoldoutTicket("rank", "rank-v1", 2, LAST))


@pytest.mark.parametrize(
    "key",
    [
        "",
        "../etc",
        "a/b",
        "a\\b",
        ".hidden",
        "..",
        "a b",
        "rank\n",
        "名稱",
        "x" * 65,
        "Rank-v1",
        "A1",
    ],
)
def test_keys_with_path_characters_are_refused(key: str) -> None:
    with pytest.raises(ValueError, match="letters"):
        require_key(key)


@pytest.mark.parametrize("key", ["rank-v1", "rank_20d.lgbm", "a1", "x" * 64])
def test_plain_keys_are_accepted(key: str) -> None:
    assert require_key(key) == key


@pytest.mark.django_db
@pytest.mark.parametrize("scope", ["Rank", "rank_20d", "../rank", ""])
def test_claim_refuses_unknown_scopes(scope: str) -> None:
    with pytest.raises(ValueError, match="scope"):
        claim_holdout(scope, "rank-v1", LAST)
    assert not HoldoutRun.objects.exists()


@pytest.mark.django_db
def test_claim_refuses_an_unsafe_run_key() -> None:
    with pytest.raises(ValueError, match="letters"):
        claim_holdout("rank", "a/b", LAST)
    assert not HoldoutRun.objects.exists()


@pytest.mark.django_db
def test_rerun_must_use_a_new_run_key() -> None:
    claim_holdout("rank", "rank-v1", LAST)

    with pytest.raises(HoldoutError, match="new run_key"):
        claim_holdout("rank", "rank-v1", LAST, force=True, reason="label fix")
    assert HoldoutRun.objects.count() == 1


@pytest.mark.django_db
@pytest.mark.parametrize(
    "row",
    [
        {"scope": "other", "attempt": 1, "run_key": "other-v1"},
        {"scope": "interval", "attempt": 1, "run_key": "rank-v1"},
    ],
)
def test_database_refuses_unknown_scopes_and_shared_run_keys(row: dict[str, object]) -> None:
    HoldoutRun.objects.create(scope="rank", attempt=1, run_key="rank-v1", last_date=LAST)

    with pytest.raises(IntegrityError), transaction.atomic():
        HoldoutRun.objects.create(**row, last_date=LAST)


@pytest.mark.django_db
def test_latest_ticket_resumes_the_newest_claim(folds: list[Fold]) -> None:
    assert latest_ticket("rank") is None
    claim_holdout("rank", "rank-v1", LAST)
    claim_holdout("rank", "rank-v2", LAST, force=True, reason="label fix")

    ticket = latest_ticket("rank")

    assert ticket == HoldoutTicket("rank", "rank-v2", 2, LAST)
    authorize_folds(folds, "rank-v2", ticket)


@pytest.mark.django_db
def test_holdout_folds_past_the_claimed_last_date_are_refused(
    long_calendar: TradingDays, folds: list[Fold]
) -> None:
    cutoff = date(2025, 6, 30)
    ticket = claim_holdout("rank", "rank-v1", cutoff)

    authorize_folds(generate_folds(long_calendar, cutoff), "rank-v1", ticket)
    with pytest.raises(HoldoutError, match="past the claimed last_date"):
        authorize_folds(folds, "rank-v1", ticket)
