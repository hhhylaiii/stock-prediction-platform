import re
import threading
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
    require_key,
)
from apps.research.models import HoldoutRun
from apps.research.walkforward import Fold, generate_folds

LAST = date(2026, 10, 30)


@pytest.fixture(scope="module")
def folds(long_calendar: TradingDays) -> list[Fold]:
    return generate_folds(long_calendar, LAST)


@pytest.mark.django_db
def test_first_claim_records_the_scope_and_returns_a_ticket() -> None:
    ticket = claim_holdout("rank", "rank-v1")

    assert ticket == HoldoutTicket("rank", "rank-v1", 1)
    row = HoldoutRun.objects.get()
    assert (row.scope, row.attempt, row.run_key, row.reason) == ("rank", 1, "rank-v1", "")
    assert str(row) == "rank #1 rank-v1"
    assert holdout_claimed("rank")
    assert not holdout_claimed("interval")


@pytest.mark.django_db
def test_second_claim_without_force_is_refused() -> None:
    claim_holdout("rank", "rank-v1")

    with pytest.raises(HoldoutError, match="already"):
        claim_holdout("rank", "rank-v2")
    assert HoldoutRun.objects.count() == 1


@pytest.mark.django_db
@pytest.mark.parametrize("reason", ["", "   "])
def test_forced_rerun_needs_a_reason(reason: str) -> None:
    claim_holdout("rank", "rank-v1")

    with pytest.raises(HoldoutError, match="reason"):
        claim_holdout("rank", "rank-v2", force=True, reason=reason)
    assert HoldoutRun.objects.count() == 1


@pytest.mark.django_db
def test_forced_rerun_with_a_reason_is_the_next_attempt() -> None:
    claim_holdout("rank", "rank-v1")

    ticket = claim_holdout("rank", "rank-v2", force=True, reason="  label bug fixed in 3.2  ")

    assert ticket == HoldoutTicket("rank", "rank-v2", 2)
    assert HoldoutRun.objects.get(attempt=2).reason == "label bug fixed in 3.2"


@pytest.mark.django_db
def test_scopes_are_claimed_independently() -> None:
    claim_holdout("rank", "rank-v1")

    assert claim_holdout("interval", "interval-v1").attempt == 1


@pytest.mark.django_db
def test_database_refuses_a_rerun_without_a_reason() -> None:
    HoldoutRun.objects.create(scope="rank", attempt=1, run_key="rank-v1")

    with pytest.raises(IntegrityError), transaction.atomic():
        HoldoutRun.objects.create(scope="rank", attempt=2, run_key="rank-v2", reason="  ")


@pytest.mark.django_db(transaction=True)
def test_two_first_claims_at_once_let_only_one_through() -> None:
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def claim(run_key: str) -> None:
        barrier.wait()
        try:
            claim_holdout("rank", run_key)
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
    ticket = claim_holdout("rank", "rank-v1")

    authorize_folds(folds, "rank-v1", ticket)


@pytest.mark.django_db
def test_ticket_for_another_run_key_is_refused(folds: list[Fold]) -> None:
    ticket = claim_holdout("rank", "rank-v1")

    with pytest.raises(HoldoutError, match="rank-v2"):
        authorize_folds(folds, "rank-v2", ticket)


@pytest.mark.django_db
def test_ticket_without_a_recorded_claim_is_refused(folds: list[Fold]) -> None:
    claim_holdout("rank", "rank-v1")

    with pytest.raises(HoldoutError, match="holdout_run"):
        authorize_folds(folds, "rank-v1", HoldoutTicket("rank", "rank-v1", 2))


@pytest.mark.parametrize(
    "key",
    ["", "../etc", "a/b", "a\\b", ".hidden", "..", "a b", "rank\n", "名稱", "x" * 65],
)
def test_keys_with_path_characters_are_refused(key: str) -> None:
    with pytest.raises(ValueError, match="letters"):
        require_key(key)


@pytest.mark.parametrize("key", ["rank-v1", "rank_20d.lgbm", "A1", "x" * 64])
def test_plain_keys_are_accepted(key: str) -> None:
    assert require_key(key) == key


@pytest.mark.django_db
@pytest.mark.parametrize(("scope", "run_key"), [("../rank", "rank-v1"), ("rank", "a/b")])
def test_claim_refuses_unsafe_names(scope: str, run_key: str) -> None:
    with pytest.raises(ValueError, match="letters"):
        claim_holdout(scope, run_key)
    assert not HoldoutRun.objects.exists()
