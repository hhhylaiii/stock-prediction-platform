"""Holdout guard: a scope runs its holdout folds once unless forced with a reason (SPEC §7)."""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from django.db import IntegrityError, transaction
from django.db.models import Max

from apps.research.models import HoldoutRun
from apps.research.walkforward import Fold, WalkForwardConfig

KEY_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}", re.ASCII)
SPEC_HOLDOUT_START = WalkForwardConfig().holdout_start


class HoldoutError(RuntimeError):
    """A holdout fold was about to run without a recorded claim, or a claim was refused."""


@dataclass(frozen=True)
class HoldoutTicket:
    scope: str
    run_key: str
    attempt: int


def require_key(value: str) -> str:
    """Return value if it is a plain name that is safe as one path segment."""
    if not KEY_PATTERN.fullmatch(value):
        raise ValueError(
            f"{value!r} must be 1 to 64 lowercase ASCII letters, digits, '_', '.' or '-', "
            "starting with a letter or digit"
        )
    return value


def _last_attempt(scope: str) -> int:
    last = HoldoutRun.objects.filter(scope=scope).aggregate(last=Max("attempt"))["last"]
    return int(last or 0)


def claim_holdout(
    scope: str, run_key: str, *, force: bool = False, reason: str = ""
) -> HoldoutTicket:
    """Record a holdout run for scope; any later run needs force and a reason."""
    require_key(scope)
    require_key(run_key)
    note = reason.strip()
    if force and not note:
        raise HoldoutError("a forced holdout rerun needs a reason")
    attempt = _last_attempt(scope) + 1 if force else 1
    try:
        with transaction.atomic(durable=True):
            HoldoutRun.objects.create(scope=scope, attempt=attempt, run_key=run_key, reason=note)
    except IntegrityError as error:
        hint = "another claim took it first" if force else "rerun only with force and a reason"
        raise HoldoutError(
            f"holdout attempt {attempt} for {scope} is already recorded; {hint}"
        ) from error
    return HoldoutTicket(scope, run_key, attempt)


def holdout_claimed(scope: str) -> bool:
    """Whether scope has run its holdout at least once."""
    return HoldoutRun.objects.filter(scope=scope).exists()


def is_holdout(fold: Fold) -> bool:
    """Holdout by its flag or by the SPEC holdout start, whatever config made the fold."""
    return fold.holdout or fold.test_end >= SPEC_HOLDOUT_START


def authorize_folds(folds: Iterable[Fold], run_key: str, ticket: HoldoutTicket | None) -> None:
    """Refuse holdout folds unless ticket is the latest recorded claim of its scope for run_key."""
    holdout = [fold.index for fold in folds if is_holdout(fold)]
    if not holdout:
        return
    if ticket is None or ticket.run_key != run_key:
        raise HoldoutError(f"folds {holdout} are holdout; claim_holdout for {run_key} first")
    recorded = HoldoutRun.objects.filter(
        scope=ticket.scope, attempt=ticket.attempt, run_key=run_key
    ).exists()
    if not recorded:
        raise HoldoutError(
            f"no holdout_run row for {ticket.scope} attempt {ticket.attempt} and {run_key}"
        )
    if ticket.attempt != _last_attempt(ticket.scope):
        raise HoldoutError(f"attempt {ticket.attempt} of {ticket.scope} has been superseded")
