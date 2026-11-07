"""Holdout guard: a scope runs its holdout folds once unless forced with a reason (SPEC §7)."""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from django.db import IntegrityError, transaction
from django.db.models import Max

from apps.research.models import HoldoutRun, HoldoutScope
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
    last_date: date


@dataclass(frozen=True)
class ProductionPass:
    """Leave for production runs to write past the holdout start once a claim froze the design."""

    scope: str
    frozen_by: str


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
    scope: str, run_key: str, last_date: date, *, force: bool = False, reason: str = ""
) -> HoldoutTicket:
    """Record a holdout run on data through last_date; a rerun needs force, a reason, a new key."""
    if scope not in HoldoutScope.values:
        raise ValueError(f"scope must be one of {HoldoutScope.values}, not {scope!r}")
    require_key(run_key)
    note = reason.strip()
    if force and not note:
        raise HoldoutError("a forced holdout rerun needs a reason")
    if HoldoutRun.objects.filter(run_key=run_key).exists():
        raise HoldoutError(f"{run_key} already holds a holdout run; a rerun needs a new run_key")
    attempt = _last_attempt(scope) + 1 if force else 1
    try:
        with transaction.atomic(durable=True):
            HoldoutRun.objects.create(
                scope=scope, attempt=attempt, run_key=run_key, last_date=last_date, reason=note
            )
    except IntegrityError as error:
        hint = "another claim took it first" if force else "rerun only with force and a reason"
        raise HoldoutError(
            f"holdout attempt {attempt} for {scope} is already recorded; {hint}"
        ) from error
    return HoldoutTicket(scope, run_key, attempt, last_date)


def holdout_claimed(scope: str) -> bool:
    """Whether scope has run its holdout at least once."""
    return HoldoutRun.objects.filter(scope=scope).exists()


def latest_ticket(scope: str) -> HoldoutTicket | None:
    """Ticket of the latest recorded claim, to resume its missing holdout folds."""
    row = HoldoutRun.objects.filter(scope=scope).order_by("-attempt").first()
    if row is None:
        return None
    return HoldoutTicket(row.scope, row.run_key, row.attempt, row.last_date)


def _not_frozen(scope: str) -> HoldoutError:
    return HoldoutError(
        f"the {scope} design is not frozen; production runs wait for its holdout claim"
    )


def production_pass(scope: str) -> ProductionPass:
    """The scope's production pass, granted only once its holdout is claimed."""
    ticket = latest_ticket(scope)
    if ticket is None:
        raise _not_frozen(scope)
    return ProductionPass(scope, ticket.run_key)


def authorize_production(run_key: str, granted: ProductionPass) -> None:
    """Refuse a production write before its scope's claim, outside its scope or on a holdout key."""
    if not holdout_claimed(granted.scope):
        raise _not_frozen(granted.scope)
    if not run_key.startswith(f"{granted.scope}-"):
        raise HoldoutError(
            f"a {granted.scope} production pass covers {granted.scope}- run_keys, not {run_key!r}"
        )
    if HoldoutRun.objects.filter(run_key=run_key).exists():
        raise HoldoutError(f"{run_key} is a holdout run; production runs use their own run_key")


def is_holdout(fold: Fold) -> bool:
    """Holdout by its flag or by the SPEC holdout start, whatever config made the fold."""
    return fold.holdout or fold.test_end >= SPEC_HOLDOUT_START


def authorize_folds(folds: Iterable[Fold], run_key: str, ticket: HoldoutTicket | None) -> None:
    """Refuse holdout folds unless ticket is the latest recorded claim of its scope for run_key."""
    holdout = [fold for fold in folds if is_holdout(fold)]
    if not holdout:
        return
    if ticket is None or ticket.run_key != run_key:
        indices = [fold.index for fold in holdout]
        raise HoldoutError(f"folds {indices} are holdout; claim_holdout for {run_key} first")
    recorded = HoldoutRun.objects.filter(
        scope=ticket.scope, attempt=ticket.attempt, run_key=run_key, last_date=ticket.last_date
    ).exists()
    if not recorded:
        raise HoldoutError(
            f"no holdout_run row for {ticket.scope} attempt {ticket.attempt} and {run_key}"
        )
    if ticket.attempt != _last_attempt(ticket.scope):
        raise HoldoutError(f"attempt {ticket.attempt} of {ticket.scope} has been superseded")
    beyond = [fold.index for fold in holdout if fold.test_end > ticket.last_date]
    if beyond:
        raise HoldoutError(f"folds {beyond} run past the claimed last_date {ticket.last_date}")
