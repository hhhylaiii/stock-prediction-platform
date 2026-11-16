"""Holdout scopes whose latest claim was never released and whose models do not serve."""

from typing import Any

from apps.api.serving import Slot, latest_days
from apps.research.models import HoldoutRun, HoldoutScope, ModelObjective

SCOPE_SLOTS: dict[str, tuple[Slot, ...]] = {
    HoldoutScope.INTERVAL: (
        Slot(ModelObjective.QUANTILE, 20),
        Slot(ModelObjective.VOLATILITY, 20),
        Slot(ModelObjective.CLASSIFICATION, 20),
    ),
}


def slot_json(slot: Slot) -> dict[str, Any]:
    return {"objective": slot.objective, "horizon": slot.horizon}


def unreleased() -> list[dict[str, Any]]:
    """Each scope above whose latest attempt has no release and none of whose slots serve."""
    claims = (
        HoldoutRun.objects.filter(scope__in=SCOPE_SLOTS)
        .order_by("scope", "-attempt")
        .values_list("scope", "run_key", "release")
    )
    latest: dict[str, tuple[str, int | None]] = {}
    for scope, run_key, release in claims:
        latest.setdefault(scope, (run_key, release))
    held = {scope: run_key for scope, (run_key, release) in latest.items() if release is None}
    if not held:
        return []
    served = latest_days(slot for scope in held for slot in SCOPE_SLOTS[scope])
    return [
        {
            "scope": scope,
            "run_key": run_key,
            "slots": [slot_json(slot) for slot in SCOPE_SLOTS[scope]],
        }
        for scope, run_key in held.items()
        if not any(slot in served for slot in SCOPE_SLOTS[scope])
    ]
