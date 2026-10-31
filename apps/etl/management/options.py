"""Argument helpers shared by the ETL management commands."""

from datetime import date

from django.core.management.base import CommandError


def as_date(value: date | str | None, flag: str) -> date | None:
    """Accept a date or an ISO string; call_command passes non-required options unparsed."""
    if value is None or isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise CommandError(f"{flag} must be an ISO date like 2026-10-28, got {value!r}") from None


def required_date(value: date | str | None, flag: str) -> date:
    parsed = as_date(value, flag)
    if parsed is None:
        raise CommandError(f"{flag} is required")
    return parsed
