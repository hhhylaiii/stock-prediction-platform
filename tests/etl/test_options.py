from datetime import date

import pytest
from django.core.management.base import CommandError

from apps.etl.management.options import as_date, required_date


def test_as_date_accepts_dates_strings_and_none() -> None:
    assert as_date(date(2026, 10, 1), "--from") == date(2026, 10, 1)
    assert as_date("2026-10-01", "--from") == date(2026, 10, 1)
    assert as_date(None, "--from") is None


def test_required_date_refuses_none() -> None:
    with pytest.raises(CommandError, match="--from is required"):
        required_date(None, "--from")
    assert required_date("2026-10-01", "--from") == date(2026, 10, 1)
