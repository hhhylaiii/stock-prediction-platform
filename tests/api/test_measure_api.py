from io import StringIO

import pytest
from django.core.management import call_command

from tests.api.conftest import Runs

pytestmark = pytest.mark.django_db


def test_every_endpoint_gets_a_timed_row(runs: Runs, calendar_in_db: list[object]) -> None:
    out = StringIO()

    call_command("measure_api", "--repeat", "2", stdout=out)

    rows = out.getvalue().splitlines()[2:]
    assert len(rows) == 11
    assert rows[0].startswith("| `/api/stocks/2330` | 200 |")
    assert all(row.split(" | ")[1] == "200" for row in rows)


def test_the_stock_and_repeat_are_options(runs: Runs, calendar_in_db: list[object]) -> None:
    out = StringIO()

    call_command("measure_api", "--stock", "6488", "--repeat", "0", stdout=out)

    assert "| `/api/stocks/6488/history` |" in out.getvalue()
