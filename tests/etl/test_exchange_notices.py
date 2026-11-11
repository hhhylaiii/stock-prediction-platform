import copy
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from apps.etl.adapters.exchange_notices import Notice, tpex_notices, twse_notices
from apps.etl.errors import SchemaChangedError

TWSE = "twse/dividend_notice/twt48u_20261111.json"
TPEX = "tpex/dividend_notice/prepost_20261111.json"


def _by_key(notices: list[Notice]) -> dict[tuple[str, date], Notice]:
    return {(notice.stock_id, notice.ex_date): notice for notice in notices}


def test_twse_rows_carry_the_ex_date_and_what_is_due(load_fixture: Callable[[str], Any]) -> None:
    notices = _by_key(twse_notices(load_fixture(TWSE)))

    assert len(notices) == 50
    assert notices["2344", date(2026, 11, 18)] == Notice(
        "2344", date(2026, 11, 18), Decimal("2.00000000"), Decimal(0)
    )
    assert notices["1437", date(2026, 12, 3)].cash_dividend == Decimal("0.62465200")


def test_an_amount_still_to_be_announced_reads_as_nothing_due(
    load_fixture: Callable[[str], Any],
) -> None:
    twse = _by_key(twse_notices(load_fixture(TWSE)))
    tpex = _by_key(tpex_notices(load_fixture(TPEX)))

    assert twse["0051", date(2026, 11, 20)].cash_dividend == 0
    assert tpex["006201", date(2026, 11, 20)].cash_dividend == 0


def test_tpex_rows_carry_the_ex_date_and_what_is_due(load_fixture: Callable[[str], Any]) -> None:
    notices = _by_key(tpex_notices(load_fixture(TPEX)))

    assert len(notices) == 65
    assert notices["6488", date(2026, 12, 29)] == Notice(
        "6488", date(2026, 12, 29), Decimal("6.00000000"), Decimal(0)
    )
    assert notices["5483", date(2027, 1, 5)].cash_dividend == Decimal("2.80000000")


def test_bonus_shares_per_share_become_their_par_value(load_fixture: Callable[[str], Any]) -> None:
    body = copy.deepcopy(load_fixture(TWSE))
    row = next(row for row in body["data"] if row[1] == "2344")
    row[4] = "0.05000000"

    notice = _by_key(twse_notices(body))["2344", date(2026, 11, 18)]

    assert notice.stock_dividend == Decimal("0.5")
    assert notice.cash_dividend == Decimal("2.00000000")


def test_rows_of_one_stock_and_day_add_up(load_fixture: Callable[[str], Any]) -> None:
    body = copy.deepcopy(load_fixture(TWSE))
    row = next(row for row in body["data"] if row[1] == "2344")
    body["data"].append([*row[:7], "0.50000000", *row[8:]])

    notices = twse_notices(body)

    assert len(notices) == 50
    assert _by_key(notices)["2344", date(2026, 11, 18)].cash_dividend == Decimal("2.5")


@pytest.mark.parametrize("text", ["", "  ", "--", "---"])
def test_a_blank_or_dashed_amount_reads_as_nothing_due(
    load_fixture: Callable[[str], Any], text: str
) -> None:
    body = copy.deepcopy(load_fixture(TWSE))
    row = next(row for row in body["data"] if row[1] == "2344")
    row[7] = text

    assert _by_key(twse_notices(body))["2344", date(2026, 11, 18)].cash_dividend == 0


@pytest.mark.parametrize("text", ["另行公告", "N/A", "-1.5"])
def test_an_amount_that_is_neither_a_figure_nor_pending_is_refused(
    load_fixture: Callable[[str], Any], text: str
) -> None:
    body = copy.deepcopy(load_fixture(TWSE))
    body["data"][0][7] = text

    with pytest.raises(SchemaChangedError):
        twse_notices(body)


def test_a_table_without_its_columns_is_refused(load_fixture: Callable[[str], Any]) -> None:
    body = copy.deepcopy(load_fixture(TPEX))
    body["tables"][0]["fields"][7] = "股利"

    with pytest.raises(SchemaChangedError, match="現金股利"):
        tpex_notices(body)
