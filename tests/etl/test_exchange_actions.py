import copy
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from apps.etl.adapters.corporate_actions import (
    CAPITAL_REDUCTION,
    EX_RIGHTS,
    PAR_VALUE_CHANGE,
    Action,
)
from apps.etl.adapters.exchange_actions import (
    TPEX_RESTARTS,
    TWSE_RESTARTS,
    ExRight,
    restart_actions,
    tpex_ex_rights,
    tpex_table,
    twse_action,
    twse_detail,
    twse_ex_rights,
)
from apps.etl.errors import SchemaChangedError

OCT_6 = date(2026, 10, 6)
OCT_7 = date(2026, 10, 7)
AUG_31 = date(2026, 8, 31)
SEP_1 = date(2026, 9, 1)
TWSE = "twse/dividend_event"
TPEX = "tpex/dividend_event"


def _ratio(after: str, before: str) -> float:
    return float(Decimal(after) / Decimal(before))


def test_twse_ex_rights_reads_each_row_of_the_span(load_fixture: Callable[[str], Any]) -> None:
    rows = {
        row.stock_id: row
        for row in twse_ex_rights(load_fixture(f"{TWSE}/twt49u_20261006_07.json"), (OCT_6, OCT_7))
    }
    assert len(rows) == 9
    assert rows["2614"].ex_date == OCT_6
    assert rows["2614"].kind == "權息"
    assert rows["2890"] == ExRight(
        "2890", OCT_7, _ratio("44.97", "45.40"), "權", Decimal("0.425362")
    )
    assert rows["6108"].kind == "息"
    assert rows["6108"].value == Decimal("1.000000")


def test_twse_ex_rights_refuses_a_row_outside_the_span(
    load_fixture: Callable[[str], Any],
) -> None:
    with pytest.raises(SchemaChangedError, match="outside"):
        twse_ex_rights(load_fixture(f"{TWSE}/twt49u_20261006_07.json"), (OCT_6, OCT_6))


def test_twse_detail_reads_cash_and_bonus_shares_per_thousand(
    load_fixture: Callable[[str], Any],
) -> None:
    assert twse_detail(load_fixture(f"{TWSE}/twt49u_detail_2890_20261007.json"), "2890") == (
        Decimal("0"),
        Decimal("0"),
    )
    body = copy.deepcopy(load_fixture(f"{TWSE}/twt49u_detail_8021_20261006.json"))
    body["data"][0][2] = "1.2 元／股"
    body["data"][0][4] = "1,050.5 股"
    assert twse_detail(body, "8021") == (Decimal("1.2"), Decimal("1050.5"))


@pytest.mark.parametrize(("column", "value"), [(2, "1.2"), (4, "50 元／股"), (2, "-1 元／股")])
def test_twse_detail_refuses_an_amount_it_cannot_read(
    load_fixture: Callable[[str], Any], column: int, value: str
) -> None:
    body = copy.deepcopy(load_fixture(f"{TWSE}/twt49u_detail_8021_20261006.json"))
    body["data"][0][column] = value
    with pytest.raises(SchemaChangedError):
        twse_detail(body, "8021")


def test_twse_detail_must_list_the_asked_code_once(load_fixture: Callable[[str], Any]) -> None:
    with pytest.raises(SchemaChangedError, match="2890"):
        twse_detail(load_fixture(f"{TWSE}/twt49u_detail_8021_20261006.json"), "2890")


def test_twse_action_pays_cash_only_rows_their_value_and_others_their_detail() -> None:
    cash = ExRight("6108", OCT_7, 0.96, "息", Decimal("1.000000"))
    assert twse_action(cash, None) == Action(
        "6108", OCT_7, Decimal("1.000000"), Decimal("0"), 0.96, EX_RIGHTS
    )
    both = ExRight("2614", OCT_6, 0.84, "權息", Decimal("2.964737"))
    assert twse_action(both, (Decimal("1.5"), Decimal("50"))) == Action(
        "2614", OCT_6, Decimal("1.5"), Decimal("0.5"), 0.84, EX_RIGHTS
    )
    with pytest.raises(SchemaChangedError, match="detail"):
        twse_action(both, None)


def _tpex(
    load_fixture: Callable[[str], Any], name: str, span: tuple[date, date]
) -> dict[str, Action]:
    return {
        action.stock_id: action for action in tpex_ex_rights(load_fixture(f"{TPEX}/{name}"), span)
    }


def test_tpex_ex_rights_reads_cash_and_bonus_shares_of_each_row(
    load_fixture: Callable[[str], Any],
) -> None:
    actions = _tpex(load_fixture, "exdailyq_20260831.json", (AUG_31, SEP_1))
    assert len(actions) == 7
    assert actions["2751"] == Action(
        "2751", AUG_31, Decimal("1.99664894"), Decimal("0"), _ratio("53.50", "55.50"), EX_RIGHTS
    )
    assert actions["2937"].cash_dividend == Decimal("0")
    assert actions["2937"].stock_dividend == Decimal("1.0000001148")
    assert actions["5392"].cash_dividend == Decimal("0.2")
    assert actions["5392"].adj_factor == _ratio("47.16", "48.30")


def test_tpex_ex_rights_reads_the_layout_before_2016_and_a_rights_issue(
    load_fixture: Callable[[str], Any],
) -> None:
    old = _tpex(load_fixture, "exdailyq_20150818.json", (date(2015, 8, 18), date(2015, 8, 19)))
    assert old["2640"].cash_dividend == Decimal("4.9")
    assert old["2640"].stock_dividend == Decimal("0.5000000237")
    rights = _tpex(load_fixture, "exdailyq_20261007.json", (OCT_7, date(2026, 10, 8)))
    assert rights["6129"] == Action(
        "6129", OCT_7, Decimal("0"), Decimal("0"), _ratio("14.11", "14.40"), EX_RIGHTS
    )


def test_tpex_ex_rights_of_a_quiet_day_and_outside_the_span(
    load_fixture: Callable[[str], Any],
) -> None:
    assert _tpex(load_fixture, "exdailyq_20261006.json", (OCT_6, OCT_7)) == {}
    with pytest.raises(SchemaChangedError, match="outside"):
        _tpex(load_fixture, "exdailyq_20261007.json", (OCT_6, OCT_6))


def test_tpex_answers_must_hold_one_table(load_fixture: Callable[[str], Any]) -> None:
    body = load_fixture(f"{TPEX}/exdailyq_20261007.json")
    with pytest.raises(SchemaChangedError, match="one table"):
        tpex_table({**body, "tables": body["tables"] * 2}, "TPEx exDailyQ")


def test_restart_actions_divide_the_reference_price_by_the_last_close(
    load_fixture: Callable[[str], Any],
) -> None:
    reduction, par_value = TPEX_RESTARTS
    reductions = restart_actions(
        tpex_table(load_fixture(f"{TPEX}/revivt_20261006_07.json"), reduction.label),
        reduction,
        (OCT_6, OCT_7),
    )
    assert reductions == [
        Action(
            "4527",
            OCT_7,
            Decimal("0"),
            Decimal("0"),
            _ratio("44.33", "40.90"),
            CAPITAL_REDUCTION,
        ),
        Action(
            "5301", OCT_7, Decimal("0"), Decimal("0"), _ratio("6.10", "4.99"), CAPITAL_REDUCTION
        ),
    ]
    changes = restart_actions(
        tpex_table(load_fixture(f"{TPEX}/pvchgrslt_20260831_0901.json"), par_value.label),
        par_value,
        (AUG_31, SEP_1),
    )
    assert changes == [Action("4747", AUG_31, Decimal("0"), Decimal("0"), 0.5, PAR_VALUE_CHANGE)]


def test_twse_par_value_restarts_read_the_recorded_answer(
    load_fixture: Callable[[str], Any],
) -> None:
    par_value = TWSE_RESTARTS[1]
    actions = restart_actions(
        load_fixture(f"{TWSE}/twtb8u_2026.json"), par_value, (date(2026, 1, 1), date(2026, 10, 31))
    )
    assert actions == [
        Action(
            "7780",
            date(2026, 1, 19),
            Decimal("0"),
            Decimal("0"),
            _ratio("18.50", "185.00"),
            PAR_VALUE_CHANGE,
        ),
        Action(
            "6949",
            date(2026, 9, 7),
            Decimal("0"),
            Decimal("0"),
            _ratio("74.50", "1490.00"),
            PAR_VALUE_CHANGE,
        ),
    ]
