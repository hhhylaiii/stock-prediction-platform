from datetime import date
from typing import Any

import pytest
from django.test import Client
from pytest_django import DjangoAssertNumQueries

from apps.api.holding import BOOK_START, MAIN_BOOK, RULE_NAME
from apps.market.models import IndexDaily
from apps.research.models import BookAction, BookStatus
from apps.research.shadow import BOOKS
from tests.api.conftest import add_model_run, add_predictions, add_stock, envelope_of
from tests.holding_books import add_closes as shared_closes
from tests.holding_books import add_fill, add_list

MAIN = MAIN_BOOK
S0, D0, D1, D2, D3 = (date(2026, 11, day) for day in (16, 17, 18, 19, 20))


def add_closes(**closes: float) -> None:
    days = {"d0": D0, "d1": D1, "d2": D2, "d3": D3}
    shared_closes({days[name]: close for name, close in closes.items()})


def holding(client: Client) -> dict[str, Any]:
    response = client.get("/api/holding")
    assert response.status_code == 200
    body = envelope_of(response)
    assert body["success"] is True
    return body


@pytest.fixture
def stocks(db: None) -> None:
    for stock_id, name in (
        ("1101", "台泥"),
        ("2330", "台積電"),
        ("2454", "聯發科"),
        ("3008", "大立光"),
    ):
        add_stock(stock_id, name)


def test_the_endpoint_names_the_research_layer_s_main_book() -> None:
    spec = BOOKS[MAIN_BOOK]

    assert spec.role == "main"
    assert (spec.rule.name, spec.start) == (RULE_NAME, BOOK_START)
    assert [name for name, other in BOOKS.items() if other.role == "main"] == [MAIN_BOOK]


def test_no_book_rows_yet_names_the_book_and_returns_empty_parts(client: Client, db: None) -> None:
    body = holding(client)

    assert body["data"] == {
        "book": MAIN,
        "rule": "H70/N30",
        "start": "2026-11-16",
        "list": None,
        "fill": None,
        "since": {"points": [], "no_prediction_days": 0, "error": None},
    }
    assert body["meta"] == {"predictions_date": None}


def test_the_latest_list_names_each_action_in_sell_keep_buy_backup_order(
    client: Client, stocks: None
) -> None:
    add_list(
        S0,
        D0,
        {
            "3008": (BookAction.BACKUP, 2, 0.93),
            "2454": (BookAction.BUY, 1, 0.97),
            "1101": (BookAction.KEEP, 0, 0.81),
            "2330": (BookAction.SELL, 0, 0.42),
        },
        backfilled=True,
    )

    data = holding(client)["data"]

    assert data["list"] == {
        "signal_date": "2026-11-16",
        "trade_date": "2026-11-17",
        "status": "listed",
        "backfilled": True,
        "held_before": 2,
        "sells": 1,
        "buys": 1,
        "backups": 1,
        "actions": [
            {"stock_id": "2330", "name": "台積電", "action": "sell", "priority": 0, "pct": 0.42},
            {"stock_id": "1101", "name": "台泥", "action": "keep", "priority": 0, "pct": 0.81},
            {"stock_id": "2454", "name": "聯發科", "action": "buy", "priority": 1, "pct": 0.97},
            {"stock_id": "3008", "name": "大立光", "action": "backup", "priority": 2, "pct": 0.93},
        ],
    }
    assert data["fill"] is None
    assert data["since"] == {"points": [], "no_prediction_days": 0, "error": None}


def test_keeps_follow_their_percentile_from_the_top(client: Client, stocks: None) -> None:
    add_list(
        S0,
        D0,
        {
            "1101": (BookAction.KEEP, 0, 0.75),
            "2330": (BookAction.KEEP, 0, 0.95),
            "2454": (BookAction.KEEP, 0, None),
        },
    )

    actions = holding(client)["data"]["list"]["actions"]

    assert [line["stock_id"] for line in actions] == ["2330", "1101", "2454"]


def test_the_list_and_the_fill_are_each_the_latest_of_their_phase(
    client: Client, stocks: None
) -> None:
    add_list(S0, D0, {"2330": (BookAction.BUY, 1, 0.95)})
    add_fill(S0, D0, 0.99, cash=0.01, paid=0.0014, traded=0.98, held=1)
    add_list(D0, D1, {"2330": (BookAction.SELL, 0, 0.5), "2454": (BookAction.BUY, 1, 0.92)})
    add_fill(
        D0, D1, 1.01, cash=0.02, paid=0.003, traded=1.9, held=1, stuck=("2330",), backfilled=True
    )
    add_list(D1, D2, {"2330": (BookAction.SELL, 0, 0.4), "2454": (BookAction.KEEP, 0, 0.9)})

    data = holding(client)["data"]

    assert data["list"]["signal_date"] == "2026-11-18"
    assert [line["action"] for line in data["list"]["actions"]] == ["sell", "keep"]
    assert data["fill"] == {
        "signal_date": "2026-11-17",
        "trade_date": "2026-11-18",
        "status": "traded",
        "value": 1.01,
        "cash": 0.02,
        "held": 1,
        "paid": 0.003,
        "traded": 1.9,
        "backfilled": True,
        "stuck": [{"stock_id": "2330", "name": "台積電"}],
    }


def test_a_day_without_predictions_has_no_actions_and_is_counted_after_the_first_fill(
    client: Client, stocks: None
) -> None:
    add_list(S0, D0, {"2330": (BookAction.BUY, 1, 0.95)})
    add_fill(S0, D0, 0.99, universe=0.4)
    add_list(D0, D1, status=BookStatus.NO_PREDICTION)
    add_fill(D0, D1, 1.0, universe=0.0, status=BookStatus.IDLE)
    add_list(D1, D2, status=BookStatus.NO_PREDICTION)

    data = holding(client)["data"]

    assert data["list"]["status"] == "no_prediction"
    assert data["list"]["actions"] == []
    assert data["fill"]["status"] == "idle"
    assert data["since"]["no_prediction_days"] == 1
    assert [point["universe"] for point in data["since"]["points"]] == [1.0, 1.0]


def test_a_first_list_without_predictions_is_not_counted(client: Client, stocks: None) -> None:
    add_list(S0, D0, status=BookStatus.NO_PREDICTION)
    add_fill(S0, D0, 1.0, status=BookStatus.IDLE)

    assert holding(client)["data"]["since"]["no_prediction_days"] == 0


def test_since_compares_the_book_with_the_universe_and_the_index_closes(
    client: Client, stocks: None
) -> None:
    add_closes(d0=20000.0, d1=20400.0, d3=20604.0)
    IndexDaily.objects.bulk_create(
        IndexDaily(index_id="TAIEX", date=day, close=15000.0) for day in (D0, D1, D2, D3)
    )
    add_list(S0, D0)
    add_fill(S0, D0, 0.99, universe=0.3)
    add_list(D0, D1)
    add_fill(D0, D1, 1.01, universe=0.01, index_return=None)
    add_list(D1, D2)
    add_fill(D1, D2, 0.97, universe=-0.02)
    add_list(D2, D3)
    add_fill(D2, D3, 0.99, universe=0.0)

    points = holding(client)["data"]["since"]["points"]

    assert points == [
        {"trade_date": "2026-11-17", "book": 1.0, "universe": 1.0, "index": 1.0},
        {
            "trade_date": "2026-11-18",
            "book": round(1.01 / 0.99, 6),
            "universe": 1.01,
            "index": 1.02,
        },
        {
            "trade_date": "2026-11-19",
            "book": round(0.97 / 0.99, 6),
            "universe": round(1.01 * 0.98, 6),
            "index": None,
        },
        {
            "trade_date": "2026-11-20",
            "book": 1.0,
            "universe": round(1.01 * 0.98, 6),
            "index": 1.0302,
        },
    ]
    assert all(isinstance(point["index"], float | None) for point in points)


def test_a_missing_fill_keeps_the_list_and_says_the_ledger_is_off(
    client: Client, stocks: None, caplog: pytest.LogCaptureFixture
) -> None:
    add_list(S0, D0, {"2330": (BookAction.BUY, 1, 0.95)})
    add_fill(S0, D0, 0.99)
    add_list(D0, D1)
    add_list(D1, D2, {"2330": (BookAction.KEEP, 0, 0.9)})
    add_fill(D1, D2, 1.0)

    data = holding(client)["data"]

    assert data["since"] == {"points": [], "no_prediction_days": 0, "error": "ledger_invalid"}
    assert data["list"]["actions"][0]["stock_id"] == "2330"
    assert data["fill"]["trade_date"] == "2026-11-19"
    assert "gap before 2026-11-19" in caplog.text


def test_candidate_and_paired_books_never_show(client: Client, stocks: None) -> None:
    add_list(S0, D0, {"2330": (BookAction.BUY, 1, 0.95)})
    add_fill(S0, D0, 0.99)
    for book in ("H70N30-C1", "H70N30-C2", "H70N30-C0-20261118"):
        add_list(D1, D2, {"2454": (BookAction.BUY, 1, 0.99)}, book=book)
        add_fill(D1, D2, 1.2, book=book, stuck=("1101",))

    response = client.get("/api/holding")
    data = envelope_of(response)["data"]

    for other in ("H70N30-C1", "H70N30-C2", "H70N30-C0-20261118", "2454", "1101", "1.2"):
        assert other not in response.content.decode()
    assert data["book"] == MAIN
    assert data["list"]["signal_date"] == "2026-11-16"
    assert [line["stock_id"] for line in data["list"]["actions"]] == ["2330"]
    assert data["fill"]["value"] == 0.99
    assert data["fill"]["stuck"] == []
    assert [point["trade_date"] for point in data["since"]["points"]] == ["2026-11-17"]


def test_meta_names_the_five_day_prediction_date_of_the_today_page(
    client: Client, stocks: None
) -> None:
    run = add_model_run("rank-2026-11.h5", "rank")
    add_predictions(run, D1, 5, {"2330": {"rank": 1, "pct": 1.0, "score": 0.9}})
    twenty = add_model_run("rank-2026-11.h20", "rank")
    add_predictions(twenty, D2, 20, {"2330": {"rank": 1, "pct": 1.0, "score": 0.9}})

    assert holding(client)["meta"] == {"predictions_date": "2026-11-18"}


def test_an_unknown_stock_keeps_its_code_without_a_name(client: Client, db: None) -> None:
    add_list(S0, D0, {"9999": (BookAction.BUY, 1, 0.95)})

    actions = holding(client)["data"]["list"]["actions"]

    assert actions == [
        {"stock_id": "9999", "name": None, "action": "buy", "priority": 1, "pct": 0.95}
    ]


def test_the_endpoint_reads_a_fixed_number_of_queries(
    client: Client, stocks: None, django_assert_max_num_queries: DjangoAssertNumQueries
) -> None:
    add_closes(d0=20000.0, d1=20400.0)
    add_list(S0, D0, {"2330": (BookAction.BUY, 1, 0.95), "2454": (BookAction.BACKUP, 2, 0.93)})
    add_fill(S0, D0, 0.99)
    add_list(D0, D1, {"2330": (BookAction.KEEP, 0, 0.9)})
    add_fill(D0, D1, 1.0, stuck=("1101",))

    with django_assert_max_num_queries(8):
        holding(client)


def test_only_reads_are_allowed(client: Client, db: None) -> None:
    response = client.post("/api/holding")

    assert response.status_code == 405
