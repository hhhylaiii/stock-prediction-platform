from datetime import date

import pytest
from django.test import Client
from pytest_django import DjangoAssertNumQueries

from apps.api.stocks import SEARCH_LIMIT
from apps.market.models import Market
from tests.api.conftest import add_stock, envelope_of

pytestmark = pytest.mark.django_db


def found(client: Client, q: str) -> list[dict[str, object]]:
    response = client.get("/api/stocks", {"q": q})
    assert response.status_code == 200
    body = envelope_of(response)
    assert body["meta"] == {"q": q.strip(), "limit": SEARCH_LIMIT}
    data: list[dict[str, object]] = body["data"]
    return data


def test_a_code_prefix_lists_the_exact_code_first_then_the_rest_by_code(client: Client) -> None:
    add_stock("23301", "某權證")
    add_stock("2330", "台積電")
    add_stock("2303", "聯電")
    add_stock("1101", "台泥")

    assert [row["stock_id"] for row in found(client, "2330")] == ["2330", "23301"]
    assert [row["stock_id"] for row in found(client, "230")] == ["2303"]


def test_a_code_matches_from_its_start_only(client: Client) -> None:
    add_stock("2330", "台積電")

    assert found(client, "330") == []


def test_each_row_carries_the_code_name_market_and_delisting(client: Client) -> None:
    add_stock("6488", "環球晶", market=Market.TPEX)

    assert found(client, "6488") == [
        {"stock_id": "6488", "name": "環球晶", "market_label": "上櫃", "delisted": False}
    ]


def test_a_name_matches_anywhere_in_it_after_the_code_matches(client: Client) -> None:
    add_stock("2330", "台積電")
    add_stock("3711", "日月光投控")
    add_stock("1234", "黑松")
    add_stock("9910", "台積電子")

    assert [row["stock_id"] for row in found(client, "台積")] == ["2330", "9910"]
    assert [row["stock_id"] for row in found(client, "投控")] == ["3711"]


def test_an_exact_name_leads_the_other_name_matches(client: Client) -> None:
    add_stock("2211", "長榮鋼")
    add_stock("2603", "長榮")
    add_stock("2618", "長榮航")

    assert [row["stock_id"] for row in found(client, "長榮")] == ["2603", "2211", "2618"]


def test_letters_match_without_regard_to_case(client: Client) -> None:
    add_stock("00631L", "元大台灣50正2")
    add_stock("2412", "中華電")
    add_stock("4904", "遠傳ABC")

    assert [row["stock_id"] for row in found(client, "00631l")] == ["00631L"]
    assert [row["stock_id"] for row in found(client, "abc")] == ["4904"]


def test_code_matches_rank_before_name_matches_with_smaller_codes(client: Client) -> None:
    add_stock("1000", "代號1101概念")
    add_stock("11015", "台泥特")
    add_stock("1101", "台泥")

    assert [row["stock_id"] for row in found(client, "1101")] == ["1101", "11015", "1000"]


def test_delisted_stocks_are_listed_and_marked(client: Client) -> None:
    add_stock("2498", "宏達電")
    add_stock("2499", "東貝", delist_date=date(2020, 6, 1))

    rows = found(client, "249")

    assert [(row["stock_id"], row["delisted"]) for row in rows] == [
        ("2498", False),
        ("2499", True),
    ]


def test_at_most_the_limit_comes_back_in_one_query(
    client: Client, django_assert_num_queries: DjangoAssertNumQueries
) -> None:
    for n in range(SEARCH_LIMIT + 5):
        add_stock(f"8{n:03d}", f"測試{n}")

    with django_assert_num_queries(1):
        rows = found(client, "8")

    assert SEARCH_LIMIT == 20
    assert [row["stock_id"] for row in rows] == [f"8{n:03d}" for n in range(SEARCH_LIMIT)]


def test_surrounding_spaces_are_dropped(client: Client) -> None:
    add_stock("2330", "台積電")

    assert [row["stock_id"] for row in found(client, "  2330 ")] == ["2330"]


def test_no_match_is_an_empty_list(client: Client) -> None:
    add_stock("2330", "台積電")

    assert found(client, "鴻海") == []


@pytest.mark.parametrize("params", [{}, {"q": ""}, {"q": "   "}, {"q": "x" * 33}])
def test_a_missing_blank_or_long_query_is_a_bad_request(
    client: Client, params: dict[str, str]
) -> None:
    response = client.get("/api/stocks", params)

    assert response.status_code == 400
    body = envelope_of(response)
    assert body["success"] is False
    assert set(body["error"]["fields"]) == {"q"}
