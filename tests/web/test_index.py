import pytest
from django.test import Client

VIEWS = ("view-today", "view-stock", "view-models")
ASSETS = ("web/app.css", "web/api.js", "web/app.js")


def test_index_serves_the_three_hash_routed_views(client: Client) -> None:
    response = client.get("/")

    assert response.status_code == 200
    html = response.content.decode()
    assert html.count('<section class="view"') == len(VIEWS)
    for view in VIEWS:
        assert f'id="{view}"' in html


def test_index_loads_its_assets_from_static_only(client: Client) -> None:
    html = client.get("/").content.decode()

    for asset in ASSETS:
        assert f"/static/{asset}" in html
    assert "<style" not in html
    assert "<script>" not in html
    assert "googleapis" not in html


def test_index_carries_no_demo_numbers_or_prototype_copy(client: Client) -> None:
    html = client.get("/").content.decode()

    for demo in ("1,842", "0.041", "LightGBM 2026-10", "示意資料", "40 個特徵", "優於舊版"):
        assert demo not in html
    assert "四項健全性檢查" in html


@pytest.mark.parametrize("asset", ASSETS)
def test_assets_are_served_without_collectstatic(client: Client, asset: str) -> None:
    response = client.get(f"/static/{asset}")

    assert response.status_code == 200
    assert int(response["Content-Length"]) > 0


@pytest.mark.parametrize("method", ["post", "put", "delete"])
def test_index_only_answers_reads(client: Client, method: str) -> None:
    assert getattr(client, method)("/").status_code == 405
