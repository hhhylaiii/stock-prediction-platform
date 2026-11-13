import pytest

from apps.market.common_stock import is_common_stock


@pytest.mark.parametrize(
    ("stock_id", "industry"),
    [("2330", "半導體業"), ("1589", "電機機械"), ("9904", "其他"), ("4207", "")],
)
def test_four_digit_codes_off_the_depositary_range_are_common_stocks(
    stock_id: str, industry: str
) -> None:
    assert is_common_stock(stock_id, industry)


@pytest.mark.parametrize(
    ("stock_id", "industry"),
    [
        ("9105", "存託憑證"),
        ("9199", "其他"),
        ("2330", "存託憑證"),
        ("0050", "ETF"),
        ("00793B", "ETF"),
        ("2891A", "金融保險"),
        ("912000", "存託憑證"),
        ("020039", "指數投資證券(ETN)"),
        ("23301", "半導體業"),
        ("", ""),
    ],
)
def test_other_codes_are_not_common_stocks(stock_id: str, industry: str) -> None:
    assert not is_common_stock(stock_id, industry)
