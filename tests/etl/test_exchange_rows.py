from typing import Any

import pytest

from apps.etl.errors import SchemaChangedError
from apps.etl.exchange_rows import parse_percent, parse_signed, stock_code


@pytest.mark.parametrize(
    ("text", "expected"), [("-3,175,021", -3_175_021), ("+12", 12), (" 0 ", 0), ("1,000", 1000)]
)
def test_signed_counts_keep_their_sign(text: str, expected: int) -> None:
    assert parse_signed(text) == expected


@pytest.mark.parametrize("text", ["", "-", "1.5", "²", "--1", "1,2x"])
def test_signed_counts_refuse_anything_else(text: str) -> None:
    with pytest.raises(SchemaChangedError, match="count"):
        parse_signed(text)


@pytest.mark.parametrize(("value", "expected"), [(68.55, 68.55), (0, 0.0), ("1,012.5", 1012.5)])
def test_percent_accepts_numbers_and_text(value: Any, expected: float) -> None:
    assert parse_percent(value) == expected


@pytest.mark.parametrize("value", [True, "abc", None])
def test_percent_refuses_flags_and_words(value: Any) -> None:
    with pytest.raises(SchemaChangedError, match="percent"):
        parse_percent(value)


@pytest.mark.parametrize(("text", "expected"), [(" 2330 ", "2330"), ("00636K", "00636K")])
def test_stock_codes_are_trimmed(text: str, expected: str) -> None:
    assert stock_code(text) == expected


@pytest.mark.parametrize("text", ["", "23 30", "233", "2330000", "a330"])
def test_odd_stock_codes_are_refused(text: str) -> None:
    with pytest.raises(SchemaChangedError, match="stock code"):
        stock_code(text)
