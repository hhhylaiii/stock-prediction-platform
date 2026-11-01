from collections.abc import Callable
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import responses
from django.core.management import call_command
from django.core.management.base import CommandError
from responses import matchers

from apps.etl.adapters import limiter_for
from apps.etl.adapters.tpex import TpexClient
from apps.etl.adapters.twse import TwseClient
from apps.etl.errors import SchemaChangedError
from apps.etl.exchange_rows import OfficialDay
from apps.etl.roc import parse_roc_date
from apps.etl.verify import Difference, compare_days, sample_pairs
from apps.market.models import DailyPrice, Market, StockMaster, TradingCalendar

TWSE_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY"
TPEX_URL = "https://www.tpex.org.tw/www/zh-tw/afterTrading/tradingStock"
EMPTY_TWSE = "twse/stock_day/6446_202312"
EMPTY_TPEX = "tpex/trading_stock/6488_201508"
JAN_4 = date(2010, 1, 4)
TWSE_FIELDS = [
    "日期",
    "成交股數",
    "成交金額",
    "開盤價",
    "最高價",
    "最低價",
    "收盤價",
    "漲跌價差",
    "成交筆數",
    "註記",
]


def _calendar(first: date, last: date, trading: set[date]) -> None:
    days = [first + timedelta(days=offset) for offset in range((last - first).days + 1)]
    TradingCalendar.objects.bulk_create(
        TradingCalendar(date=day, is_trading_day=day in trading) for day in days
    )


def _fixture_days(load_fixture: Callable[[str], Any], *names: str) -> set[date]:
    found = set()
    for name in names:
        payload = load_fixture(f"{name}.json")
        rows = payload["tables"][0]["data"] if "tables" in payload else payload["data"]
        found |= {parse_roc_date(str(row[0]).rstrip("*")) for row in rows}
    return found


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("apps.etl.ratelimit.time.sleep", lambda seconds: None)


@pytest.fixture
def month(load_fixture: Callable[[str], Any]) -> Callable[[str, str, str, str], None]:
    def _mock(market: str, stock: str, yyyymm: str, name: str) -> None:
        if market == "twse":
            query = {"date": f"{yyyymm}01", "stockNo": stock, "response": "json"}
            match = matchers.query_param_matcher(query)
            responses.get(TWSE_URL, json=load_fixture(f"{name}.json"), match=[match])
            return
        form = {"code": stock, "date": f"{yyyymm[:4]}/{yyyymm[4:]}/01", "response": "json"}
        match = matchers.urlencoded_params_matcher(form)
        responses.post(TPEX_URL, json=load_fixture(f"{name}.json"), match=[match])

    return _mock


@responses.activate
@pytest.mark.django_db
def test_official_rows_carry_close_and_volume_in_shares(month: Callable[..., None]) -> None:
    month("twse", "2330", "201001", "twse/stock_day/2330_201001")
    month("tpex", "6446", "201608", "tpex/trading_stock/6446_201608")
    twse = TwseClient.build(limiter_for("twse")).stock_days("2330", date(2010, 1, 1))
    tpex = TpexClient.build(limiter_for("tpex")).stock_days("6446", date(2016, 8, 1))
    assert twse is not None
    assert twse[0] == OfficialDay(JAN_4, Decimal("64.90"), 39_511_138, 1)
    assert tpex[0] == OfficialDay(date(2016, 8, 1), Decimal("184.50"), 729_000, 1000)


@responses.activate
@pytest.mark.django_db
@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (["115/10/01", "0", "0", "--", "--", "--", "--", "", "0", ""], None),
        (["115/10/01", "1,000", "0"], "rows are malformed"),
        (["115/10/01", "1,000", "0", "1", "1", "1", "abc", "", "0", ""], "unexpected price"),
        (["115/10/01", "-5", "0", "1", "1", "1", "1", "", "0", ""], "unexpected count"),
    ],
)
def test_official_rows_are_checked(row: list[str], expected: str | None) -> None:
    responses.get(TWSE_URL, json={"stat": "OK", "fields": TWSE_FIELDS, "data": [row]})
    client = TwseClient.build(limiter_for("twse"))
    if expected is None:
        days = client.stock_days("2330", date(2026, 10, 1))
        assert days == [OfficialDay(date(2026, 10, 1), None, 0, 1)]
        return
    with pytest.raises(SchemaChangedError, match=expected):
        client.stock_days("2330", date(2026, 10, 1))


@responses.activate
@pytest.mark.django_db
def test_a_changed_column_layout_is_a_schema_change() -> None:
    moved = ["日期", "成交金額", *TWSE_FIELDS[2:]]
    row = ["115/10/01", "1,000", "0", "1", "1", "1", "1", "", "0", ""]
    responses.get(TWSE_URL, json={"stat": "OK", "fields": moved, "data": [row]})
    with pytest.raises(SchemaChangedError, match="fields changed"):
        TwseClient.build(limiter_for("twse")).stock_days("2330", date(2026, 10, 1))


def _official(day: date, close: str | None, volume: int, unit: int = 1) -> OfficialDay:
    return OfficialDay(day, None if close is None else Decimal(close), volume, unit)


def test_tpex_volumes_match_within_half_a_thousand_shares() -> None:
    official = {JAN_4: _official(JAN_4, "64.90", 200_000, 1000)}
    assert compare_days("6488", official, {JAN_4: (Decimal("64.90"), 200_500)}) == []
    assert compare_days("6488", official, {JAN_4: (Decimal("64.90"), 199_500)}) == []


def test_matching_days_have_no_differences() -> None:
    official = {
        JAN_4: _official(JAN_4, "64.90", 729_000, 1000),
        date(2010, 1, 5): _official(date(2010, 1, 5), None, 0),
    }
    stored = {JAN_4: (Decimal("64.9000"), 729_431), date(2010, 1, 5): (None, 0)}
    assert compare_days("2330", official, stored) == []


def test_each_kind_of_difference_is_reported() -> None:
    day5, day6, day7 = date(2010, 1, 5), date(2010, 1, 6), date(2010, 1, 7)
    official = {
        JAN_4: _official(JAN_4, "64.90", 100),
        day5: _official(day5, "65.00", 200_000, 1000),
        day6: _official(day6, "66.00", 300),
    }
    stored = {JAN_4: (Decimal("64.80"), 100), day5: (Decimal("65.00"), 200_600), day7: (None, 0)}
    assert compare_days("2330", official, stored) == [
        Difference("2330", JAN_4, "close", "64.80", "64.90"),
        Difference("2330", day5, "volume", "200600", "200000"),
        Difference("2330", day6, "missing", "-", "66.00 / 300"),
        Difference("2330", day7, "extra", "None / 0", "-"),
    ]


def _prices(stock: str, *days: date) -> None:
    DailyPrice.objects.bulk_create(
        DailyPrice(stock_id=stock, date=day, close=10, volume=1, turnover=10) for day in days
    )


@pytest.mark.django_db
def test_sampling_is_repeatable_and_stays_inside_the_years() -> None:
    for code in ("1101", "2330", "2884"):
        StockMaster.objects.create(stock_id=code, name=code, market=Market.TWSE)
        _prices(code, date(2009, 3, 2), date(2011, 3, 1), date(2012, 3, 1), date(2026, 3, 2))
    StockMaster.objects.create(stock_id="6488", name="6488", market=Market.TPEX)
    _prices("6488", date(2012, 3, 1))
    first = sample_pairs(2, 2, seed=7, last_year=2025)
    assert first == sample_pairs(2, 2, seed=7, last_year=2025)
    assert len(first) == 4
    assert {year for _, year in first} == {2011, 2012}
    assert len({code for code, _ in first}) == 2
    assert "6488" not in {code for code, _ in sample_pairs(5, 2, seed=7, last_year=2025)}
    assert sample_pairs(5, 3, seed=7, last_year=2025) == []


def _stored_2330() -> None:
    StockMaster.objects.create(stock_id="2330", name="台積電", market=Market.TWSE)
    DailyPrice.objects.create(
        stock_id="2330", date=JAN_4, close=Decimal("64.90"), volume=39_511_138, turnover=1
    )
    DailyPrice.objects.create(
        stock_id="2330",
        date=date(2010, 1, 5),
        close=Decimal("64.40"),
        volume=38_394_084,
        turnover=1,
    )


def _empty_rest_of_2010(month: Callable[..., None], stock: str) -> None:
    for number in range(2, 13):
        month("twse", stock, f"2010{number:02d}", EMPTY_TWSE)
        month("tpex", stock, f"2010{number:02d}", EMPTY_TPEX)


@responses.activate
@pytest.mark.django_db
def test_differences_are_written_to_the_report_and_fail_the_command(
    month: Callable[..., None], tmp_path: Path, load_fixture: Callable[[str], Any]
) -> None:
    january = _fixture_days(load_fixture, "twse/stock_day/2330_201001")
    _calendar(date(2010, 1, 1), date(2010, 12, 31), january)
    _stored_2330()
    month("twse", "2330", "201001", "twse/stock_day/2330_201001")
    _empty_rest_of_2010(month, "2330")
    report = tmp_path / "verify.md"
    with pytest.raises(CommandError, match="19 differences, 0 failed in 1 stock-year"):
        call_command("verify_backfill", "--pair", "2330:2010", "--output", str(report))
    text = report.read_text(encoding="utf-8")
    assert "| 2330 | 2010 | 20 | 2 | 19 |  |" in text
    assert "block trades" in text
    assert "| 2330 | 2010-01-05 | close | 64.4000 | 64.50 |" in text
    assert "| 2330 | 2010-01-06 | missing |" in text


@responses.activate
@pytest.mark.django_db
def test_stored_days_outside_the_current_market_are_looked_up_on_the_other(
    month: Callable[..., None], tmp_path: Path, load_fixture: Callable[[str], Any]
) -> None:
    august = _fixture_days(load_fixture, "tpex/trading_stock/6446_201608")
    _calendar(date(2016, 8, 1), date(2016, 8, 31), august)
    StockMaster.objects.create(stock_id="6446", name="藥華藥", market=Market.TWSE)
    DailyPrice.objects.create(
        stock_id="6446", date=date(2016, 8, 1), close=Decimal("184.50"), volume=729_400, turnover=1
    )
    month("twse", "6446", "201608", EMPTY_TWSE)
    month("tpex", "6446", "201608", "tpex/trading_stock/6446_201608")
    with pytest.raises(CommandError, match="22 differences"):
        call_command(
            "verify_backfill", "--pair", "6446:2016-08", "--output", str(tmp_path / "r.md")
        )
    assert "| 6446 | 2016-08-01 |" not in (tmp_path / "r.md").read_text(encoding="utf-8")


@responses.activate
@pytest.mark.django_db
def test_a_matching_pair_passes(month: Callable[..., None], tmp_path: Path) -> None:
    _calendar(date(2010, 2, 1), date(2010, 2, 28), set())
    StockMaster.objects.create(stock_id="2330", name="台積電", market=Market.TWSE)
    month("twse", "2330", "201002", EMPTY_TWSE)
    month("tpex", "2330", "201002", EMPTY_TPEX)
    call_command("verify_backfill", "--pair", "2330:2010-02", "--output", str(tmp_path / "ok.md"))
    assert "0 differences" in (tmp_path / "ok.md").read_text(encoding="utf-8")


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("args", "message"),
    [
        ([], "no stock has prices to sample"),
        (["--pair", "2330"], "--pair takes CODE:YEAR or CODE:YYYY-MM"),
        (["--pair", "9999:2010"], "not in stock_master: 9999"),
        (["--pair", "2330:2099"], "2330:2099 is in the future"),
        (["--pair", "2330:2099-01"], "2330:2099-01 is in the future"),
    ],
)
def test_bad_requests_are_command_errors(args: list[str], message: str) -> None:
    with pytest.raises(CommandError, match=message):
        call_command("verify_backfill", *args)


@responses.activate
@pytest.mark.django_db
def test_tpex_stocks_ask_tpex_first_and_long_lists_are_cut(
    month: Callable[..., None], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apps.etl.management.commands.verify_backfill.DETAIL_LIMIT", 1)
    _calendar(date(2015, 9, 1), date(2015, 9, 30), {date(2015, 9, 25), date(2015, 9, 30)})
    StockMaster.objects.create(stock_id="6488", name="環球晶", market=Market.TPEX)
    month("tpex", "6488", "201509", "tpex/trading_stock/6488_201509")
    with pytest.raises(CommandError, match="2 differences"):
        call_command(
            "verify_backfill", "--pair", "6488:2015-09", "--output", str(tmp_path / "r.md")
        )
    text = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "| 6488 | 2015-09-25 | missing |" in text
    assert "1 more differences not listed" in text


@responses.activate
@pytest.mark.django_db
def test_a_month_the_current_market_covers_only_in_part_asks_the_other(
    month: Callable[..., None], tmp_path: Path, load_fixture: Callable[[str], Any]
) -> None:
    twse_days = _fixture_days(load_fixture, "twse/stock_day/6446_202401")
    both = twse_days | _fixture_days(load_fixture, "tpex/trading_stock/6446_202401")
    _calendar(date(2024, 1, 1), date(2024, 1, 31), both)
    StockMaster.objects.create(stock_id="6446", name="藥華藥", market=Market.TWSE)
    _prices("6446", *twse_days)
    month("twse", "6446", "202401", "twse/stock_day/6446_202401")
    month("tpex", "6446", "202401", "tpex/trading_stock/6446_202401")
    report = tmp_path / "r.md"
    with pytest.raises(CommandError, match="27 differences"):
        call_command("verify_backfill", "--pair", "6446:2024-01", "--output", str(report))
    text = report.read_text(encoding="utf-8")
    assert text.count("| missing |") == 17
    assert "| 6446 | 2024-01-02 | missing |" in text


@responses.activate
@pytest.mark.django_db
def test_a_failed_span_is_reported_and_the_others_still_run(
    month: Callable[..., None], tmp_path: Path
) -> None:
    _calendar(date(2009, 12, 1), date(2010, 2, 28), {date(2009, 12, 1)})
    for code in ("2330", "2884"):
        StockMaster.objects.create(stock_id=code, name=code, market=Market.TWSE)
    query = {"date": "20100201", "stockNo": "2330", "response": "json"}
    responses.get(TWSE_URL, json={"stat": "error"}, match=[matchers.query_param_matcher(query)])
    month("twse", "2884", "201002", EMPTY_TWSE)
    month("tpex", "2884", "200912", EMPTY_TPEX)
    args = ["--pair", "2330:2010-02", "--pair", "2884:2010-02", "--pair", "2884:2009-12"]
    with pytest.raises(CommandError, match="0 differences, 2 failed in 3 stock-years"):
        call_command("verify_backfill", *args, "--output", str(tmp_path / "r.md"))
    text = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "| 2330 | 2010-02 | 0 | 0 | 0 | SchemaChangedError: TWSE stock month stat" in text
    assert "| 2884 | 2010-02 | 0 | 0 | 0 |  |" in text
    assert "| 2884 | 2009-12 | 0 | 0 | 0 | a market cannot answer 2009-12 |" in text


@pytest.mark.django_db
def test_an_empty_calendar_is_a_command_error() -> None:
    StockMaster.objects.create(stock_id="2330", name="台積電", market=Market.TWSE)
    with pytest.raises(CommandError, match="trading_calendar is empty"):
        call_command("verify_backfill", "--pair", "2330:2010-02")
