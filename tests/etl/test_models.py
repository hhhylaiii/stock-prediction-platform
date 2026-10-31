from datetime import date

from apps.etl.models import EtlRun, RawTwse


def test_string_representations() -> None:
    raw = RawTwse(dataset="daily_price", request_params={"date": "20261028"})
    assert str(raw) == "daily_price {'date': '20261028'}"
    run = EtlRun(
        source="twse",
        dataset="daily_price",
        start_date=date(2026, 10, 27),
        end_date=date(2026, 10, 28),
        status=EtlRun.Status.SUCCESS,
    )
    assert str(run) == "twse/daily_price 2026-10-27..2026-10-28 success"
