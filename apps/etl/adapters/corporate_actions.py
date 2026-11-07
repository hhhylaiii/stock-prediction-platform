"""Ex-dividend, ex-rights, capital reduction and par value change days (plan 2.7)."""

import logging
from collections.abc import Iterable, Mapping
from datetime import date
from decimal import Decimal
from typing import Any, NamedTuple

import pandas as pd
from django.db import transaction

from apps.etl.adapters.finmind_datasets import DIVIDEND_PLACES, EVENT_COLUMNS
from apps.etl.errors import SchemaChangedError
from apps.etl.models import DataQualityLog
from apps.market.adjust import refresh_adj_factor
from apps.market.models import DividendEvent, StockMaster

logger = logging.getLogger(__name__)
DIVIDEND_EVENT = "dividend_event"
EX_RIGHTS = "ex_rights"
EX_RIGHTS_DETAIL = "ex_rights_detail"
CAPITAL_REDUCTION = "capital_reduction"
PAR_VALUE_CHANGE = "par_value_change"
MISMATCH_CHECK = "dividend_source_mismatch"
AMOUNTS = ("cash_dividend", "stock_dividend")
FACTOR_TOLERANCE = 1e-6
PAR_VALUE = Decimal(10)
PER_THOUSAND = Decimal(1000)
ACTION_COLUMNS = [*EVENT_COLUMNS[:5], "table"]


class Action(NamedTuple):
    """One table's row: what a stock pays and how its price restarts on one day."""

    stock_id: str
    ex_date: date
    cash_dividend: Decimal
    stock_dividend: Decimal
    adj_factor: float | None
    table: str


def price_factor(after: Decimal | None, before: Decimal | None, where: str) -> float | None:
    """Reference price ÷ the close before; blank when either price is missing."""
    if after is None or before is None or after <= 0 or before <= 0:
        logger.warning("%s has no usable reference price: %s ÷ %s", where, after, before)
        return None
    return float(after / before)


def stock_value(per_thousand: Decimal) -> Decimal:
    """Bonus shares per 1,000 held, as 元 per share at a 10 元 par like FinMind's."""
    return per_thousand / PER_THOUSAND * PAR_VALUE


def in_span(day: date, span: tuple[date, date], where: str) -> date:
    """A row dated outside what was asked means the answer is not the one requested."""
    if not span[0] <= day <= span[1]:
        raise SchemaChangedError(f"{where} lists {day}, outside {span[0]}..{span[1]}")
    return day


def action_frame(actions: Iterable[Action]) -> pd.DataFrame:
    return pd.DataFrame(
        [action._asdict() for action in actions], columns=ACTION_COLUMNS, dtype=object
    )


def events_from(frame: pd.DataFrame) -> pd.DataFrame:
    """One event per stock and day; factors multiply and amounts add up across tables."""
    events = []
    for (stock_id, ex_date), rows in frame.groupby(["stock_id", "ex_date"], sort=True):
        if len(rows) > 1:
            logger.warning("%s %s: %s on one day", stock_id, ex_date, ", ".join(rows["table"]))
        factors = [factor for factor in rows["adj_factor"] if factor is not None]
        product = None
        for factor in factors:
            product = factor if product is None else product * factor
        events.append(
            {
                "ex_date": ex_date,
                "stock_id": stock_id,
                **{name: _amount(rows[name]) for name in AMOUNTS},
                "adj_factor": product,
                "announce_date": None,
            }
        )
    return pd.DataFrame(events, columns=EVENT_COLUMNS, dtype=object)


def write_events(frame: pd.DataFrame, source: str, day: date) -> int:
    """New events and blank factors are written; differing stored figures are only logged."""
    records = [_record(raw) for raw in frame.to_dict("records")]
    listed = set(
        StockMaster.objects.filter(
            stock_id__in={record["stock_id"] for record in records}
        ).values_list("stock_id", flat=True)
    )
    records = [record for record in records if record["stock_id"] in listed]
    stored = _stored(records)
    new: list[DividendEvent] = []
    filled: list[DividendEvent] = []
    differing: list[dict[str, Any]] = []
    for record in records:
        row = stored.get((record["stock_id"], record["ex_date"]))
        if row is None:
            new.append(DividendEvent(**record))
            continue
        if row.adj_factor is None and record["adj_factor"] is not None:
            row.adj_factor = record["adj_factor"]
            filled.append(row)
        fields = _differences(row, record)
        if fields:
            differing.append(
                {"stock_id": row.stock_id, "ex_date": row.ex_date.isoformat(), "fields": fields}
            )
    adjusted = {row.stock_id for row in [*new, *filled] if row.adj_factor is not None}
    with transaction.atomic():
        DividendEvent.objects.bulk_create(new)
        DividendEvent.objects.bulk_update(filled, ["adj_factor"])
        _log(source, day, differing)
        for stock_id in sorted(adjusted):
            refresh_adj_factor(stock_id)
    return len(new) + len(filled)


def _amount(values: Iterable[Decimal]) -> Decimal:
    total = sum(values, Decimal(0))
    return total.quantize(DIVIDEND_PLACES).normalize()


def _record(raw: Mapping[Any, Any]) -> dict[str, Any]:
    return {str(name): value for name, value in raw.items()}


def _stored(records: list[dict[str, Any]]) -> dict[tuple[str, date], DividendEvent]:
    rows = DividendEvent.objects.filter(
        stock_id__in={record["stock_id"] for record in records},
        ex_date__in={record["ex_date"] for record in records},
    )
    return {(row.stock_id, row.ex_date): row for row in rows}


def _differences(row: DividendEvent, record: Mapping[str, Any]) -> dict[str, list[Any]]:
    """Stored against the exchange's figure, for each amount or factor that differs."""
    fields: dict[str, list[Any]] = {
        name: [str(getattr(row, name).normalize()), str(record[name])]
        for name in AMOUNTS
        if getattr(row, name).quantize(DIVIDEND_PLACES) != record[name].quantize(DIVIDEND_PLACES)
    }
    factor = record["adj_factor"]
    stored = row.adj_factor
    if (
        factor is not None
        and stored is not None
        and abs(stored - factor) > FACTOR_TOLERANCE * factor
    ):
        fields["adj_factor"] = [stored, factor]
    return fields


def _log(source: str, day: date, differing: list[dict[str, Any]]) -> None:
    """One row per run day and source: a rerun replaces it, agreement removes it."""
    DataQualityLog.objects.filter(
        date=day, check_name=MISMATCH_CHECK, detail__source=source
    ).delete()
    if differing:
        DataQualityLog.objects.create(
            date=day,
            check_name=MISMATCH_CHECK,
            severity=DataQualityLog.Severity.INFO,
            detail={"source": source, "events": differing},
        )
