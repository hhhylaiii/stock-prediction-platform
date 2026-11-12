"""Ex-dividend, ex-rights, capital reduction and par value change days (plan 2.7)."""

import logging
import math
from collections.abc import Iterable, Mapping
from datetime import date
from decimal import Decimal
from typing import Any, NamedTuple

import pandas as pd
from django.conf import settings
from django.db import transaction

from apps.etl.adapters.finmind_datasets import DIVIDEND_PLACES, EVENT_COLUMNS
from apps.etl.dividend_dates import exchanges_lead_from
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
    """One event per stock and day; factors multiply and amounts add up across tables.

    A day with any table's factor blank keeps its factor blank rather than a partial product.
    """
    events = []
    for (stock_id, ex_date), rows in frame.groupby(["stock_id", "ex_date"], sort=True):
        if len(rows) > 1:
            logger.warning("%s %s: %s on one day", stock_id, ex_date, ", ".join(rows["table"]))
        factors = list(rows["adj_factor"])
        product = None if any(factor is None for factor in factors) else math.prod(factors)
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
    """New events and blank factors are written; stored figures that differ are logged.

    From the day the exchanges lead, the results table also replaces what differs, unless
    DIVIDEND_RESULTS_SETTLE is off.
    """
    records = [_record(raw) for raw in frame.to_dict("records")]
    listed = set(
        StockMaster.objects.filter(
            stock_id__in={record["stock_id"] for record in records}
        ).values_list("stock_id", flat=True)
    )
    records = [record for record in records if record["stock_id"] in listed]
    stored = _stored(records)
    settling = bool(settings.DIVIDEND_RESULTS_SETTLE)
    leads_from = exchanges_lead_from()
    new: list[DividendEvent] = []
    changed: list[DividendEvent] = []
    differing: list[dict[str, Any]] = []
    for record in records:
        row = stored.get((record["stock_id"], record["ex_date"]))
        if row is None:
            new.append(DividendEvent(**record))
            continue
        fields = _differences(row, record)
        settles = settling and bool(fields) and row.ex_date >= leads_from
        blank = row.adj_factor is None and record["adj_factor"] is not None
        if settles:
            row.cash_dividend, row.stock_dividend = (record[name] for name in AMOUNTS)
        if record["adj_factor"] is not None and (settles or blank):
            row.adj_factor = record["adj_factor"]
        if settles or blank:
            changed.append(row)
        if fields:
            differing.append(_difference(row, fields, settles))
    adjusted = {row.stock_id for row in [*new, *changed] if row.adj_factor is not None}
    with transaction.atomic():
        DividendEvent.objects.bulk_create(new)
        DividendEvent.objects.bulk_update(changed, [*AMOUNTS, "adj_factor"])
        _log(source, day, differing)
        for stock_id in sorted(adjusted):
            refresh_adj_factor(stock_id)
    return len(new) + len(changed)


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


def _plain(value: Decimal) -> str:
    """An amount written out in full, so 70 never reads 7E+1."""
    return format(value.normalize(), "f")


def _differences(row: DividendEvent, record: Mapping[str, Any]) -> dict[str, list[Any]]:
    """Stored against the exchange's figure, for each amount or factor that differs."""
    fields: dict[str, list[Any]] = {
        name: [_plain(getattr(row, name)), _plain(record[name])]
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


def _difference(row: DividendEvent, fields: dict[str, list[Any]], settled: bool) -> dict[str, Any]:
    entry = {"stock_id": row.stock_id, "ex_date": row.ex_date.isoformat(), "fields": fields}
    return {**entry, "overwritten": True} if settled else entry


def _log(source: str, day: date, differing: list[dict[str, Any]]) -> None:
    """One row per run day and source: a rerun replaces it, agreement removes it.

    What the day already overwrote stays on record, as a rerun finds nothing left to differ.
    """
    logs = DataQualityLog.objects.filter(date=day, check_name=MISMATCH_CHECK, detail__source=source)
    again = {(entry["stock_id"], entry["ex_date"]) for entry in differing}
    kept = [
        entry
        for log in logs
        for entry in log.detail.get("events", [])
        if entry.get("overwritten") and (entry["stock_id"], entry["ex_date"]) not in again
    ]
    differing = sorted([*kept, *differing], key=lambda entry: (entry["stock_id"], entry["ex_date"]))
    logs.delete()
    if differing:
        DataQualityLog.objects.create(
            date=day,
            check_name=MISMATCH_CHECK,
            severity=DataQualityLog.Severity.INFO,
            detail={"source": source, "events": differing},
        )
