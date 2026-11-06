"""MOPS figures of each quarter's own filing replace FinMind's where they differ (plan K18)."""

from collections.abc import Sequence
from typing import Any

import pandas as pd
from django.conf import settings
from django.db import transaction

from apps.etl.adapters.mops import STATEMENT_SCAN
from apps.etl.errors import SchemaChangedError
from apps.etl.filings import quarter_end
from apps.etl.models import DataQualityLog, EtlRun, RawMops
from apps.etl.mops_history import STATEMENT_HISTORY, STATEMENT_VIEWER
from apps.etl.statement_scan import SCAN_COLUMNS, figure_differences
from apps.etl.statement_summary import (
    MARKETS,
    REPORTS,
    Summary,
    open_periods,
    parse_summary,
    previous_period,
    statement_rows,
)
from apps.etl.viewer_history import viewer_frame
from apps.market.models import FinancialStatement

OVERWRITE_CHECK = "statement_overwritten"
REPLACED = ("revenue", "net_income", "operating_cf", "debt_ratio", "roe")
FILLED = ("eps",)
NOT_KEPT = "檔案不存在!"
ROC_OFFSET = 1911


def overwrite_statements(frame: pd.DataFrame) -> int:
    """Stored rows of the same type take MOPS's figures; the log keeps FinMind's first value."""
    scanned = first_scanned()
    if scanned is not None:
        frame = frame[frame["period"] < scanned]
    stored = {
        (row.stock_id, row.period, row.fin_type): row
        for row in FinancialStatement.objects.filter(
            stock_id__in=frame["stock_id"].unique().tolist(),
            period__in=frame["period"].unique().tolist(),
        )
    }
    changes: dict[str, list[dict[str, Any]]] = {}
    updated: dict[tuple[str, str, str], set[str]] = {}
    for raw in frame.to_dict("records"):
        record = {str(name): value for name, value in raw.items()}
        key = (record["stock_id"], record["period"], record["fin_type"])
        row = stored.get(key)
        fields = {} if row is None else _taken(figure_differences(row, record))
        if row is None or not fields:
            continue
        for name in fields:
            setattr(row, name, record[name])
        updated[key] = updated.get(key, set()) | fields.keys()
        entry = {"period": row.period, "fin_type": row.fin_type, "fields": fields}
        changes.setdefault(row.stock_id, []).append(entry)
    with transaction.atomic():
        for key, names in updated.items():
            stored[key].save(update_fields=sorted(names))
        for stock_id, entries in changes.items():
            _log(stock_id, entries)
    return len(updated)


def first_scanned() -> str | None:
    """The first quarter the daily scan read as being filed; from it on, the first sight stands."""
    run = EtlRun.objects.filter(source="mops", dataset=STATEMENT_SCAN).order_by("start_date")
    first = run.first()
    if first is None:
        return None
    periods = open_periods(first.start_date, int(settings.STATEMENT_SCAN_DAYS_AFTER))
    return min(periods) if periods else None


def _taken(differences: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """Replaced figures where MOPS has one; EPS only where FinMind has none."""
    return {
        name: pair
        for name, pair in differences.items()
        if pair[1] is not None and (name in REPLACED or (name in FILLED and pair[0] is None))
    }


def _log(stock_id: str, entries: list[dict[str, Any]]) -> None:
    """One row per stock, dated at its last quarter changed; a field keeps FinMind's first value."""
    logged = DataQualityLog.objects.filter(check_name=OVERWRITE_CHECK, detail__stock_id=stock_id)
    merged = {(e["period"], e["fin_type"]): e for log in logged for e in log.detail["changes"]}
    for entry in entries:
        key = (entry["period"], entry["fin_type"])
        earlier = merged.get(key, {"fields": {}})["fields"]
        fields = {**earlier}
        for name, (before, after) in entry["fields"].items():
            fields[name] = [earlier[name][0] if name in earlier else before, after]
        merged[key] = {**entry, "fields": fields}
    logged.delete()
    DataQualityLog.objects.create(
        date=quarter_end(max(period for period, _ in merged)),
        check_name=OVERWRITE_CHECK,
        severity=DataQualityLog.Severity.INFO,
        detail={"stock_id": stock_id, "changes": [merged[key] for key in sorted(merged)]},
    )


def summary_frame() -> pd.DataFrame:
    """Single-quarter rows from the summary pages the history backfill kept.

    Without the quarter before, only what needs none is read: balances and first quarters.
    """
    pages = _latest(STATEMENT_HISTORY, ("report", "TYPEK", "year", "season"))
    periods = sorted({_summary_period(params) for params in pages})
    summaries = {period: _summary(period, pages) for period in periods}
    empty: Summary = {report: {} for report in REPORTS}
    records = []
    for period in periods:
        before = summaries.get(previous_period(period), empty)
        for row in statement_rows(summaries[period], before, int(period[-1])):
            records.append({**row, "period": period})
    return pd.DataFrame(records, columns=SCAN_COLUMNS, dtype=object)


def viewer_frame_from_raw(stock_ids: Sequence[str] | None = None) -> pd.DataFrame:
    """Single-quarter rows from the viewer pages the delisted backfill kept.

    A stock with a kept page that does not parse is listed in attrs["unreadable"] and skipped.
    """
    keys = ("CO_ID", "SYEAR", "SSEASON", "REPORT_ID")
    asked = {"request_params__CO_ID__in": list(stock_ids)} if stock_ids else {}
    pages = _latest(STATEMENT_VIEWER, keys, **asked)
    by_stock: dict[str, set[str]] = {}
    for code, year, season, _ in pages:
        if not stock_ids or code in stock_ids:
            by_stock.setdefault(code, set()).add(f"{year}-Q{season}")
    frames, unreadable = [], []
    for code, periods in sorted(by_stock.items()):

        def page(period: str, kind: str, code: str = code) -> str:
            return pages.get((code, period[:4], period[-1], kind), NOT_KEPT)

        try:
            frames.append(viewer_frame(code, sorted(periods), set(), page, SCAN_COLUMNS))
        except SchemaChangedError:
            unreadable.append(code)
    found = [part for part in frames if not part.empty]
    frame = pd.concat(found, ignore_index=True) if found else pd.DataFrame(columns=SCAN_COLUMNS)
    frame.attrs["unreadable"] = unreadable
    return frame


def _latest(dataset: str, keys: tuple[str, ...], **match: object) -> dict[tuple[str, ...], str]:
    """The page stored last for each request."""
    rows = RawMops.objects.filter(dataset=dataset, **match).order_by("fetched_at", "id")
    return {
        tuple(str(params[key]) for key in keys): payload["html"]
        for params, payload in rows.values_list("request_params", "payload").iterator()
    }


def _summary_period(params: tuple[str, ...]) -> str:
    return f"{int(params[2]) + ROC_OFFSET}-Q{int(params[3])}"


def _summary(period: str, pages: dict[tuple[str, ...], str]) -> Summary:
    """Both markets of the three summaries, as the backfill read them; a filed one is not empty."""
    year, season = str(int(period[:4]) - ROC_OFFSET), f"{int(period[-1]):02d}"
    found: dict[str, dict[str, dict[str, float | None]]] = {}
    for report in REPORTS:
        found[report] = {}
        for market in MARKETS:
            page = pages.get((report, market, year, season))
            companies = {} if page is None else parse_summary(page, market, int(season), report)
            if not companies:
                raise SchemaChangedError(f"MOPS {report} {market} {period} kept no companies")
            if companies.keys() & found[report].keys():
                raise SchemaChangedError(f"MOPS {report} {period} lists a company twice")
            found[report] |= companies
    return found
