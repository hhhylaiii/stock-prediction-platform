"""Rebuild stock_master from FinMind listings and the TWSE and TPEx delisting lists."""

from argparse import ArgumentParser
from typing import Any, NamedTuple

import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from apps.etl.adapters import limiter_for
from apps.etl.adapters.finmind import FinmindClient
from apps.etl.adapters.tpex import TpexClient
from apps.etl.adapters.twse import TwseClient
from apps.etl.errors import SourceError
from apps.etl.models import EtlRun
from apps.etl.ratelimit import RetryableError
from apps.etl.runs import tracked_run
from apps.etl.stock_master import (
    build_stock_master,
    check_listed_count,
    suspended_codes,
    tpex_delistings,
    twse_delistings,
    unlisted_codes,
)
from apps.etl.upsert import upsert
from apps.market.models import StockMaster

UPDATED_FIELDS = ("name", "market", "industry", "delist_date")
UNLISTED_SAMPLE = 10
DATASET = "stock_master"
SELECTED_DATASET = "stock_master_selected"
DETAIL_LIMIT = 50


class Found(NamedTuple):
    """What a write does to the stored rows, with a line for each row it touches."""

    lines: list[str]
    created: list[str]
    changed: list[str]


class Command(BaseCommand):
    help = (
        "Upsert listed, officially delisted and halted common stocks into stock_master (SPEC §5)."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--stock",
            action="append",
            dest="stocks",
            help="write only this code (repeatable); every other row is left as it is",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="print the rows a real run would create or change; nothing stays",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        stocks = sorted(set(options["stocks"] or []))
        dry_run = bool(options["dry_run"])
        try:
            if dry_run:
                with transaction.atomic():
                    lines = self._load(stocks, dry_run)
                    transaction.set_rollback(True)
            else:
                lines = self._load(stocks, dry_run)
        except (SourceError, RetryableError) as exc:
            raise CommandError(f"{type(exc).__name__}: {exc}") from exc
        for line in lines:
            self.stdout.write(line)
        if dry_run:
            self.stdout.write("nothing was written")

    def _load(self, stocks: list[str], dry_run: bool) -> list[str]:
        """One fetch and write for every mode; returns the lines to print."""
        today = timezone.localdate()
        dataset = SELECTED_DATASET if stocks else DATASET
        with tracked_run("stock_master", dataset, today, today) as run:
            info = FinmindClient.build(limiter_for("finmind")).data("stock_info", "TaiwanStockInfo")
            twse = twse_delistings(TwseClient.build(limiter_for("twse")).suspend_listing())
            tpex = tpex_delistings(TpexClient.build(limiter_for("tpex")).delisted())
            master = build_stock_master(info, twse, tpex)
            listed = master.loc[master["delist_date"].isna(), "stock_id"].tolist()
            suspended = suspended_codes(info, master)
            check_listed_count(len(listed) - len(suspended), self._last_listed_count())
            chosen = _chosen(master, stocks)
            found = _differences(chosen)
            run.rows = upsert(StockMaster, chosen, ("stock_id",), UPDATED_FIELDS)
            if stocks:
                run.checkpoint = {
                    "stocks": stocks,
                    "created": found.created,
                    "changed": found.changed,
                }
                return [*_capped(found.lines), _tally(found, dry_run)]
            unlisted = sorted(
                set(unlisted_codes(info, master)) | set(self._vanished_from_db(listed))
            )
            run.checkpoint = {
                "listed": len(listed),
                "delisted": len(master) - len(listed),
                "unlisted_without_delisting": len(unlisted),
                "suspended": suspended,
            }
        notes = _notes(unlisted, suspended)
        if dry_run:
            return [*_capped(found.lines), _tally(found, dry_run), *notes]
        gone = len(master) - len(listed)
        return [f"stock_master: {len(listed)} listed, {gone} delisted ({run.rows} rows)", *notes]

    @staticmethod
    def _last_listed_count() -> int | None:
        """Stocks the last full load found in its newest snapshot; the halted are not in one."""
        last = (
            EtlRun.objects.filter(
                source="stock_master", dataset=DATASET, status=EtlRun.Status.SUCCESS
            )
            .order_by("-run_id")
            .first()
        )
        count = last.checkpoint.get("listed") if last is not None else None
        if last is None or count is None:
            return None
        return int(count) - len(last.checkpoint.get("suspended") or [])

    @staticmethod
    def _vanished_from_db(listed: list[str]) -> list[str]:
        """Codes once loaded as listed that vanished without an official delisting date."""
        rows = StockMaster.objects.filter(delist_date__isnull=True).exclude(stock_id__in=listed)
        return sorted(rows.values_list("stock_id", flat=True))


def _chosen(master: pd.DataFrame, stocks: list[str]) -> pd.DataFrame:
    """Every row, or the codes asked for, each of which the sources must give."""
    if not stocks:
        return master
    missing = sorted(set(stocks) - set(master["stock_id"]))
    if missing:
        raise CommandError(f"neither listed, halted nor delisted: {', '.join(missing)}")
    return master.loc[master["stock_id"].isin(stocks)].reset_index(drop=True)


def _differences(frame: pd.DataFrame) -> Found:
    """The frame against the stored rows, before it is written."""
    rows = StockMaster.objects.filter(stock_id__in=list(frame["stock_id"]))
    stored = {row[0]: row[1:] for row in rows.values_list("stock_id", *UPDATED_FIELDS)}
    found = Found([], [], [])
    for record in frame.to_dict("records"):
        code = str(record["stock_id"])
        new = tuple(None if pd.isna(record[name]) else record[name] for name in UPDATED_FIELDS)
        old = stored.get(code)
        if old is None:
            found.created.append(code)
            found.lines.append(f"create {code} {new[0]} {new[1]} {new[2] or '-'} {_state(new[3])}")
        elif old != new:
            found.changed.append(code)
            found.lines.extend(
                f"change {code} {name} {_shown(was)} -> {_shown(now)}"
                for name, was, now in zip(UPDATED_FIELDS, old, new, strict=True)
                if was != now
            )
    return found


def _state(delisted: object) -> str:
    return "listed" if delisted is None else f"delisted {delisted}"


def _shown(value: object) -> str:
    return repr(str(value)) if isinstance(value, str) else str(value)


def _capped(lines: list[str]) -> list[str]:
    if len(lines) <= DETAIL_LIMIT:
        return lines
    return [*lines[:DETAIL_LIMIT], f"and {len(lines) - DETAIL_LIMIT} more lines"]


def _tally(found: Found, dry_run: bool) -> str:
    created, changed = len(found.created), len(found.changed)
    if dry_run:
        return f"stock_master: {created} to create, {changed} to change"
    return f"stock_master: {created} created, {changed} changed"


def _notes(unlisted: list[str], suspended: list[str]) -> list[str]:
    notes = []
    if unlisted:
        sample = ", ".join(unlisted[:UNLISTED_SAMPLE])
        notes.append(f"{len(unlisted)} codes no longer listed and on no delisting list: {sample}")
    if suspended:
        count, kept = len(suspended), ", ".join(suspended)
        notes.append(
            f"{count} common stocks the newest snapshot no longer lists, kept as listed: {kept}"
        )
    return notes
