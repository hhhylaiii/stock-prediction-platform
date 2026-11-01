"""Rebuild stock_master from FinMind listings and the TWSE and TPEx delisting lists."""

from typing import Any

from django.core.management.base import BaseCommand, CommandError
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
    tpex_delistings,
    twse_delistings,
    unlisted_codes,
)
from apps.etl.upsert import upsert
from apps.market.models import StockMaster

UPDATED_FIELDS = ("name", "market", "industry", "delist_date")
UNLISTED_SAMPLE = 10


class Command(BaseCommand):
    help = "Upsert every listed and officially delisted stock into stock_master (SPEC §5)."

    def handle(self, *args: Any, **options: Any) -> None:
        today = timezone.localdate()
        try:
            with tracked_run("stock_master", "stock_master", today, today) as run:
                info = FinmindClient.build(limiter_for("finmind")).data(
                    "stock_info", "TaiwanStockInfo"
                )
                twse = twse_delistings(TwseClient.build(limiter_for("twse")).suspend_listing())
                tpex = tpex_delistings(TpexClient.build(limiter_for("tpex")).delisted())
                master = build_stock_master(info, twse, tpex)
                listed = master.loc[master["delist_date"].isna(), "stock_id"].tolist()
                check_listed_count(len(listed), self._last_listed_count())
                run.rows = upsert(StockMaster, master, ("stock_id",), UPDATED_FIELDS)
                unlisted = sorted(
                    set(unlisted_codes(info, master)) | set(self._vanished_from_db(listed))
                )
                run.checkpoint = {
                    "listed": len(listed),
                    "delisted": len(master) - len(listed),
                    "unlisted_without_delisting": len(unlisted),
                }
        except (SourceError, RetryableError) as exc:
            raise CommandError(f"{type(exc).__name__}: {exc}") from exc
        self.stdout.write(
            f"stock_master: {len(listed)} listed, {len(master) - len(listed)} delisted "
            f"({run.rows} rows)"
        )
        if unlisted:
            sample = ", ".join(unlisted[:UNLISTED_SAMPLE])
            self.stdout.write(
                f"{len(unlisted)} codes no longer listed and on no delisting list: {sample}"
            )

    @staticmethod
    def _last_listed_count() -> int | None:
        last = (
            EtlRun.objects.filter(
                source="stock_master", dataset="stock_master", status=EtlRun.Status.SUCCESS
            )
            .order_by("-run_id")
            .first()
        )
        count = last.checkpoint.get("listed") if last is not None else None
        return int(count) if count is not None else None

    @staticmethod
    def _vanished_from_db(listed: list[str]) -> list[str]:
        """Codes once loaded as listed that vanished without an official delisting date."""
        rows = StockMaster.objects.filter(delist_date__isnull=True).exclude(stock_id__in=listed)
        return sorted(rows.values_list("stock_id", flat=True))
