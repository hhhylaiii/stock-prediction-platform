"""Rebuild stock_master from FinMind listings and the TWSE and TPEx delisting lists."""

from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from apps.etl.adapters import limiter_for
from apps.etl.adapters.base import SourceError
from apps.etl.adapters.finmind import FinmindClient
from apps.etl.adapters.tpex import TpexClient
from apps.etl.adapters.twse import TwseClient
from apps.etl.ratelimit import RetryableError
from apps.etl.runs import tracked_run
from apps.etl.stock_master import build_stock_master, tpex_delistings, twse_delistings
from apps.etl.upsert import upsert
from apps.market.models import StockMaster

UPDATED_FIELDS = ("name", "market", "industry", "delist_date")


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
                run.rows = upsert(StockMaster, master, ("stock_id",), UPDATED_FIELDS)
        except (SourceError, RetryableError) as exc:
            raise CommandError(f"{type(exc).__name__}: {exc}") from exc
        delisted = int(master["delist_date"].notna().sum())
        self.stdout.write(
            f"stock_master: {len(master) - delisted} listed, {delisted} delisted ({run.rows} rows)"
        )
