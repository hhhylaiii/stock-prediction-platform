"""Standard-layer tables shared by ETL, research and the API (SPEC §4)."""

from django.db import models

INF = float("inf")
MONTH_PERIOD = r"^[0-9]{4}-(0[1-9]|1[0-2])$"
QUARTER_PERIOD = r"^[0-9]{4}-Q[1-4]$"


def finite(*fields: str) -> models.Q:
    """Each float column is NULL or a finite number; Postgres sorts NaN above Infinity."""
    condition = models.Q()
    for field in fields:
        condition &= models.Q(**{f"{field}__gt": -INF, f"{field}__lt": INF})
    return condition


def non_negative(*fields: str) -> models.Q:
    condition = models.Q()
    for field in fields:
        condition &= models.Q(**{f"{field}__gte": 0})
    return condition


class Market(models.TextChoices):
    TWSE = "TWSE", "上市"
    TPEX = "TPEx", "上櫃"


class StockMaster(models.Model):
    """One row per listed or OTC stock; delisted stocks stay with delist_date set (SPEC §8.3)."""

    stock_id = models.CharField(max_length=10, primary_key=True)
    name = models.CharField(max_length=64)
    market = models.CharField(max_length=4, choices=Market.choices)
    industry = models.CharField(max_length=64, blank=True, default="")
    list_date = models.DateField(null=True, blank=True)
    delist_date = models.DateField(null=True, blank=True)

    class Meta:
        db_table = "stock_master"

    def __str__(self) -> str:
        return f"{self.stock_id} {self.name}"


class TradingCalendar(models.Model):
    """Every calendar day of the loaded range and whether TWSE trades on it (SPEC §8.7)."""

    date = models.DateField(primary_key=True)
    is_trading_day = models.BooleanField()
    note = models.CharField(max_length=64, blank=True, default="")

    class Meta:
        db_table = "trading_calendar"

    def __str__(self) -> str:
        return f"{self.date} {'open' if self.is_trading_day else 'closed'}"


class DailyPrice(models.Model):
    """Unadjusted daily bar; adjusted price = price × adj_factor at read time (SPEC §4)."""

    pk = models.CompositePrimaryKey("stock_id", "date")
    stock_id = models.CharField(max_length=10)
    date = models.DateField()
    open = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True)
    high = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True)
    low = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True)
    close = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True)
    volume = models.BigIntegerField(verbose_name="成交量（股）")
    turnover = models.BigIntegerField(verbose_name="成交金額（元）")
    adj_factor = models.FloatField(default=1.0)

    class Meta:
        db_table = "daily_price"
        indexes = [models.Index(fields=["date"], name="daily_price_date_idx")]
        constraints = [
            models.CheckConstraint(
                condition=models.Q(volume__gte=0), name="daily_price_volume_gte_0"
            ),
            models.CheckConstraint(
                condition=models.Q(turnover__gte=0), name="daily_price_turnover_gte_0"
            ),
            models.CheckConstraint(
                condition=models.Q(adj_factor__gt=0, adj_factor__lt=float("inf")),
                name="daily_price_adj_factor_finite_positive",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.stock_id} {self.date}"


class DividendEvent(models.Model):
    """Ex-right or ex-dividend event; adj_factor = reference price / previous close (SPEC §8.2)."""

    pk = models.CompositePrimaryKey("stock_id", "ex_date")
    stock_id = models.CharField(max_length=10)
    ex_date = models.DateField()
    cash_dividend = models.DecimalField(
        max_digits=16, decimal_places=8, default=0, verbose_name="現金股利（元／股）"
    )
    stock_dividend = models.DecimalField(
        max_digits=16, decimal_places=8, default=0, verbose_name="股票股利（元／股，面額 10 元）"
    )
    adj_factor = models.FloatField(null=True, blank=True)

    class Meta:
        db_table = "dividend_event"
        constraints = [
            models.CheckConstraint(
                condition=non_negative("cash_dividend", "stock_dividend"),
                name="dividend_event_amounts_gte_0",
            ),
            models.CheckConstraint(
                condition=models.Q(adj_factor__isnull=True)
                | models.Q(adj_factor__gt=0, adj_factor__lt=INF),
                name="dividend_event_adj_factor_finite_positive",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.stock_id} ex {self.ex_date}"


class DailyChip(models.Model):
    """Institutional flows, margin and lending in shares; each source fills its own columns."""

    pk = models.CompositePrimaryKey("stock_id", "date")
    stock_id = models.CharField(max_length=10)
    date = models.DateField()
    fi_net = models.BigIntegerField(null=True, blank=True, verbose_name="外資買賣超（股）")
    it_net = models.BigIntegerField(null=True, blank=True, verbose_name="投信買賣超（股）")
    dealer_self_net = models.BigIntegerField(
        null=True, blank=True, verbose_name="自營商自行買賣超（股）"
    )
    dealer_hedge_net = models.BigIntegerField(
        null=True, blank=True, verbose_name="自營商避險買賣超（股）"
    )
    margin_balance = models.BigIntegerField(null=True, blank=True, verbose_name="融資餘額（股）")
    margin_limit = models.BigIntegerField(null=True, blank=True, verbose_name="融資限額（股）")
    short_balance = models.BigIntegerField(null=True, blank=True, verbose_name="融券餘額（股）")
    sbl_short_balance = models.BigIntegerField(
        null=True, blank=True, verbose_name="借券賣出餘額（股）"
    )
    fi_holding_pct = models.FloatField(null=True, blank=True, verbose_name="外資持股比率（%）")
    day_trade_volume = models.BigIntegerField(
        null=True, blank=True, verbose_name="當沖成交股數（股）"
    )

    class Meta:
        db_table = "daily_chip"
        indexes = [models.Index(fields=["date"], name="daily_chip_date_idx")]
        constraints = [
            models.CheckConstraint(
                condition=non_negative(
                    "margin_balance",
                    "margin_limit",
                    "short_balance",
                    "sbl_short_balance",
                    "day_trade_volume",
                ),
                name="daily_chip_balances_gte_0",
            ),
            models.CheckConstraint(
                condition=models.Q(fi_holding_pct__gte=0, fi_holding_pct__lte=100),
                name="daily_chip_fi_holding_pct_range",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.stock_id} {self.date}"


class WeeklyHolding(models.Model):
    """TDCC shareholding distribution; levels 1–15 by size, 16 adjustment, 17 total."""

    pk = models.CompositePrimaryKey("stock_id", "data_date", "level")
    stock_id = models.CharField(max_length=10)
    data_date = models.DateField()
    level = models.PositiveSmallIntegerField()
    holders = models.BigIntegerField(verbose_name="人數")
    shares = models.BigIntegerField(verbose_name="股數（股）")
    pct = models.FloatField(verbose_name="占集保庫存比例（%）")
    publish_date = models.DateField()

    class Meta:
        db_table = "weekly_holding"
        indexes = [models.Index(fields=["publish_date"], name="weekly_holding_publish_idx")]
        constraints = [
            models.CheckConstraint(
                condition=models.Q(level__gte=1, level__lte=17), name="weekly_holding_level_range"
            ),
            models.CheckConstraint(
                condition=models.Q(publish_date__gt=models.F("data_date")),
                name="weekly_holding_published_after_data_date",
            ),
            models.CheckConstraint(
                condition=models.Q(level=16) | non_negative("holders", "shares", "pct"),
                name="weekly_holding_counts_gte_0",
            ),
            models.CheckConstraint(condition=finite("pct"), name="weekly_holding_pct_finite"),
        ]

    def __str__(self) -> str:
        return f"{self.stock_id} {self.data_date} L{self.level}"
