"""Standard-layer tables shared by ETL, research and the API (SPEC §4)."""

from django.db import models


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
    """Unadjusted daily bar; adjusted price = price × adj_factor at read time (SPEC §4).

    OHLC come from the regular session; volume and turnover also count odd-lot, after-hours
    fixed-price and block trades, as in TWSE MI_INDEX and STOCK_DAY.
    """

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
