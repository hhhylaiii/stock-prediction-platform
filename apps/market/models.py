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


class MonthlyRevenue(models.Model):
    """Monthly revenue in TWD; yoy and mom are fractions (0.1 = 10%), visible from announce_date."""

    pk = models.CompositePrimaryKey("stock_id", "period")
    stock_id = models.CharField(max_length=10)
    period = models.CharField(max_length=7, verbose_name="期間（YYYY-MM）")
    revenue = models.BigIntegerField(verbose_name="營收（元）")
    yoy = models.FloatField(null=True, blank=True)
    mom = models.FloatField(null=True, blank=True)
    announce_date = models.DateField()

    class Meta:
        db_table = "monthly_revenue"
        indexes = [models.Index(fields=["announce_date"], name="monthly_revenue_announce_idx")]
        constraints = [
            models.CheckConstraint(
                condition=models.Q(period__regex=MONTH_PERIOD), name="monthly_revenue_period_format"
            ),
            models.CheckConstraint(condition=finite("yoy", "mom"), name="monthly_revenue_finite"),
        ]

    def __str__(self) -> str:
        return f"{self.stock_id} {self.period}"


class FinType(models.TextChoices):
    CONSOLIDATED = "consolidated", "合併"
    INDIVIDUAL = "individual", "個體"


class FinancialStatement(models.Model):
    """Single-quarter figures in TWD (Q4 = annual minus Q1–Q3); ratios are fractions."""

    pk = models.CompositePrimaryKey("stock_id", "period", "fin_type")
    stock_id = models.CharField(max_length=10)
    period = models.CharField(max_length=7, verbose_name="期間（YYYY-Qn）")
    fin_type = models.CharField(
        max_length=16, choices=FinType.choices, default=FinType.CONSOLIDATED
    )
    eps = models.DecimalField(
        max_digits=12, decimal_places=4, null=True, blank=True, verbose_name="EPS（元）"
    )
    revenue = models.BigIntegerField(null=True, blank=True, verbose_name="營收（元）")
    gross_margin = models.FloatField(null=True, blank=True)
    op_margin = models.FloatField(null=True, blank=True)
    net_income = models.BigIntegerField(null=True, blank=True, verbose_name="淨利（元）")
    roe = models.FloatField(null=True, blank=True)
    debt_ratio = models.FloatField(null=True, blank=True)
    operating_cf = models.BigIntegerField(null=True, blank=True, verbose_name="營業現金流（元）")
    announce_date = models.DateField()

    class Meta:
        db_table = "financial_statement"
        indexes = [models.Index(fields=["announce_date"], name="financial_stmt_announce_idx")]
        constraints = [
            models.CheckConstraint(
                condition=models.Q(period__regex=QUARTER_PERIOD),
                name="financial_statement_period_format",
            ),
            models.CheckConstraint(
                condition=models.Q(fin_type__in=FinType.values),
                name="financial_statement_fin_type_valid",
            ),
            models.CheckConstraint(
                condition=finite("gross_margin", "op_margin", "roe", "debt_ratio"),
                name="financial_statement_finite",
            ),
            models.CheckConstraint(
                condition=non_negative("debt_ratio"), name="financial_statement_debt_ratio_gte_0"
            ),
        ]

    def __str__(self) -> str:
        return f"{self.stock_id} {self.period} {self.fin_type}"


class IndexId(models.TextChoices):
    TAIEX = "TAIEX", "加權指數"
    TPEX = "TPEX", "櫃買指數"
    SOX = "SOX", "費城半導體指數"
    TSM_ADR = "TSM_ADR", "台積電 ADR"
    USDTWD = "USDTWD", "美元兌台幣"


class IndexDaily(models.Model):
    """Daily close of market series; US series keep their own US trading date (SPEC §6)."""

    pk = models.CompositePrimaryKey("index_id", "date")
    index_id = models.CharField(max_length=16, choices=IndexId.choices)
    date = models.DateField()
    close = models.DecimalField(max_digits=16, decimal_places=4)
    volume = models.BigIntegerField(null=True, blank=True)

    class Meta:
        db_table = "index_daily"
        constraints = [
            models.CheckConstraint(
                condition=models.Q(index_id__in=IndexId.values), name="index_daily_index_id_valid"
            ),
            models.CheckConstraint(
                condition=models.Q(close__gt=0), name="index_daily_close_positive"
            ),
            models.CheckConstraint(
                condition=non_negative("volume"), name="index_daily_volume_gte_0"
            ),
        ]

    def __str__(self) -> str:
        return f"{self.index_id} {self.date}"


class MacroIndicator(models.Model):
    """Monthly macro series such as the business-cycle signal, visible from announce_date."""

    pk = models.CompositePrimaryKey("indicator_id", "period")
    indicator_id = models.CharField(max_length=32)
    period = models.CharField(max_length=7, verbose_name="期間（YYYY-MM）")
    value = models.FloatField()
    announce_date = models.DateField()

    class Meta:
        db_table = "macro_indicator"
        indexes = [models.Index(fields=["announce_date"], name="macro_indicator_announce_idx")]
        constraints = [
            models.CheckConstraint(
                condition=models.Q(period__regex=MONTH_PERIOD), name="macro_indicator_period_format"
            ),
            models.CheckConstraint(condition=finite("value"), name="macro_indicator_value_finite"),
        ]

    def __str__(self) -> str:
        return f"{self.indicator_id} {self.period}"


class ShareChangeReason(models.TextChoices):
    UNKNOWN = "unknown", "未知"
    CAPITAL_INCREASE = "capital_increase", "增資"
    CAPITAL_REDUCTION = "capital_reduction", "減資"
    TREASURY_CANCELLATION = "treasury_cancellation", "庫藏股註銷"


class SharesOutstanding(models.Model):
    """Shares outstanding from effective_date until the next row for the same stock."""

    pk = models.CompositePrimaryKey("stock_id", "effective_date")
    stock_id = models.CharField(max_length=10)
    effective_date = models.DateField()
    shares = models.BigIntegerField(verbose_name="流通股數（股）")
    reason = models.CharField(
        max_length=32, choices=ShareChangeReason.choices, default=ShareChangeReason.UNKNOWN
    )

    class Meta:
        db_table = "shares_outstanding"
        constraints = [
            models.CheckConstraint(
                condition=models.Q(shares__gt=0), name="shares_outstanding_shares_positive"
            ),
            models.CheckConstraint(
                condition=models.Q(reason__in=ShareChangeReason.values),
                name="shares_outstanding_reason_valid",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.stock_id} from {self.effective_date}"


class DailyValuation(models.Model):
    """Exchange-published P/E, P/B and dividend yield (a fraction); P/E is blank for losses."""

    pk = models.CompositePrimaryKey("stock_id", "date")
    stock_id = models.CharField(max_length=10)
    date = models.DateField()
    pe = models.FloatField(null=True, blank=True)
    pb = models.FloatField(null=True, blank=True)
    dividend_yield = models.FloatField(null=True, blank=True)

    class Meta:
        db_table = "daily_valuation"
        indexes = [models.Index(fields=["date"], name="daily_valuation_date_idx")]
        constraints = [
            models.CheckConstraint(
                condition=models.Q(pe__gt=0, pb__gt=0) & non_negative("dividend_yield"),
                name="daily_valuation_ranges",
            ),
            models.CheckConstraint(
                condition=finite("pe", "pb", "dividend_yield"), name="daily_valuation_finite"
            ),
        ]

    def __str__(self) -> str:
        return f"{self.stock_id} {self.date}"


class DispositionType(models.TextChoices):
    ATTENTION = "attention", "注意"
    DISPOSITION = "disposition", "處置"
    FULL_DELIVERY = "full_delivery", "全額交割"


class DispositionEvent(models.Model):
    """Attention, disposition or full-delivery period; end_date is blank while still open."""

    pk = models.CompositePrimaryKey("stock_id", "start_date", "type")
    stock_id = models.CharField(max_length=10)
    start_date = models.DateField()
    type = models.CharField(max_length=16, choices=DispositionType.choices)
    end_date = models.DateField(null=True, blank=True)

    class Meta:
        db_table = "disposition_event"
        indexes = [models.Index(fields=["start_date"], name="disposition_start_idx")]
        constraints = [
            models.CheckConstraint(
                condition=models.Q(type__in=DispositionType.values),
                name="disposition_event_type_valid",
            ),
            models.CheckConstraint(
                condition=models.Q(end_date__isnull=True)
                | models.Q(end_date__gte=models.F("start_date")),
                name="disposition_event_end_after_start",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.stock_id} {self.type} from {self.start_date}"
