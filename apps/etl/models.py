"""Raw source responses and ETL run bookkeeping (SPEC §4, §5)."""

from django.db import models
from django.utils import timezone


class RawRecord(models.Model):
    """One unmodified source response, kept so normalize can rerun without refetching."""

    dataset = models.CharField(max_length=64)
    request_params = models.JSONField(default=dict)
    payload = models.JSONField()
    fetched_at = models.DateTimeField(default=timezone.now)
    source_hash = models.CharField(max_length=64)

    class Meta:
        abstract = True

    def __str__(self) -> str:
        return f"{self.dataset} {self.request_params}"


class RawTwse(RawRecord):
    class Meta:
        db_table = "raw_twse"
        constraints = [
            models.UniqueConstraint(
                fields=["dataset", "source_hash"], name="raw_twse_dataset_hash_uniq"
            )
        ]
        indexes = [
            models.Index(fields=["dataset", "fetched_at"], name="raw_twse_dataset_fetched_idx")
        ]


class RawFinmind(RawRecord):
    class Meta:
        db_table = "raw_finmind"
        constraints = [
            models.UniqueConstraint(
                fields=["dataset", "source_hash"], name="raw_finmind_dataset_hash_uniq"
            )
        ]
        indexes = [
            models.Index(fields=["dataset", "fetched_at"], name="raw_finmind_ds_fetched_idx")
        ]


class RawTpex(RawRecord):
    class Meta:
        db_table = "raw_tpex"
        constraints = [
            models.UniqueConstraint(
                fields=["dataset", "source_hash"], name="raw_tpex_dataset_hash_uniq"
            )
        ]
        indexes = [
            models.Index(fields=["dataset", "fetched_at"], name="raw_tpex_dataset_fetched_idx")
        ]


class RawTdcc(RawRecord):
    class Meta:
        db_table = "raw_tdcc"
        constraints = [
            models.UniqueConstraint(
                fields=["dataset", "source_hash"], name="raw_tdcc_dataset_hash_uniq"
            )
        ]
        indexes = [
            models.Index(fields=["dataset", "fetched_at"], name="raw_tdcc_dataset_fetched_idx")
        ]


class RawYahoo(RawRecord):
    class Meta:
        db_table = "raw_yahoo"
        constraints = [
            models.UniqueConstraint(
                fields=["dataset", "source_hash"], name="raw_yahoo_dataset_hash_uniq"
            )
        ]
        indexes = [
            models.Index(fields=["dataset", "fetched_at"], name="raw_yahoo_dataset_fetched_idx")
        ]


class RawDatagov(RawRecord):
    class Meta:
        db_table = "raw_datagov"
        constraints = [
            models.UniqueConstraint(
                fields=["dataset", "source_hash"], name="raw_datagov_dataset_hash_uniq"
            )
        ]
        indexes = [
            models.Index(fields=["dataset", "fetched_at"], name="raw_datagov_ds_fetched_idx")
        ]


class RawMops(RawRecord):
    class Meta:
        db_table = "raw_mops"
        constraints = [
            models.UniqueConstraint(
                fields=["dataset", "source_hash"], name="raw_mops_dataset_hash_uniq"
            )
        ]
        indexes = [models.Index(fields=["dataset", "fetched_at"], name="raw_mops_ds_fetched_idx")]


class EtlRun(models.Model):
    """One ETL execution with its outcome and a checkpoint for resuming."""

    class Status(models.TextChoices):
        RUNNING = "running"
        SUCCESS = "success"
        FAILED = "failed"

    run_id = models.BigAutoField(primary_key=True)
    source = models.CharField(max_length=32)
    dataset = models.CharField(max_length=64)
    start_date = models.DateField()
    end_date = models.DateField()
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.RUNNING)
    rows = models.IntegerField(default=0)
    error = models.TextField(blank=True, default="")
    checkpoint = models.JSONField(default=dict)
    started_at = models.DateTimeField(default=timezone.now)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "etl_run"
        indexes = [
            models.Index(
                fields=["source", "dataset", "start_date"], name="etl_run_source_dataset_idx"
            )
        ]

    def __str__(self) -> str:
        return f"{self.source}/{self.dataset} {self.start_date}..{self.end_date} {self.status}"


class DataQualityLog(models.Model):
    """One quality-check result for a day; any error stops that day's predictions (SPEC §5)."""

    class Severity(models.TextChoices):
        INFO = "info"
        WARNING = "warning"
        ERROR = "error"

    id = models.BigAutoField(primary_key=True)
    date = models.DateField()
    check_name = models.CharField(max_length=64)
    severity = models.CharField(max_length=16, choices=Severity.choices)
    detail = models.JSONField(default=dict)

    class Meta:
        db_table = "data_quality_log"
        indexes = [models.Index(fields=["date", "severity"], name="dq_log_date_severity_idx")]
        constraints = [
            models.CheckConstraint(
                condition=models.Q(severity__in=["info", "warning", "error"]),
                name="data_quality_log_severity_valid",
            )
        ]

    def __str__(self) -> str:
        return f"{self.date} {self.check_name} {self.severity}"
