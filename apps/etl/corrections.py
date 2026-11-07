"""Checks that log one row per stock to record a data correction rather than a day's quality."""

CORRECTION_CHECKS = frozenset(
    {
        "statement_filed_late",
        "statement_filed_later",
        "statement_restated",
        "statement_overwritten",
        "statement_margin_cleared",
        "statement_viewer_mismatch",
        "fi_holding_zero_replaced",
    }
)
