"""Checks that record a data correction, dated by its run or its data, not a day's quality."""

CORRECTION_CHECKS = frozenset(
    {
        "statement_filed_late",
        "statement_filed_later",
        "statement_restated",
        "statement_overwritten",
        "statement_margin_cleared",
        "statement_viewer_mismatch",
        "fi_holding_zero_replaced",
        "market_history_rebuilt",
    }
)
