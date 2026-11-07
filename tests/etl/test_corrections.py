from apps.etl import cumulative_margins, filings, mops_history, statement_overwrite, zero_holdings
from apps.etl.corrections import CORRECTION_CHECKS


def test_every_correction_check_is_registered() -> None:
    written = {
        filings.FILED_LATE_CHECK,
        filings.FILED_LATER_CHECK,
        filings.RESTATED_CHECK,
        statement_overwrite.OVERWRITE_CHECK,
        cumulative_margins.MARGIN_CHECK,
        mops_history.MISMATCH_CHECKS[mops_history.STATEMENT_VIEWER],
        zero_holdings.REPLACED_CHECK,
        "market_history_rebuilt",
    }

    assert written == CORRECTION_CHECKS
