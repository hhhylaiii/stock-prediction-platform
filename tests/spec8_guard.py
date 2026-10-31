"""Turn a skipped or xfailed spec8 test into a failure; SPEC §8 tests must actually run."""

from collections.abc import Generator

import pytest
from pluggy import Result


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, Result[pytest.TestReport], None]:
    outcome = yield
    report = outcome.get_result()
    if report.skipped and item.get_closest_marker("spec8") is not None:
        if hasattr(report, "wasxfail"):
            del report.wasxfail
        report.outcome = "failed"
        report.longrepr = f"spec8 test {item.nodeid} was skipped or xfailed; it must run"
