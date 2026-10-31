import subprocess
import sys

import pytest

MODULES = ["apps.etl.client", "apps.etl.raw", "apps.etl.errors", "apps.etl.stock_master"]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_first_in_a_fresh_process(module: str) -> None:
    code = f"import django; django.setup(); import {module}"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env={"DJANGO_SETTINGS_MODULE": "config.settings.test", "PATH": ""},
    )
    assert result.returncode == 0, result.stderr
