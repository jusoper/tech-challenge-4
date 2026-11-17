from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from credit_scoring.validation.schema import validate_dataframe, validation_report


@pytest.fixture()
def reference_df(project_root: Path) -> pd.DataFrame:
    path = project_root / "data" / "reference" / "credit_reference.csv"
    if not path.exists():
        pytest.skip("Gere os dados antes: poetry run python scripts/generate_credit_data.py")
    return pd.read_csv(path)


@pytest.fixture()
def invalid_df(project_root: Path) -> pd.DataFrame:
    path = project_root / "data" / "invalid" / "credit_invalid.csv"
    if not path.exists():
        pytest.skip("Gere os dados antes: poetry run python scripts/generate_credit_data.py")
    return pd.read_csv(path)


def test_reference_passes_contract(reference_df: pd.DataFrame) -> None:
    validated = validate_dataframe(reference_df)
    assert len(validated) >= 5000
    assert validation_report(reference_df)["passed"] is True


def test_invalid_fails_contract(invalid_df: pd.DataFrame) -> None:
    report = validation_report(invalid_df)
    assert report["passed"] is False
    assert report.get("n_failures", 0) > 0
