from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from credit_scoring.drift.detection import feature_drift_metrics
from credit_scoring.drift.simulation import simulate_production_drift


def test_simulate_and_detect_drift(project_root: Path) -> None:
    path = project_root / "data" / "reference" / "credit_reference.csv"
    if not path.exists():
        pytest.skip("Gere os dados antes")
    reference = pd.read_csv(path)
    production = simulate_production_drift(reference)
    summary = feature_drift_metrics(reference, production, psi_threshold=0.25)
    assert (
        "income" in summary["drifted_features"] or "employment_years" in summary["drifted_features"]
    )
    assert summary["n_drifted"] >= 2
