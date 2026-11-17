from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from credit_scoring.training.baseline import save_artifacts, train_baseline
from credit_scoring.validation.schema import validate_dataframe


def test_train_baseline_produces_metrics(project_root: Path, tmp_path: Path) -> None:
    path = project_root / "data" / "reference" / "credit_reference.csv"
    if not path.exists():
        pytest.skip("Gere os dados antes")
    df = validate_dataframe(pd.read_csv(path))
    model, metrics, _, _ = train_baseline(df, n_estimators=20, max_depth=4)
    assert "roc_auc" in metrics
    assert metrics["roc_auc"] > 0.5
    paths = save_artifacts(model, metrics, tmp_path)
    assert paths["model"].exists()
    assert paths["metrics"].exists()
