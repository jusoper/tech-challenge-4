from __future__ import annotations

import json
from pathlib import Path

import pytest

from credit_scoring.monitoring.dashboard import render_monitoring_dashboard


def test_dashboard_and_drift_artifacts(project_root: Path, tmp_path: Path) -> None:
    metrics_path = project_root / "models" / "artifacts" / "metrics.json"
    drift_path = project_root / "models" / "artifacts" / "drift_summary.json"
    html_drift = project_root / "reports" / "drift" / "drift_report.html"
    if not metrics_path.exists() or not drift_path.exists():
        pytest.skip("Rode o pipeline antes: poetry run python scripts/run_monitoring_pipeline.py")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    drift = json.loads(drift_path.read_text(encoding="utf-8"))
    assert drift["n_drifted"] >= 2
    assert html_drift.exists()

    out = render_monitoring_dashboard(
        pipeline_events=[{"timestamp": "t", "stage": "test", "status": "success", "message": "ok"}],
        metrics=metrics,
        drift_summary=drift,
        output_path=tmp_path / "index.html",
    )
    assert out.exists()
    assert "PSI" in out.read_text(encoding="utf-8")


def test_mlruns_experiment_exists(project_root: Path) -> None:
    mlruns = project_root / "mlruns"
    if not mlruns.exists():
        pytest.skip("MLflow ainda não rodou")
    # experiment dirs are numeric under mlruns/
    assert any(mlruns.iterdir()), "esperado pelo menos um experiment MLflow"
