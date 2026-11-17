#!/usr/bin/env python3
"""Pipeline completo: validar → treinar → simular drift → detectar → MLflow."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_scoring.monitoring.dashboard import render_monitoring_dashboard  # noqa: E402
from credit_scoring.training.baseline import load_model  # noqa: E402
from credit_scoring.utils.config import load_config, project_path  # noqa: E402
from credit_scoring.utils.logging_utils import log_event  # noqa: E402


def run_script(name: str, *extra_args: str) -> int:
    cmd = [sys.executable, str(ROOT / "scripts" / name), *extra_args]
    return subprocess.call(cmd, cwd=str(ROOT))


def load_jsonl_events(log_dir: Path) -> list[dict]:
    events: list[dict] = []
    if not log_dir.exists():
        return events
    for path in sorted(log_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
    return events


def main() -> int:
    cfg = load_config()
    log_dir = project_path(cfg["monitoring"]["log_dir"])
    tracking_uri = str(project_path(cfg["monitoring"]["tracking_uri"]))
    experiment = cfg["monitoring"]["experiment_name"]

    steps = [
        ("generate_credit_data.py", []),
        ("validate_data.py", []),
        ("train_baseline.py", []),
        ("simulate_drift.py", []),
        ("detect_drift.py", []),
    ]

    for script, args in steps:
        code = run_script(script, *args)
        if code != 0:
            log_event(
                "pipeline",
                "failed",
                message=f"Etapa {script} falhou com código {code}",
                log_dir=log_dir,
            )
            return code

    # Demonstra bloqueio do lote inválido (contrato falhando)
    invalid_code = run_script(
        "validate_data.py",
        "--path",
        str(project_path(cfg["dataset"]["invalid_path"])),
    )
    if invalid_code == 0:
        log_event(
            "pipeline",
            "failed",
            message="Lote inválido passou no contrato — esperado falhar",
            log_dir=log_dir,
        )
        return 3

    metrics_path = project_path(cfg["model"]["artifact_dir"]) / "metrics.json"
    drift_path = project_path(cfg["model"]["artifact_dir"]) / "drift_summary.json"
    model_path = project_path(cfg["model"]["artifact_dir"]) / "baseline_model.joblib"
    html_drift = project_path(cfg["drift"]["report_dir"]) / "drift_report.html"
    dashboard_path = project_path(cfg["monitoring"]["dashboard_path"])

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    drift_summary = json.loads(drift_path.read_text(encoding="utf-8"))
    events = load_jsonl_events(log_dir)
    render_monitoring_dashboard(
        pipeline_events=events,
        metrics=metrics,
        drift_summary=drift_summary,
        output_path=dashboard_path,
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="credit-scoring-monitoring"):
        mlflow.log_param("dataset_n_samples", cfg["dataset"]["n_samples"])
        mlflow.log_param("psi_threshold", cfg["drift"]["psi_threshold"])
        mlflow.log_param("shifted_features", ",".join(cfg["drift"]["shifted_features"]))
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))
        mlflow.log_metric("n_drifted_features", float(drift_summary["n_drifted"]))
        if model_path.exists():
            mlflow.sklearn.log_model(load_model(model_path), artifact_path="model")
        if html_drift.exists():
            mlflow.log_artifact(str(html_drift), artifact_path="drift")
        if dashboard_path.exists():
            mlflow.log_artifact(str(dashboard_path), artifact_path="monitoring")
        mlflow.log_artifact(str(drift_path), artifact_path="drift")

    log_event(
        "pipeline",
        "success",
        message="Pipeline completo com MLflow e dashboard",
        extras={
            "dashboard": str(dashboard_path),
            "experiment": experiment,
            "n_drifted": drift_summary["n_drifted"],
        },
        log_dir=log_dir,
    )
    print(
        json.dumps(
            {
                "dashboard": str(dashboard_path),
                "drift_report": str(html_drift),
                "mlflow_tracking_uri": tracking_uri,
                "experiment": experiment,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
