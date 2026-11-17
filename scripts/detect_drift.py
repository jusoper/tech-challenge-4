#!/usr/bin/env python3
"""Detecta drift Referência vs Produção com Evidently + PSI/KS (A07/A08)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_scoring.drift.detection import (  # noqa: E402
    build_evidently_report,
    feature_drift_metrics,
    save_drift_summary,
)
from credit_scoring.utils.config import load_config, project_path  # noqa: E402
from credit_scoring.utils.logging_utils import log_event  # noqa: E402


def main() -> int:
    cfg = load_config()
    log_dir = project_path(cfg["monitoring"]["log_dir"])
    ref_path = project_path(cfg["dataset"]["reference_path"])
    prod_path = project_path(cfg["dataset"]["production_path"])
    report_dir = project_path(cfg["drift"]["report_dir"])
    html_path = report_dir / "drift_report.html"
    summary_path = project_path(cfg["model"]["artifact_dir"]) / "drift_summary.json"

    if not ref_path.exists() or not prod_path.exists():
        log_event(
            "detect_drift",
            "failed",
            message="Referência ou produção ausente",
            log_dir=log_dir,
        )
        return 1

    reference = pd.read_csv(ref_path)
    production = pd.read_csv(prod_path)

    summary = feature_drift_metrics(
        reference,
        production,
        psi_threshold=cfg["drift"]["psi_threshold"],
    )
    build_evidently_report(reference, production, html_path)
    save_drift_summary(summary, summary_path)

    print(json.dumps(summary, indent=2))
    log_event(
        "detect_drift",
        "success" if summary["n_drifted"] else "warning",
        message=f"Drift em {summary['n_drifted']} features; HTML={html_path}",
        extras={
            "drifted_features": summary["drifted_features"],
            "report": str(html_path),
        },
        log_dir=log_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
