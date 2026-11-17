#!/usr/bin/env python3
"""Simula Dataset de Produção com drift em ≥2 features (Data Drift A06)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_scoring.drift.simulation import simulate_production_drift  # noqa: E402
from credit_scoring.training.baseline import load_model  # noqa: E402
from credit_scoring.utils.config import load_config, project_path  # noqa: E402
from credit_scoring.utils.logging_utils import log_event  # noqa: E402
from credit_scoring.validation.schema import FEATURE_COLUMNS  # noqa: E402


def main() -> int:
    cfg = load_config()
    log_dir = project_path(cfg["monitoring"]["log_dir"])
    ref_path = project_path(cfg["dataset"]["reference_path"])
    prod_path = project_path(cfg["dataset"]["production_path"])
    model_path = project_path(cfg["model"]["artifact_dir"]) / "baseline_model.joblib"

    if not ref_path.exists():
        log_event("simulate_drift", "failed", message="Referência ausente", log_dir=log_dir)
        return 1

    reference = pd.read_csv(ref_path)
    drift_cfg = cfg["drift"]
    production = simulate_production_drift(
        reference,
        income_multiplier=drift_cfg["income_multiplier"],
        employment_years_shift=drift_cfg["employment_years_shift"],
        random_seed=cfg["dataset"]["random_seed"],
    )

    if model_path.exists():
        model = load_model(model_path)
        production["prediction"] = model.predict(production[FEATURE_COLUMNS])
        production["prediction_proba"] = model.predict_proba(production[FEATURE_COLUMNS])[:, 1]

    prod_path.parent.mkdir(parents=True, exist_ok=True)
    production.to_csv(prod_path, index=False)

    summary = {
        "path": str(prod_path),
        "n_rows": len(production),
        "shifted_features": drift_cfg["shifted_features"],
        "income_mean_ref": float(reference["income"].mean()),
        "income_mean_prod": float(production["income"].mean()),
        "employment_years_mean_ref": float(reference["employment_years"].mean()),
        "employment_years_mean_prod": float(production["employment_years"].mean()),
    }
    print(json.dumps(summary, indent=2))
    log_event(
        "simulate_drift",
        "success",
        message="Dataset de produção com drift gerado",
        extras=summary,
        log_dir=log_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
