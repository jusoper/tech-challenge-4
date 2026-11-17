"""Detecção de drift com Evidently + PSI/KS (Data Drift A07/A08)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from scipy import stats

from credit_scoring.validation.schema import FEATURE_COLUMNS


def calculate_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index — limiar de mercado: >0.25 = severo (Aula 07)."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    quantiles = np.linspace(0, 100, bins + 1)
    breakpoints = np.unique(np.percentile(ref, quantiles))
    if len(breakpoints) < 3:
        breakpoints = np.linspace(ref.min(), ref.max() + 1e-9, bins + 1)

    ref_counts = np.histogram(ref, bins=breakpoints)[0].astype(float)
    cur_counts = np.histogram(cur, bins=breakpoints)[0].astype(float)

    ref_perc = ref_counts / ref_counts.sum()
    cur_perc = cur_counts / cur_counts.sum()
    # evita log(0)
    ref_perc = np.where(ref_perc == 0, 1e-6, ref_perc)
    cur_perc = np.where(cur_perc == 0, 1e-6, cur_perc)
    return float(np.sum((cur_perc - ref_perc) * np.log(cur_perc / ref_perc)))


def ks_test(reference: np.ndarray, current: np.ndarray) -> dict[str, float]:
    statistic, p_value = stats.ks_2samp(reference, current)
    return {"statistic": float(statistic), "p_value": float(p_value)}


def feature_drift_metrics(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    features: list[str] | None = None,
    psi_threshold: float = 0.25,
) -> dict[str, Any]:
    features = features or FEATURE_COLUMNS
    per_feature: dict[str, Any] = {}
    drifted: list[str] = []

    for col in features:
        psi = calculate_psi(reference[col].to_numpy(), production[col].to_numpy())
        ks = ks_test(reference[col].to_numpy(), production[col].to_numpy())
        is_drift = psi > psi_threshold or ks["p_value"] < 0.05
        per_feature[col] = {
            "psi": psi,
            "ks_statistic": ks["statistic"],
            "ks_p_value": ks["p_value"],
            "drift_detected": is_drift,
        }
        if is_drift:
            drifted.append(col)

    return {
        "psi_threshold": psi_threshold,
        "drifted_features": drifted,
        "n_drifted": len(drifted),
        "features": per_feature,
    }


def build_evidently_report(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    report_path: Path,
) -> Path:
    """Gera HTML com DataDriftPreset (Evidently — Aula 08)."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # Evidently 0.7+ API
    report = Report([DataDriftPreset()], include_tests=True)
    result = report.run(
        reference_data=reference[FEATURE_COLUMNS],
        current_data=production[FEATURE_COLUMNS],
    )
    result.save_html(str(report_path))
    return report_path


def save_drift_summary(summary: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
