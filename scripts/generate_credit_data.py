#!/usr/bin/env python3
"""Gera dataset sintético de crédito ≥5k + lote inválido (Validação A01 / Privacy A04)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_scoring.utils.config import load_config, project_path  # noqa: E402


def generate_reference(n_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 71, size=n_samples)
    income = rng.lognormal(mean=8.5, sigma=0.45, size=n_samples).clip(800, 200_000)
    employment_years = rng.uniform(0, 35, size=n_samples)
    loan_amount = rng.uniform(2_000, 120_000, size=n_samples)
    credit_history_score = rng.normal(650, 80, size=n_samples).clip(300, 900)
    num_credit_lines = rng.integers(0, 15, size=n_samples)
    debt_ratio = (loan_amount / np.maximum(income, 1)).clip(0.05, 3.5)

    # Probabilidade de default correlacionada a risco
    logit = (
        -2.2
        + 0.015 * (age - 40)
        - 0.00002 * income
        - 0.06 * employment_years
        + 0.00001 * loan_amount
        - 0.004 * credit_history_score
        + 0.8 * debt_ratio
        + 0.05 * num_credit_lines
    )
    prob = 1 / (1 + np.exp(-logit))
    default = (rng.random(n_samples) < prob).astype(int)

    return pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(n_samples)],
            "age": age.astype(int),
            "income": np.round(income, 2),
            "employment_years": np.round(employment_years, 2),
            "loan_amount": np.round(loan_amount, 2),
            "credit_history_score": np.round(credit_history_score, 2),
            "num_credit_lines": num_credit_lines.astype(int),
            "debt_ratio": np.round(debt_ratio, 4),
            "default": default,
        }
    )


def generate_invalid(reference: pd.DataFrame) -> pd.DataFrame:
    """Lote com violações: nulos, idade < 18, renda negativa, duplicatas."""
    invalid = reference.head(200).copy()
    invalid.loc[invalid.index[0:10], "age"] = 16
    invalid.loc[invalid.index[10:20], "income"] = -1500.0
    invalid.loc[invalid.index[20:30], "income"] = np.nan
    invalid.loc[invalid.index[30:40], "loan_amount"] = np.nan
    # duplicatas de customer_id
    invalid.loc[invalid.index[40:50], "customer_id"] = invalid.loc[invalid.index[0], "customer_id"]
    return invalid


def main() -> int:
    cfg = load_config()
    ds = cfg["dataset"]
    reference = generate_reference(ds["n_samples"], ds["random_seed"])
    invalid = generate_invalid(reference)

    ref_path = project_path(ds["reference_path"])
    inv_path = project_path(ds["invalid_path"])
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    inv_path.parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(ref_path, index=False)
    invalid.to_csv(inv_path, index=False)

    print(f"Referência: {ref_path} ({len(reference)} linhas)")
    print(f"Inválido:   {inv_path} ({len(invalid)} linhas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
