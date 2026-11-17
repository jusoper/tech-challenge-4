"""Simula Data/Covariate Drift deslocando features (Data Drift A06)."""

from __future__ import annotations

import pandas as pd


def simulate_production_drift(
    reference: pd.DataFrame,
    *,
    income_multiplier: float = 1.45,
    employment_years_shift: float = -2.5,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Gera dataset de produção com mudança econômica simulada.

    - income ↑ (inflação / novos perfis com renda declarada maior)
    - employment_years ↓ (mais clientes recentes / informalidade)
    """
    production = reference.copy()
    rng = pd.Series(range(len(production)), dtype="float64")
    # ruído leve para não ficar deslocamento determinístico puro
    noise = (rng * random_seed % 97) / 97.0 - 0.5

    production["income"] = (production["income"] * income_multiplier + noise * 500).clip(
        lower=0, upper=500_000
    )
    production["employment_years"] = (
        production["employment_years"] + employment_years_shift + noise * 0.3
    ).clip(lower=0, upper=50)

    # debt_ratio sobe levemente com o cenário macro (efeito colateral)
    production["debt_ratio"] = (production["debt_ratio"] * 1.15).clip(upper=5)

    return production
