"""Contrato de dados com Pandera DataFrameModel (Validação A05 — schema-as-code)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series


class CreditApplicationSchema(pa.DataFrameModel):
    """Contrato rígido para applications de crédito.

    Regras alinhadas às dimensões de qualidade (A01) e ao hands-on financeiro (A05):
    - Completude: campos obrigatórios sem nulos
    - Validade: ranges de idade, renda e valor do empréstimo
    - Unicidade: customer_id único
    """

    customer_id: Series[str] = pa.Field(nullable=False, unique=True)
    age: Series[int] = pa.Field(ge=18, le=90, nullable=False)
    income: Series[float] = pa.Field(ge=0, le=500_000, nullable=False)
    employment_years: Series[float] = pa.Field(ge=0, le=50, nullable=False)
    loan_amount: Series[float] = pa.Field(ge=500, le=500_000, nullable=False)
    credit_history_score: Series[float] = pa.Field(ge=0, le=1000, nullable=False)
    num_credit_lines: Series[int] = pa.Field(ge=0, le=30, nullable=False)
    debt_ratio: Series[float] = pa.Field(ge=0, le=5, nullable=False)
    default: Series[int] = pa.Field(isin=[0, 1], nullable=False)

    class Config:
        strict = True
        coerce = True


FEATURE_COLUMNS = [
    "age",
    "income",
    "employment_years",
    "loan_amount",
    "credit_history_score",
    "num_credit_lines",
    "debt_ratio",
]

TARGET_COLUMN = "default"


def validate_dataframe(df: pd.DataFrame, lazy: bool = True) -> pd.DataFrame:
    """Valida o DataFrame; em falha levanta SchemaErrors (fail-fast do contrato)."""
    return CreditApplicationSchema.validate(df, lazy=lazy)


def validation_report(df: pd.DataFrame) -> dict[str, Any]:
    """Retorna resumo pass/fail sem interromper o caller (para logs/dashboard)."""
    try:
        validate_dataframe(df, lazy=True)
        return {"passed": True, "n_rows": len(df), "errors": []}
    except pa.errors.SchemaErrors as exc:
        failures = exc.failure_cases.to_dict(orient="records")
        return {
            "passed": False,
            "n_rows": len(df),
            "n_failures": len(failures),
            "errors": failures[:50],
        }
