#!/usr/bin/env python3
"""Treina baseline só após validação do contrato (gate de qualidade)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_scoring.training.baseline import save_artifacts, train_baseline  # noqa: E402
from credit_scoring.utils.config import load_config, project_path  # noqa: E402
from credit_scoring.utils.logging_utils import log_event  # noqa: E402
from credit_scoring.validation.schema import validate_dataframe  # noqa: E402


def main() -> int:
    cfg = load_config()
    log_dir = project_path(cfg["monitoring"]["log_dir"])
    ref_path = project_path(cfg["dataset"]["reference_path"])

    if not ref_path.exists():
        log_event("train", "failed", message="Dataset de referência ausente", log_dir=log_dir)
        return 1

    df = pd.read_csv(ref_path)
    try:
        df = validate_dataframe(df)
    except Exception as exc:  # noqa: BLE001 — gate explícito
        log_event(
            "train",
            "failed",
            message=f"Validação bloqueou o treino: {exc}",
            log_dir=log_dir,
        )
        return 2

    model_cfg = cfg["model"]
    model, metrics, _, _ = train_baseline(
        df,
        test_size=model_cfg["test_size"],
        random_seed=model_cfg["random_seed"],
        n_estimators=model_cfg["n_estimators"],
        max_depth=model_cfg["max_depth"],
    )
    paths = save_artifacts(model, metrics, project_path(model_cfg["artifact_dir"]))
    print(
        json.dumps(
            {"metrics": metrics, "artifacts": {k: str(v) for k, v in paths.items()}}, indent=2
        )
    )

    log_event(
        "train",
        "success",
        message="Baseline treinado",
        extras={"roc_auc": metrics["roc_auc"], "f1": metrics["f1"]},
        log_dir=log_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
