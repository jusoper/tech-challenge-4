#!/usr/bin/env python3
"""Valida CSV contra contrato Pandera; fail-fast se violar (Validação A05/A08)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_scoring.utils.config import load_config, project_path  # noqa: E402
from credit_scoring.utils.logging_utils import log_event  # noqa: E402
from credit_scoring.validation.schema import validation_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Valida dataset de crédito (Pandera).")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="CSV a validar (default: dataset de referência).",
    )
    args = parser.parse_args()

    cfg = load_config()
    csv_path = Path(args.path) if args.path else project_path(cfg["dataset"]["reference_path"])
    log_dir = project_path(cfg["monitoring"]["log_dir"])

    if not csv_path.exists():
        log_event(
            "validation",
            "failed",
            message=f"Arquivo não encontrado: {csv_path}",
            log_dir=log_dir,
        )
        return 1

    df = pd.read_csv(csv_path)
    report = validation_report(df)
    print(json.dumps(report, indent=2, default=str))

    if report["passed"]:
        log_event(
            "validation",
            "success",
            message=f"Contrato OK para {csv_path.name}",
            extras={"n_rows": report["n_rows"], "path": str(csv_path)},
            log_dir=log_dir,
        )
        return 0

    log_event(
        "validation",
        "failed",
        message=f"Contrato violado — ingestão bloqueada ({csv_path.name})",
        extras={
            "n_rows": report["n_rows"],
            "n_failures": report.get("n_failures", 0),
            "path": str(csv_path),
        },
        log_dir=log_dir,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
