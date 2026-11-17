"""Logs estruturados JSON (Monitoramento A05 — logs/métricas/traces)."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def get_logger(name: str = "credit_scoring") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def log_event(
    stage: str,
    status: str,
    *,
    message: str = "",
    extras: dict[str, Any] | None = None,
    log_dir: Path | None = None,
) -> dict[str, Any]:
    """Emite e opcionalmente persiste um evento JSON do pipeline."""
    event: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "status": status,
        "message": message,
    }
    if extras:
        event.update(extras)

    get_logger().info(json.dumps(event, ensure_ascii=False))

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        out = log_dir / f"{stage}.jsonl"
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    return event
