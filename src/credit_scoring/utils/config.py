"""Carrega configs/default.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or DEFAULT_CONFIG
    with config_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def project_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)
