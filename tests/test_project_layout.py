from __future__ import annotations

from pathlib import Path


def test_required_paths_exist(project_root: Path) -> None:
    required = [
        "pyproject.toml",
        "configs/default.yaml",
        "src/credit_scoring/validation/schema.py",
        "src/credit_scoring/training/baseline.py",
        "src/credit_scoring/drift/detection.py",
        "scripts/generate_credit_data.py",
        "scripts/validate_data.py",
        "scripts/train_baseline.py",
        "scripts/simulate_drift.py",
        "scripts/detect_drift.py",
        "scripts/run_monitoring_pipeline.py",
        "etapas-tech-challenge.txt",
    ]
    for rel in required:
        assert (project_root / rel).exists(), f"missing {rel}"
