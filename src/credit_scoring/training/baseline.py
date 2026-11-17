"""Treino do classificador baseline de credit scoring (scikit-learn)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from credit_scoring.validation.schema import FEATURE_COLUMNS, TARGET_COLUMN


def train_baseline(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_seed: int = 42,
    n_estimators: int = 100,
    max_depth: int = 8,
) -> tuple[RandomForestClassifier, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    proba = model.predict_proba(x_test)[:, 1]
    pred = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
    }
    return model, metrics, x_test, y_test.to_frame(name=TARGET_COLUMN)


def save_artifacts(
    model: RandomForestClassifier,
    metrics: dict[str, Any],
    artifact_dir: Path,
) -> dict[str, Path]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "baseline_model.joblib"
    metrics_path = artifact_dir / "metrics.json"
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"model": model_path, "metrics": metrics_path}


def load_model(model_path: Path) -> RandomForestClassifier:
    return joblib.load(model_path)
