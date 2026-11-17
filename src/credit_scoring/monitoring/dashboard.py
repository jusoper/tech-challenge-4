"""Gera dashboard HTML simples de saúde do pipeline (Monitoramento A01/A05)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_monitoring_dashboard(
    *,
    pipeline_events: list[dict[str, Any]],
    metrics: dict[str, Any] | None,
    drift_summary: dict[str, Any] | None,
    output_path: Path,
) -> Path:
    metrics = metrics or {}
    drift_summary = drift_summary or {}
    drifted = drift_summary.get("drifted_features", [])
    psi_threshold = drift_summary.get("psi_threshold", 0.25)

    rows = []
    for event in pipeline_events:
        rows.append(
            "<tr>"
            f"<td>{event.get('timestamp', '')}</td>"
            f"<td>{event.get('stage', '')}</td>"
            f"<td class='{event.get('status', '')}'>{event.get('status', '')}</td>"
            f"<td>{event.get('message', '')}</td>"
            "</tr>"
        )

    feature_rows = []
    for name, info in (drift_summary.get("features") or {}).items():
        flag = "ALERTA" if info.get("drift_detected") else "OK"
        feature_rows.append(
            "<tr>"
            f"<td>{name}</td>"
            f"<td>{info.get('psi', 0):.4f}</td>"
            f"<td>{info.get('ks_p_value', 1):.4g}</td>"
            f"<td class='{'failed' if info.get('drift_detected') else 'success'}'>{flag}</td>"
            "</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <title>Monitoramento — Credit Scoring Fase 4</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 2rem; background: #f7f4ef; color: #1c1a17; }}
    h1, h2 {{ font-weight: 600; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; background: #fff; }}
    th, td {{ border: 1px solid #d9d2c5; padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: #ebe4d7; }}
    .success {{ color: #1b6b3a; font-weight: 600; }}
    .failed, .ALERTA {{ color: #9b1c1c; font-weight: 600; }}
    .card {{
      background: #fff; border: 1px solid #d9d2c5;
      padding: 1rem 1.25rem; margin-bottom: 1.5rem;
    }}
    code {{ background: #efe8dc; padding: 0.1rem 0.35rem; }}
  </style>
</head>
<body>
  <h1>Dashboard de Observabilidade — Credit Scoring</h1>
  <p>
    Pipeline de validação, treino e detecção de drift (Fase 4).
    Limiar PSI = <code>{psi_threshold}</code>.
  </p>

  <div class="card">
    <h2>Saúde do modelo (baseline)</h2>
    <ul>
      <li>ROC-AUC: <strong>{metrics.get("roc_auc", "—")}</strong></li>
      <li>F1: <strong>{metrics.get("f1", "—")}</strong></li>
      <li>Accuracy: <strong>{metrics.get("accuracy", "—")}</strong></li>
      <li>Amostras treino/teste: {metrics.get("n_train", "—")} / {metrics.get("n_test", "—")}</li>
    </ul>
  </div>

  <div class="card">
    <h2>Alertas de drift</h2>
    <p>Features com drift: <strong>{", ".join(drifted) if drifted else "nenhuma"}</strong>
       ({drift_summary.get("n_drifted", 0)} alertas)</p>
    <table>
      <thead><tr><th>Feature</th><th>PSI</th><th>KS p-value</th><th>Status</th></tr></thead>
      <tbody>
        {"".join(feature_rows) or "<tr><td colspan='4'>Sem métricas de drift.</td></tr>"}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>Eventos do pipeline (logs)</h2>
    <table>
      <thead><tr><th>Timestamp</th><th>Stage</th><th>Status</th><th>Message</th></tr></thead>
      <tbody>
        {"".join(rows) or "<tr><td colspan='4'>Sem eventos.</td></tr>"}
      </tbody>
    </table>
  </div>

  <p><small>Gerado automaticamente — alinhado a Monitoramento A05/A06 e Data Drift A07.</small></p>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    # também salva snapshot JSON para testes
    snapshot = {
        "metrics": metrics,
        "drift_summary": drift_summary,
        "n_events": len(pipeline_events),
    }
    output_path.with_suffix(".json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return output_path
