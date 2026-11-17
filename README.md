# Tech Challenge FIAP — Fase 4

Observabilidade contínua, detecção de **Data Drift**, qualidade de dados (**Pandera**) e governança **LGPD** para um modelo de **Credit Scoring** em cenário de fintech.

> Status: **Etapas 1–4 concluídas** (código + relatórios + documentação). Vídeo STAR: gravar com o roteiro em [`docs/roteiro-video-star.md`](docs/roteiro-video-star.md).

Repositório: https://github.com/jusoper/tech-challenge-4

## Contexto

Uma fintech de concessão de crédito suspeita de **degradação silenciosa** do modelo de scoring após mudanças econômicas (inflação, novos perfis de clientes). Este projeto implementa a camada de **Sustentação e Confiabilidade**: contratos de dados, simulação/detecção de drift, observabilidade e documentação de governança.

## Stack (material da Fase 4)

| Camada | Ferramenta | Referência nas aulas |
|--------|------------|----------------------|
| Data contracts | Pandera `DataFrameModel` | Validação A05 / A07 |
| Modelo baseline | scikit-learn RandomForest | Enunciado / Drift A01 |
| Detecção de drift | Evidently `DataDriftPreset` + PSI/KS | Drift A07 / A08 |
| Observabilidade | MLflow + logs JSON + dashboard HTML | Monitoramento A05 / A06 |
| Governança | README LGPD / PbD / causalidade | Privacy A01 / A03 / A04 |

## Arquitetura do pipeline

```
Dataset Referência ──► Pandera Contract ──► Treino baseline
                              │                    │
                              ▼                    ▼
                     lote inválido          Dataset Produção
                     (bloqueado)            (drift simulado)
                                                   │
                     Evidently + PSI/KS ◄──────────┘
                              │
                              ▼
                     MLflow + dashboard HTML + logs JSON
```

## Setup

```bash
poetry install
poetry run python scripts/run_monitoring_pipeline.py
poetry run pytest
```

Passos individuais:

```bash
poetry run python scripts/generate_credit_data.py
poetry run python scripts/validate_data.py
poetry run python scripts/validate_data.py --path data/invalid/credit_invalid.csv  # deve falhar
poetry run python scripts/train_baseline.py
poetry run python scripts/simulate_drift.py
poetry run python scripts/detect_drift.py
```

Artefatos gerados:

- `reports/drift/drift_report.html` — relatório Evidently
- `reports/monitoring/index.html` — dashboard de saúde/alertas
- `models/artifacts/metrics.json` / `drift_summary.json`
- `mlruns/` — experiment tracking local (`mlflow ui --backend-store-uri mlruns`)

## Resultados (execução local)

| Item | Valor |
|------|-------|
| Amostras referência | 6.000 |
| ROC-AUC baseline | ~0.72 |
| Features com drift | `income`, `employment_years`, `debt_ratio` |
| PSI income | ~0.64 (> 0.25 → severo) |
| Contrato no lote inválido | **falha** (nulos, idade &lt; 18, renda negativa, duplicatas) |

## Governança e LGPD (Privacy by Design)

### Dados pessoais e minimização

O dataset de desenvolvimento é **sintético** (Privacy A04 — dados sintéticos como alternativa para inovação com menor risco de PII real). Em produção, o pipeline trataria como dados pessoais/sensíveis financeiros:

- Identificador (`customer_id`) pseudonimizado
- Features de renda, histórico e endividamento com finalidade específica de **análise de crédito**
- Princípio de **minimização**: apenas colunas necessárias ao scoring e ao monitoramento de drift

### Base legal (LGPD)

Para decisão automatizada de crédito, a base típica é **execução de contrato / procedimentos preliminares** (interesse legítimo + transparência), com direito à revisão humana quando aplicável (art. 20 LGPD). O titular deve ser informado sobre critérios gerais do modelo e canais de contestação.

### Retenção

- Dados de treino/referência: retenção limitada ao ciclo de vida do modelo + prazo de auditoria (ex.: 5 anos para obrigações regulatórias de crédito, alinhado à política interna).
- Logs de validação/drift: retenção operacional (ex.: 90–180 dias) com agregação posterior.
- Relatórios Evidently/MLflow: versionados como artefatos de auditoria, sem PII desnecessária.

### Privacy by Default

Contratos Pandera atuam como **gate fail-fast**: dados fora do domínio não entram no treino/serving. Monitoramento contínuo (PSI limiar 0,25 — Drift A07) dispara alerta antes da degradação silenciosa virar prejuízo sistêmico.

### Mitigação de vieses

- Avaliar disparidade de taxa de default/aprovação por faixas demográficas quando disponíveis (aqui: idade).
- Retreinar com amostra atualizada após drift confirmado (concept/covariate — Drift A06).
- Documentar features sensíveis e justificar uso sob necessidade e proporcionalidade.

## Análise causal da degradação

Hipótese de negócio alinhada ao enunciado e à Aula 06 (concept vs covariate/label drift):

1. **Choque macro** (inflação / novos perfis) altera a distribuição de `income` e `employment_years` (covariate drift).
2. O modelo, treinado na distribuição antiga, gera scores desalinhados ao risco real.
3. `debt_ratio` sobe no cenário simulado, amplificando o impacto nas predições.
4. Sem contratos + drift monitoring, a queda de qualidade seria **silenciosa**; com Evidently/PSI/KS + dashboard, o alerta é explícito e auditável no MLflow.

## Estrutura do repositório

```
src/credit_scoring/     # validation, training, drift, monitoring
scripts/                # generate / validate / train / drift / pipeline
data/                   # reference, production, invalid
models/artifacts/       # métricas e sumário de drift
reports/                # Evidently + dashboard
docs/roteiro-video-star.md
disciplinas/*/extracts/ # material das aulas (referência)
etapas-tech-challenge.txt
```

## Critérios de avaliação (checklist)

- [x] Validação de Dados (25%) — Pandera intercepta anomalias
- [x] Detecção de Drift (25%) — simulação + Evidently HTML + PSI/KS
- [x] Observabilidade e Logs (20%) — MLflow + dashboard + JSON logs
- [x] Governança LGPD (15%) — esta seção
- [ ] Vídeo STAR (15%) — gravar com [`docs/roteiro-video-star.md`](docs/roteiro-video-star.md)

## Commits

Orientação do grupo: commit por tarefa/bloco (`chore` / `docs` / `feat` / `refactor` / `test`).
