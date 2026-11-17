# Roteiro do Vídeo STAR — Tech Challenge Fase 4 (≤ 5 min)

## Situation (~45s)

Uma fintech de crédito tem um modelo de **Credit Scoring** em produção. Com inflação e mudança de perfil de clientes, a diretoria suspeita de **degradação silenciosa**: o modelo continua respondendo, mas as decisões pioram sem alerta claro.

Mostrar: contexto de negócio + risco de Data/Concept Drift.

## Task (~45s)

Entregar a camada de **Sustentação e Confiabilidade**:

1. Contratos de qualidade (Pandera) para bloquear dados ruins
2. Simulação e detecção de drift (Evidently + PSI/KS)
3. Observabilidade (MLflow + logs + dashboard)
4. Documentação de governança LGPD e análise causal

## Action (~2 min 30s)

Demonstração prática:

1. **Validação**: rodar contrato no dataset de referência (passa) e no lote inválido (falha / bloqueia ingestão)
2. **Treino**: baseline RandomForest após o gate de qualidade
3. **Drift**: gerar produção com shift em `income` e `employment_years`; abrir relatório Evidently
4. **Observabilidade**: mostrar dashboard (`reports/monitoring/index.html`) e run no MLflow com métricas/artefatos
5. Citar alinhamento às aulas (Pandera A05, Evidently A08, PSI A07, MLflow A06, LGPD Privacy)

## Result (~1 min)

- Relatório aponta features degradadas
- Alertas PSI > 0,25 / KS significativos
- Pipeline auditável (logs JSON + MLflow)
- README com LGPD (PII, base legal, retenção) e causalidade da degradação
- Lição: sem contrato + drift monitoring, o modelo “quebra em silêncio”

## Checklist de gravação

- [ ] Terminal limpo com comandos prontos
- [ ] HTML Evidently e dashboard abertos
- [ ] README na seção de Governança
- [ ] Tempo total ≤ 5 minutos
