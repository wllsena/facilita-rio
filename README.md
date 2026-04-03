# Facilita Rio — Busca Inteligente de Serviços Públicos

Cidadãos raramente sabem o nome exato do serviço público que precisam. A busca começa com necessidades vagas ("meu cachorro está doente"), incompletas ("IPTU"), ou ambíguas ("problema na rua"). Este sistema resolve dois problemas:

1. **Buscar**: dado texto em linguagem natural, retornar serviços relevantes mesmo com vocabulário divergente entre busca e serviço
2. **Recomendar**: sugerir serviços relacionados que complementem a jornada do cidadão (ex: quem busca maternidade pode precisar de kit enxoval)

Para isso, combina busca por palavras-chave (BM25) e por significado (embeddings E5), com fusão por posição (RRF), reordenação adaptativa (cross-encoder), expansão automática de sinônimos, e recomendações baseadas em jornadas reais do cidadão.

### TL;DR — Resultados-chave

- **nDCG@5**: 0.939 (main, 75 queries) · 0.817 (holdout, 15 queries)
- **MRR@10**: 0.993 · **Recall@10**: 0.946
- **Latência**: p50=56ms, p99=88ms (benchmark 60 medições)
- **73 testes** (pytest + hypothesis), **75 queries** de avaliação + **15 holdout**
- **5 variantes** no ablation study com significância estatística (Fisher's randomization, p<0.05)
- Sistema funciona **100% sem LLM externo** — GPT-4o-mini é enhancement opcional

### Observações sobre o catálogo

50 serviços em PT-BR, 12 temas. Três padrões de nomenclatura afetam a busca:

- **Informacionais** (~8): prefixo "Informações sobre..." cria ambiguidade no BM25 — diferenciação está no conteúdo.
- **Ação/emissão** (~10): termos específicos (IPTU, castração) são precisos para BM25.
- **Atendimento/vistoria** (~12): "atendimento" aparece em saúde, animais E violência — embeddings >0.88 entre contextos diferentes.

Categorias densas (Saúde: 8) permitem recomendações ricas; categorias com 1-2 serviços dependem de recomendações cross-category.

---

## Execução Rápida

```bash
# Docker (recomendado)
docker compose up --build            # http://localhost:8000

# Ou local
pip install -e ".[test]"             # instala app + dependências de teste
uvicorn app.main:app --reload        # http://localhost:8000

# Testes e avaliação
pytest tests/ -v                     # 73 testes
python -m evaluation.evaluate        # ablation study + holdout + latência + significância
python -m evaluation.check_regression # CI-ready: exit 0=ok, 1=regressed

# LLM (opcional — sistema funciona 100% sem)
export OPENAI_API_KEY=sua-chave-aqui
```

### Interface

Autocomplete accent-insensitive; scores por componente visíveis; recomendações com explicação; aviso de baixa confiança para queries fora de escopo:

| Tela inicial com sugestões | Resultados + scores + recomendações |
|---|---|
| ![Home](screenshots/01-home.png) | ![Busca](screenshots/02-search-results.png) |

| Aviso de baixa confiança (fora de escopo) | Detalhe do serviço + recomendações |
|---|---|
| ![Low confidence](screenshots/03-low-confidence.png) | ![Detalhe](screenshots/04-service-detail.png) |

---

## 1. Arquitetura

```
            Busca do Cidadão
                  │
       Normalização + Expansão + Cache
                  │
      ┌───────────┴───────────┐
 BM25+RSLP (top-20)    E5+FAISS (top-20)
      └───────────┬───────────┘
                  │
        Fusão RRF (sem. 2x, lex 1x)
                  │
         Reranker (adaptativo)
                  │
      ┌───────────┴───────────┐
 Resultados + debug    Recomendações + explicações
```

### Componentes

| Componente | Tecnologia | O que faz |
|------------|-----------|-----------|
| **BM25** | rank_bm25 + RSLP + ~80 stopwords PT-BR | Busca por palavras-chave. Stemmer: "vacinação"→"vacin". Nome com 3x peso. |
| **Semântico** | E5-small + FAISS flat | Vetores 384-dim. "meu cachorro está doente" ≈ "atendimento clínico em animais". |
| **Fusão RRF** | Weighted Reciprocal Rank Fusion | Combina rankings por posição (não por score). Semântico 2x, BM25 1x. |
| **Reranker** | Cross-encoder mMARCO | Relê cada par busca+serviço. Ativo só quando confiante. +0.001 nDCG@5 a 50 docs, preparado para 1200. |
| **Recomendação** | Vizinhos semânticos + clusters + jornadas | 4 sinais: similaridade, categoria (+0.20), cluster (+0.10), jornada (+0.15). |
| **Expansão** | 18 padrões, zero-latência | "árvore caiu"→"remoção de árvore", "fome"→"cozinha comunitária". |
| **Cache** | LRU (256 entradas) | Buscas repetidas em <1ms. Em produção: Redis. |
| **LLM** | GPT-4o-mini (opcional) | Expansão + intenção. Fallback gracioso sem API key. |

### Observabilidade

- **Prometheus**: histogramas de latência (total + reranker), contagem de requests (`/metrics`)
- **Logs** (structlog JSON): query, latência, scores, ativação do reranker
- **Health**: `GET /health` — modelos, serviços indexados, startup, LLM
- **Debug na UI**: scores individuais por resultado; recomendações com explicação; `rerank_ms` separado
- **Confiança**: `low_confidence` flag (limiar 0.84) com aviso na interface
- **Regressão**: `check_regression.py` — CI-ready, exit 1 se piorou

---

## 2. Decisões Arquiteturais e Trade-offs

### Por que busca híbrida?

- **BM25 sozinho** falha com vocabulário divergente: "buraco na rua" vs "reparo de deformação na pista"
- **Semântico sozinho** perde siglas exatas (IPTU, ISS, BPC, EJA)
- **Combinando os dois** via RRF, aproveitamos o melhor de cada

Sem stopwords, "minha esposa está grávida" retornava "Habite-se" (match em "minha"). Correções: stopwords PT-BR + RSLP + peso semântico 2x.

### Por que RRF e não soma de scores?

BM25 dá notas de 0-15, semântico de -1 a 1. RRF combina por *posição*, não por score — escalas incompatíveis.

### Reranker: por que manter com impacto marginal?

mMARCO produz scores negativos para buscas em PT coloquial. Solução: **blending adaptativo** (`blended = rrf × (1 + 0.15 × ce_norm)`) — só ajusta quando confiante. Honestidade: +0.001 nDCG@5, ~80ms, ~300MB. Mantido porque a 1200 serviços ganha valor, e o mecanismo se auto-desativa quando incapaz.

### Alternativas rejeitadas

| Alternativa | Razão |
|-------------|-------|
| **Elasticsearch** | Excesso para 50 docs; BM25 em memória basta. |
| **Qdrant/Pinecone** | FAISS busca 50 vetores em <1ms. Banco externo sem benefício. |
| **Embeddings do JSON** (768-dim) | Modelo desconhecido. E5 garante consistência busca↔documento. |
| **`search_content` do JSON** | Redundância + artefatos. Doc próprio dá controle sobre pesos. |
| **RAG / LLM como ranker** | Custo, latência, não-reproduzível. Pipeline determinístico é mensurável. |
| **bge-reranker-v2-m3** | 568M/2.3GB — excessivo. mMARCO (135M) é proporcional. |
| **spaCy** | 200MB+ para o que RSLP (<1MB) resolve. |
| **bm25s** | 4x mais rápido, mas diferença <1ms a 50 docs. Escolha certa a 1200. |

---

## 3. Recomendação

Para os 3 primeiros resultados, o sistema gera recomendações com quatro sinais:

```
score(rec) = similaridade_semântica + bônus_categoria(+0.20) + bônus_cluster(+0.10) + bônus_jornada(+0.15)
```

O protótipo inicial (só similaridade E5) falhava em dois cenários: (1) **ruído cross-category** — "Castração gratuita" aparecia para "Cozinha comunitária" por compartilharem "gratuito". Correção: bônus de categoria + filtro cross-category (similaridade ≥ 0.87). (2) **Relações causais** — maternidade→kit enxoval→Bolsa Família não são semanticamente próximos. Correção: jornadas cidadãs curadas. O cluster (+0.10) funciona como tiebreaker.

### Jornadas do cidadão

12 jornadas curadas capturam relações que similaridade semântica não alcança:

| Jornada | Sequência | Lógica |
|---------|-----------|--------|
| **Gestante** | Maternidade → Kit enxoval → Bolsa Família → Vacinação | Gravidez ao pós-parto |
| **Tributária** | IPTU 2ª via → Débitos → Parcelamento → Certidão negativa | Regularização fiscal |
| **Animal** | Clínica → Castração → SISBICHO | Responsabilidade do tutor |
| **Emprego** | Vagas → PcD → EJA | Qualificação para reinserção |
| **Acolhimento** | Pop. de rua → Cozinha comunitária → Emprego | Necessidades básicas |
| **Escolar** | Matrícula → Merenda → Acompanhamento | Ciclo escolar |
| **Saúde** | Atenção primária → UPA → Vacinação → Insumos | Níveis de atendimento |
| **Imóvel** | Habite-se → Minha Casa → Certidão negativa | Regularização habitacional |

Validação: mini-QREL de 8 buscas com 15 recomendações esperadas — 100% encontradas.

### Exemplo concreto

Busca: **"minha esposa está grávida"** → Resultado #1: Maternidades (0.049), #2: Kit Enxoval (0.048). Recomendações via jornada gestante: Bolsa Família (1.07), Vacinação (1.05) — não apareceriam sem a jornada.

---

## 4. Estratégia de Avaliação

### 4.1 Ground Truth

Sem gabarito fornecido, criamos **75 buscas** em 6 categorias: direta (10), natural (20), sinônimo (10), ambígua (13), extrema (15, typos+gíria), negativa (7, fora de escopo). Relevância gradada: **3**=resolve, **2**=pertinente, **1**=tangencial. Negativas excluídas das métricas.

### 4.2 Ablation Study

| Variante | nDCG@5 | nDCG@10 | P@5 | MRR@10 | R@10 |
|----------|--------|---------|-----|--------|------|
| BM25 only | 0.799 | 0.821 | 0.303 | 0.842 | 0.890 |
| Semantic only | 0.887 | 0.895 | 0.329 | 0.933 | 0.921 |
| Semantic + expansão | 0.908 | 0.916 | 0.329 | 0.965 | 0.921 |
| Hybrid (sem reranker) | 0.939 | 0.944 | 0.344 | 0.993 | 0.946 |
| **Full (hybrid + reranker)** | **0.939** | **0.945** | **0.344** | **0.993** | **0.946** |

68 buscas positivas. Significância via Fisher's randomization (p<0.05). Contribuição isolada: fusão BM25+semântico (+0.031) > expansão (+0.021) > reranker (+0.001). P@5 baixo reflete catálogo: maioria das buscas tem 1-3 relevantes entre 50.

### 4.3 Análise por Categoria

| Categoria | nDCG@5 | MRR@10 | Observação |
|-----------|--------|--------|------------|
| Direta (10) | 0.99 | 1.00 | Trivial |
| Natural (20) | 0.94 | 0.98 | Expansão resolveu "árvore caiu" e "moradia popular" |
| Sinônimo (10) | 0.96 | 1.00 | Expansão resolveu "refeição gratuita" |
| Ambígua (13) | 0.86 | 1.00 | MRR perfeito mas ordenação subsequente difícil |
| Extrema (15) | 0.95 | 1.00 | Typos, gíria |
| Negativa (7) | — | — | Limiar 0.84: 7/7 detectadas, 8/68 falsos alarmes |

### 4.4 Qualidade das Recomendações

| Métrica | Valor | Significado |
|---------|-------|-----------|
| Cobertura | 97.1% | % de buscas com recomendações |
| Coerência categórica | 76.1% | % de recs na mesma categoria dos resultados |
| Taxa de jornadas | 39.7% | % de buscas com link de jornada cidadã |
| Deduplicação | 100% | Nenhuma rec repete resultados |
| Precisão de jornadas | 100% (15/15) | Mini-QREL de 8 buscas com 15 recs esperadas |

A coerência de 76% é intencional — 24% são cross-category por design (ex: emprego → EJA/educação).

### 4.5 Análise de Falhas

7 causas-raiz corrigidas: sem stopwords → "minha esposa" matchava "Habite-se" (+0.08 nDCG@5); sem stemming → "vacinação"≠"vacinas" (+0.02); pesos iguais BM25:semântico (+0.04); reranker sobrescrevia RRF → influência ≤15% (+0.01); reranker com score negativo em PT → desativação automática (+0.003); vocabulário disjunto "árvore caiu" (+0.009); falso positivo por raiz "refeição gratuita"→castração (+0.005).

Tentativas que falharam: "gratuit" como stopword (sem efeito), peso semântico 3x e 2.5x (regressão MRR).

### 4.6 Holdout — Validação de Generalização

As 75 buscas foram tuneadas iterativamente — nDCG@5=0.939 é parcialmente métrica de treino. Mitigação: **15 buscas holdout** criadas após todo ajuste, vocabulário distinto, zero sobreposição (verificada por teste).

| Métrica | Main (75q) | Holdout (15q) | Gap |
|---------|-----------|--------------|-----|
| **nDCG@5** | 0.939 | 0.817 | 0.122 |
| **MRR@10** | 0.993 | 0.867 | 0.126 |
| **Recall@10** | 0.946 | **1.000** | -0.054 |

| Categoria holdout | nDCG@5 | MRR@10 | n |
|-------------------|--------|--------|---|
| Natural | 0.90 | 1.00 | 7 |
| Synonym | 1.00 | 1.00 | 3 |
| Edge | 0.59 | 0.50 | 3 |
| Ambiguous | 0.60 | 0.75 | 2 |

**Padrões recorrentes**: (1) confusão E5 no prefixo "atendimento" (saúde/animais/violência — embeddings não desambiguam); (2) queries genéricas de 1-2 palavras sem signal léxico; (3) duplicatas no catálogo (2x Bolsa Família). Diagnóstico por query: `python -m evaluation.evaluate`.

**Conclusão**: desempenho real estimado em **0.85-0.90 nDCG@5**. Natural (0.90) e sinônimo (1.00) generalizam bem. Gap concentrado em queries genéricas e confusão semântica.

### 4.7 Monitoramento de Regressões

Script CI-ready (`check_regression.py`): compara contra baseline, limiares nDCG@5>0.01 e MRR@10>0.015.

```
Metric          Baseline    Current      Delta     Status
ndcg@5            0.9391     0.9391    +0.0000         ok
mrr@10            0.9926     0.9926    +0.0000         ok
PASSED: no regressions detected
```

---

## 5. Escalabilidade: de 50 para 1200 Serviços

### 5.1 Componente por Componente

| Aspecto | 50 (atual) | 1200 (produção) |
|---------|-----------|-----------------|
| **BM25** | rank_bm25 (~0.1ms) | Elasticsearch PT-BR ou bm25s. |
| **Vetores** | FAISS flat (<1ms) | FAISS IVF-PQ / HNSW ou Qdrant/Weaviate. |
| **Reranker** | Top-20 (~80ms) | Mesmo custo. Ganha valor com mais candidatos. |
| **Embeddings** | Gerados na init (~5s) | Pré-computados em disco, atualização incremental. |
| **Recomendação** | 12 clusters, 12 jornadas | 30-50 clusters, jornadas via co-ocorrência em logs. |
| **Avaliação** | 75 buscas manuais | 200+ buscas + click-through + A/B testing. |
| **Cache** | LRU local (256) | Redis distribuído, TTL 5min. |

O design do pipeline (buscar → fundir → refinar → recomendar) é independente da escala.

### 5.2 Transições

**50 → 200**: BM25 e FAISS flat continuam adequados. Clusters: 20-25. Jornadas manuais: ~30. Criar 50+ queries de avaliação. Sinônimos: ~40-50 padrões.

**200 → 500** (inflexão): bm25s (4x mais rápido). FAISS IVFPQ. Reranker ativo (+0.02-0.04 nDCG@5 esperado). Recalibrar pesos RRF. Jornadas automáticas via logs. Embeddings pré-computados.

**500 → 1200** (produção): Elasticsearch com per-field boosting. Qdrant/Weaviate para atualização incremental. Expansão semiautomática (zero clicks → LLM → revisão humana). Avaliação: 200+ queries + click-through + sintéticas via LLM.

**Atualização de serviços**: diff → embedding incremental + reindex + cluster rebuild + queries canary. <30s incremental.

### 5.3 Estimativas

| Métrica | 50 (medido) | 200 (est.) | 500 (est.) | 1200 (est.) |
|---------|-------------|------------|------------|-------------|
| Memória | ~500MB | ~520MB | ~550MB | ~600MB |
| Latência p50 | **56ms** | ~60ms | ~70ms | ~85ms |
| Latência p99 | **88ms** | ~100ms | ~120ms | ~160ms |
| Startup | ~5s | ~8s | ~15s | ~25s (pré-cache: <3s) |
| Custo/mês | $0 | $30 | $50 | $70-120 |

### 5.4 Operação

- **Alertas**: p99 > 500ms, zero resultados > 5%, reranker inativo > 90%
- **Deploy**: blue-green + sanidade automática (10 buscas QREL, nDCG@5 > 0.85)
- **A/B testing**: split 90/10, MRR implícito por 1 semana
- **Metadata**: monitorar serviços nunca retornados → enriquecimento de conteúdo

---

## 6. Limitações Atuais

- **Viés de anotador único**: todas as 75+15 queries e relevâncias foram criadas por um anotador. Risco: "árvore caiu" como relevância 3 para "remoção" mas 2 para "poda" — outro anotador poderia inverter. Mitigação: holdout + diversidade de categorias. Em produção: múltiplos anotadores + inter-annotator agreement.

- **Queries genéricas de categoria**: "saúde pública", "meio ambiente" — holdout mostra nDCG@5 ~0.60. Sem signal léxico para 1-2 palavras. Em produção: click-through seria essencial.

- **Confusão semântica "atendimento"**: E5-small gera embeddings similares (>0.88) para "atendimento clínico em animais" vs "atendimento a vítimas de violência". Requer encoder maior ou field-level embeddings.

- **Catálogo com duplicatas**: 2x Bolsa Família (IDs distintos) confunde ranking — variante não-anotada sobe para top-5. Pipeline não deduplicou.

- **Reranker pouco calibrado para PT**: mMARCO treinado com traduções. Mecanismo adaptativo mitiga mas não resolve.

- **Jornadas manuais**: 12 jornadas à mão. A 1200 serviços: descoberta automática via logs.

- **Expansão limitada**: 18 padrões (+0.021 nDCG@5). A 1200: LLM complementa.

- **Detecção fora-de-escopo**: gap de 0.059 entre positivas e negativas. Suficiente para aviso, não rejeição.

- **Filtro cross-category**: limiar 0.87 pode ser restritivo para categorias com pouca diversidade.

---

## 7. Diferenciais Implementados

| Diferencial | Detalhe |
|-------------|---------|
| **Ablation study** | 5 variantes + Fisher's randomization (p<0.05) via ranx |
| **Análise de falhas** | 7 causas-raiz corrigidas e medidas. Tentativas falhas documentadas. |
| **Holdout validation** | 15 buscas pós-tuning. Gap reportado honestamente. |
| **Transparência** | Scores por componente na UI. Explicações nas recomendações. |
| **Regressão** | Script CI-ready com limiares configuráveis |
| **Autocomplete** | `/api/suggest` accent-insensitive |
| **LLM gracioso** | 100% funcional sem API key. LLM é enhancement, não dependência. |

---

## Estrutura do Projeto

```
facilita-rio/
├── README.md, pyproject.toml, Dockerfile, docker-compose.yml
├── .env.example                       # OPENAI_API_KEY (opcional)
├── servicos_selecionados.json         # Catálogo de 50 serviços
├── app/
│   ├── main.py                        # FastAPI app, rotas, pipeline, cache
│   ├── config.py, models.py           # Configuração + schemas Pydantic
│   ├── indexing/                       # loader, bm25_index, vector_index, clusters
│   ├── search/                        # pipeline, hybrid RRF, reranker, query_processor
│   ├── recommendation/recommender.py  # 4 sinais + jornadas cidadãs
│   ├── observability/                 # structlog + Prometheus
│   └── templates/                     # Jinja2 + Tailwind CSS
├── evaluation/                        # 75+15 queries, ablation, regression check
└── tests/                             # 73 testes (pytest + hypothesis)
```
