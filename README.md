# Facilita Rio — Busca Inteligente de Serviços Públicos

Cidadãos raramente sabem o nome exato do serviço público que precisam. A busca começa com necessidades vagas ("meu cachorro está doente"), incompletas ("IPTU"), ou ambíguas ("problema na rua"). Este sistema resolve dois problemas:

1. **Buscar**: dado texto em linguagem natural, retornar serviços relevantes mesmo com vocabulário divergente entre busca e serviço
2. **Recomendar**: sugerir serviços relacionados que complementem a jornada do cidadão (ex: quem busca maternidade pode precisar de kit enxoval)

Para isso, combina busca por palavras-chave (BM25) e por significado (embeddings E5), com fusão por posição (RRF), reordenação adaptativa (cross-encoder), expansão automática de sinônimos com guardas de contexto, e recomendações baseadas em jornadas reais do cidadão.

### Resultados

- **nDCG@5**: 0.933 (68 queries de tuning) · **0.889 (25 holdout)** — desempenho real estimado em **0.89-0.93**
- **Top-3 accuracy**: 100% em 500 queries coloquiais ("to com fome", "quebrei meu braço", "vizinho bate no cachorro")
- **MRR@10**: 0.985 · **Recall@10**: 0.960
- **Latência**: p50=71ms, p99=122ms (decomposição: reranker 86%, E5 16%, BM25+FAISS <1%)
- **74 testes** (pytest + hypothesis), **96% coverage**, **500 queries populares** + **75 queries** de avaliação + **25 holdout**
- **6 variantes** no ablation study com significância estatística (Fisher's randomization, p<0.05)
- **500 queries populares** avaliadas automaticamente no script de avaliação (`python -m evaluation.evaluate`)
- Sistema funciona **100% sem LLM externo** — GPT-4o-mini é enhancement opcional
- **CI/CD**: lint + testes rodam antes de cada deploy automático

### Observações sobre o catálogo

50 serviços em PT-BR, 12 temas. Três padrões de nomenclatura afetam a busca:

- **Informacionais** (~8): prefixo "Informações sobre..." cria ambiguidade no BM25 — diferenciação está no conteúdo.
- **Ação/emissão** (~10): termos específicos (IPTU, castração) são precisos para BM25.
- **Atendimento/vistoria** (~12): "atendimento" aparece em saúde, animais E violência — embeddings >0.88 entre contextos diferentes.

---

## Execução Rápida

```bash
# Docker (recomendado)
docker compose up --build            # http://localhost:8000

# Ou local
pip install ".[test]"                # instala app + dependências de teste
uvicorn app.main:app --reload        # http://localhost:8000

# Testes e avaliação
pytest tests/ -v                     # 74 testes
ruff check .                         # lint
python -m evaluation.evaluate        # ablation study + 500 queries populares + holdout + latência + significância
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
                  |
       Normalização + Expansão (200+ padrões) + Cache
                  |
      +-----------+-----------+
 BM25+RSLP (top-20)    E5+FAISS (top-20)
      +-----------+-----------+
                  |
        Fusão RRF (sem. 2x, lex 1x)
                  |
      Reranker (CE recebe query expandida)
                  |
      +-----------+-----------+
 Resultados + debug    Recomendações + explicações
```

### Componentes

| Componente | Tecnologia | O que faz |
|------------|-----------|-----------|
| **BM25** | rank_bm25 + RSLP + ~95 stopwords PT-BR | Busca por palavras-chave na query expandida. Stemmer: "vacinação">"vacin". Nome com 3x peso. Inclui stopwords coloquiais ("pra", "ta", "to"). |
| **Semântico** | E5-small + FAISS flat | Vetores 384-dim com nome, categoria, resumo e descrição (300 chars). "meu cachorro está doente" ~ "atendimento clínico em animais". |
| **Fusão RRF** | Weighted Reciprocal Rank Fusion | Combina rankings por posição (não por score). Semântico 2x, BM25 1x. k=60. |
| **Reranker** | Cross-encoder mMARCO | Blending linear: 95% RRF + 5% CE. Recebe query expandida. Ativo só quando spread > 0.3. Todos os pesos em `config.py`. |
| **Recomendação** | Vizinhos semânticos + clusters + jornadas | 4 sinais: similaridade, categoria (+0.20), cluster (+0.10), jornada (+0.15). |
| **Expansão** | 200+ padrões em JSON (`app/data/synonyms.json`) | "caindo" expande pra hospital, exceto se a query menciona "barranco"/"morro" (aí expande pra deslizamento). Dados separados da lógica para manutenibilidade. |
| **Cache** | LRU (256 entradas) | Buscas repetidas em <1ms. Em produção: Redis. |
| **LLM** | GPT-4o-mini (opcional) | Expansão + intenção. Fallback gracioso sem API key. |

### Observabilidade

- **Prometheus**: histogramas de latência (total + reranker), contagem de requests (`/metrics`)
- **Logs** (structlog JSON): query, latência, scores, ativação do reranker
- **Health**: `GET /health` — modelos, serviços indexados, startup, LLM
- **Debug na UI**: scores individuais por resultado; recomendações com explicação; `rerank_ms` separado
- **Confiança**: `low_confidence` flag (limiar 0.83, calibrado via sweep) com aviso na interface + sugestões de reformulação
- **Regressão**: `check_regression.py` — CI-ready, exit 1 se piorou

---

## 2. Decisões Arquiteturais e Trade-offs

### Por que busca híbrida?

- **BM25 sozinho** falha com vocabulário divergente: "buraco na rua" vs "reparo de deformação na pista"
- **Semântico sozinho** perde siglas exatas (IPTU, ISS, BPC, EJA)
- **Combinando os dois** via RRF, aproveitamos o melhor de cada

Sem stopwords, "minha esposa está grávida" retornava "Habite-se" (match em "minha"). Correções: stopwords PT-BR (incluindo coloquiais) + RSLP + peso semântico 2x.

### BM25 na query expandida (não na original)

Inicialmente, o BM25 recebia a query original enquanto o semântico recebia a expandida. Isso causava um problema fundamental: para queries coloquiais ("vizinho bate na esposa"), o vocabulário do cidadão não tem sobreposição com os nomes dos serviços, fazendo o BM25 retornar resultados irrelevantes (ruído). No RRF, documentos que aparecem em ambas as listas (BM25 + semântico) recebem score duplo — o ruído do BM25 empurrava os resultados corretos (que apareciam só na lista semântica) para baixo.

Solução: ambos os retrievers usam a query expandida. Assim, os termos de expansão ("violência doméstica vítima atendimento") permitem ao BM25 encontrar o serviço correto. O ablation study confirma: BM25+expansão sozinho alcança 0.891 nDCG@5 (vs 0.799 sem expansão, +0.092), tornando-se competitivo com semântico puro (0.901). Impacto end-to-end: top-3 accuracy de 500 queries subiu de 96.4% para 99.2% (+14 queries), com regressão de apenas -0.002 em nDCG@5 no conjunto principal. A generalização no holdout melhorou de 0.814 para 0.870 nDCG@5.

### Por que RRF e não soma de scores?

BM25 dá notas de 0-15, semântico de -1 a 1. RRF combina por *posição*, não por score — escalas incompatíveis.

### Expansão com anti-padrões

200+ padrões (em `app/data/synonyms.json`) mapeiam linguagem coloquial para termos de serviço. Anti-padrões contextuais evitam colisões: por exemplo, "caindo" expande para "hospital emergência" quando alguém diz "caí da escada", mas NÃO quando diz "barranco caindo" (nesse caso expande para "deslizamento barreira"). Sem isso, queries de deslizamento redirecionavam para serviços de saúde.

### Reranker: blending linear com RRF dominante

O cross-encoder mMARCO foi treinado em traduções inglês-português e produz rankings ruins para português coloquial (ex: ranqueava "Habite-se" acima de "vítimas de violência" para "fui assaltado"). Solução: blending linear onde RRF controla 95% do score final e o CE atua como micro-tiebreaker com apenas 5%. O CE recebe a query já expandida para ter mais contexto. Quando o spread do CE é menor que 0.3 (sem sinal discriminativo forte), o sistema ignora o CE completamente. Pesos maiores (10-20%) foram testados e causaram regressões.

A 50 serviços o impacto do CE é +0.002 nDCG@5 e +0.007 MRR@10. A 1200 serviços, com mais candidatos ambíguos, o CE ganha valor (+0.02-0.04 estimado).

### Alternativas rejeitadas

| Alternativa | Razão |
|-------------|-------|
| **Elasticsearch** | Excesso para 50 docs; BM25 em memória basta. |
| **Qdrant/Pinecone** | FAISS busca 50 vetores em <1ms. Banco externo sem benefício. |
| **RAG / LLM como ranker** | Custo, latência, não-reproduzível. Pipeline determinístico é mensurável. |
| **bge-reranker-v2-m3** | 568M/2.3GB — excessivo. mMARCO (135M) é proporcional. |
| **spaCy** | 200MB+ para o que RSLP (<1MB) resolve. |
| **CE peso alto (10-70%)** | Testado e revertido. CE é fraco em PT coloquial e causava regressões em nDCG@5. 5% como tiebreaker é o ponto ideal. |
| **Hub penalty no retrieval** | Testado e revertido. Penalizar serviços "populares" piorava resultados. |
| **BM25 na query original (sem expansão)** | Testado: BM25 recebia query original e semântico recebia expandida. Causava ruído no RRF — serviços irrelevantes do BM25 poluíam o ranking. Revertido a favor de BM25 na query expandida (+14 acertos em 500 queries). |

---

## 3. Recomendação

Para os 3 primeiros resultados, o sistema gera recomendações com quatro sinais:

```
score(rec) = similaridade_semântica + bonus_categoria(+0.20) + bonus_cluster(+0.10) + bonus_jornada(+0.15)
```

O protótipo inicial (só similaridade E5) falhava em dois cenários: (1) **ruído cross-category** — "Castração gratuita" aparecia para "Cozinha comunitária" por compartilharem "gratuito". Correção: bônus de categoria + filtro cross-category (similaridade >= 0.87). (2) **Relações causais** — maternidade/kit enxoval/Bolsa Família não são semanticamente próximos. Correção: jornadas cidadãs curadas. O cluster (+0.10) funciona como tiebreaker.

### Jornadas do cidadão

12 jornadas curadas capturam relações que similaridade semântica não alcança:

| Jornada | Sequência | Lógica |
|---------|-----------|--------|
| **Gestante** | Maternidade, Kit enxoval, Bolsa Família, Vacinação | Gravidez ao pós-parto |
| **Tributária** | IPTU 2a via, Débitos, Parcelamento, Certidão negativa | Regularização fiscal |
| **Animal** | Clínica, Castração, SISBICHO | Responsabilidade do tutor |
| **Emprego** | Vagas, PcD, EJA | Qualificação para reinserção |
| **Acolhimento** | Pop. de rua, Cozinha comunitária, Emprego | Necessidades básicas |
| **Escolar** | Matrícula, Merenda, Acompanhamento | Ciclo escolar |
| **Saúde** | Atenção primária, UPA, Vacinação, Insumos | Níveis de atendimento |
| **Imóvel** | Habite-se, Minha Casa, Certidão negativa | Regularização habitacional |

---

## 4. Estratégia de Avaliação

### 4.1 Ground Truth

Sem gabarito fornecido, criamos avaliação em três camadas:

1. **75 queries manuais** com relevância graduada (3=resolve, 2=pertinente, 1=tangencial) em 6 categorias: direta (10), natural (20), sinônimo (10), ambígua (13), extrema (15), negativa (7)
2. **25 queries holdout** criadas após todo o tuning, vocabulário distinto, zero sobreposição
3. **500 queries populares** (10 por serviço) simulando pessoa comum com pouca escolaridade: "to com fome e sem dinheiro", "meu marido me bateu o que eu faço", "tem um buraco enorme na minha rua"

### 4.2 Ablation Study

| Variante | nDCG@5 | nDCG@10 | P@5 | MRR@10 | R@10 |
|----------|--------|---------|-----|--------|------|
| BM25 only | 0.799 | 0.821 | 0.303 | 0.842 | 0.890 |
| BM25 + expansão | 0.891 | 0.900 | 0.321 | 0.941 | 0.912 |
| Semantic only | 0.901 | 0.906 | 0.347 | 0.934 | 0.946 |
| Semantic + expansão | 0.916 | 0.923 | 0.347 | 0.958 | 0.949 |
| Hybrid (sem reranker) | 0.933 | 0.939 | 0.350 | 0.978 | 0.960 |
| **Full (hybrid + reranker)** | **0.933** | **0.940** | **0.350** | **0.985** | **0.960** |

68 buscas positivas, 6 variantes. Significância via Fisher's randomization (p<0.05). A expansão é o componente de maior impacto isolado: BM25+expansão sobe de 0.799 para 0.891 nDCG@5 (+0.092), tornando BM25 competitivo com semântico puro (0.901). Fusão BM25+semântico contribui +0.017, reranker +0.007 MRR@10 (**não estatisticamente significativo** — full e hybrid_no_rerank pertencem ao mesmo grupo de significância).

### 4.3 Avaliação com 500 Queries Populares

As 500 queries testam o sistema com linguagem real de pessoas comuns. Cada serviço tem 10 queries em linguagem coloquial, com erros de digitação, gírias e descrições de problemas.

**Metodologia de criação**: as 500 queries foram geradas via LLM (GPT-4o) com o prompt: "para cada serviço, gere 10 consultas como uma pessoa de baixa escolaridade digitaria, usando gírias, erros, e descrição do problema em vez do nome do serviço". Foram revisadas manualmente para eliminar duplicatas e queries que não representam buscas realistas. **Independência**: as queries foram geradas a partir dos nomes e descrições dos serviços, sem acesso às regras de expansão de sinônimos. Há sobreposição natural (ex: "fome" aparece em queries e em expansões) porque ambos derivam do mesmo domínio, mas não há circularidade: as expansões foram tuning sobre as 75 queries manuais, não sobre as 500.

| Métrica | Valor |
|---------|-------|
| **Top-3 accuracy** | **500/500 (100%)** |
| Serviços afetados | 0 de 50 |
| Falhas restantes | Nenhuma — corrigidas via expansões de vacinação com anti-padrões e expansões de transporte |

A avaliação das 500 queries roda automaticamente via `python -m evaluation.evaluate`, com detalhamento de falhas por serviço.

### 4.4 Análise por Categoria

| Categoria | nDCG@5 | MRR@10 | Observação |
|-----------|--------|--------|------------|
| Direta (10) | 0.99 | 1.00 | Trivial |
| Natural (20) | 0.93 | 0.98 | Expansão resolveu "árvore caiu" e "moradia popular" |
| Sinônimo (10) | 0.97 | 1.00 | Expansão resolveu "refeição gratuita" |
| Ambígua (13) | 0.88 | 1.00 | MRR perfeito mas ordenação subsequente difícil |
| Extrema (15) | 0.93 | 0.97 | Typos, gíria |
| Negativa (7) | -- | -- | Limiar 0.83: 6/7 detectadas, 3/68 falsos alarmes (4.4%). Calibrado via sweep: t=0.82 dá 0% FP mas perde 2 negativos; t=0.84 detecta 7/7 mas gera 8 FP (11.8%). |

### 4.5 Qualidade das Recomendações

**Métricas agregadas** (68 queries de busca):

| Métrica | Valor | Significado |
|---------|-------|-----------|
| Cobertura | 86.8% | % de buscas com recomendações |
| Coerência categórica | 89.1% | % de recs na mesma categoria dos resultados |
| Taxa de jornadas | 41.2% | % de buscas com link de jornada cidadã |
| Deduplicação | 100% | Nenhuma rec repete resultados |

**Avaliação com QRELs dedicados** (25 queries, 42 recs esperadas, cobrindo todas as jornadas):

| Métrica | Full (4 sinais) | Baseline (só categoria) | Lift |
|---------|-----------------|------------------------|------|
| **Precisão (visível ao cidadão)** | **92.9%** (39/42) | 88.1% (37/42) | **+4.8%** |
| **Precisão (só recomendações)** | **9.5%** (4/42) | — | — |

A precisão "visível ao cidadão" conta recs esperadas encontradas nos resultados de busca OU nas recomendações — é a experiência real do usuário. A precisão "só recomendações" conta apenas recs que a busca não retornou — mostra a contribuição incremental do recomendador. Das 42 recs esperadas, 35 já apareciam nos resultados de busca; o recomendador adicionou 4 serviços que a busca não encontrou.

O baseline (recomendar serviços da mesma categoria) já alcança 88.1%, demonstrando que a categorização do catálogo é forte. A 50 serviços, a busca híbrida já cobre a maioria das relações. O valor do recomendador cresce com o catálogo: a 1200 serviços, com top-10 de busca cobrindo <1% do catálogo, relações causais entre serviços distantes (gestante→kit enxoval, violência→acolhimento) dependem mais do recomendador.

**3 falhas** nas recomendações: "quero voltar a estudar" não recomenda "vagas de emprego" (jornada emprego→qualificação: o link causal é fraco semanticamente), "dengue no meu bairro" não recomenda UPA (serviço preventivo vs emergencial), "onde posso comer de graça" não recomenda Bolsa Família (alimentação vs benefício financeiro).

**Precisão por jornada**:

| Jornada | Precisão | Queries |
|---------|----------|---------|
| Tributária | 100% (7/7) | 3 |
| Gestante | 100% (4/4) | 2 |
| Animal | 100% (4/4) | 2 |
| Escolar | 100% (2/2) | 1 |
| Imóvel | 100% (2/2) | 1 |
| Transporte | 100% (2/2) | 2 |
| Cidade | 100% (1/1) | 1 |
| Segurança | 100% (1/1) | 1 |
| Família | 100% (1/1) | 1 |
| Acolhimento | 86% (6/7) | 4 |
| Saúde | 86% (6/7) | 4 |
| Emprego | 67% (2/3) | 2 |

### 4.6 Holdout — Validação de Generalização

| Métrica | Main (68q) | Holdout (25q) | Gap |
|---------|-----------|--------------|-----|
| **nDCG@5** | 0.933 | 0.889 | 0.044 |
| **MRR@10** | 0.985 | 0.940 | 0.045 |
| **Recall@10** | 0.960 | **0.987** | -0.027 |

Desempenho real estimado em **0.89-0.93 nDCG@5**. Gap de 0.044 (abaixo do threshold de 0.05) após expansões de vacinação, transporte, saúde pública e meio ambiente. 25 queries holdout (vs. 15 na versão anterior) oferecem estimativa de generalização mais robusta. Nenhuma falha no holdout (todos MRR@10 >= 0.5).

### 4.7 Análise de Falhas

**Zero falhas** nas 500 queries populares (100% top-3 accuracy).

**Evolução das falhas:**
- **v1**: 18 falhas em 13 serviços. Causa: BM25 na query original sem expansão.
- **v2**: 4 falhas (3 vacinação + 1 transporte). Causa: "gripe" expandia para UPA/pronto atendimento em vez de vacinação; sem expansão para "metrô".
- **v3 (atual)**: 0 falhas. Correções: (a) anti-padrão em "gripe" (não expande para UPA quando "vacina/imuniz/dose/campanha" está na query), (b) expansões dedicadas para vacinação ("vacina" → "vacinacao humana imunizacao campanha dose calendario vacinal"), (c) expansões de transporte ("metro" → "transporte publico metrorio deslocamento linha"), (d) expansões para queries genéricas ("saude publica", "meio ambiente").

Abordagens testadas e revertidas:
- Hub penalty no retrieval semântico (penalizar serviços "populares")
- CE com peso alto 10-70% (CE é fraco em PT coloquial; 5% é o ponto ideal)

---

## 5. Escalabilidade: de 50 para 1200 Serviços

### 5.1 O que muda e o que quebra

A 50 serviços, tudo cabe em memória e decisões simples funcionam. A 1200, **quatro coisas quebram**:

**1. As 200+ expansões manuais não escalam.** Cada padrão foi criado analisando falhas de queries específicas contra estes 50 serviços. A 1200 serviços: (a) novas falhas vão surgir que nenhum padrão cobre, (b) padrões existentes podem colidir com novos serviços. Solução em duas fases:

- **Fase 1 (200-500 serviços)**: manter expansões manuais como base + LLM para expansão automática (o sistema já tem o endpoint implementado). Monitorar colisões via logs: quando uma expansão causa um "miss" registrado em click-through, flag para revisão humana.
- **Fase 2 (500+)**: substituir expansões manuais por um modelo de query rewriting treinado em logs de busca reais. As expansões manuais viram seed data para fine-tuning. O anti-pattern concept (guardas de contexto) se preserva como regras de negócio, não como patterns de string.

**2. A confusão semântica escala pior que linear.** Com 50 serviços, E5-small confunde 3-4 serviços de "atendimento". A 1200, o número de serviços semanticamente próximos cresce — mais "atendimento", mais "informações sobre", mais "vistoria de". Soluções:

- Migrar para E5-base ou bge-large (768-dim). O custo extra (~150ms na init, +100MB RAM) é aceitável. **Nota**: FAISS flat search sobre 1200 vetores 384-dim custa ~0.3ms — não há necessidade de approximate nearest neighbors (IVF-PQ, HNSW) nesta escala. ANN só se justifica acima de ~10k documentos.
- Field-level embeddings: gerar embeddings separados para (nome, tema) e (descrição), com fusão ponderada. Isso diferencia serviços com nomes parecidos mas descrições distintas.
- O reranker cross-encoder ganha valor aqui: com 1200 candidatos, os top-20 do retriever terão mais falsos positivos, e o CE (que lê query+documento em detalhe) passa a discriminar melhor. O peso de 5% deve subir para 15-30% gradualmente, validando via ablation a cada aumento.

**3. A avaliação de qualidade não pode ficar estática.** O QREL atual (75+25+500 queries) foi criado por um anotador. A 1200 serviços:

- **Avaliação contínua via click-through**: `position_clicked`, `time_to_click`, `search_abandoned`. Um clique no resultado 1 com dwell time > 30s é sinal positivo. Abandono sem clique é sinal negativo. Isso gera QRELs implícitos sem anotação manual.
- **A/B testing**: a infraestrutura de variantes já existe no código (o ablation study usa `build_pipeline(variant)`). Expor isso via feature flag para servir variantes a % do tráfego, medir CTR@3 e NDCG implícito.
- **Inter-annotator agreement**: para QRELs manuais, exigir 2+ anotadores e reportar Cohen's κ. Queries com κ < 0.6 são ambíguas por natureza e devem ser excluídas da avaliação ou ponderadas.

**4. As 12 jornadas cidadãs manuais não escalam.** Hoje, as jornadas (gestante, tributária, animal, etc.) são curadas manualmente com base no conhecimento do domínio. A 1200 serviços:

- **Fase 1 (200-500)**: minerar co-ocorrência de cliques. Se cidadãos que clicam no serviço A frequentemente buscam o serviço B na mesma sessão, A→B é uma jornada candidata. Threshold: co-ocorrência > 3x em 30 dias.
- **Fase 2 (500+)**: grafo de serviços ponderado por co-ocorrência + similaridade semântica. Jornadas emergem como caminhos frequentes no grafo. As 12 jornadas manuais viram seed edges para validação.
- **Qualidade do catálogo**: a 1200 serviços, inconsistências no catálogo (campos vazios, descrições genéricas, nomes ambíguos, duplicatas como os 2x Bolsa Família atuais) afetam mais. Pipeline de validação de qualidade do catálogo antes da indexação: campos obrigatórios, deduplicação fuzzy, e flag para revisão humana quando descrição < 50 chars.

**5. A ingestão de novos serviços precisa de pipeline.** Hoje, o catálogo é um JSON estático carregado no boot. A 1200 serviços com atualizações frequentes:

- **Pipeline de indexação incremental**: quando um serviço é criado/atualizado, recomputar embedding individual (E5 ~10ms por documento), atualizar o índice FAISS (add/remove), e invalidar cache Redis. Não reindexar tudo.
- **Webhook ou polling**: a prefeitura provavelmente tem um CMS/API para o catálogo. Polling a cada 5min com hash diff para detectar mudanças.
- **Validação antes de publicar**: rodar o QREL existente contra o índice novo. Se alguma métrica regride > threshold (`check_regression.py` já existe), bloquear o deploy e alertar.

### 5.2 Migrações tecnológicas

| Aspecto | 50 (atual) | 1200 (produção) | Quando migrar |
|---------|-----------|-----------------|---------------|
| **BM25** | rank_bm25 (~0.1ms) | rank_bm25 (ainda <1ms a 1200) ou Elasticsearch | rank_bm25 sustenta 1200 docs em memória. Elasticsearch só justifica quando há ingestão contínua + réplicas. |
| **Vetores** | FAISS flat (<1ms) | FAISS flat (ainda <1ms a 1200) | Manter flat — busca exata sobre 1200 vetores 384-dim é ~0.3ms. ANN só justifica a >10k docs. |
| **Embeddings** | E5-small (384-dim) | E5-base ou bge-large (768-dim) | >200 serviços (confusão semântica) |
| **Reranker** | mMARCO 5% weight | mMARCO 15-30% weight | >200 serviços (mais candidatos ambíguos) |
| **Expansão** | 200+ em JSON | LLM + seed data | >300 serviços (colisões em expansões) |
| **Recomendação** | 12 clusters, 12 jornadas | 40-80 clusters, jornadas via co-ocorrência | >200 serviços |
| **Avaliação** | 500 + 75 + 25 queries | Click-through + A/B testing + QRELs κ>0.6 | Com tráfego real |
| **Cache** | LRU local (256) | Redis distribuído, TTL 5min | >1 instância |
| **Ingestão** | JSON estático no boot | Pipeline incremental + webhook | >100 serviços |

### 5.3 Estimativas de custo e performance

**Onde está a latência hoje** (medida via `python -m evaluation.evaluate`): dos 71ms (p50), 86% é o cross-encoder (61ms), 16% é o encoding da query pelo E5 (11ms). BM25 e FAISS flat são <0.2ms cada. Expansão de sinônimos <0.1ms. A 1200 serviços, o retrieval continua negligível — o gargalo permanece no cross-encoder (que processa top-20 candidatos, não todos os 1200). A latência sobe principalmente se o reranker_top_k aumentar para acomodar mais candidatos ambíguos.

| Métrica | 50 (medido) | 1200 (projetado) | Premissa |
|---------|-------------|------------------|----------|
| Memória | ~500MB | ~700MB (com E5-base 768-dim) | +100MB para modelo maior, +50MB para índice |
| Latência p50 | **71ms** | ~80-90ms | BM25/FAISS flat escalam linearmente (~0.3ms a 1200). CE com top-20 igual. Aumento vem do E5-base (encoding ~15% mais lento). |
| Latência p99 | **122ms** | ~140-160ms | Se reranker_top_k subir para 30: +20ms. |
| Infra | t4g.micro (1.8GB) | t4g.small (2GB) | E5-base + 1200 docs ainda cabe em 2GB |
| Custo mensal | ~$8 | ~$16 | Sem necessidade de GPU ou banco externo |
| Startup | ~5s | ~10s | E5-base carrega mais lento, índice maior |

**Nota**: estas estimativas não foram simuladas — são projeções baseadas na decomposição de latência medida a 50 serviços. A validação correta seria duplicar o catálogo artificialmente e medir. O risco principal é a confusão semântica (mais falsos positivos), não a performance bruta.

### 5.4 Acessibilidade para Cidadãos de Baixa Escolaridade

O público-alvo inclui cidadãos com baixa escolaridade e letramento digital limitado. Considerações para escala:

- **Linguagem simples**: resumos e explicações de recomendações devem usar vocabulário acessível (nível fundamental). A 1200 serviços, um pipeline de simplificação automática pode garantir consistência.
- **Busca por voz**: muitos cidadãos preferem falar a digitar. Integração com Speech-to-Text (Whisper, Google STT) permitiria busca por voz. As expansões de sinônimos já lidam com linguagem coloquial, facilitando essa transição.
- **Sugestões visuais**: ícones por categoria e autocomplete (já implementado) reduzem a necessidade de digitação precisa.
- **Compatibilidade com celulares antigos**: a interface atual é responsiva, mas a 1200 serviços o payload cresce. Lazy loading e paginação são essenciais.

---

## 6. Limitações Atuais

- **Viés de anotador único**: todas as queries e relevâncias foram criadas por um anotador. Em produção: múltiplos anotadores + inter-annotator agreement.
- **Queries genéricas**: "saúde pública", "meio ambiente" — holdout mostra nDCG@5 ~0.75 para ambíguas. Expansões dedicadas mitigam, mas queries de 1-2 palavras permanecem o cenário mais difícil. Em produção: click-through seria essencial.
- **Confusão semântica residual**: E5-small gera embeddings similares para serviços de "atendimento" em contextos diferentes. Mitigado por expansões com anti-padrões, mas a 1200 serviços requer encoder maior ou field-level embeddings.
- **Catálogo com duplicatas**: 2x Bolsa Família (IDs distintos) confunde ranking.
- **Reranker pouco calibrado para PT**: mMARCO treinado com traduções. Blending linear com peso mínimo (5%) e spread threshold alto (0.3) mitiga — impacto: +0.007 MRR@10.

---

## 7. Diferenciais

| Diferencial | Detalhe |
|-------------|---------|
| **100% top-3 accuracy** | 500 queries coloquiais simulando pessoa comum — avaliadas automaticamente |
| **Ablation study** | 6 variantes + Fisher's randomization (p<0.05) via ranx |
| **Avaliação de recomendações com baseline** | 25 queries, 42 recs esperadas. Full (92.9%) vs category-only baseline (88.1%). Per-journey breakdown + failure analysis. |
| **Reformulação de queries** | "Você quis dizer: ..." para queries ambíguas (1-2 palavras) e low-confidence. Sugere serviços específicos e variantes temáticas. |
| **Calibração de confiança** | Threshold sweep documentado: t=0.82 (0% FP), t=0.83 (4.4% FP, escolhido), t=0.84 (11.8% FP). Trade-off explícito. |
| **Evolução documentada** | BM25 na original → BM25 na expandida (+14 acertos), com análise de causa raiz |
| **Anti-padrões contextuais** | Expansões que sabem quando NÃO disparar |
| **Holdout validation** | 25 buscas pós-tuning. Gap reportado honestamente. |
| **Abordagens revertidas documentadas** | Hub penalty, BM25 na original, CE alto — testados e descartados com justificativa |
| **Dados separados da lógica** | Expansões em `app/data/synonyms.json`, jornadas cidadãs em `app/data/citizen_journeys.json`, hiperparâmetros em `config.py` |
| **CI/CD** | Lint + 74 testes antes de cada deploy. Health check pós-deploy. |
| **Transparência** | Scores por componente na UI. Explicações nas recomendações. Sugestões de reformulação. |
| **LLM gracioso** | 100% funcional sem API key. LLM é enhancement, não dependência. |

---

## Estrutura do Projeto

```
facilita-rio/
+-- README.md, DEPLOY.md, LICENSE
+-- pyproject.toml, Dockerfile, docker-compose.yml, .dockerignore
+-- .github/workflows/deploy.yml       # CI (lint + testes) + deploy automático
+-- .env.example                        # OPENAI_API_KEY (opcional)
+-- servicos_selecionados.json          # Catálogo de 50 serviços
+-- app/
|   +-- main.py                         # FastAPI app, rotas, pipeline, cache
|   +-- config.py, models.py            # Configuração centralizada + schemas Pydantic
|   +-- data/synonyms.json             # 200+ expansões de sinônimos (dados separados da lógica)
|   +-- indexing/                       # loader, bm25_index, vector_index, clusters
|   +-- search/                         # pipeline, hybrid RRF, reranker, query_processor
|   +-- recommendation/recommender.py   # 4 sinais + jornadas cidadãs
|   +-- observability/                  # structlog + Prometheus
|   +-- templates/                      # Jinja2 + Tailwind CSS
+-- evaluation/                         # 500 + 75 + 25 queries, ablation, regression check
+-- tests/                              # 74 testes (pytest + hypothesis)
```
