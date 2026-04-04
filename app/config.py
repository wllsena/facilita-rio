"""Application configuration — all tunables in one place."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "servicos_selecionados.json"


@dataclass(frozen=True)
class SearchConfig:
    # Embedding model (multilingual, retrieval-optimized)
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384

    # Reranker (multilingual, trained on mMARCO)
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranker_max_length: int = 512
    reranker_desc_snippet: int = 500  # chars of descricao_completa for reranker input

    # Retrieval parameters
    bm25_top_k: int = 20
    semantic_top_k: int = 20
    rrf_k: int = 60  # RRF constant
    rerank_top_k: int = 10  # final results after reranking
    # RRF weights — semantic gets 2x BM25 in rank fusion.
    # Justified by ablation: semantic alone (0.901 nDCG@5) > BM25+expansion (0.891).
    # Sweep on 68 queries: 1:1→0.924, 1.5:1→0.929, 2:1→0.933, 3:1→0.931.
    # 2:1 is the sweet spot — higher suppresses BM25's exact-match advantage on siglas (IPTU, ISS).
    semantic_weight: float = 2.0
    bm25_weight: float = 1.0

    # Reranker blending: final_score = ce_weight * ce_norm + (1 - ce_weight) * rrf_norm
    # mMARCO CE is weak on PT colloquial — 5% as micro-tiebreaker only (10-70% caused regressions)
    ce_weight: float = 0.05
    ce_spread_threshold: float = 0.3  # minimum CE spread to activate blending

    # Recommendation parameters
    rec_seed_count: int = 3  # top-N results used as seed for recommendations
    rec_semantic_neighbors: int = 8
    rec_category_boost: float = 0.20
    rec_cluster_boost: float = 0.10
    rec_cross_category_min_sim: float = 0.87  # minimum cosine sim for cross-category recs (above median in 50-service corpus)
    rec_max_results: int = 5
    n_clusters: int = 12  # agglomerative clustering (tighter clusters reduce noise)

    # Confidence threshold — when the best semantic match is below this value,
    # the query is likely out-of-scope. Flags the response as low_confidence.
    # Calibrated via sweep: t=0.83 catches 8/9 negatives with 6% FP on 83 positive queries.
    # t=0.84 catches 9/9 but has 14.5% FP; t=0.82 has 0% FP but misses 2/9 negatives.
    # The 5 FP at t=0.83 are genuinely ambiguous queries ("cultura e lazer", "problema na rua")
    # where showing suggestions is helpful rather than harmful.
    confidence_threshold: float = 0.83

    # LLM (optional — system works fully without it)
    openai_model: str = "gpt-4o-mini"
    llm_enabled: bool = field(default_factory=lambda: bool(os.getenv("OPENAI_API_KEY")))
    llm_timeout: float = 10.0
    llm_max_tokens: int = 150


CONFIG = SearchConfig()
