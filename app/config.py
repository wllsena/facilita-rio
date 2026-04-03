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
    confidence_threshold: float = 0.84

    # LLM (optional — system works fully without it)
    openai_model: str = "gpt-4o-mini"
    llm_enabled: bool = field(default_factory=lambda: bool(os.getenv("OPENAI_API_KEY")))
    llm_timeout: float = 10.0
    llm_max_tokens: int = 150


CONFIG = SearchConfig()
