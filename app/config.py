"""Application configuration — all tunables in one place."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "servicos_selecionados.json"


@dataclass(frozen=True)
class SearchConfig:
    # Embedding model
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384

    # Cross-encoder reranker
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranker_max_length: int = 512
    reranker_desc_snippet: int = 500

    # Retrieval
    bm25_top_k: int = 20
    semantic_top_k: int = 20
    rrf_k: int = 60
    rerank_top_k: int = 10
    semantic_weight: float = 2.0
    bm25_weight: float = 1.0

    # Reranker blending: final = ce_weight * ce_norm + (1 - ce_weight) * rrf_norm
    ce_weight: float = 0.02
    ce_spread_threshold: float = 0.3

    # Recommendations
    rec_seed_count: int = 3
    rec_semantic_neighbors: int = 8
    rec_category_boost: float = 0.20
    rec_cluster_boost: float = 0.10
    rec_journey_boost: float = 0.30
    rec_cross_category_min_sim: float = 0.87
    rec_max_results: int = 5
    n_clusters: int = 12

    # Out-of-scope detection: flag queries whose best cosine sim is below this
    confidence_threshold: float = 0.83

    # LLM (optional — system works fully without it)
    openai_model: str = "gpt-5.4-mini"
    llm_reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] = "high"
    llm_enabled: bool = field(default_factory=lambda: bool(os.getenv("OPENAI_API_KEY")))
    llm_timeout: float = 10.0
    llm_max_tokens: int = 2048
    llm_system_prompt: str = (
        "You are an assistant that helps users find services in a catalog. "
        "Given a user query, extract:\n"
        "1. The main intent (1 short phrase)\n"
        "2. Expanded search terms in the catalog's language (synonyms, related terms)\n"
        "Reply ONLY in JSON: {\"intent\": \"...\", \"expanded\": \"...\"}"
    )


CONFIG = SearchConfig()
