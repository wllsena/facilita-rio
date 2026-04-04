"""Pydantic schemas for API request/response and internal data."""

from __future__ import annotations

from typing import NamedTuple

from pydantic import BaseModel, Field


class RetrievalCandidate(NamedTuple):
    """A candidate from hybrid retrieval (BM25 + semantic via RRF)."""

    doc_id: str
    rrf_score: float
    bm25_score: float | None
    semantic_score: float | None


class RerankResult(NamedTuple):
    """A reranked result with all component scores."""

    doc_id: str
    rrf_score: float
    bm25_score: float | None
    semantic_score: float | None
    blended_score: float


class Service(BaseModel):
    """Parsed public service from the catalog."""

    id: str  # slug
    nome: str
    resumo: str
    descricao_completa: str
    tema: str
    orgao_gestor: list[str]
    custo: str
    publico: list[str]
    tempo_atendimento: str
    instrucoes: str
    resultado: str
    search_content: str


class SearchResult(BaseModel):
    """A single search result with per-component scores for explainability."""

    service: Service
    score: float
    bm25_score: float | None = None
    semantic_score: float | None = None
    reranker_score: float | None = None


class RecommendedService(BaseModel):
    """A recommended service with a human-readable reason."""

    service: Service
    score: float
    reason: str


class SearchRequest(BaseModel):
    """POST body for the search API."""

    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=50)


class SearchResponse(BaseModel):
    """Full search response including results, recommendations, and debug info."""

    query: str
    results: list[SearchResult]
    recommendations: list[RecommendedService]
    suggested_queries: list[str] = Field(default_factory=list)
    latency_ms: float
    rerank_ms: float = 0.0
    max_semantic_score: float | None = None
    low_confidence: bool = False
    debug: dict | None = None
