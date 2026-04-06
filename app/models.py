from __future__ import annotations

from typing import NamedTuple

from pydantic import BaseModel, Field


class RetrievalCandidate(NamedTuple):
    doc_id: str
    rrf_score: float
    bm25_score: float | None
    semantic_score: float | None


class RerankResult(NamedTuple):
    doc_id: str
    rrf_score: float
    bm25_score: float | None
    semantic_score: float | None
    blended_score: float


class Service(BaseModel):
    id: str
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


class SearchResult(BaseModel):
    service: Service
    score: float
    bm25_score: float | None = None
    semantic_score: float | None = None
    match_reason: str | None = None


class RecommendedService(BaseModel):
    service: Service
    score: float
    reason: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=50)


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    recommendations: list[RecommendedService]
    suggested_queries: list[str] = Field(default_factory=list)
    latency_ms: float
    rerank_ms: float = 0.0
    max_semantic_score: float | None = None
    low_confidence: bool = False
    debug: dict | None = None
