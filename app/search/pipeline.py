"""Search pipeline: expand → retrieve → rerank → recommend."""

from __future__ import annotations

import time

import structlog
from unidecode import unidecode

from app.config import CONFIG
from app.models import SearchResponse, SearchResult, Service
from app.recommendation.recommender import Recommender
from app.search.hybrid import HybridRetriever
from app.search.query_processor import enrich_query_with_llm, expand_query
from app.search.reranker import Reranker

logger = structlog.get_logger()


class SearchPipeline:

    def __init__(
        self,
        services_map: dict[str, Service],
        retriever: HybridRetriever,
        reranker: Reranker,
        recommender: Recommender,
    ) -> None:
        self._services_map = services_map
        self._retriever = retriever
        self._reranker = reranker
        self._recommender = recommender

    @staticmethod
    def _suggest_queries(
        query: str, results: list[SearchResult], low_confidence: bool,
    ) -> list[str]:
        if not results or (not low_confidence and len(query.split()) > 2):
            return []
        seen: set[str] = set()
        suggestions: list[str] = []
        for r in results[:5]:
            name = r.service.nome
            if unidecode(name.lower()) not in seen:
                seen.add(unidecode(name.lower()))
                suggestions.append(name)
            if len(suggestions) >= 3:
                break
        return suggestions

    @staticmethod
    def _explain_match(service: Service, semantic_score: float | None) -> str:
        parts = []
        if semantic_score and semantic_score >= 0.85:
            parts.append(f"similaridade {semantic_score:.0%}")
        if service.tema:
            parts.append(service.tema)
        return " · ".join(parts) if parts else ""

    async def execute(self, query: str, top_k: int = 10) -> SearchResponse:
        t0 = time.time()
        debug_info: dict = {}

        expanded_query = expand_query(query)
        if expanded_query != query:
            debug_info["expanded_query"] = expanded_query

        intent = ""
        if CONFIG.llm_enabled:
            enrichment = await enrich_query_with_llm(query)
            if enrichment["expanded"] != query:
                expanded_query = enrichment["expanded"]
                debug_info["expanded_query"] = expanded_query
            intent = enrichment["intent"]
            if intent:
                debug_info["intent"] = intent

        candidates = self._retriever.search(query, expanded_query=expanded_query, top_k=top_k * 2)
        max_sem = max((c.semantic_score for c in candidates if c.semantic_score is not None), default=None)

        t_rerank = time.time()
        reranked = self._reranker.rerank(
            query, candidates, self._services_map, top_k=top_k,
            expanded_query=expanded_query,
        )
        rerank_ms = (time.time() - t_rerank) * 1000

        results = []
        for r in reranked:
            service = self._services_map.get(r.doc_id)
            if service:
                results.append(SearchResult(
                    service=service,
                    score=round(r.blended_score, 4),
                    bm25_score=round(r.bm25_score, 4) if r.bm25_score is not None else None,
                    semantic_score=round(r.semantic_score, 4) if r.semantic_score is not None else None,
                    match_reason=self._explain_match(service, r.semantic_score),
                ))

        result_ids = [r.service.id for r in results]
        recommendations = self._recommender.recommend(result_ids)
        latency_ms = (time.time() - t0) * 1000
        low_conf = max_sem is not None and max_sem < CONFIG.confidence_threshold
        suggested = self._suggest_queries(query, results, low_conf)

        response = SearchResponse(
            query=query,
            results=results,
            recommendations=recommendations,
            suggested_queries=suggested,
            latency_ms=round(latency_ms, 1),
            rerank_ms=round(rerank_ms, 1),
            max_semantic_score=round(max_sem, 4) if max_sem is not None else None,
            low_confidence=low_conf,
            debug=debug_info or None,
        )

        logger.info(
            "search_completed",
            query=query, results=len(results), recommendations=len(recommendations),
            latency_ms=round(latency_ms, 1), top_score=results[0].score if results else None,
        )

        return response
