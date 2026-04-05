"""Search pipeline: expand → retrieve → rerank → recommend."""

from __future__ import annotations

import time

import structlog
from unidecode import unidecode

from app.config import CONFIG
from app.indexing.bm25_index import PT_STOPWORDS
from app.models import RerankResult, SearchResponse, SearchResult, Service
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
        if not results:
            return []

        query_lower = unidecode(query.lower().strip())
        query_words = set(query_lower.split())
        is_short = len(query_words) <= 2

        if not low_confidence and not is_short:
            return []

        suggestions: list[str] = []
        seen: set[str] = set()

        for r in results[:5]:
            name_norm = unidecode(r.service.nome.lower().strip())
            if len(query_words & set(name_norm.split())) >= len(query_words) and not low_confidence:
                continue
            if name_norm not in seen:
                suggestions.append(r.service.nome)
                seen.add(name_norm)
            if is_short and r.service.tema:
                cat_suggestion = f"{query} ({r.service.tema.lower()})"
                cat_norm = unidecode(cat_suggestion.lower())
                if cat_norm not in seen:
                    suggestions.append(cat_suggestion)
                    seen.add(cat_norm)
            if len(suggestions) >= 3:
                break

        return suggestions[:3]

    @staticmethod
    def _explain_match(r: RerankResult, service: Service, query: str) -> str:
        query_words = set(unidecode(query.lower()).split())
        name_words = set(unidecode(service.nome.lower()).split())

        parts = []
        overlap = query_words & name_words - PT_STOPWORDS
        if overlap:
            parts.append(f"nome contém '{' '.join(sorted(overlap))}'")

        semantic = r.semantic_score or 0
        if semantic >= 0.90:
            parts.append(f"similaridade semântica alta ({semantic:.0%})")
        elif semantic >= 0.85:
            parts.append(f"similaridade semântica ({semantic:.0%})")

        if (r.bm25_score or 0) > 0 and not overlap:
            parts.append("palavras-chave encontradas na descrição")

        if service.tema:
            parts.append(f"categoria: {service.tema}")

        return "; ".join(parts) if parts else "serviço relacionado à busca"

    async def execute(self, query: str, top_k: int = 10) -> SearchResponse:
        t0 = time.time()
        debug_info: dict = {}

        expanded_query = expand_query(query)

        intent = ""
        if CONFIG.llm_enabled:
            enrichment = await enrich_query_with_llm(query)
            if enrichment["expanded"] != query:
                expanded_query = enrichment["expanded"]
            intent = enrichment["intent"]
            if intent:
                debug_info["intent"] = intent
                debug_info["expanded_query"] = expanded_query

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
                    reranker_score=round(r.blended_score, 4),
                    match_reason=self._explain_match(r, service, query),
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
