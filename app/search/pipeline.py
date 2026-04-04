"""Search pipeline orchestration — single entry point for the full search flow."""

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
    """Encapsulates the full search flow: normalize → expand → retrieve → rerank → recommend.

    Extracted from main.py to improve testability and separation of concerns.
    The pipeline is stateless per-request; all state lives in the injected components.
    """

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
        query: str,
        results: list[SearchResult],
        low_confidence: bool,
    ) -> list[str]:
        """Generate query reformulation suggestions based on top results.

        Suggests more specific queries when:
        - The query is short/ambiguous (1-2 words) and top results have distinct names
        - The query has low confidence (out-of-scope detection triggered)
        - The top result name offers a more precise formulation

        Returns up to 3 suggested queries, deduplicated and different from the original.
        """
        if not results:
            return []

        query_lower = unidecode(query.lower().strip())
        query_words = set(query_lower.split())
        suggestions: list[str] = []
        seen_normalized: set[str] = set()

        # For low-confidence queries, suggest based on top results (the system's best guesses)
        # For short/ambiguous queries, suggest more specific alternatives
        is_short = len(query_words) <= 2

        if not low_confidence and not is_short:
            return []

        for r in results[:5]:
            # Use the service name as a suggested query
            name = r.service.nome
            name_normalized = unidecode(name.lower().strip())

            # Skip if too similar to original query
            name_words = set(name_normalized.split())
            overlap = len(query_words & name_words)
            if overlap >= len(query_words) and not low_confidence:
                continue

            if name_normalized not in seen_normalized:
                suggestions.append(name)
                seen_normalized.add(name_normalized)

            # Also suggest theme-based reformulation for ambiguous queries
            if is_short and r.service.tema:
                tema_suggestion = f"{query} ({r.service.tema.lower()})"
                tema_norm = unidecode(tema_suggestion.lower())
                if tema_norm not in seen_normalized:
                    suggestions.append(tema_suggestion)
                    seen_normalized.add(tema_norm)

            if len(suggestions) >= 3:
                break

        return suggestions[:3]

    async def execute(self, query: str, top_k: int = 10) -> SearchResponse:
        """Run the full search pipeline.

        Expects a pre-normalized query (normalize_query is called by the
        caller for cache-key consistency). Skips redundant normalization.
        """
        t0 = time.time()
        debug_info: dict = {}

        # Local synonym expansion (always active, zero-cost)
        expanded_query = expand_query(query)

        # Optional LLM enrichment (supplements local expansion)
        intent = ""
        if CONFIG.llm_enabled:
            enrichment = await enrich_query_with_llm(query)
            if enrichment["expanded"] != query:
                expanded_query = enrichment["expanded"]
            intent = enrichment["intent"]
            if intent:
                debug_info["intent"] = intent
                debug_info["expanded_query"] = expanded_query

        # Hybrid retrieval
        candidates = self._retriever.search(query, expanded_query=expanded_query, top_k=top_k * 2)

        # Max cosine similarity — confidence signal for out-of-scope detection
        max_sem = max((c.semantic_score for c in candidates if c.semantic_score is not None), default=None)

        # Reranking
        t_rerank = time.time()
        reranked = self._reranker.rerank(
            query, candidates, self._services_map, top_k=top_k,
            expanded_query=expanded_query,
        )
        rerank_ms = (time.time() - t_rerank) * 1000

        # Build search results
        results = []
        for r in reranked:
            service = self._services_map.get(r.doc_id)
            if service:
                results.append(
                    SearchResult(
                        service=service,
                        score=round(r.blended_score, 4),
                        bm25_score=round(r.bm25_score, 4) if r.bm25_score is not None else None,
                        semantic_score=round(r.semantic_score, 4) if r.semantic_score is not None else None,
                        reranker_score=round(r.blended_score, 4),
                    )
                )

        # Recommendations
        result_ids = [r.service.id for r in results]
        recommendations = self._recommender.recommend(result_ids)

        latency_ms = (time.time() - t0) * 1000

        # Flag low-confidence results �� likely out-of-scope queries
        low_conf = max_sem is not None and max_sem < CONFIG.confidence_threshold

        # Query reformulation suggestions for ambiguous/low-confidence queries
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
            query=query,
            results=len(results),
            recommendations=len(recommendations),
            latency_ms=round(latency_ms, 1),
            rerank_ms=round(rerank_ms, 1),
            top_score=results[0].score if results else None,
        )

        return response
