"""Search pipeline orchestration — single entry point for the full search flow."""

from __future__ import annotations

import time

import structlog

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
        max_sem = max((c[3] for c in candidates if c[3] is not None), default=None)

        # Reranking
        t_rerank = time.time()
        reranked = self._reranker.rerank(
            query, candidates, self._services_map, top_k=top_k,
            expanded_query=expanded_query,
        )
        rerank_ms = (time.time() - t_rerank) * 1000

        # Build search results
        results = []
        for doc_id, _rrf_score, bm25_score, sem_score, reranker_score in reranked:
            service = self._services_map.get(doc_id)
            if service:
                results.append(
                    SearchResult(
                        service=service,
                        score=round(reranker_score, 4),
                        bm25_score=round(bm25_score, 4) if bm25_score is not None else None,
                        semantic_score=round(sem_score, 4) if sem_score is not None else None,
                        reranker_score=round(reranker_score, 4),
                    )
                )

        # Recommendations
        result_ids = [r.service.id for r in results]
        recommendations = self._recommender.recommend(result_ids)

        latency_ms = (time.time() - t0) * 1000

        # Flag low-confidence results — likely out-of-scope queries
        low_conf = max_sem is not None and max_sem < CONFIG.confidence_threshold

        response = SearchResponse(
            query=query,
            results=results,
            recommendations=recommendations,
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
