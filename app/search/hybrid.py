"""Hybrid retrieval: BM25 + semantic search fused with Weighted Reciprocal Rank Fusion."""

from __future__ import annotations

import structlog

from app.config import CONFIG
from app.models import RetrievalCandidate

logger = structlog.get_logger()


def weighted_rrf(
    ranked_lists: list[tuple[list[tuple[str, float]], float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Weighted Reciprocal Rank Fusion.

    Each entry is (ranked_list, weight). The standard RRF score for a doc
    is multiplied by the list's weight, then summed across lists.
    """
    scores: dict[str, float] = {}
    for ranked_list, weight in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight * (1.0 / (k + rank))

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


class HybridRetriever:
    """Combines BM25 and semantic search via Weighted RRF.

    Both BM25 and semantic search use the expanded query (with synonym
    expansions) when available. This ensures BM25 can find services even when
    the citizen's original vocabulary has zero overlap with service names —
    e.g., "vizinho bate na esposa" expands to include "violência doméstica
    vítima atendimento", which BM25 can then match against the violence
    service.

    Initial approach used BM25 on the original query only, but this caused
    BM25 to contribute noise for colloquial queries (matching common words
    in unrelated services) while the correct service only appeared in semantic
    results. Since RRF rewards documents appearing in both lists, the noise
    from BM25 pushed correct results down.

    Semantic results get higher weight (default 2.0 vs 1.0 for BM25) because
    evaluation showed semantic search is far more accurate for natural language
    queries, while BM25 excels at exact keyword matches.
    """

    def __init__(self, bm25_index, vector_index) -> None:
        self._bm25 = bm25_index
        self._vector = vector_index

    def search(
        self,
        query: str,
        expanded_query: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalCandidate]:
        """Return RetrievalCandidate tuples with per-component scores.

        Uses expanded_query for semantic search if provided (LLM enrichment).
        """
        top_k = top_k or CONFIG.rerank_top_k

        # Both retrievers use expanded query when available.
        # Expansion terms help BM25 find services whose names don't overlap
        # with colloquial citizen vocabulary (e.g., "barranco" → "deslizamento").
        retrieval_query = expanded_query or query

        bm25_results = self._bm25.search(retrieval_query, top_k=CONFIG.bm25_top_k)
        semantic_results = self._vector.search(retrieval_query, top_k=CONFIG.semantic_top_k)

        # Build score lookups
        bm25_scores = {doc_id: score for doc_id, score in bm25_results}
        semantic_scores = {doc_id: score for doc_id, score in semantic_results}

        # Weighted RRF: semantic gets 2x the contribution of BM25
        fused = weighted_rrf(
            [
                (bm25_results, CONFIG.bm25_weight),
                (semantic_results, CONFIG.semantic_weight),
            ],
            k=CONFIG.rrf_k,
        )

        # Attach individual scores
        results = []
        for doc_id, rrf_score in fused[:top_k * 2]:  # fetch extra for reranker
            results.append(RetrievalCandidate(
                doc_id=doc_id,
                rrf_score=rrf_score,
                bm25_score=bm25_scores.get(doc_id),
                semantic_score=semantic_scores.get(doc_id),
            ))

        logger.debug(
            "hybrid_search",
            query=query,
            bm25_hits=len(bm25_results),
            semantic_hits=len(semantic_results),
            fused_hits=len(results),
        )

        return results
