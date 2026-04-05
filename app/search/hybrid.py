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
    scores: dict[str, float] = {}
    for ranked_list, weight in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight * (1.0 / (k + rank))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:

    def __init__(self, bm25_index, vector_index) -> None:
        self._bm25 = bm25_index
        self._vector = vector_index

    def search(
        self,
        query: str,
        expanded_query: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalCandidate]:
        top_k = top_k or CONFIG.rerank_top_k
        retrieval_query = expanded_query or query

        bm25_results = self._bm25.search(retrieval_query, top_k=CONFIG.bm25_top_k)
        semantic_results = self._vector.search(retrieval_query, top_k=CONFIG.semantic_top_k)

        bm25_scores = dict(bm25_results)
        semantic_scores = dict(semantic_results)

        fused = weighted_rrf(
            [(bm25_results, CONFIG.bm25_weight), (semantic_results, CONFIG.semantic_weight)],
            k=CONFIG.rrf_k,
        )

        return [
            RetrievalCandidate(
                doc_id=doc_id, rrf_score=rrf_score,
                bm25_score=bm25_scores.get(doc_id), semantic_score=semantic_scores.get(doc_id),
            )
            for doc_id, rrf_score in fused[:top_k * 2]
        ]
