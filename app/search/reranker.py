"""Cross-encoder reranker with adaptive RRF-primary blending."""

from __future__ import annotations

import numpy as np
import structlog
from sentence_transformers import CrossEncoder

from app.config import CONFIG
from app.models import RerankResult, RetrievalCandidate, Service

logger = structlog.get_logger()


class Reranker:

    def __init__(self) -> None:
        logger.info("loading_reranker", model=CONFIG.reranker_model)
        self._model = CrossEncoder(CONFIG.reranker_model, max_length=CONFIG.reranker_max_length)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        services_map: dict[str, Service],
        top_k: int | None = None,
        expanded_query: str | None = None,
        ce_weight: float | None = None,
    ) -> list[RerankResult]:
        top_k = top_k or CONFIG.rerank_top_k
        alpha = ce_weight if ce_weight is not None else CONFIG.ce_weight

        if not candidates:
            return []

        ce_query = expanded_query or query

        pairs = []
        valid_candidates = []
        for c in candidates:
            service = services_map.get(c.doc_id)
            if service:
                desc_snippet = service.descricao_completa[:CONFIG.reranker_desc_snippet]
                doc_text = f"{service.nome}. {service.resumo} {desc_snippet}"
                pairs.append((ce_query, doc_text))
                valid_candidates.append(c)

        if not pairs:
            return []

        ce_scores = self._model.predict(pairs, show_progress_bar=False)
        rrf_scores = np.array([c.rrf_score for c in valid_candidates])
        ce_scores = np.array(ce_scores, dtype=np.float64)

        ce_spread = float(ce_scores.max() - ce_scores.min())

        if ce_spread > CONFIG.ce_spread_threshold:
            ce_norm = _min_max_normalize(ce_scores)
            rrf_norm = _min_max_normalize(rrf_scores)
            blended = alpha * ce_norm + (1.0 - alpha) * rrf_norm
        else:
            blended = rrf_scores.copy()

        results = [
            RerankResult(
                doc_id=valid_candidates[i].doc_id,
                rrf_score=valid_candidates[i].rrf_score,
                bm25_score=valid_candidates[i].bm25_score,
                semantic_score=valid_candidates[i].semantic_score,
                blended_score=float(blended_score),
            )
            for i, blended_score in enumerate(blended)
        ]
        results.sort(key=lambda x: x.blended_score, reverse=True)
        return results[:top_k]


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-9:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)
