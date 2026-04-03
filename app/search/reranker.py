"""Cross-encoder reranker for fine-grained relevance scoring."""

from __future__ import annotations

import numpy as np
import structlog
from sentence_transformers import CrossEncoder

from app.config import CONFIG
from app.models import Service

logger = structlog.get_logger()


class Reranker:
    """Multilingual cross-encoder reranker (mMARCO-trained).

    Uses RRF-primary linear blending:
        final_score = CE_WEIGHT * ce_norm + (1 - CE_WEIGHT) * rrf_norm

    The mMARCO cross-encoder was trained on English MS MARCO translations and
    struggles with Portuguese colloquial queries (e.g., ranks "Habite-se" above
    "vítimas de violência" for "fui assaltado"). RRF (BM25+semantic) provides
    the primary signal; the CE nudges ordering when it has discriminative signal
    but cannot override the retrieval ranking.
    """

    CE_WEIGHT = 0.20  # CE fine-tunes, RRF dominates (mMARCO is weak on PT colloquial)

    def __init__(self) -> None:
        logger.info("loading_reranker", model=CONFIG.reranker_model)
        self._model = CrossEncoder(CONFIG.reranker_model, max_length=CONFIG.reranker_max_length)
        logger.info("reranker_loaded")

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float, float | None, float | None]],
        services_map: dict[str, Service],
        top_k: int | None = None,
        expanded_query: str | None = None,
        ce_weight: float | None = None,
    ) -> list[tuple[str, float, float | None, float | None, float]]:
        """Rerank candidates using RRF-primary linear blending.

        Returns (service_id, rrf_score, bm25_score, semantic_score, blended_score) tuples.
        """
        top_k = top_k or CONFIG.rerank_top_k
        alpha = ce_weight if ce_weight is not None else self.CE_WEIGHT

        if not candidates:
            return []

        # Use expanded query for CE input when available — gives mMARCO
        # richer context for colloquial queries
        ce_query = expanded_query or query

        # Build query-document pairs for the cross-encoder
        pairs = []
        valid_candidates = []
        for doc_id, rrf_score, bm25_score, sem_score in candidates:
            service = services_map.get(doc_id)
            if service:
                max_chars = CONFIG.reranker_desc_snippet
                desc_snippet = service.descricao_completa[:max_chars] if service.descricao_completa else ""
                doc_text = f"{service.nome}. {service.resumo} {desc_snippet}"
                pairs.append((ce_query, doc_text))
                valid_candidates.append((doc_id, rrf_score, bm25_score, sem_score))

        if not pairs:
            return []

        # Score with cross-encoder
        ce_scores = self._model.predict(pairs, show_progress_bar=False)

        rrf_scores = np.array([c[1] for c in valid_candidates])
        ce_scores = np.array(ce_scores, dtype=np.float64)

        # Adaptive: blend when CE has discriminative signal (sufficient spread
        # between best and worst scores). The mMARCO cross-encoder often produces
        # negative scores for Portuguese colloquial queries, but the *relative*
        # ordering is still useful. Only fall back to RRF when all CE scores are
        # nearly identical (no discrimination).
        ce_spread = float(ce_scores.max() - ce_scores.min())

        if ce_spread > 0.1:
            # CE-primary linear blending: cross-encoder drives the ranking,
            # RRF acts as tiebreaker.
            ce_norm = self._min_max_normalize(ce_scores)
            rrf_norm = self._min_max_normalize(rrf_scores)
            blended = alpha * ce_norm + (1.0 - alpha) * rrf_norm
        else:
            blended = rrf_scores.copy()

        # Combine and sort by blended score
        results = []
        for i, blended_score in enumerate(blended):
            doc_id, rrf_score, bm25_score, sem_score = valid_candidates[i]
            results.append((doc_id, rrf_score, bm25_score, sem_score, float(blended_score)))

        results.sort(key=lambda x: x[4], reverse=True)

        logger.debug(
            "reranked",
            query=query,
            top_blended=results[0][4] if results else None,
            ce_spread=round(ce_spread, 4),
            ce_active=ce_spread > 0.1,
        )

        return results[:top_k]

    @staticmethod
    def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range.

        When all scores are identical (no discriminative signal), returns zeros
        so the cross-encoder contributes nothing to the blend.
        """
        min_s = scores.min()
        max_s = scores.max()
        if max_s - min_s < 1e-9:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)
