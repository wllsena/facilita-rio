"""Recommendation engine: semantic neighbors + category affinity + cluster membership + citizen journeys."""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from app.config import CONFIG
from app.indexing.cluster_builder import ClusterIndex
from app.indexing.vector_index import VectorIndex
from app.models import RecommendedService, Service

logger = structlog.get_logger()


def _load_citizen_journeys() -> dict[str, list[tuple[str, str]]]:
    """Load citizen journey connections from JSON data file.

    Separates domain data from logic — same principle as synonyms.json.
    """
    path = Path(__file__).parent.parent / "data" / "citizen_journeys.json"
    with open(path) as f:
        data = json.load(f)
    return {
        source_id: [(target, reason) for target, reason in links]
        for source_id, links in data["journeys"].items()
    }


CITIZEN_JOURNEYS = _load_citizen_journeys()
JOURNEY_BOOST = 0.15


class Recommender:
    """Generate service recommendations based on search results."""

    def __init__(
        self,
        services_map: dict[str, Service],
        vector_index: VectorIndex,
        cluster_index: ClusterIndex,
    ) -> None:
        self._services = services_map
        self._vector_index = vector_index
        self._cluster_index = cluster_index

    def recommend(
        self,
        result_ids: list[str],
        top_k: int | None = None,
    ) -> list[RecommendedService]:
        """Generate recommendations based on the top search results.

        Scoring (4 signals):
          - semantic_similarity to search results (primary signal)
          - same tema_geral bonus
          - same semantic cluster bonus
          - citizen journey bonus (hand-curated service connections)
        """
        top_k = top_k or CONFIG.rec_max_results
        if not result_ids:
            return []

        exclude = set(result_ids)
        candidate_scores: dict[str, float] = {}
        candidate_reasons: dict[str, list[str]] = {}

        seed_ids = result_ids[:CONFIG.rec_seed_count]

        for seed_id in seed_ids:
            # Strategy A: semantic neighbors
            neighbors = self._vector_index.get_neighbors(
                seed_id, top_k=CONFIG.rec_semantic_neighbors
            )
            for neighbor_id, sim_score in neighbors:
                if neighbor_id in exclude:
                    continue

                score = sim_score
                reasons = [f"similar a '{self._services[seed_id].nome}'"]

                # Category boost
                neighbor_svc = self._services.get(neighbor_id)
                seed_svc = self._services.get(seed_id)
                same_category = neighbor_svc and seed_svc and neighbor_svc.tema == seed_svc.tema

                # Filter out low-similarity cross-category noise
                if not same_category and sim_score < CONFIG.rec_cross_category_min_sim:
                    continue

                if same_category:
                    score += CONFIG.rec_category_boost
                    reasons.append(f"mesma categoria ({neighbor_svc.tema})")

                # Cluster boost
                if self._cluster_index.same_cluster(seed_id, neighbor_id):
                    score += CONFIG.rec_cluster_boost
                    reasons.append("mesmo grupo temático")

                # Keep best score across seeds
                if neighbor_id not in candidate_scores or score > candidate_scores[neighbor_id]:
                    candidate_scores[neighbor_id] = score
                    candidate_reasons[neighbor_id] = reasons

            # Strategy B: citizen journey connections
            journey_links = CITIZEN_JOURNEYS.get(seed_id, [])
            for linked_id, journey_reason in journey_links:
                if linked_id in exclude or linked_id not in self._services:
                    continue
                journey_score = candidate_scores.get(linked_id, CONFIG.rec_cross_category_min_sim) + JOURNEY_BOOST
                journey_reasons = candidate_reasons.get(linked_id, [])
                if journey_reason not in journey_reasons:
                    journey_reasons = journey_reasons + [journey_reason]
                if linked_id not in candidate_scores or journey_score > candidate_scores[linked_id]:
                    candidate_scores[linked_id] = journey_score
                    candidate_reasons[linked_id] = journey_reasons

        # Sort by score and take top-k
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)

        recommendations = []
        for cand_id, score in sorted_candidates[:top_k]:
            service = self._services.get(cand_id)
            if service:
                reason_parts = candidate_reasons.get(cand_id, [])
                reason_text = "; ".join(reason_parts) if reason_parts else "serviço relacionado"
                recommendations.append(
                    RecommendedService(service=service, score=round(score, 4), reason=reason_text)
                )

        logger.debug(
            "recommendations_generated",
            seed_count=len(seed_ids),
            candidates_evaluated=len(candidate_scores),
            returned=len(recommendations),
        )

        return recommendations
