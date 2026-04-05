"""Recommendation engine: semantic neighbors + category + clusters + citizen journeys."""

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
    path = Path(__file__).parent.parent / "data" / "citizen_journeys.json"
    if not path.exists():
        logger.info("no_journeys_file", path=str(path))
        return {}
    with open(path) as f:
        data = json.load(f)
    return {
        source_id: [(target, reason) for target, reason in links]
        for source_id, links in data["journeys"].items()
    }


CITIZEN_JOURNEYS = _load_citizen_journeys()


class Recommender:

    def __init__(
        self,
        services_map: dict[str, Service],
        vector_index: VectorIndex,
        cluster_index: ClusterIndex,
    ) -> None:
        self._services = services_map
        self._vector_index = vector_index
        self._cluster_index = cluster_index

    def recommend(self, result_ids: list[str], top_k: int | None = None) -> list[RecommendedService]:
        top_k = top_k or CONFIG.rec_max_results
        if not result_ids:
            return []

        exclude = set(result_ids)
        scores: dict[str, float] = {}
        reasons: dict[str, list[str]] = {}

        for seed_id in result_ids[:CONFIG.rec_seed_count]:
            seed_svc = self._services.get(seed_id)
            for neighbor_id, sim_score in self._vector_index.get_neighbors(seed_id, top_k=CONFIG.rec_semantic_neighbors):
                if neighbor_id in exclude:
                    continue

                neighbor_svc = self._services.get(neighbor_id)
                same_category = neighbor_svc and seed_svc and neighbor_svc.tema == seed_svc.tema

                if not same_category and sim_score < CONFIG.rec_cross_category_min_sim:
                    continue

                score = sim_score
                r = [f"similar a '{seed_svc.nome}'"] if seed_svc else []

                if same_category:
                    score += CONFIG.rec_category_boost
                    r.append(f"mesma categoria ({neighbor_svc.tema})")

                if self._cluster_index.same_cluster(seed_id, neighbor_id):
                    score += CONFIG.rec_cluster_boost
                    r.append("mesmo grupo temático")

                if neighbor_id not in scores or score > scores[neighbor_id]:
                    scores[neighbor_id] = score
                    reasons[neighbor_id] = r

            for linked_id, journey_reason in CITIZEN_JOURNEYS.get(seed_id, []):
                if linked_id in exclude or linked_id not in self._services:
                    continue
                base = scores.get(linked_id, CONFIG.rec_cross_category_min_sim)
                journey_score = base + CONFIG.rec_journey_boost
                r = reasons.get(linked_id, [])
                if journey_reason not in r:
                    r = r + [journey_reason]
                if linked_id not in scores or journey_score > scores[linked_id]:
                    scores[linked_id] = journey_score
                    reasons[linked_id] = r

        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        recommendations = []
        for cand_id, score in sorted_candidates[:top_k]:
            service = self._services.get(cand_id)
            if service:
                reason_parts = reasons.get(cand_id, [])
                recommendations.append(RecommendedService(
                    service=service,
                    score=round(score, 4),
                    reason="; ".join(reason_parts) if reason_parts else "serviço relacionado",
                ))

        logger.debug("recommendations_generated", seeds=len(result_ids[:CONFIG.rec_seed_count]), returned=len(recommendations))
        return recommendations
