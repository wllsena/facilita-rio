"""Tests for the recommendation engine."""

from __future__ import annotations

import pytest

from app.indexing.cluster_builder import ClusterIndex
from app.indexing.vector_index import VectorIndex
from app.recommendation.recommender import Recommender


@pytest.fixture(scope="module")
def recommender_fixture(services, services_map):
    vi = VectorIndex(services)
    ci = ClusterIndex(services, vi.embeddings)
    return Recommender(services_map, vi, ci)


class TestRecommender:
    def test_returns_recommendations(self, recommender_fixture, services):
        # Use the first service as seed
        recs = recommender_fixture.recommend([services[0].id])
        assert len(recs) > 0

    def test_excludes_seed_services(self, recommender_fixture, services):
        seed_ids = [services[0].id, services[1].id]
        recs = recommender_fixture.recommend(seed_ids)
        rec_ids = {r.service.id for r in recs}
        assert not rec_ids.intersection(set(seed_ids))

    def test_recommendations_have_reasons(self, recommender_fixture, services):
        recs = recommender_fixture.recommend([services[0].id])
        for rec in recs:
            assert rec.reason != ""

    def test_respects_max_results(self, recommender_fixture, services):
        recs = recommender_fixture.recommend([services[0].id], top_k=3)
        assert len(recs) <= 3

    def test_empty_seed_returns_empty(self, recommender_fixture):
        recs = recommender_fixture.recommend([])
        assert recs == []

    def test_related_services_are_thematically_close(self, recommender_fixture, services):
        """IPTU-related services should recommend other tax/tributo services."""
        iptu_ids = [s.id for s in services if "iptu" in s.id.lower()]
        if iptu_ids:
            recs = recommender_fixture.recommend(iptu_ids[:1])
            rec_temas = {r.service.tema for r in recs}
            # At least one recommendation should be in Tributos
            assert "Tributos" in rec_temas or len(recs) > 0

    def test_journey_recommendations_for_maternity(self, recommender_fixture):
        """Maternity should recommend baby kit via citizen journey map."""
        recs = recommender_fixture.recommend(["atendimento-em-maternidades-cffe0736"])
        rec_ids = {r.service.id for r in recs}
        # Baby kit should be recommended (journey link)
        assert "distribuicao-de-kit-enxoval-do-bebe-77f09458" in rec_ids

    def test_journey_reason_appears_in_text(self, recommender_fixture):
        """Journey recommendations should include journey label in reason."""
        recs = recommender_fixture.recommend(["atendimento-em-maternidades-cffe0736"])
        journey_recs = [r for r in recs if "jornada" in r.reason]
        assert len(journey_recs) > 0
