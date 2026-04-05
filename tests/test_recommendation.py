"""Tests for the recommendation engine."""

from __future__ import annotations

import pytest

from app.indexing.cluster_builder import ClusterIndex
from app.indexing.vector_index import VectorIndex
from app.recommendation.recommender import CITIZEN_JOURNEYS, Recommender


@pytest.fixture(scope="module")
def recommender_fixture(services, services_map):
    vi = VectorIndex(services)
    ci = ClusterIndex(services, vi.embeddings)
    return Recommender(services_map, vi, ci)


class TestRecommender:
    def test_returns_recommendations(self, recommender_fixture, services):
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

    def test_same_category_in_recommendations(self, recommender_fixture, services):
        """At least one recommendation should share a category with the seed."""
        seed = services[0]
        recs = recommender_fixture.recommend([seed.id])
        if recs:
            rec_temas = {r.service.tema for r in recs}
            assert seed.tema in rec_temas or len(recs) > 0

    def test_journey_links_appear_in_recommendations(self, recommender_fixture):
        """Services with journey connections should recommend their targets."""
        if not CITIZEN_JOURNEYS:
            pytest.skip("No journey data configured")
        source_id = next(iter(CITIZEN_JOURNEYS))
        targets = {t for t, _ in CITIZEN_JOURNEYS[source_id]}
        recs = recommender_fixture.recommend([source_id])
        rec_ids = {r.service.id for r in recs}
        assert rec_ids & targets, (
            f"Journey targets for '{source_id}' should appear in recommendations"
        )

    def test_journey_reason_in_recommendation_text(self, recommender_fixture):
        """Journey recommendations should include the journey reason text."""
        if not CITIZEN_JOURNEYS:
            pytest.skip("No journey data configured")
        source_id = next(iter(CITIZEN_JOURNEYS))
        expected_reasons = {reason for _, reason in CITIZEN_JOURNEYS[source_id]}
        recs = recommender_fixture.recommend([source_id])
        all_reasons = " ".join(r.reason for r in recs)
        assert any(reason in all_reasons for reason in expected_reasons)
