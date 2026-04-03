"""Tests for search components — BM25, vector, hybrid, reranker."""

from __future__ import annotations

import pytest

from app.indexing.bm25_index import BM25Index
from app.indexing.vector_index import VectorIndex
from app.search.hybrid import HybridRetriever, weighted_rrf


class TestBM25:
    def test_exact_match(self, services):
        idx = BM25Index(services)
        results = idx.search("IPTU segunda via", top_k=5)
        ids = [r[0] for r in results]
        assert any("iptu" in rid for rid in ids)

    def test_returns_scores(self, services):
        idx = BM25Index(services)
        results = idx.search("vacinação", top_k=5)
        assert len(results) > 0
        for _doc_id, score in results:
            assert score > 0

    def test_empty_query_returns_empty(self, services):
        idx = BM25Index(services)
        results = idx.search("", top_k=5)
        assert results == []

    def test_max_results_respected(self, services):
        idx = BM25Index(services)
        results = idx.search("serviço público", top_k=3)
        assert len(results) <= 3


@pytest.fixture(scope="module")
def vector_index(services):
    return VectorIndex(services)


class TestVectorIndex:
    def test_semantic_search(self, services, vector_index):
        results = vector_index.search("meu cachorro está doente", top_k=5)
        ids = [r[0] for r in results]
        # Should find animal-related services
        assert any("animal" in rid or "clinico" in rid or "bicho" in rid for rid in ids)

    def test_get_neighbors(self, services, vector_index):
        # Find neighbors of IPTU service
        iptu_id = next(s.id for s in services if "iptu" in s.id.lower() and "consulta" in s.id.lower())
        neighbors = vector_index.get_neighbors(iptu_id, top_k=3)
        assert len(neighbors) > 0
        # Neighbors should not include the source
        assert all(nid != iptu_id for nid, _ in neighbors)

    def test_returns_cosine_similarity(self, services, vector_index):
        results = vector_index.search("vacinação", top_k=5)
        for _, score in results:
            assert -1.0 <= score <= 1.0


class TestWeightedRRF:
    def test_basic_fusion(self):
        list_a = [("doc1", 10.0), ("doc2", 8.0), ("doc3", 6.0)]
        list_b = [("doc2", 0.9), ("doc3", 0.8), ("doc4", 0.7)]
        fused = weighted_rrf([(list_a, 1.0), (list_b, 1.0)])
        ids = [doc_id for doc_id, _ in fused]
        # doc2 should rank high (appears in both lists)
        assert "doc2" in ids[:2]

    def test_empty_lists(self):
        result = weighted_rrf([([], 1.0), ([], 1.0)])
        assert result == []

    def test_weight_matters(self):
        list_a = [("doc1", 10.0)]
        list_b = [("doc2", 10.0)]
        # With equal weight, both get the same score
        fused_equal = weighted_rrf([(list_a, 1.0), (list_b, 1.0)])
        scores = {doc_id: s for doc_id, s in fused_equal}
        assert abs(scores["doc1"] - scores["doc2"]) < 1e-9
        # With 3x weight on list_b, doc2 should score higher
        fused_weighted = weighted_rrf([(list_a, 1.0), (list_b, 3.0)])
        scores = {doc_id: s for doc_id, s in fused_weighted}
        assert scores["doc2"] > scores["doc1"]


class TestHybridRetriever:
    def test_hybrid_returns_results(self, services):
        bm25 = BM25Index(services)
        vector = VectorIndex(services)
        hybrid = HybridRetriever(bm25, vector)
        results = hybrid.search("preciso de emprego")
        assert len(results) > 0

    def test_hybrid_contains_scores(self, services):
        bm25 = BM25Index(services)
        vector = VectorIndex(services)
        hybrid = HybridRetriever(bm25, vector)
        results = hybrid.search("multa de trânsito")
        for _doc_id, rrf_score, _bm25_score, _sem_score in results:
            assert rrf_score > 0
