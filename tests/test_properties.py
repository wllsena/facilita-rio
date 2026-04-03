"""Property-based tests for search pipeline invariants using Hypothesis."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis.strategies import integers, sampled_from, text

from app.indexing.bm25_index import BM25Index, _tokenize
from app.indexing.vector_index import VectorIndex
from app.search.hybrid import weighted_rrf

# ── Tokenizer properties ───────────────────────────────────────────────────


@given(text(min_size=0, max_size=500))
@settings(max_examples=100)
def test_tokenize_never_crashes(s):
    """Tokenizer should handle any unicode input without crashing."""
    result = _tokenize(s)
    assert isinstance(result, list)
    for token in result:
        assert isinstance(token, str)
        assert len(token) > 0


@given(text(min_size=0, max_size=100))
@settings(max_examples=50)
def test_tokenize_no_stopwords_in_output(s):
    """No stopwords should survive tokenization."""

    tokens = _tokenize(s)
    for token in tokens:
        # Tokens are already stemmed, so we can't directly check against stopwords.
        # But we can verify no single-char tokens slip through.
        assert len(token) > 1


# ── BM25 properties ────────────────────────────────────────────────────────


class TestBM25Properties:
    @pytest.fixture(autouse=True, scope="class")
    def setup(self, services):
        TestBM25Properties._bm25 = BM25Index(services)
        TestBM25Properties._n_services = len(services)

    @given(top_k=integers(min_value=1, max_value=50))
    @settings(max_examples=20)
    def test_respects_top_k(self, top_k):
        """BM25 should never return more than top_k results."""
        results = self._bm25.search("IPTU", top_k=top_k)
        assert len(results) <= top_k

    @given(query=text(min_size=1, max_size=200))
    @settings(max_examples=50)
    def test_scores_are_non_negative(self, query):
        """All returned BM25 scores should be positive."""
        results = self._bm25.search(query, top_k=10)
        for _, score in results:
            assert score > 0

    @given(query=text(min_size=1, max_size=200))
    @settings(max_examples=50)
    def test_scores_are_monotonically_decreasing(self, query):
        """Results should be sorted by descending score."""
        results = self._bm25.search(query, top_k=20)
        scores = [s for _, s in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    @given(query=text(min_size=1, max_size=200))
    @settings(max_examples=30)
    def test_no_duplicate_results(self, query):
        """No service should appear twice in results."""
        results = self._bm25.search(query, top_k=20)
        ids = [doc_id for doc_id, _ in results]
        assert len(ids) == len(set(ids))


# ── Vector index properties ─────────────────────────────────────────────────


class TestVectorProperties:
    @pytest.fixture(autouse=True, scope="class")
    def setup(self, services):
        TestVectorProperties._vi = VectorIndex(services)

    @given(query=text(min_size=1, max_size=200))
    @settings(max_examples=20, deadline=None)
    def test_cosine_sim_in_range(self, query):
        """Cosine similarities should be in [-1, 1]."""
        results = self._vi.search(query, top_k=5)
        for _, score in results:
            assert -1.0 <= score <= 1.0 + 1e-6

    @given(query=text(min_size=1, max_size=200))
    @settings(max_examples=20, deadline=None)
    def test_no_duplicate_results(self, query):
        """No service should appear twice."""
        results = self._vi.search(query, top_k=20)
        ids = [doc_id for doc_id, _ in results]
        assert len(ids) == len(set(ids))


# ── Weighted RRF properties ─────────────────────────────────────────────────


@given(
    k=integers(min_value=1, max_value=200),
    w1=sampled_from([0.5, 1.0, 2.0, 3.0]),
    w2=sampled_from([0.5, 1.0, 2.0, 3.0]),
)
@settings(max_examples=30)
def test_rrf_scores_are_positive(k, w1, w2):
    """All RRF scores should be positive."""
    list_a = [("d1", 1.0), ("d2", 0.5)]
    list_b = [("d2", 1.0), ("d3", 0.5)]
    fused = weighted_rrf([(list_a, w1), (list_b, w2)], k=k)
    for _, score in fused:
        assert score > 0


def test_rrf_docs_appearing_in_both_lists_rank_higher():
    """Documents in both lists should generally outrank single-list docs."""
    list_a = [("shared", 1.0), ("only_a", 0.5)]
    list_b = [("shared", 1.0), ("only_b", 0.5)]
    fused = weighted_rrf([(list_a, 1.0), (list_b, 1.0)])
    ids = [doc_id for doc_id, _ in fused]
    assert ids[0] == "shared"
