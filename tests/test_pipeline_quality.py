"""Pipeline quality regression tests — verify the full pipeline returns relevant results.

These tests run real queries through expand → hybrid → rerank and check that
the top-ranked service is among the expected relevant services. Quality checks
are loaded from evaluation/test_queries.json so they stay in sync with the
evaluation suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import DATA_PATH
from app.indexing.bm25_index import BM25Index
from app.indexing.loader import load_services
from app.indexing.vector_index import VectorIndex
from app.search.hybrid import HybridRetriever
from app.search.query_processor import expand_query
from app.search.reranker import Reranker


@pytest.fixture(scope="module")
def pipeline():
    """Build full search pipeline once for all tests in this module."""
    services = load_services(DATA_PATH)
    services_map = {s.id: s for s in services}
    bm25 = BM25Index(services)
    vector = VectorIndex(services)
    retriever = HybridRetriever(bm25, vector)
    reranker_model = Reranker()
    return services_map, retriever, reranker_model


def _search(pipeline, query: str, top_k: int = 10) -> list[tuple[str, float]]:
    """Run full pipeline: expand → hybrid → rerank."""
    services_map, retriever, reranker_model = pipeline
    expanded = expand_query(query)
    candidates = retriever.search(query, expanded_query=expanded, top_k=top_k * 2)
    reranked = reranker_model.rerank(query, candidates, services_map, top_k=top_k)
    return [(doc_id, score) for doc_id, _, _, _, score in reranked]


def _load_quality_checks() -> list[tuple[str, str, str]]:
    """Load a sample of queries from each category in test_queries.json.

    Returns (query, expected_id, test_id) tuples. Picks the first query
    from each non-negative category that has a grade-3 relevant service.
    """
    path = Path(__file__).resolve().parent.parent / "evaluation" / "test_queries.json"
    with open(path) as f:
        data = json.load(f)

    seen_categories: set[str] = set()
    checks = []
    for q in data["queries"]:
        cat = q["category"]
        if cat == "negative" or cat in seen_categories:
            continue
        # Find the best-graded relevant service
        best_id = max(q["relevant"], key=lambda k: q["relevant"][k], default=None)
        if best_id and q["relevant"][best_id] >= 3:
            checks.append((q["query"], best_id, q["id"]))
            seen_categories.add(cat)
    return checks


QUALITY_CHECKS = _load_quality_checks()


@pytest.mark.parametrize(
    "query,expected_id,test_id",
    QUALITY_CHECKS,
    ids=[qid for _, _, qid in QUALITY_CHECKS],
)
def test_top3_result_is_correct(pipeline, query, expected_id, test_id):
    """The most relevant service should appear in top-3 for representative queries."""
    results = _search(pipeline, query)
    assert len(results) > 0, f"No results for '{query}'"
    top_ids = [r[0] for r in results[:3]]
    assert expected_id in top_ids, (
        f"[{test_id}] Expected '{expected_id}' in top-3 for '{query}', got {top_ids}"
    )


def test_all_queries_return_results(pipeline):
    """Pipeline should return results for common natural language queries."""
    path = Path(__file__).resolve().parent.parent / "evaluation" / "test_queries.json"
    with open(path) as f:
        data = json.load(f)

    natural_queries = [q["query"] for q in data["queries"] if q["category"] == "natural"]
    for query in natural_queries[:6]:
        results = _search(pipeline, query)
        assert len(results) >= 3, f"Too few results ({len(results)}) for '{query}'"


def test_max_semantic_score_is_confidence_signal(pipeline):
    """Positive queries should have higher max cosine similarity than out-of-scope queries."""
    _, retriever, _ = pipeline
    vi = retriever._vector

    path = Path(__file__).resolve().parent.parent / "evaluation" / "test_queries.json"
    with open(path) as f:
        data = json.load(f)

    positive = [q["query"] for q in data["queries"] if q["category"] != "negative"][:3]
    negative = [q["query"] for q in data["queries"] if q["category"] == "negative"][:3]

    pos_sims = [vi.search(q, top_k=1)[0][1] for q in positive]
    neg_sims = [vi.search(q, top_k=1)[0][1] for q in negative]

    avg_pos = sum(pos_sims) / len(pos_sims)
    avg_neg = sum(neg_sims) / len(neg_sims)

    assert avg_pos > avg_neg, (
        f"Positive queries ({avg_pos:.4f}) should have higher cosine sim "
        f"than negative queries ({avg_neg:.4f})"
    )


def test_low_confidence_flag(pipeline):
    """Positive queries should have higher cosine similarity than clearly out-of-scope queries."""
    from app.config import CONFIG

    _, retriever, _ = pipeline
    vi = retriever._vector

    path = Path(__file__).resolve().parent.parent / "evaluation" / "test_queries.json"
    with open(path) as f:
        data = json.load(f)

    # First positive query should be above threshold
    pos_query = next(q["query"] for q in data["queries"] if q["category"] == "direct")
    pos_results = vi.search(pos_query, top_k=1)
    assert pos_results[0][1] >= CONFIG.confidence_threshold

    # Find a negative query that is clearly below threshold
    neg_queries = [q["query"] for q in data["queries"] if q["category"] == "negative"]
    neg_sims = [(vi.search(q, top_k=1)[0][1], q) for q in neg_queries]
    min_sim, min_query = min(neg_sims)
    assert min_sim < CONFIG.confidence_threshold, (
        f"Lowest negative query '{min_query}' has cosine {min_sim:.4f}, "
        f"expected below {CONFIG.confidence_threshold}"
    )


def test_query_reformulation_for_ambiguous_queries():
    """Short/ambiguous queries should get reformulation suggestions."""
    from app.models import SearchResult, Service
    from app.search.pipeline import SearchPipeline

    mock_service = Service(
        id="test", nome="Emissão de Documento Oficial", resumo="", descricao_completa="",
        tema="Documentos", orgao_gestor=[], custo="", publico=[], tempo_atendimento="",
        instrucoes="", resultado="",
    )
    results = [SearchResult(service=mock_service, score=0.9)]

    # Short query with words NOT in the service name → should get suggestions
    suggestions = SearchPipeline._suggest_queries("imposto", results, low_confidence=False)
    assert len(suggestions) > 0, "Ambiguous 1-word query should get suggestions"

    # Long specific query → should NOT get suggestions
    suggestions = SearchPipeline._suggest_queries(
        "this is a very specific long query about something", results, low_confidence=False,
    )
    assert len(suggestions) == 0, "Specific multi-word query should not get suggestions"
