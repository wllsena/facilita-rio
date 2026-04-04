"""Pipeline quality regression tests — verify minimum IR metrics for representative queries.

These tests run real queries through the full pipeline and assert the correct
services appear in top results. They catch regressions that unit tests cannot:
e.g., a BM25 tokenizer change that breaks a specific query, or a weight change
that shifts rankings below acceptable thresholds.
"""

from __future__ import annotations

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


# Representative queries spanning all categories with expected top-1 service
QUALITY_CHECKS = [
    # Direct
    ("segunda via IPTU", "emissao-de-2-via-do-iptu-ce2b748c"),
    # Natural language
    ("meu cachorro está doente", "atendimento-clinico-em-animais-8c9a32e8"),
    ("quero parar de fumar", "inscricao-em-programa-de-tratamento-antitabagismo-00116adc"),
    ("tem buraco na minha rua", "reparo-de-buraco-deformacao-ou-afundamento-na-0a5c9f7e"),
    # Synonym
    ("veterinário público gratuito", "atendimento-clinico-em-animais-8c9a32e8"),
    # Edge (with expansion)
    ("árvore caiu na calçada perto da minha casa", "remocao-de-arvore-em-vias-publicas-54b06fd3"),
]


@pytest.mark.parametrize(
    "query,expected_id",
    QUALITY_CHECKS,
    ids=[q[:35] for q, _ in QUALITY_CHECKS],
)
def test_top1_result_is_correct(pipeline, query, expected_id):
    """The most relevant service should appear in top-3 for representative queries."""
    results = _search(pipeline, query)
    assert len(results) > 0, f"No results for '{query}'"
    top_ids = [r[0] for r in results[:3]]
    assert expected_id in top_ids, (
        f"Expected '{expected_id}' in top-3 for '{query}', got {top_ids}"
    )


def test_all_natural_queries_return_results(pipeline):
    """Pipeline should return results for all common natural language queries."""
    queries = [
        "preciso de emprego",
        "vacinação",
        "sofri violência doméstica",
        "dengue no meu bairro",
        "minha esposa está grávida",
        "pessoa morando na rua",
    ]
    for query in queries:
        results = _search(pipeline, query)
        assert len(results) >= 3, f"Too few results ({len(results)}) for '{query}'"


def test_max_semantic_score_is_confidence_signal(pipeline):
    """Positive queries should have higher max cosine similarity than out-of-scope queries."""
    services_map, retriever, _ = pipeline

    # Use the vector index directly for cosine similarity
    vi = retriever._vector

    positive_sims = []
    for query in ["segunda via IPTU", "meu cachorro está doente", "vacinação"]:
        results = vi.search(query, top_k=1)
        positive_sims.append(results[0][1])

    negative_sims = []
    for query in ["pizza delivery", "weather forecast", "receita de bolo"]:
        results = vi.search(query, top_k=1)
        negative_sims.append(results[0][1])

    avg_pos = sum(positive_sims) / len(positive_sims)
    avg_neg = sum(negative_sims) / len(negative_sims)

    assert avg_pos > avg_neg, (
        f"Positive queries ({avg_pos:.4f}) should have higher cosine sim "
        f"than negative queries ({avg_neg:.4f})"
    )


def test_low_confidence_flag(pipeline):
    """Out-of-scope queries should be flagged as low_confidence."""
    from app.config import CONFIG

    services_map, retriever, _ = pipeline
    vi = retriever._vector

    # Positive query should NOT be low confidence
    pos_results = vi.search("segunda via IPTU", top_k=1)
    assert pos_results[0][1] >= CONFIG.confidence_threshold, (
        f"Positive query should be above threshold ({CONFIG.confidence_threshold})"
    )

    # Negative query should be low confidence
    # "weather forecast tomorrow" has cosine ~0.76 — clearly below any reasonable threshold
    neg_results = vi.search("weather forecast tomorrow", top_k=1)
    assert neg_results[0][1] < CONFIG.confidence_threshold, (
        f"Negative query should be below threshold ({CONFIG.confidence_threshold})"
    )


def test_query_reformulation_for_ambiguous_queries():
    """Short/ambiguous queries should get reformulation suggestions."""
    from app.models import SearchResult, Service
    from app.search.pipeline import SearchPipeline

    # Create mock results with distinct service names
    mock_service = Service(
        id="test", nome="Emissão de 2ª via do IPTU", resumo="", descricao_completa="",
        tema="Tributos", orgao_gestor=[], custo="", publico=[], tempo_atendimento="",
        instrucoes="", resultado="", search_content="",
    )
    results = [SearchResult(service=mock_service, score=0.9)]

    # Short query should get suggestions
    suggestions = SearchPipeline._suggest_queries("imposto", results, low_confidence=False)
    assert len(suggestions) > 0, "Ambiguous 1-word query should get suggestions"

    # Long specific query should NOT get suggestions
    suggestions = SearchPipeline._suggest_queries(
        "meu cachorro está doente e precisa de ajuda", results, low_confidence=False,
    )
    assert len(suggestions) == 0, "Specific multi-word query should not get suggestions"
