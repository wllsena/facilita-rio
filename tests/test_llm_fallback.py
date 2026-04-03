"""Tests for query processing: normalization, local expansion, LLM fallback."""

from __future__ import annotations

from app.search.query_processor import enrich_query_with_llm, expand_query, normalize_query


def test_normalize_query_strips_whitespace():
    assert normalize_query("  hello   world  ") == "hello world"


def test_normalize_query_empty():
    assert normalize_query("") == ""


# ── Local synonym expansion ───────────────────────────────────────────────


def test_expand_query_tree_fell():
    """'árvore caiu' should expand with 'remoção de árvore'."""
    result = expand_query("árvore caiu na calçada")
    assert "remocao de arvore" in result.lower() or "remoção" in result.lower()


def test_expand_query_hunger():
    """'fome' should expand with 'cozinha comunitária'."""
    result = expand_query("refeição gratuita para quem passa fome")
    assert "cozinha comunitaria" in result.lower()


def test_expand_query_no_match():
    """Queries without known patterns should be returned unchanged."""
    result = expand_query("segunda via IPTU")
    assert result == "segunda via IPTU"


def test_expand_query_dengue():
    """'dengue no bairro' should expand with 'aedes aegypti'."""
    result = expand_query("dengue no bairro")
    assert "aedes" in result.lower()


# ── LLM enrichment fallback ──────────────────────────────────────────────


async def test_enrich_query_without_api_key(monkeypatch):
    """Without OPENAI_API_KEY, enrichment returns the original query unchanged."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = await enrich_query_with_llm("meu cachorro está doente")
    assert result["original"] == "meu cachorro está doente"
    assert result["expanded"] == "meu cachorro está doente"
    assert result["intent"] == ""


async def test_enrich_query_with_invalid_key(monkeypatch):
    """With an invalid API key, enrichment falls back gracefully."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-invalid-test-key")
    result = await enrich_query_with_llm("vacinação")
    # Should return original query on any error
    assert result["original"] == "vacinação"
    assert result["expanded"] == "vacinação"
