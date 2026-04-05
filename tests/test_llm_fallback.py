"""Tests for query processing: normalization, local expansion, LLM fallback."""

from __future__ import annotations

import pytest

from app.search.query_processor import (
    SYNONYM_EXPANSIONS,
    enrich_query_with_llm,
    expand_query,
    normalize_query,
)


def test_normalize_query_strips_whitespace():
    assert normalize_query("  hello   world  ") == "hello world"


def test_normalize_query_empty():
    assert normalize_query("") == ""


# ── Local synonym expansion ───────────────────────────────────────────────


def test_expand_query_applies_matching_pattern():
    """A query containing a synonym pattern gets additional terms appended."""
    # Pick the first pattern without anti-patterns
    simple = [(p, e) for p, e, ap in SYNONYM_EXPANSIONS if not ap]
    assert simple, "synonyms.json should have at least one entry without anti-patterns"
    pattern, expansion = simple[0]
    result = expand_query(pattern)
    assert len(result) > len(pattern), f"Pattern '{pattern}' should be expanded"
    assert expansion.split()[0] in result.lower()


def test_expand_query_no_match():
    """Queries without known patterns should be returned unchanged."""
    result = expand_query("xyzzy foobar nonsense")
    assert result == "xyzzy foobar nonsense"


def test_expand_query_anti_pattern_prevents_expansion():
    """Anti-patterns should prevent expansion from firing."""
    entries_with_anti = [(p, e, ap) for p, e, ap in SYNONYM_EXPANSIONS if ap]
    if not entries_with_anti:
        pytest.skip("No entries with anti-patterns in synonyms.json")
    pattern, expansion, anti_patterns = entries_with_anti[0]
    anti = next(iter(anti_patterns))
    # Query contains both pattern and anti-pattern → should NOT expand with that entry
    result = expand_query(f"{pattern} {anti}")
    assert expansion not in result.lower()


# ── LLM enrichment fallback ──────────────────────────────────────────────


async def test_enrich_query_without_api_key(monkeypatch):
    """Without OPENAI_API_KEY, enrichment returns the original query unchanged."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = await enrich_query_with_llm("test query")
    assert result["original"] == "test query"
    assert result["expanded"] == "test query"
    assert result["intent"] == ""


async def test_enrich_query_with_invalid_key(monkeypatch):
    """With an invalid API key, enrichment falls back gracefully."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-invalid-test-key")
    result = await enrich_query_with_llm("test query")
    assert result["original"] == "test query"
    assert result["expanded"] == "test query"
