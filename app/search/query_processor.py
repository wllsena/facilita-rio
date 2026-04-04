"""Query preprocessing: normalization, local synonym expansion, optional LLM enrichment."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import structlog
from unidecode import unidecode

logger = structlog.get_logger()


# ── Local query expansion ─────────────────────────────────────────────────
# Lightweight, zero-latency synonym expansion for known vocabulary gaps.
# Patterns are loaded from app/data/synonyms.json — separates data from logic
# so expansions can be maintained, reviewed, and tested independently.
# Each entry has: pattern, expansion, optional anti_patterns.
# When anti_patterns is set and ANY anti-pattern appears in the query,
# the expansion is skipped — prevents collisions like "caindo" matching
# both "caí da escada" and "barranco caindo".
SynonymEntry = tuple[str, str, frozenset[str] | None]

_SYNONYMS_PATH = Path(__file__).resolve().parent.parent / "data" / "synonyms.json"


def _load_synonyms() -> list[SynonymEntry]:
    """Load synonym expansions from JSON, converting to internal tuple format."""
    with open(_SYNONYMS_PATH) as f:
        data = json.load(f)
    entries: list[SynonymEntry] = []
    for item in data["expansions"]:
        anti = frozenset(item["anti_patterns"]) if item.get("anti_patterns") else None
        entries.append((item["pattern"], item["expansion"], anti))
    return entries


SYNONYM_EXPANSIONS: list[SynonymEntry] = _load_synonyms()


def expand_query(query: str) -> str:
    """Expand query with local synonyms for known vocabulary gaps.

    Returns the original query with additional terms appended if any
    pattern matches. Supports anti-patterns: if a guard set is provided
    and any anti-pattern appears in the query, the expansion is skipped.
    """
    normalized = unidecode(query.lower())
    additions = []
    for pattern, expansion, anti_patterns in SYNONYM_EXPANSIONS:
        if pattern in normalized:
            if anti_patterns and any(ap in normalized for ap in anti_patterns):
                continue
            additions.append(expansion)

    if additions:
        expanded = f"{query} {' '.join(additions)}"
        logger.debug("query_expanded_local", original=query, expanded=expanded)
        return expanded
    return query


def normalize_query(query: str) -> str:
    """Basic normalization: strip, collapse whitespace."""
    return re.sub(r"\s+", " ", query.strip())


async def enrich_query_with_llm(query: str) -> dict:
    """Use LLM to extract intent and suggest query expansion.

    Returns dict with keys: original, expanded, intent.
    Falls back gracefully if LLM is unavailable.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"original": query, "expanded": query, "intent": ""}

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)
        from app.config import CONFIG

        response = await client.chat.completions.create(
            model=CONFIG.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é um assistente que ajuda cidadãos a encontrar serviços públicos "
                        "da Prefeitura do Rio de Janeiro. Dada uma consulta do usuário, extraia:\n"
                        "1. A intenção principal (1 frase curta)\n"
                        "2. Termos de busca expandidos em português (sinônimos, termos relacionados)\n"
                        "Responda APENAS em JSON: {\"intent\": \"...\", \"expanded\": \"...\"}"
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=CONFIG.llm_max_tokens,
            timeout=CONFIG.llm_timeout,
            response_format={"type": "json_object"},
        )

        parsed = json.loads(response.choices[0].message.content or "{}")
        return {
            "original": query,
            "expanded": parsed.get("expanded", query),
            "intent": parsed.get("intent", ""),
        }
    except Exception as e:
        logger.warning("llm_enrichment_failed", error=str(e))

    return {"original": query, "expanded": query, "intent": ""}
