"""Query preprocessing: normalization, synonym expansion, optional LLM enrichment."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import structlog
from unidecode import unidecode

logger = structlog.get_logger()

SynonymEntry = tuple[str, str, frozenset[str] | None]

_SYNONYMS_PATH = Path(__file__).resolve().parent.parent / "data" / "synonyms.json"


def _load_synonyms() -> list[SynonymEntry]:
    if not _SYNONYMS_PATH.exists():
        logger.info("no_synonyms_file", path=str(_SYNONYMS_PATH))
        return []
    with open(_SYNONYMS_PATH) as f:
        data = json.load(f)
    return [
        (item["pattern"], item["expansion"],
         frozenset(item["anti_patterns"]) if item.get("anti_patterns") else None)
        for item in data["expansions"]
    ]


SYNONYM_EXPANSIONS: list[SynonymEntry] = _load_synonyms()


def _pattern_matches(pattern: str, normalized: str) -> bool:
    """Match pattern against normalized query.

    Single-word: substring match. Multi-word: all words present (any order).
    """
    words = pattern.split()
    if len(words) > 1:
        query_words = set(normalized.split())
        return all(w in query_words for w in words)
    return pattern in normalized


def expand_query(query: str) -> str:
    normalized = unidecode(query.lower())
    additions = [
        expansion
        for pattern, expansion, anti_patterns in SYNONYM_EXPANSIONS
        if _pattern_matches(pattern, normalized)
        and not (anti_patterns and any(ap in normalized for ap in anti_patterns))
    ]
    if additions:
        expanded = f"{query} {' '.join(additions)}"
        logger.debug("query_expanded_local", original=query, expanded=expanded)
        return expanded
    return query


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip())


async def enrich_query_with_llm(query: str) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"original": query, "expanded": query, "intent": ""}

    try:
        from openai import AsyncOpenAI

        from app.config import CONFIG

        client = AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model=CONFIG.openai_model,
            messages=[
                {"role": "system", "content": CONFIG.llm_system_prompt},
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
