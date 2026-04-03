"""BM25 lexical index over service documents with Portuguese stemming."""

from __future__ import annotations

import re

import nltk
from rank_bm25 import BM25Okapi
from unidecode import unidecode

from app.models import Service

# Download RSLP stemmer data (no-op if already present)
nltk.download("rslp", quiet=True)
from nltk.stem import RSLPStemmer  # noqa: E402

_stemmer = RSLPStemmer()

# Portuguese stopwords — high-frequency words that add noise to BM25 retrieval.
# Includes articles, prepositions, pronouns, common verbs, and connectives.
PT_STOPWORDS = frozenset({
    # Articles
    "o", "a", "os", "as", "um", "uma", "uns", "umas",
    # Prepositions / contractions
    "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "por", "pelo", "pela", "pelos", "pelas", "ao", "aos",
    "com", "sem", "para", "ate", "entre", "sobre", "sob",
    # Pronouns
    "eu", "tu", "ele", "ela", "eles", "elas",
    "meu", "minha", "meus", "minhas", "seu", "sua", "seus", "suas",
    "esse", "essa", "este", "esta", "isso", "isto", "aquele", "aquela",
    "me", "te", "se", "lhe", "lhes",
    # Demonstratives / relative
    "que", "qual", "quais", "quem", "onde", "como", "quando",
    # Common verbs / auxiliaries
    "ser", "estar", "ter", "haver", "ir", "vir", "fazer", "poder", "dever",
    "foi", "era", "estou", "tem", "sao", "pode",
    "preciso", "quero", "posso", "tenho",
    # Connectives
    "ou", "mas", "porem", "porque", "pois", "ja", "tambem",
    "ainda", "muito", "mais", "menos", "tao", "bem", "mal",
    # Others
    "nao", "sim", "aqui", "ali", "la", "ha", "so",
    # Colloquial contractions (ubiquitous in informal queries, no signal)
    "pra", "pro", "ta", "to", "ne", "ai", "tipo", "perto",
    "faco", "faz", "fica", "tava", "tou",
})


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip accents, remove stopwords, and stem with RSLP."""
    text = unidecode(text.lower())
    tokens = re.split(r"[^a-z0-9]+", text)
    return [_stemmer.stem(t) for t in tokens if len(t) > 1 and t not in PT_STOPWORDS]


class BM25Index:
    """In-memory BM25 index for lexical retrieval with Portuguese stemming."""

    def __init__(self, services: list[Service]) -> None:
        self._services = services
        self._id_to_idx = {s.id: i for i, s in enumerate(services)}

        # Build corpus: combine name (boosted via repetition), tema, resumo, and full
        # description. Name is repeated 3x to approximate field boosting (rank_bm25
        # doesn't support per-field weights). Empirically, 3x keeps name terms dominant
        # for exact keyword matches (IPTU, ISS) without overwhelming description content.
        # Tema (category) is included so broad queries like "saúde pública" or "meio
        # ambiente" get a BM25 signal from the category label, not just content overlap.
        # Note: the JSON provides a `search_content` field, but it contains markdown
        # artifacts and is ~70-80% redundant with nome+resumo+descricao_completa.
        # Using it would dilute term frequency signals and trigger BM25 length
        # normalization penalties. Constructing our own document gives control
        # over field weighting and consistency with the semantic search pipeline.
        corpus = []
        for s in services:
            doc = f"{s.nome} {s.nome} {s.nome} {s.tema} {s.resumo} {s.descricao_completa}"
            corpus.append(_tokenize(doc))

        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Return (service_id, score) pairs sorted by descending BM25 score."""
        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:
                results.append((self._services[idx].id, score))
        return results
