"""BM25 lexical index with Portuguese stemming."""

from __future__ import annotations

import re

import nltk
from rank_bm25 import BM25Okapi
from unidecode import unidecode

from app.models import Service

nltk.download("rslp", quiet=True)
from nltk.stem import RSLPStemmer  # noqa: E402

_stemmer = RSLPStemmer()

PT_STOPWORDS = frozenset({
    "o", "a", "os", "as", "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "por", "pelo", "pela", "pelos", "pelas", "ao", "aos",
    "com", "sem", "para", "ate", "entre", "sobre", "sob",
    "eu", "tu", "ele", "ela", "eles", "elas",
    "meu", "minha", "meus", "minhas", "seu", "sua", "seus", "suas",
    "esse", "essa", "este", "esta", "isso", "isto", "aquele", "aquela",
    "me", "te", "se", "lhe", "lhes",
    "que", "qual", "quais", "quem", "onde", "como", "quando",
    "ser", "estar", "ter", "haver", "ir", "vir", "fazer", "poder", "dever",
    "foi", "era", "estou", "tem", "sao", "pode",
    "preciso", "quero", "posso", "tenho",
    "ou", "mas", "porem", "porque", "pois", "ja", "tambem",
    "ainda", "muito", "mais", "menos", "tao", "bem", "mal",
    "nao", "sim", "aqui", "ali", "la", "ha", "so",
    "pra", "pro", "ta", "to", "ne", "ai", "tipo", "perto",
    "faco", "faz", "fica", "tava", "tou",
})


def _tokenize(text: str) -> list[str]:
    text = unidecode(text.lower())
    tokens = re.split(r"[^a-z0-9]+", text)
    return [_stemmer.stem(t) for t in tokens if len(t) > 1 and t not in PT_STOPWORDS]


class BM25Index:

    def __init__(self, services: list[Service]) -> None:
        self._services = services
        corpus = [
            _tokenize(f"{s.nome} {s.nome} {s.nome} {s.tema} {s.resumo} {s.descricao_completa}")
            for s in services
        ]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        top_indices = scores.argsort()[::-1][:top_k]
        return [(self._services[idx].id, float(scores[idx])) for idx in top_indices if scores[idx] > 0]
