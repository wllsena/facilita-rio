"""FAISS vector index for semantic retrieval using sentence-transformers."""

from __future__ import annotations

import faiss
import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

from app.config import CONFIG
from app.models import Service

logger = structlog.get_logger()


class VectorIndex:

    def __init__(self, services: list[Service], model: SentenceTransformer | None = None) -> None:
        self._services = services
        self._id_to_idx = {s.id: i for i, s in enumerate(services)}

        if model is None:
            logger.info("loading_embedding_model", model=CONFIG.embedding_model)
            model = SentenceTransformer(CONFIG.embedding_model)
        self._model = model

        docs = [
            f"passage: {s.nome}. Categoria: {s.tema}. {s.resumo} {s.descricao_completa[:300]}"
            for s in services
        ]
        self._embeddings = self._model.encode(docs, normalize_embeddings=True, show_progress_bar=False)
        self._embeddings = np.array(self._embeddings, dtype=np.float32)

        self._index = faiss.IndexFlatIP(self._embeddings.shape[1])
        self._index.add(self._embeddings)

    @property
    def model(self) -> SentenceTransformer:
        return self._model

    @property
    def embeddings(self) -> np.ndarray:
        return self._embeddings

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        q_emb = self._model.encode(
            [f"query: {query}"], normalize_embeddings=True, show_progress_bar=False
        )
        q_emb = np.array(q_emb, dtype=np.float32)
        scores, indices = self._index.search(q_emb, min(top_k, len(self._services)))
        return [
            (self._services[idx].id, float(score))
            for score, idx in zip(scores[0], indices[0], strict=True) if idx >= 0
        ]

    def get_neighbors(self, service_id: str, top_k: int = 10) -> list[tuple[str, float]]:
        idx = self._id_to_idx.get(service_id)
        if idx is None:
            return []
        vec = self._embeddings[idx : idx + 1]
        scores, indices = self._index.search(vec, min(top_k + 1, len(self._services)))
        return [
            (self._services[ni].id, float(score))
            for score, ni in zip(scores[0], indices[0], strict=True)
            if ni >= 0 and ni != idx
        ][:top_k]
