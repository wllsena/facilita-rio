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
    """FAISS flat index with E5 embeddings for semantic search."""

    def __init__(self, services: list[Service], model: SentenceTransformer | None = None) -> None:
        self._services = services
        self._id_to_idx = {s.id: i for i, s in enumerate(services)}

        if model is None:
            logger.info("loading_embedding_model", model=CONFIG.embedding_model)
            model = SentenceTransformer(CONFIG.embedding_model)
        self._model = model

        # Build document embeddings
        # E5 models expect "passage: " prefix for documents.
        # Include tema + first 300 chars of descricao_completa for richer context.
        # Without this, embeddings are too shallow and cluster tightly (0.81-0.87),
        # causing irrelevant services (e.g., "merenda escolar") to rank high for
        # unrelated queries (e.g., "quebrei meu braço").
        docs = [
            f"passage: {s.nome}. Categoria: {s.tema}. {s.resumo} {s.descricao_completa[:300]}"
            for s in services
        ]
        logger.info("encoding_documents", count=len(docs))
        self._embeddings = self._model.encode(docs, normalize_embeddings=True, show_progress_bar=False)
        self._embeddings = np.array(self._embeddings, dtype=np.float32)

        # Build FAISS flat index (exact search — fast enough for 50-1200 docs)
        self._index = faiss.IndexFlatIP(self._embeddings.shape[1])
        self._index.add(self._embeddings)

        logger.info("vector_index_built", num_vectors=self._index.ntotal, dim=self._embeddings.shape[1])

    @property
    def model(self) -> SentenceTransformer:
        return self._model

    @property
    def embeddings(self) -> np.ndarray:
        return self._embeddings

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Return (service_id, cosine_similarity) pairs."""
        # E5 models expect "query: " prefix for queries
        q_emb = self._model.encode(
            [f"query: {query}"], normalize_embeddings=True, show_progress_bar=False
        )
        q_emb = np.array(q_emb, dtype=np.float32)

        scores, indices = self._index.search(q_emb, min(top_k, len(self._services)))

        results = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx >= 0:
                results.append((self._services[idx].id, float(score)))
        return results

    def get_neighbors(self, service_id: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Find nearest neighbors of a service in embedding space."""
        idx = self._id_to_idx.get(service_id)
        if idx is None:
            return []

        vec = self._embeddings[idx : idx + 1]
        scores, indices = self._index.search(vec, min(top_k + 1, len(self._services)))

        results = []
        for score, neighbor_idx in zip(scores[0], indices[0], strict=True):
            if neighbor_idx >= 0 and neighbor_idx != idx:
                results.append((self._services[neighbor_idx].id, float(score)))
        return results[:top_k]
