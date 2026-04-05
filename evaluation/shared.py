"""Shared evaluation components — avoids reloading models per variant."""

from __future__ import annotations

from typing import NamedTuple

from app.config import DATA_PATH
from app.indexing.bm25_index import BM25Index
from app.indexing.loader import load_services
from app.indexing.vector_index import VectorIndex
from app.models import Service
from app.search.reranker import Reranker


class SharedComponents(NamedTuple):
    services: list[Service]
    services_map: dict[str, Service]
    bm25_index: BM25Index
    vector_index: VectorIndex
    reranker: Reranker


def build_shared_components() -> SharedComponents:
    services = load_services(DATA_PATH)
    services_map = {s.id: s for s in services}
    return SharedComponents(services, services_map, BM25Index(services), VectorIndex(services), Reranker())
