"""FastAPI application — app factory, lifespan, and core search logic."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from cachetools import LRUCache
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import CONFIG, DATA_PATH
from app.indexing.bm25_index import BM25Index
from app.indexing.cluster_builder import ClusterIndex
from app.indexing.loader import load_services
from app.indexing.vector_index import VectorIndex
from app.models import SearchResponse
from app.observability import (
    CACHE_HITS,
    INDEX_SIZE,
    RERANKER_LATENCY,
    SEARCH_LATENCY,
    SEARCH_REQUESTS,
    SEARCH_RESULT_COUNT,
    setup_logging,
)
from app.recommendation.recommender import Recommender
from app.search.hybrid import HybridRetriever
from app.search.pipeline import SearchPipeline
from app.search.query_processor import normalize_query
from app.search.reranker import Reranker

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    t0 = time.time()

    services = load_services(DATA_PATH)
    app.state.services_map = {s.id: s for s in services}
    INDEX_SIZE.set(len(services))

    bm25_index = BM25Index(services)
    vector_index = VectorIndex(services)
    cluster_index = ClusterIndex(services, vector_index.embeddings)
    hybrid_retriever = HybridRetriever(bm25_index, vector_index)
    reranker = Reranker()

    app.state.recommender = Recommender(app.state.services_map, vector_index, cluster_index)
    app.state.pipeline = SearchPipeline(
        app.state.services_map, hybrid_retriever, reranker, app.state.recommender,
    )
    app.state.search_cache = LRUCache(maxsize=256)
    app.state.startup_time = time.time() - t0

    logger.info("startup_complete", duration_s=round(app.state.startup_time, 2), services=len(services))
    yield
    logger.info("shutdown")


app = FastAPI(title=CONFIG.app_title, version="0.1.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

from app.routers.api import router as api_router  # noqa: E402
from app.routers.pages import router as pages_router  # noqa: E402

app.include_router(api_router)
app.include_router(pages_router)


async def execute_search(query: str, top_k: int) -> SearchResponse:
    t0 = time.time()
    SEARCH_REQUESTS.inc()
    query = normalize_query(query)

    cache_key = f"{query}:{top_k}"
    cached = app.state.search_cache.get(cache_key)
    if cached is not None:
        CACHE_HITS.inc()
        latency_ms = (time.time() - t0) * 1000
        return cached.model_copy(update={"latency_ms": round(latency_ms, 1)})

    response = await app.state.pipeline.execute(query, top_k)

    SEARCH_LATENCY.observe(response.latency_ms / 1000)
    SEARCH_RESULT_COUNT.observe(len(response.results))
    RERANKER_LATENCY.observe(response.rerank_ms / 1000)
    app.state.search_cache[cache_key] = response

    return response
