"""FastAPI application — search & recommendation API + web UI."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from cachetools import LRUCache
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import CONFIG, DATA_PATH
from app.indexing.bm25_index import BM25Index
from app.indexing.cluster_builder import ClusterIndex
from app.indexing.loader import load_services
from app.indexing.vector_index import VectorIndex
from app.models import SearchRequest, SearchResponse, Service
from app.observability.logging import setup_logging
from app.observability.metrics import (
    CACHE_HITS,
    INDEX_SIZE,
    RERANKER_LATENCY,
    SEARCH_LATENCY,
    SEARCH_REQUESTS,
    SEARCH_RESULT_COUNT,
)
from app.recommendation.recommender import Recommender
from app.search.hybrid import HybridRetriever
from app.search.pipeline import SearchPipeline
from app.search.reranker import Reranker

logger = structlog.get_logger()

# ── Global state filled at startup ──────────────────────────────────────────

services_map: dict[str, Service] = {}
recommender: Recommender | None = None
startup_time: float = 0
_pipeline: SearchPipeline | None = None

# LRU cache for full search responses (key: hash of query+top_k)
_search_cache: LRUCache = LRUCache(maxsize=256)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and build indices at startup."""
    global services_map, recommender, startup_time, _pipeline

    setup_logging()
    t0 = time.time()

    logger.info("loading_services", path=str(DATA_PATH))
    services = load_services(DATA_PATH)
    services_map = {s.id: s for s in services}
    INDEX_SIZE.set(len(services))

    logger.info("building_bm25_index")
    bm25_index = BM25Index(services)

    logger.info("building_vector_index")
    vector_index = VectorIndex(services)

    logger.info("building_cluster_index")
    cluster_index = ClusterIndex(services, vector_index.embeddings)

    hybrid_retriever = HybridRetriever(bm25_index, vector_index)

    logger.info("loading_reranker")
    reranker = Reranker()

    recommender = Recommender(services_map, vector_index, cluster_index)

    _pipeline = SearchPipeline(services_map, hybrid_retriever, reranker, recommender)

    startup_time = time.time() - t0
    logger.info("startup_complete", duration_s=round(startup_time, 2), services=len(services))

    yield

    logger.info("shutdown")


# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="facilita Rio",
    description="Busca inteligente de serviços públicos do município do Rio de Janeiro",
    version="0.1.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app)

templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── API endpoints ───────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    """Readiness check — returns service count, startup time, and LLM status."""
    return {
        "status": "ok",
        "services_loaded": len(services_map),
        "startup_time_s": round(startup_time, 2),
        "llm_enabled": CONFIG.llm_enabled,
    }


@app.post("/api/search", response_model=SearchResponse)
async def api_search(req: SearchRequest):
    """Core search endpoint — returns results + recommendations."""
    return await _execute_search(req.query, req.top_k)


@app.get("/api/search", response_model=SearchResponse)
async def api_search_get(
    q: str = Query(..., min_length=2, max_length=500),
    top_k: int = Query(default=10, ge=1, le=50),
):
    """GET variant for easy browser/curl testing."""
    return await _execute_search(q, top_k)


@app.get("/api/suggest")
async def api_suggest(q: str = Query("", max_length=200)):
    """Return query suggestions based on partial input for inline autocomplete.

    Uses accent-insensitive matching (via unidecode) so that "vacinacao" matches
    "Vacinação" and "arvore" matches "Árvore". Supports both full-string and
    token-level matching: "poda arvore" matches any service containing both tokens.
    """
    if len(q.strip()) < 2:
        return {"suggestions": []}

    from unidecode import unidecode

    q_norm = unidecode(q.strip().lower())
    q_tokens = q_norm.split()
    suggestions: list[tuple[int, int, str]] = []  # (priority, length, name)

    for service in services_map.values():
        name_norm = unidecode(service.nome.lower())
        resumo_norm = unidecode(service.resumo[:120].lower())

        # Full-string substring match (highest priority)
        if q_norm in name_norm:
            priority = 0 if name_norm.startswith(q_norm) else 1
            suggestions.append((priority, len(service.nome), service.nome))
        elif q_norm in resumo_norm:
            suggestions.append((2, len(service.nome), service.nome))
        # Token-level match: all query tokens appear in name or resumo
        elif len(q_tokens) > 1 and all(
            t in name_norm or t in resumo_norm for t in q_tokens
        ):
            suggestions.append((3, len(service.nome), service.nome))

    suggestions.sort()
    return {"suggestions": [name for _, _, name in suggestions[:8]]}


@app.get("/api/service/{service_id}")
async def api_service_detail(service_id: str):
    """Get a single service by ID."""
    from fastapi.responses import JSONResponse

    service = services_map.get(service_id)
    if not service:
        return JSONResponse(status_code=404, content={"error": "Service not found"})
    return service


# ── Web UI ──────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "search.html")


@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request, q: str = ""):
    """Search page with results."""
    if not q.strip():
        return templates.TemplateResponse(request, "search.html")

    response = await _execute_search(q, top_k=CONFIG.rerank_top_k)
    return templates.TemplateResponse(
        request,
        "search.html",
        {"query": q, "response": response},
    )


@app.get("/service/{service_id}", response_class=HTMLResponse)
async def service_detail_page(request: Request, service_id: str):
    """Service detail page with recommendations."""
    service = services_map.get(service_id)
    if not service:
        return templates.TemplateResponse(
            request, "search.html", {"error": "Serviço não encontrado"}
        )

    # Generate recommendations for this specific service
    recs = recommender.recommend([service_id]) if recommender else []

    return templates.TemplateResponse(
        request,
        "service_detail.html",
        {"service": service, "recommendations": recs},
    )


# ── Core search logic ──────────────────────────────────────────────────────


async def _execute_search(query: str, top_k: int) -> SearchResponse:
    """Run the full search pipeline with caching and metrics.

    Delegates to SearchPipeline for the actual search logic. This function
    adds caching and Prometheus metrics on top.
    """
    t0 = time.time()
    SEARCH_REQUESTS.inc()

    from app.search.query_processor import normalize_query

    query = normalize_query(query)

    # Check cache (keyed on normalized query + top_k)
    cache_key = f"{query}:{top_k}"
    cached = _search_cache.get(cache_key)
    if cached is not None:
        CACHE_HITS.inc()
        latency_ms = (time.time() - t0) * 1000
        return cached.model_copy(update={"latency_ms": round(latency_ms, 1)})

    response = await _pipeline.execute(query, top_k)

    SEARCH_LATENCY.observe(response.latency_ms / 1000)
    SEARCH_RESULT_COUNT.observe(len(response.results))
    RERANKER_LATENCY.observe(response.rerank_ms / 1000)

    # Store in cache
    _search_cache[cache_key] = response

    return response
