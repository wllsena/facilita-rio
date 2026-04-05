from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from unidecode import unidecode

from app.config import CONFIG
from app.models import SearchRequest, SearchResponse

router = APIRouter()


@router.get("/health")
async def health(request: Request):
    return {
        "status": "ok",
        "services_loaded": len(request.app.state.services_map),
        "startup_time_s": round(request.app.state.startup_time, 2),
        "llm_enabled": CONFIG.llm_enabled,
    }


@router.post("/api/search", response_model=SearchResponse)
async def api_search(req: SearchRequest):
    from app.main import execute_search
    return await execute_search(req.query, req.top_k)


@router.get("/api/search", response_model=SearchResponse)
async def api_search_get(
    q: str = Query(..., min_length=2, max_length=500),
    top_k: int = Query(default=10, ge=1, le=50),
):
    from app.main import execute_search
    return await execute_search(q, top_k)


@router.get("/api/suggest")
async def api_suggest(request: Request, q: str = Query("", max_length=200)):
    if len(q.strip()) < 2:
        return {"suggestions": []}

    q_norm = unidecode(q.strip().lower())
    q_tokens = q_norm.split()
    suggestions: list[tuple[int, int, str]] = []

    for service in request.app.state.services_map.values():
        name_norm = unidecode(service.nome.lower())
        resumo_norm = unidecode(service.resumo[:120].lower())

        if q_norm in name_norm:
            priority = 0 if name_norm.startswith(q_norm) else 1
            suggestions.append((priority, len(service.nome), service.nome))
        elif q_norm in resumo_norm:
            suggestions.append((2, len(service.nome), service.nome))
        elif len(q_tokens) > 1 and all(t in name_norm or t in resumo_norm for t in q_tokens):
            suggestions.append((3, len(service.nome), service.nome))

    suggestions.sort()
    return {"suggestions": [name for _, _, name in suggestions[:8]]}


@router.get("/api/service/{service_id}")
async def api_service_detail(request: Request, service_id: str):
    service = request.app.state.services_map.get(service_id)
    if not service:
        return JSONResponse(status_code=404, content={"error": "Service not found"})
    return service
