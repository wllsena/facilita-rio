from __future__ import annotations

import random
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config import CONFIG

router = APIRouter()

templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    examples = _sample_service_names(request)
    return templates.TemplateResponse(request, "search.html", {"examples": examples})


@router.get("/search", response_class=HTMLResponse)
async def search_page(request: Request, q: str = ""):
    if not q.strip():
        examples = _sample_service_names(request)
        return templates.TemplateResponse(request, "search.html", {"examples": examples})

    from app.main import execute_search
    response = await execute_search(q, top_k=CONFIG.rerank_top_k)
    return templates.TemplateResponse(request, "search.html", {"query": q, "response": response})


@router.get("/service/{service_id}", response_class=HTMLResponse)
async def service_detail_page(request: Request, service_id: str):
    services_map = request.app.state.services_map
    recommender = request.app.state.recommender

    service = services_map.get(service_id)
    if not service:
        return templates.TemplateResponse(request, "search.html", {"error": "Serviço não encontrado"})

    recs = recommender.recommend([service_id]) if recommender else []
    return templates.TemplateResponse(request, "service_detail.html", {"service": service, "recommendations": recs})


def _sample_service_names(request: Request, n: int = 8) -> list[str]:
    services = list(request.app.state.services_map.values())
    sample = random.sample(services, min(n, len(services)))
    return [s.nome for s in sample]
