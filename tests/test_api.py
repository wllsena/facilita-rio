"""Tests for the FastAPI endpoints."""

from __future__ import annotations

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture(scope="module")
async def client():
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


class TestHealthEndpoint:
    async def test_health(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["services_loaded"] == 50


class TestSearchAPI:
    async def test_search_get(self, client):
        resp = await client.get("/api/search", params={"q": "IPTU"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) > 0
        assert data["latency_ms"] > 0
        assert "rerank_ms" in data
        assert data["rerank_ms"] >= 0

    async def test_search_post(self, client):
        resp = await client.post("/api/search", json={"query": "vacinação", "top_k": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) <= 5

    async def test_search_returns_recommendations(self, client):
        resp = await client.get("/api/search", params={"q": "segunda via IPTU"})
        data = resp.json()
        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)

    async def test_search_result_structure(self, client):
        resp = await client.get("/api/search", params={"q": "dengue"})
        data = resp.json()
        result = data["results"][0]
        assert "service" in result
        assert "score" in result
        assert "nome" in result["service"]
        assert "tema" in result["service"]

    async def test_empty_query_rejected(self, client):
        resp = await client.get("/api/search", params={"q": ""})
        assert resp.status_code == 422


class TestServiceDetail:
    async def test_service_detail(self, client):
        resp = await client.get("/api/service/informacoes-sobre-vacinacao-humana-728a6848")
        assert resp.status_code == 200
        data = resp.json()
        assert data["nome"] == "Informações sobre vacinação humana"

    async def test_service_not_found(self, client):
        resp = await client.get("/api/service/nonexistent-service")
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data


class TestSuggestAPI:
    async def test_suggest_returns_results(self, client):
        resp = await client.get("/api/suggest", params={"q": "IPTU"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["suggestions"]) > 0

    async def test_suggest_short_query_empty(self, client):
        resp = await client.get("/api/suggest", params={"q": "I"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["suggestions"] == []

    async def test_suggest_max_results(self, client):
        resp = await client.get("/api/suggest", params={"q": "a"})
        data = resp.json()
        assert len(data["suggestions"]) <= 8


class TestWebUI:
    async def test_home_page(self, client):
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "Facilita Rio" in resp.text

    async def test_search_page(self, client):
        resp = await client.get("/search", params={"q": "escola"})
        assert resp.status_code == 200
        assert "resultado" in resp.text.lower() or "escola" in resp.text.lower()
