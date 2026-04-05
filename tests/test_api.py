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


@pytest.fixture(scope="module")
async def sample_service_name(client):
    """Get a real service name from the loaded catalog for use in tests."""
    resp = await client.get("/health")
    assert resp.json()["services_loaded"] > 0
    # Use POST endpoint (no min_length restriction on SearchRequest) to get any service
    resp = await client.post("/api/search", json={"query": "serviço", "top_k": 1})
    return resp.json()["results"][0]["service"]["nome"]


class TestHealthEndpoint:
    async def test_health(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["services_loaded"] > 0


class TestSearchAPI:
    async def test_search_get(self, client, sample_service_name):
        resp = await client.get("/api/search", params={"q": sample_service_name})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) > 0
        assert data["latency_ms"] > 0
        assert "rerank_ms" in data
        assert data["rerank_ms"] >= 0

    async def test_search_post(self, client, sample_service_name):
        resp = await client.post("/api/search", json={"query": sample_service_name, "top_k": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) <= 5

    async def test_search_returns_recommendations(self, client, sample_service_name):
        resp = await client.get("/api/search", params={"q": sample_service_name})
        data = resp.json()
        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)

    async def test_search_result_structure(self, client, sample_service_name):
        resp = await client.get("/api/search", params={"q": sample_service_name})
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
    async def test_service_detail(self, client, sample_service_name):
        # Search to get a real service ID
        search_resp = await client.get("/api/search", params={"q": sample_service_name})
        first_id = search_resp.json()["results"][0]["service"]["id"]
        resp = await client.get(f"/api/service/{first_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == first_id
        assert data["nome"] != ""

    async def test_service_not_found(self, client):
        resp = await client.get("/api/service/nonexistent-service")
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data


class TestSuggestAPI:
    async def test_suggest_returns_results(self, client, sample_service_name):
        # Use first few chars of a real service name
        prefix = sample_service_name[:4]
        resp = await client.get("/api/suggest", params={"q": prefix})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["suggestions"]) > 0

    async def test_suggest_short_query_empty(self, client):
        resp = await client.get("/api/suggest", params={"q": "I"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["suggestions"] == []

    async def test_suggest_max_results(self, client):
        resp = await client.get("/api/suggest", params={"q": "aa"})
        data = resp.json()
        assert len(data["suggestions"]) <= 8


class TestWebUI:
    async def test_home_page(self, client):
        resp = await client.get("/")
        assert resp.status_code == 200

    async def test_search_page(self, client, sample_service_name):
        resp = await client.get("/search", params={"q": sample_service_name})
        assert resp.status_code == 200
