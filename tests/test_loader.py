"""Tests for data loading and parsing."""

from __future__ import annotations

from app.config import DATA_PATH
from app.indexing.loader import load_services


def test_load_services_count():
    services = load_services(DATA_PATH)
    assert len(services) == 50


def test_load_services_fields():
    services = load_services(DATA_PATH)
    s = services[0]
    assert s.id != ""
    assert s.nome != ""
    assert s.resumo != ""
    assert s.tema != ""


def test_all_services_have_unique_ids():
    services = load_services(DATA_PATH)
    ids = [s.id for s in services]
    assert len(ids) == len(set(ids))


def test_all_services_have_tema():
    services = load_services(DATA_PATH)
    for s in services:
        assert s.tema != "", f"Service {s.id} missing tema"
