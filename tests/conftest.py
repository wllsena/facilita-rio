"""Shared test fixtures."""

from __future__ import annotations

import pytest

from app.config import DATA_PATH
from app.indexing.loader import load_services
from app.models import Service


@pytest.fixture(scope="session")
def services() -> list[Service]:
    """Load all services once for the test session."""
    return load_services(DATA_PATH)


@pytest.fixture(scope="session")
def services_map(services) -> dict[str, Service]:
    return {s.id: s for s in services}
