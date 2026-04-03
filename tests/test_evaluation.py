"""Tests for the evaluation script structure and QREL validity."""

from __future__ import annotations

import json
from pathlib import Path

from app.config import DATA_PATH
from app.indexing.loader import load_services

QUERIES_PATH = Path(__file__).parent.parent / "evaluation" / "test_queries.json"
HOLDOUT_PATH = Path(__file__).parent.parent / "evaluation" / "holdout_queries.json"


def test_test_queries_file_exists():
    assert QUERIES_PATH.exists()


def test_test_queries_valid_json():
    with open(QUERIES_PATH) as f:
        data = json.load(f)
    assert "queries" in data
    assert len(data["queries"]) >= 60


def test_all_qrel_service_ids_exist():
    """Every service ID in the QRELs must exist in the actual catalog."""
    services = load_services(DATA_PATH)
    valid_ids = {s.id for s in services}

    with open(QUERIES_PATH) as f:
        data = json.load(f)

    bad_ids = []
    for q in data["queries"]:
        for sid in q["relevant"]:
            if sid not in valid_ids:
                bad_ids.append((q["id"], sid))

    assert bad_ids == [], f"Invalid service IDs in QRELs: {bad_ids}"


def test_all_queries_have_required_fields():
    with open(QUERIES_PATH) as f:
        data = json.load(f)

    for q in data["queries"]:
        assert "id" in q, "Missing 'id' in query"
        assert "category" in q, f"Missing 'category' in {q['id']}"
        assert "query" in q, f"Missing 'query' in {q['id']}"
        assert "relevant" in q, f"Missing 'relevant' in {q['id']}"
        assert len(q["query"]) > 0, f"Empty query in {q['id']}"


def test_query_categories_are_valid():
    with open(QUERIES_PATH) as f:
        data = json.load(f)

    valid_categories = {"direct", "natural", "synonym", "ambiguous", "edge", "negative"}
    for q in data["queries"]:
        assert q["category"] in valid_categories, (
            f"Invalid category '{q['category']}' in {q['id']}"
        )


def test_relevance_grades_are_valid():
    with open(QUERIES_PATH) as f:
        data = json.load(f)

    for q in data["queries"]:
        if q["category"] == "negative":
            assert q["relevant"] == {}, (
                f"Negative query {q['id']} should have empty relevance"
            )
        else:
            for sid, grade in q["relevant"].items():
                assert grade in (1, 2, 3), (
                    f"Invalid grade {grade} for {sid} in {q['id']}"
                )


# ── Holdout query validation ──────────────────────────────────────────────


def test_holdout_queries_file_exists():
    assert HOLDOUT_PATH.exists()


def test_holdout_qrel_service_ids_exist():
    """Every service ID in holdout QRELs must exist in the actual catalog."""
    services = load_services(DATA_PATH)
    valid_ids = {s.id for s in services}

    with open(HOLDOUT_PATH) as f:
        data = json.load(f)

    bad_ids = []
    for q in data["queries"]:
        for sid in q["relevant"]:
            if sid not in valid_ids:
                bad_ids.append((q["id"], sid))

    assert bad_ids == [], f"Invalid service IDs in holdout QRELs: {bad_ids}"


def test_holdout_queries_no_overlap_with_main():
    """Holdout queries must NOT overlap with main evaluation queries."""
    with open(QUERIES_PATH) as f:
        main_data = json.load(f)
    with open(HOLDOUT_PATH) as f:
        holdout_data = json.load(f)

    main_texts = {q["query"].lower().strip() for q in main_data["queries"]}
    holdout_texts = {q["query"].lower().strip() for q in holdout_data["queries"]}

    overlap = main_texts & holdout_texts
    assert overlap == set(), f"Holdout queries overlap with main: {overlap}"
