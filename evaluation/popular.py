"""Popular queries — colloquial query top-3 accuracy."""

from __future__ import annotations

import json
from pathlib import Path

from .shared import SharedComponents
from .variants import run_search


def evaluate_popular_queries(*, shared: SharedComponents) -> dict:
    """Evaluate top-3 accuracy on 500 colloquial citizen queries."""
    popular_path = Path(__file__).parent / "queries_populares.json"
    if not popular_path.exists():
        return {}

    with open(popular_path) as f:
        data = json.load(f)

    print(f"\n{'='*60}")
    print("POPULAR QUERIES — 500 colloquial queries (top-3 accuracy)")
    print(f"{'='*60}")

    services_map = shared.services_map
    bm25_index = shared.bm25_index
    vector_index = shared.vector_index
    reranker_model = shared.reranker

    total = 0
    hits = 0
    failures_by_service: dict[str, list[str]] = {}

    for service_entry in data["queries"]:
        expected_id = service_entry["service_id"]
        service_name = service_entry["service_name"]

        for query in service_entry["queries"]:
            total += 1
            results = run_search(
                query, bm25_index, vector_index, services_map, reranker_model, "full", 10
            )
            top3_ids = [doc_id for doc_id, _ in results[:3]]

            if expected_id in top3_ids:
                hits += 1
            else:
                if service_name not in failures_by_service:
                    failures_by_service[service_name] = []
                failures_by_service[service_name].append(query)

    accuracy = hits / total if total else 0
    misses = total - hits

    print(f"\n  Total queries:     {total}")
    print(f"  Top-3 hits:        {hits}")
    print(f"  Top-3 misses:      {misses}")
    print(f"  Top-3 accuracy:    {accuracy:.1%}")

    if failures_by_service:
        print(f"\n  Failures by service ({len(failures_by_service)} services affected):")
        for svc_name, queries in sorted(failures_by_service.items(), key=lambda x: -len(x[1])):
            print(f"    {svc_name} ({len(queries)} failures):")
            for q in queries[:3]:
                print(f"      - \"{q}\"")
            if len(queries) > 3:
                print(f"      ... and {len(queries) - 3} more")

    return {
        "total_queries": total,
        "top3_hits": hits,
        "top3_misses": misses,
        "top3_accuracy": round(accuracy, 4),
        "failures_by_service": {k: v for k, v in failures_by_service.items()},
    }
