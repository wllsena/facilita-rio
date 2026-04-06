"""Semantic weight sweep — finds optimal semantic-vs-BM25 balance with empirical data."""

from __future__ import annotations

import json
from pathlib import Path

from ranx import Qrels, Run, evaluate

from app.config import CONFIG
from app.models import RetrievalCandidate
from app.search.hybrid import weighted_rrf
from app.search.query_processor import expand_query

from .shared import SharedComponents


def sweep_semantic_weight(*, shared: SharedComponents) -> dict:
    """Sweep semantic weight from 1.0 to 3.0 and report nDCG@5 and MRR@10."""
    print(f"\n{'='*60}")
    print("SEMANTIC WEIGHT SWEEP")
    print(f"{'='*60}")

    queries_path = Path(__file__).parent / "test_queries.json"
    with open(queries_path) as f:
        queries = json.load(f)["queries"]

    positive_queries = [q for q in queries if q["category"] != "negative"]
    weights = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    # Also evaluate on holdout if available
    holdout_path = Path(__file__).parent / "holdout_queries.json"
    holdout_queries = []
    if holdout_path.exists():
        with open(holdout_path) as f:
            holdout_queries = [
                q for q in json.load(f)["queries"]
                if q.get("relevant")
            ]

    results_table: list[dict] = []

    for sem_w in weights:
        row = {"semantic_weight": sem_w}

        for label, query_set in [("main", positive_queries), ("holdout", holdout_queries)]:
            if not query_set:
                continue

            qrels_dict: dict[str, dict[str, int]] = {}
            run_dict: dict[str, dict[str, float]] = {}

            for q in query_set:
                qid = q["id"]
                qrels_dict[qid] = q["relevant"]
                query = q["query"]
                expanded = expand_query(query)

                bm25_results = shared.bm25_index.search(expanded, top_k=CONFIG.bm25_top_k)
                semantic_results = shared.vector_index.search(expanded, top_k=CONFIG.semantic_top_k)

                fused = weighted_rrf(
                    [(bm25_results, CONFIG.bm25_weight), (semantic_results, sem_w)],
                    k=CONFIG.rrf_k,
                )

                reranked = shared.reranker.rerank(
                    query,
                    [
                        RetrievalCandidate(
                            doc_id=doc_id, rrf_score=rrf_score,
                            bm25_score=dict(bm25_results).get(doc_id),
                            semantic_score=dict(semantic_results).get(doc_id),
                        )
                        for doc_id, rrf_score in fused[:20]
                    ],
                    shared.services_map, top_k=10, expanded_query=expanded,
                )
                run_dict[qid] = {r.doc_id: float(r.blended_score) for r in reranked}

            qrels = Qrels(qrels_dict)
            run = Run(run_dict)
            metrics = evaluate(qrels, run, ["ndcg@5", "mrr@10"])
            row[f"{label}_ndcg@5"] = round(float(metrics["ndcg@5"]), 4)
            row[f"{label}_mrr@10"] = round(float(metrics["mrr@10"]), 4)

        results_table.append(row)

    # Print results
    has_holdout = any("holdout_ndcg@5" in r for r in results_table)
    header = f"  {'Weight':>8} {'nDCG@5':>10} {'MRR@10':>10}"
    if has_holdout:
        header += f" {'H-nDCG@5':>10} {'H-MRR@10':>10}"
    header += f" {'Note':>15}"
    print(f"\n{header}")
    print("  " + "-" * (65 if has_holdout else 50))

    for row in results_table:
        note = "← active" if abs(row["semantic_weight"] - CONFIG.semantic_weight) < 1e-6 else ""
        line = f"  {row['semantic_weight']:>8.1f} {row['main_ndcg@5']:>10.4f} {row['main_mrr@10']:>10.4f}"
        if has_holdout:
            line += f" {row.get('holdout_ndcg@5', 0):>10.4f} {row.get('holdout_mrr@10', 0):>10.4f}"
        line += f" {note:>15}"
        print(line)

    best = max(results_table, key=lambda r: r["main_ndcg@5"])
    print(f"\n  Best main nDCG@5: {best['main_ndcg@5']:.4f} at semantic_weight={best['semantic_weight']:.1f}")
    if has_holdout:
        best_ho = max(results_table, key=lambda r: r.get("holdout_ndcg@5", 0))
        print(f"  Best holdout nDCG@5: {best_ho['holdout_ndcg@5']:.4f} at semantic_weight={best_ho['semantic_weight']:.1f}")

    return {"sweep": results_table, "best_weight": best["semantic_weight"]}
