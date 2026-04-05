"""Cross-encoder weight sweep — finds optimal CE weight with empirical data."""

from __future__ import annotations

import json
from pathlib import Path

from ranx import Qrels, Run, evaluate

from app.config import CONFIG

from .shared import SharedComponents


def sweep_ce_weight(*, shared: SharedComponents) -> dict:
    """Sweep cross-encoder weight from 0% to 30% and report nDCG@5 and MRR@10."""
    print(f"\n{'='*60}")
    print("CROSS-ENCODER WEIGHT SWEEP")
    print(f"{'='*60}")

    queries_path = Path(__file__).parent / "test_queries.json"
    with open(queries_path) as f:
        queries = json.load(f)["queries"]

    positive_queries = [q for q in queries if q["category"] != "negative"]
    weights = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]

    results_table: list[dict] = []

    for ce_w in weights:
        qrels_dict: dict[str, dict[str, int]] = {}
        run_dict: dict[str, dict[str, float]] = {}

        for q in positive_queries:
            qid = q["id"]
            qrels_dict[qid] = q["relevant"]

            # Run full pipeline with custom CE weight
            from app.search.hybrid import HybridRetriever
            from app.search.query_processor import expand_query

            query = q["query"]
            expanded = expand_query(query)
            retriever = HybridRetriever(shared.bm25_index, shared.vector_index)
            candidates = retriever.search(query, expanded_query=expanded, top_k=20)
            reranked = shared.reranker.rerank(
                query, candidates, shared.services_map, top_k=10,
                expanded_query=expanded, ce_weight=ce_w,
            )
            run_dict[qid] = {r.doc_id: float(r.blended_score) for r in reranked}

        qrels = Qrels(qrels_dict)
        run = Run(run_dict)
        metrics = evaluate(qrels, run, ["ndcg@5", "mrr@10"])

        row = {
            "ce_weight": ce_w,
            "ndcg@5": round(float(metrics["ndcg@5"]), 4),
            "mrr@10": round(float(metrics["mrr@10"]), 4),
        }
        results_table.append(row)

    print(f"\n  {'CE Weight':>10} {'nDCG@5':>10} {'MRR@10':>10} {'Note':>20}")
    print("  " + "-" * 55)
    for row in results_table:
        note = "← active" if abs(row["ce_weight"] - CONFIG.ce_weight) < 1e-6 else ""
        print(f"  {row['ce_weight']:>10.0%} {row['ndcg@5']:>10.4f} {row['mrr@10']:>10.4f} {note:>20}")

    best = max(results_table, key=lambda r: r["ndcg@5"])
    print(f"\n  Best nDCG@5: {best['ndcg@5']:.4f} at ce_weight={best['ce_weight']:.0%}")

    return {"sweep": results_table, "best_weight": best["ce_weight"]}
