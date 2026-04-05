"""Holdout validation — evaluate on queries created after all tuning."""

from __future__ import annotations

import json
from pathlib import Path

from ranx import Qrels, Run, evaluate

from .shared import SharedComponents
from .variants import METRICS, run_search


def evaluate_holdout(*, shared: SharedComponents) -> dict | None:
    """Evaluate the full pipeline on holdout queries never used during development."""
    holdout_path = Path(__file__).parent / "holdout_queries.json"
    if not holdout_path.exists():
        return None

    with open(holdout_path) as f:
        data = json.load(f)
    holdout_queries = data["queries"]

    n_positive = sum(1 for q in holdout_queries if q.get("relevant"))
    n_negative = len(holdout_queries) - n_positive

    print(f"\n{'='*60}")
    print(f"HOLDOUT VALIDATION — {n_positive} positive + {n_negative} negative unseen queries")
    print(f"{'='*60}")
    print("These queries were created AFTER all tuning was finalized.")
    print("Metrics here validate generalization, not training-set fit.\n")

    services_map = shared.services_map
    bm25_index = shared.bm25_index
    vector_index = shared.vector_index
    reranker_model = shared.reranker

    positive_queries = [q for q in holdout_queries if q.get("relevant")]
    qrels_dict: dict[str, dict[str, int]] = {}
    run_dict: dict[str, dict[str, float]] = {}
    per_query_results = []

    for q in positive_queries:
        qid = q["id"]
        query = q["query"]
        qrels_dict[qid] = q["relevant"]

        results = run_search(
            query, bm25_index, vector_index, services_map, reranker_model, "full", 10
        )
        run_dict[qid] = {doc_id: float(score) for doc_id, score in results}

        per_query_results.append({
            "query_id": qid,
            "query": query,
            "category": q["category"],
            "top_results": [doc_id for doc_id, _ in results[:5]],
        })

    qrels = Qrels(qrels_dict)
    run = Run(run_dict, name="full_holdout")
    results = evaluate(qrels, run, METRICS)

    print(f"Holdout results (full pipeline) — {len(positive_queries)} queries:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

    # Per-category breakdown
    categories = sorted({pq["category"] for pq in per_query_results})
    for cat in categories:
        cat_qids = [pq["query_id"] for pq in per_query_results if pq["category"] == cat]
        if not cat_qids:
            continue
        cat_qrels = Qrels({qid: qrels_dict[qid] for qid in cat_qids})
        cat_run = Run({qid: run_dict[qid] for qid in cat_qids})
        cat_results = evaluate(cat_qrels, cat_run, ["ndcg@5", "mrr@10"])
        print(f"\n  {cat} queries ({len(cat_qids)}):")
        print(f"    nDCG@5: {cat_results['ndcg@5']:.4f}  |  MRR@10: {cat_results['mrr@10']:.4f}")

    # Per-query failure analysis
    sorted_qids = sorted(qrels_dict.keys())
    pq_mrr_arr = evaluate(qrels, run, "mrr@10", return_mean=False)
    qid_to_mrr = {qid: float(pq_mrr_arr[i]) for i, qid in enumerate(sorted_qids)}

    failures = []
    for pq in per_query_results:
        mrr_val = qid_to_mrr.get(pq["query_id"], 0.0)
        if mrr_val < 0.5:
            failures.append(pq | {"mrr@10": mrr_val})

    if failures:
        print(f"\n  Holdout failure cases (MRR@10 < 0.5): {len(failures)}")
        for f in failures:
            print(f"    [{f['query_id']}] \"{f['query']}\" -> MRR={f['mrr@10']:.4f}, top3={f['top_results'][:3]}")
    else:
        print("\n  No failure cases (all MRR@10 >= 0.5)")

    # Comparison with main eval set
    print("\n  Generalization check:")
    print(f"    Holdout set nDCG@5:      {results['ndcg@5']:.4f}")
    main_ndcg5 = _get_main_ndcg5()
    print(f"    Main eval set nDCG@5:    {main_ndcg5:.4f}")
    gap = abs(results["ndcg@5"] - main_ndcg5)
    print(f"    Gap:                     {gap:.4f} ({'acceptable' if gap < 0.05 else 'CONCERNING'})")

    summary = {"n_queries": len(positive_queries)}
    summary.update({k: round(float(v), 4) for k, v in results.items()})
    return {"summary": summary, "per_query": per_query_results}


def _get_main_ndcg5() -> float:
    """Read the main eval nDCG@5 from current results file."""
    results_path = Path(__file__).parent / "results" / "evaluation_results.json"
    if results_path.exists():
        try:
            with open(results_path) as f:
                data = json.load(f)
            if "full" in data:
                return data["full"]["summary"].get("ndcg@5", 0.0)
        except (json.JSONDecodeError, KeyError):
            pass
    return 0.0
