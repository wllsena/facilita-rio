"""Evaluation orchestrator — runs all evaluation components and produces comparison report."""

from __future__ import annotations

import json
from pathlib import Path

from ranx import compare

from app.observability import setup_logging

from .ce_sweep import sweep_ce_weight
from .failure_analysis import analyze_failures
from .holdout import evaluate_holdout
from .latency import benchmark_latency
from .popular import evaluate_popular_queries
from .recommendations import evaluate_recommendations
from .semantic_weight_sweep import sweep_semantic_weight
from .shared import build_shared_components
from .variants import METRICS, evaluate_variant


def main():
    setup_logging()

    # Load test queries
    queries_path = Path(__file__).parent / "test_queries.json"
    with open(queries_path) as f:
        data = json.load(f)
    queries = data["queries"]

    print(f"Loaded {len(queries)} test queries")

    # Build shared components once (avoids reloading models 6 times)
    print("Building shared pipeline components...")
    shared = build_shared_components()

    # Run ablation study
    variants = ["bm25_only", "bm25_expanded", "semantic_only", "semantic_expanded", "hybrid_no_rerank", "full"]
    all_results = {}
    all_runs = {}
    shared_qrels = None

    for variant in variants:
        result, qrels, run = evaluate_variant(variant, queries, shared=shared)
        all_results[variant] = result
        all_runs[variant] = run
        shared_qrels = qrels

    rec_results = evaluate_recommendations(queries, shared=shared)
    all_results["recommendations"] = rec_results

    holdout_results = evaluate_holdout(shared=shared)
    if holdout_results:
        all_results["holdout"] = holdout_results

    popular_results = evaluate_popular_queries(shared=shared)
    if popular_results:
        all_results["popular_queries"] = popular_results

    latency_results = benchmark_latency(queries, shared=shared)
    all_results["latency"] = latency_results

    failure_results = analyze_failures(shared=shared)
    all_results["failure_analysis"] = failure_results

    ce_sweep_results = sweep_ce_weight(shared=shared)
    all_results["ce_sweep"] = ce_sweep_results

    sem_sweep_results = sweep_semantic_weight(shared=shared)
    all_results["semantic_weight_sweep"] = sem_sweep_results

    # Save results
    results_path = Path(__file__).parent / "results" / "evaluation_results.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n\nResults saved to {results_path}")

    # Print comparison table
    print("\n" + "=" * 70)
    print("ABLATION COMPARISON")
    print("=" * 70)
    header = f"{'Variant':<28}"
    for m in METRICS:
        header += f" {m:>12}"
    print(header)
    print("-" * 88)
    for variant in variants:
        s = all_results[variant]["summary"]
        row = f"{variant:<28}"
        for m in METRICS:
            row += f" {s.get(m, 0):>12.4f}"
        print(row)

    # Statistical significance comparison
    if shared_qrels and len(all_runs) > 1:
        print("\n" + "=" * 70)
        print("STATISTICAL SIGNIFICANCE (Fisher's randomization test, p<0.05)")
        print("=" * 70)
        runs_list = [all_runs[v] for v in variants]
        try:
            report = compare(
                shared_qrels,
                runs=runs_list,
                metrics=["ndcg@5", "mrr@10"],
                max_p=0.05,
            )
            print(report)
        except Exception as e:
            print(f"  Could not run significance test: {e}")


if __name__ == "__main__":
    main()
