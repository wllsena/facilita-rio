"""Latency benchmarking — per-component timing decomposition."""

from __future__ import annotations

import time

import numpy as np

from app.config import CONFIG
from app.models import RetrievalCandidate
from app.search.hybrid import weighted_rrf
from app.search.query_processor import expand_query

from .shared import SharedComponents


def benchmark_latency(queries: list[dict], n_warmup: int = 3, n_runs: int = 3,
                      *, shared: SharedComponents) -> dict:
    """Benchmark end-to-end search latency with per-component decomposition."""
    print(f"\n{'='*60}")
    print("LATENCY BENCHMARK")
    print(f"{'='*60}")

    services_map = shared.services_map
    bm25_index = shared.bm25_index
    vector_index = shared.vector_index
    reranker_model = shared.reranker

    from app.search.hybrid import HybridRetriever

    positive_queries = [q for q in queries if q["category"] != "negative"]
    sample = positive_queries[:20]

    # Warmup
    for q in sample[:n_warmup]:
        expanded = expand_query(q["query"])
        retriever = HybridRetriever(bm25_index, vector_index)
        candidates = retriever.search(q["query"], expanded_query=expanded, top_k=20)
        if reranker_model:
            reranker_model.rerank(q["query"], candidates, services_map, top_k=10, expanded_query=expanded)

    # Benchmark with per-component timing
    total_latencies = []
    expand_latencies = []
    bm25_latencies = []
    semantic_latencies = []
    rrf_latencies = []
    reranker_latencies = []

    for _ in range(n_runs):
        for q in sample:
            t_total = time.perf_counter()

            t0 = time.perf_counter()
            expanded = expand_query(q["query"])
            expand_latencies.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            bm25_results = bm25_index.search(expanded, top_k=CONFIG.bm25_top_k)
            bm25_latencies.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            semantic_results = vector_index.search(expanded, top_k=CONFIG.semantic_top_k)
            semantic_latencies.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            fused = weighted_rrf(
                [(bm25_results, CONFIG.bm25_weight), (semantic_results, CONFIG.semantic_weight)],
                k=CONFIG.rrf_k,
            )
            bm25_scores = dict(bm25_results)
            semantic_scores = dict(semantic_results)
            candidates = [
                RetrievalCandidate(doc_id, rrf_score, bm25_scores.get(doc_id), semantic_scores.get(doc_id))
                for doc_id, rrf_score in fused[:20]
            ]
            rrf_latencies.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            if reranker_model:
                reranker_model.rerank(
                    q["query"], candidates, services_map, top_k=10, expanded_query=expanded
                )
            reranker_latencies.append((time.perf_counter() - t0) * 1000)

            total_latencies.append((time.perf_counter() - t_total) * 1000)

    def _percentiles(arr):
        a = np.array(arr)
        return {
            "mean": round(float(a.mean()), 2),
            "p50": round(float(np.percentile(a, 50)), 2),
            "p90": round(float(np.percentile(a, 90)), 2),
            "p99": round(float(np.percentile(a, 99)), 2),
        }

    total_stats = _percentiles(total_latencies)
    components = {
        "expansion": _percentiles(expand_latencies),
        "bm25": _percentiles(bm25_latencies),
        "semantic": _percentiles(semantic_latencies),
        "rrf_fusion": _percentiles(rrf_latencies),
        "reranker": _percentiles(reranker_latencies),
    }

    n_measurements = len(total_latencies)
    stats = {
        "n_queries": len(sample),
        "n_runs": n_runs,
        "total_measurements": n_measurements,
        "mean_ms": total_stats["mean"],
        "p50_ms": total_stats["p50"],
        "p90_ms": total_stats["p90"],
        "p99_ms": total_stats["p99"],
        "min_ms": round(float(np.array(total_latencies).min()), 1),
        "max_ms": round(float(np.array(total_latencies).max()), 1),
        "components": components,
    }

    print(f"\nFull pipeline latency ({n_measurements} measurements, {len(sample)} queries x {n_runs} runs):")
    print(f"  total: mean={total_stats['mean']:.1f}ms  p50={total_stats['p50']:.1f}ms  p90={total_stats['p90']:.1f}ms  p99={total_stats['p99']:.1f}ms")
    print("\n  Per-component decomposition (p50):")
    for name, c in components.items():
        pct = (c["p50"] / total_stats["p50"] * 100) if total_stats["p50"] > 0 else 0
        print(f"    {name:<12} p50={c['p50']:>7.2f}ms  ({pct:>5.1f}%)")

    return stats
