"""Scalability benchmark — measure search latency with synthetic catalogs of increasing size.

Usage:
    python -m evaluation.benchmark_scale
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import numpy as np

from app.config import CONFIG, DATA_PATH
from app.indexing.bm25_index import BM25Index
from app.indexing.loader import load_services
from app.indexing.vector_index import VectorIndex
from app.models import RetrievalCandidate, Service
from app.search.hybrid import weighted_rrf
from app.search.query_processor import expand_query
from app.search.reranker import Reranker


def _load_benchmark_queries(n: int = 10) -> list[str]:
    """Load a sample of queries from test_queries.json for benchmarking."""
    path = Path(__file__).parent / "test_queries.json"
    with open(path) as f:
        data = json.load(f)
    positive = [q["query"] for q in data["queries"] if q["category"] != "negative"]
    return positive[:n]


def create_synthetic_catalog(base_services: list[Service], target_size: int) -> list[Service]:
    """Duplicate services with unique IDs to simulate a larger catalog."""
    synthetic = list(base_services)  # keep originals
    n_base = len(base_services)

    suffixes = [f"region-{chr(65 + i)}" for i in range(20)]

    i = 0
    while len(synthetic) < target_size:
        base = base_services[i % n_base]
        suffix_idx = (i // n_base) % len(suffixes)
        suffix = suffixes[suffix_idx]
        variant_num = i // (n_base * len(suffixes)) + 1

        new_id = f"{base.id}-{suffix}-v{variant_num}-{uuid.uuid4().hex[:6]}"
        new_service = Service(
            id=new_id,
            nome=f"{base.nome} ({suffix.replace('-', ' ').title()} {variant_num})",
            resumo=base.resumo,
            descricao_completa=base.descricao_completa,
            tema=base.tema,
            orgao_gestor=base.orgao_gestor,
            custo=base.custo,
            publico=base.publico,
            tempo_atendimento=base.tempo_atendimento,
            instrucoes=base.instrucoes,
            resultado=base.resultado,
        )
        synthetic.append(new_service)
        i += 1

    return synthetic


def benchmark_at_scale(
    services: list[Service],
    reranker: Reranker,
    queries: list[str],
    n_warmup: int = 2,
    n_runs: int = 3,
) -> dict:
    """Benchmark search latency at a given catalog size."""
    services_map = {s.id: s for s in services}

    # Build indexes
    t_index_start = time.perf_counter()
    bm25_index = BM25Index(services)
    t_bm25_done = time.perf_counter()
    vector_index = VectorIndex(services)
    t_index_done = time.perf_counter()

    indexing_ms = {
        "bm25": round((t_bm25_done - t_index_start) * 1000, 1),
        "vector": round((t_index_done - t_bm25_done) * 1000, 1),
        "total": round((t_index_done - t_index_start) * 1000, 1),
    }

    # Warmup
    for q in queries[:n_warmup]:
        expanded = expand_query(q)
        bm25_results = bm25_index.search(expanded, top_k=CONFIG.bm25_top_k)
        semantic_results = vector_index.search(expanded, top_k=CONFIG.semantic_top_k)
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
        reranker.rerank(q, candidates, services_map, top_k=10, expanded_query=expanded)

    # Benchmark
    total_latencies = []
    expand_latencies = []
    bm25_latencies = []
    semantic_latencies = []
    rrf_latencies = []
    reranker_latencies = []

    for _ in range(n_runs):
        for q in queries:
            t_total = time.perf_counter()

            t0 = time.perf_counter()
            expanded = expand_query(q)
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
            reranker.rerank(q, candidates, services_map, top_k=10, expanded_query=expanded)
            reranker_latencies.append((time.perf_counter() - t0) * 1000)

            total_latencies.append((time.perf_counter() - t_total) * 1000)

    def _stats(arr):
        a = np.array(arr)
        return {
            "mean": round(float(a.mean()), 2),
            "p50": round(float(np.percentile(a, 50)), 2),
            "p90": round(float(np.percentile(a, 90)), 2),
            "p99": round(float(np.percentile(a, 99)), 2),
        }

    return {
        "n_services": len(services),
        "n_queries": len(queries),
        "n_runs": n_runs,
        "n_measurements": len(total_latencies),
        "indexing_ms": indexing_ms,
        "total": _stats(total_latencies),
        "components": {
            "expansion": _stats(expand_latencies),
            "bm25": _stats(bm25_latencies),
            "semantic": _stats(semantic_latencies),
            "rrf_fusion": _stats(rrf_latencies),
            "reranker": _stats(reranker_latencies),
        },
        "memory_estimate_mb": _estimate_memory(services),
    }


def _estimate_memory(services: list[Service]) -> float:
    """Rough memory estimate for the index footprint."""
    n = len(services)
    # E5-small: 384-dim float32 per service = 384 * 4 bytes
    vector_bytes = n * 384 * 4
    # BM25: rough estimate ~2KB per service (tokenized corpus)
    bm25_bytes = n * 2048
    # Total in MB
    return round((vector_bytes + bm25_bytes) / (1024 * 1024), 1)


def main():
    print("=" * 70)
    print("SCALABILITY BENCHMARK — Measured Latency at Different Catalog Sizes")
    print("=" * 70)

    base_services = load_services(DATA_PATH)
    print(f"\nBase catalog: {len(base_services)} services")

    benchmark_queries = _load_benchmark_queries()
    print(f"Benchmark queries: {len(benchmark_queries)}")

    print("Loading reranker model (shared across benchmarks)...")
    reranker = Reranker()

    scale_points = [50, 200, 500, 1000]
    results = []

    for target_size in scale_points:
        print(f"\n{'─' * 50}")
        print(f"Benchmarking at {target_size} services...")
        print(f"{'─' * 50}")

        if target_size <= len(base_services):
            services = base_services[:target_size]
        else:
            services = create_synthetic_catalog(base_services, target_size)

        print(f"  Catalog size: {len(services)} services")

        result = benchmark_at_scale(services, reranker, benchmark_queries)
        results.append(result)

        total = result["total"]
        idx = result["indexing_ms"]
        mem = result["memory_estimate_mb"]
        print(f"  Indexing: BM25={idx['bm25']:.0f}ms, Vector={idx['vector']:.0f}ms, Total={idx['total']:.0f}ms")
        print(f"  Latency: p50={total['p50']:.1f}ms, p90={total['p90']:.1f}ms, p99={total['p99']:.1f}ms")
        print(f"  Index memory estimate: {mem:.1f}MB")

        # Component breakdown
        print("  Components (p50):")
        for name, stats in result["components"].items():
            pct = (stats["p50"] / total["p50"] * 100) if total["p50"] > 0 else 0
            print(f"    {name:<12} {stats['p50']:>7.2f}ms  ({pct:>5.1f}%)")

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY: Latency Scaling")
    print(f"{'=' * 70}")
    print(f"{'Services':>10} | {'p50 (ms)':>10} | {'p90 (ms)':>10} | {'p99 (ms)':>10} | {'Index (ms)':>10} | {'Mem (MB)':>10}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['n_services']:>10} | "
            f"{r['total']['p50']:>10.1f} | "
            f"{r['total']['p90']:>10.1f} | "
            f"{r['total']['p99']:>10.1f} | "
            f"{r['indexing_ms']['total']:>10.0f} | "
            f"{r['memory_estimate_mb']:>10.1f}"
        )

    # Validate projections
    print(f"\n{'=' * 70}")
    print("PROJECTION VALIDATION")
    print(f"{'=' * 70}")
    if len(results) >= 2:
        base = results[0]  # 50 services
        largest = results[-1]  # 1000 services
        p50_ratio = largest["total"]["p50"] / base["total"]["p50"] if base["total"]["p50"] > 0 else 0
        print(f"  {base['n_services']}→{largest['n_services']} services: "
              f"p50 went from {base['total']['p50']:.1f}ms to {largest['total']['p50']:.1f}ms "
              f"({p50_ratio:.2f}x)")

        # BM25 scaling
        bm25_base = base["components"]["bm25"]["p50"]
        bm25_large = largest["components"]["bm25"]["p50"]
        bm25_ratio = bm25_large / bm25_base if bm25_base > 0 else 0
        print(f"  BM25: {bm25_base:.2f}ms ��� {bm25_large:.2f}ms ({bm25_ratio:.1f}x) — "
              f"{'linear as projected' if bm25_ratio < largest['n_services'] / base['n_services'] * 1.5 else 'WORSE than linear'}")

        # Semantic scaling
        sem_base = base["components"]["semantic"]["p50"]
        sem_large = largest["components"]["semantic"]["p50"]
        sem_ratio = sem_large / sem_base if sem_base > 0 else 0
        print(f"  Semantic: {sem_base:.2f}ms → {sem_large:.2f}ms ({sem_ratio:.1f}x) — "
              f"{'mostly E5 encoding (constant)' if sem_ratio < 2 else 'scaling concern'}")

        # Reranker (should be constant — always processes top-20)
        re_base = base["components"]["reranker"]["p50"]
        re_large = largest["components"]["reranker"]["p50"]
        re_ratio = re_large / re_base if re_base > 0 else 0
        print(f"  Reranker: {re_base:.2f}ms → {re_large:.2f}ms ({re_ratio:.1f}x) — "
              f"{'constant as projected (top-20 fixed)' if re_ratio < 1.3 else 'unexpected scaling'}")

    # Save results
    results_path = Path(__file__).parent / "results" / "scale_benchmark.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
