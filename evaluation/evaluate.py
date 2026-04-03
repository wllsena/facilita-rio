"""Evaluation script — compute IR metrics using ranx against hand-crafted relevance judgments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ranx import Qrels, Run, compare, evaluate

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CONFIG, DATA_PATH
from app.indexing.bm25_index import BM25Index
from app.indexing.loader import load_services
from app.indexing.vector_index import VectorIndex
from app.observability.logging import setup_logging
from app.search.hybrid import HybridRetriever, weighted_rrf
from app.search.query_processor import expand_query
from app.search.reranker import Reranker

METRICS = ["ndcg@5", "ndcg@10", "precision@5", "mrr@10", "recall@10"]


# ── Evaluation pipeline ────────────────────────────────────────────────────


def build_pipeline(variant: str = "full"):
    """Build search pipeline for a given variant."""
    services = load_services(DATA_PATH)
    services_map = {s.id: s for s in services}

    bm25_index = BM25Index(services)
    vector_index = VectorIndex(services)

    reranker_model = None
    if variant == "full":
        reranker_model = Reranker()

    return services_map, bm25_index, vector_index, reranker_model


def run_search(
    query: str,
    bm25_index: BM25Index,
    vector_index: VectorIndex,
    services_map: dict,
    reranker_model: Reranker | None,
    variant: str = "full",
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Run search and return ranked list of (service_id, score) pairs."""
    # Apply local synonym expansion (same as production pipeline)
    expanded = expand_query(query)

    if variant == "bm25_only":
        return bm25_index.search(query, top_k=top_k)

    elif variant == "semantic_only":
        return vector_index.search(query, top_k=top_k)

    elif variant == "semantic_expanded":
        return vector_index.search(expanded, top_k=top_k)

    elif variant == "hybrid_no_rerank":
        bm25_results = bm25_index.search(query, top_k=CONFIG.bm25_top_k)
        semantic_results = vector_index.search(expanded, top_k=CONFIG.semantic_top_k)
        fused = weighted_rrf(
            [(bm25_results, 1.0), (semantic_results, 2.0)], k=CONFIG.rrf_k
        )
        return fused[:top_k]

    else:  # full pipeline
        retriever = HybridRetriever(bm25_index, vector_index)
        candidates = retriever.search(query, expanded_query=expanded, top_k=top_k * 2)
        if reranker_model:
            reranked = reranker_model.rerank(query, candidates, services_map, top_k=top_k)
            return [(doc_id, score) for doc_id, _rrf, _bm25, _sem, score in reranked]
        return [(doc_id, rrf) for doc_id, rrf, *_ in candidates[:top_k]]


def evaluate_variant(
    variant: str, queries: list[dict], top_k: int = 10
) -> tuple[dict, Qrels, Run]:
    """Evaluate a pipeline variant across all test queries using ranx."""
    print(f"\n{'='*60}")
    print(f"Evaluating variant: {variant}")
    print(f"{'='*60}")

    services_map, bm25_index, vector_index, reranker_model = build_pipeline(variant)

    # Separate negative queries (no relevant docs — cannot use standard IR metrics)
    positive_queries = [q for q in queries if q["category"] != "negative"]
    negative_queries = [q for q in queries if q["category"] == "negative"]

    # Build ranx Qrels and Run dicts (positive queries only)
    qrels_dict: dict[str, dict[str, int]] = {}
    run_dict: dict[str, dict[str, float]] = {}

    per_query_results = []

    for q in positive_queries:
        qid = q["id"]
        query = q["query"]
        qrels_dict[qid] = q["relevant"]

        # Run search
        results = run_search(
            query, bm25_index, vector_index, services_map, reranker_model, variant, top_k
        )
        run_dict[qid] = {doc_id: float(score) for doc_id, score in results}

        per_query_results.append({
            "query_id": qid,
            "query": query,
            "category": q["category"],
            "top_results": [doc_id for doc_id, _ in results[:5]],
        })

    # Evaluate positive queries with ranx
    qrels = Qrels(qrels_dict)
    run = Run(run_dict, name=variant)

    results = evaluate(qrels, run, METRICS)

    # Print summary
    print(f"\nResults ({variant}) — {len(positive_queries)} positive queries:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

    # Per-category breakdown using ranx per-query evaluation
    categories = sorted({pq["category"] for pq in per_query_results})
    for cat in categories:
        cat_qids = [pq["query_id"] for pq in per_query_results if pq["category"] == cat]
        if not cat_qids:
            continue
        # Filter qrels and run to this category
        cat_qrels = Qrels({qid: qrels_dict[qid] for qid in cat_qids})
        cat_run = Run({qid: run_dict[qid] for qid in cat_qids})
        cat_results = evaluate(cat_qrels, cat_run, ["ndcg@5", "mrr@10"])
        print(f"\n  {cat} queries ({len(cat_qids)}):")
        print(f"    nDCG@5: {cat_results['ndcg@5']:.4f}  |  MRR@10: {cat_results['mrr@10']:.4f}")

    # Negative query analysis — measure false positive rate + confidence signal
    if negative_queries:
        print(f"\n  Negative queries ({len(negative_queries)}) — out-of-scope robustness:")

        # Compute avg positive query top score AND max cosine similarity for context
        pos_top_scores = []
        pos_max_cosines = []
        for pq in per_query_results:
            if pq["query_id"] in run_dict and run_dict[pq["query_id"]]:
                pos_top_scores.append(max(run_dict[pq["query_id"]].values()))
            # Max cosine similarity for confidence analysis
            sem_results = vector_index.search(pq["query"], top_k=1)
            if sem_results:
                pos_max_cosines.append(sem_results[0][1])
        avg_pos_top = sum(pos_top_scores) / len(pos_top_scores) if pos_top_scores else 0
        avg_pos_cosine = sum(pos_max_cosines) / len(pos_max_cosines) if pos_max_cosines else 0
        print(f"    (avg positive query top_score: {avg_pos_top:.4f} for comparison)")
        print(f"    (avg positive query max_cosine: {avg_pos_cosine:.4f})")

        neg_results = []
        neg_max_cosines = []
        for q in negative_queries:
            # Get BM25 hit count separately for diagnostic signal
            bm25_hits = len(bm25_index.search(q["query"], top_k=top_k))

            # Max cosine similarity — confidence signal for out-of-scope detection
            sem_results = vector_index.search(q["query"], top_k=1)
            max_cosine = sem_results[0][1] if sem_results else 0.0
            neg_max_cosines.append(max_cosine)

            neg_search = run_search(
                q["query"], bm25_index, vector_index, services_map, reranker_model, variant, top_k
            )
            n_returned = len(neg_search)
            top_score = neg_search[0][1] if neg_search else 0.0
            neg_results.append({
                "query_id": q["id"],
                "query": q["query"],
                "category": "negative",
                "top_results": [doc_id for doc_id, _ in neg_search[:5]],
                "n_returned": n_returned,
                "top_score": top_score,
                "max_cosine": max_cosine,
                "bm25_hits": bm25_hits,
            })
            print(
                f"    [{q['id']}] \"{q['query']}\" -> "
                f"{n_returned} results, top_score={top_score:.4f}, "
                f"max_cosine={max_cosine:.4f}, bm25_hits={bm25_hits}"
            )
        per_query_results.extend(neg_results)

        # Confidence threshold analysis
        avg_neg_cosine = sum(neg_max_cosines) / len(neg_max_cosines) if neg_max_cosines else 0
        cosine_gap = avg_pos_cosine - avg_neg_cosine
        # Test threshold at 0.84 (midpoint between avg positive and negative)
        threshold = 0.84
        neg_flagged = sum(1 for c in neg_max_cosines if c < threshold)
        pos_flagged = sum(1 for c in pos_max_cosines if c < threshold)
        print("\n  Confidence analysis (max cosine similarity):")
        print(f"    avg positive: {avg_pos_cosine:.4f}")
        print(f"    avg negative: {avg_neg_cosine:.4f}")
        print(f"    gap: {cosine_gap:.4f}")
        print(f"    threshold={threshold}: flags {neg_flagged}/{len(neg_max_cosines)} negatives, "
              f"{pos_flagged}/{len(pos_max_cosines)} false positives")

    # Failure analysis — compute per-query MRR using per-query evaluation
    # (ranx returns per-query values in alphabetical qid order)
    sorted_qids = sorted(qrels_dict.keys())
    pq_mrr_arr = evaluate(qrels, run, "mrr@10", return_mean=False)
    qid_to_mrr = {qid: float(pq_mrr_arr[i]) for i, qid in enumerate(sorted_qids)}

    failures = []
    for pq in per_query_results:
        qid = pq["query_id"]
        mrr_val = qid_to_mrr.get(qid, 0.0)
        pq["mrr@10"] = mrr_val
        if pq["category"] != "negative" and mrr_val < 0.5:
            failures.append(pq)

    if failures:
        print(f"\n  Failure cases (MRR@10 < 0.5): {len(failures)}")
        for f in failures[:5]:
            print(f"    [{f['query_id']}] \"{f['query']}\" -> MRR={f['mrr@10']:.4f}, top3={f['top_results'][:3]}")

    summary = {"variant": variant, "n_queries": len(positive_queries), "n_negative": len(negative_queries)}
    summary.update({k: round(float(v), 4) for k, v in results.items()})

    return {"summary": summary, "per_query": per_query_results}, qrels, run


def evaluate_recommendations(queries: list[dict]) -> dict:
    """Evaluate recommendation quality — the second core deliverable.

    Measures:
    - Category coherence: % of recs sharing a category with search results
    - Journey hit rate: % of queries where a citizen-journey link appears
    - Deduplication: % of recs NOT in search results (should be 100%)
    - Rec precision: for queries with expected recs, % that match
    """
    from app.indexing.cluster_builder import ClusterIndex
    from app.recommendation.recommender import CITIZEN_JOURNEYS, Recommender

    print(f"\n{'='*60}")
    print("RECOMMENDATION QUALITY EVALUATION")
    print(f"{'='*60}")

    services = load_services(DATA_PATH)
    services_map = {s.id: s for s in services}
    bm25_index = BM25Index(services)
    vector_index = VectorIndex(services)
    cluster_index = ClusterIndex(services, vector_index.embeddings)
    reranker_model = Reranker()
    recommender = Recommender(services_map, vector_index, cluster_index)

    # Expected recommendations for key queries (mini rec-QREL).
    # These capture citizen journeys: "if you searched for X, you should see Y."
    REC_QRELS: dict[str, list[str]] = {
        # Tax journey: IPTU → consult payments, installments, clearance cert
        "segunda via IPTU": [
            "iptu-consulta-a-pagamentos-e-debito-automatico-b175364b",
            "parcelamento-de-debitos-em-divida-ativa-6ba1f0f4",
            "certidao-negativa-de-debito-nada-consta-439306e1",
        ],
        # Pregnancy journey: maternity → baby kit, Bolsa Família
        "minha esposa está grávida": [
            "distribuicao-de-kit-enxoval-do-bebe-77f09458",
            "informacoes-sobre-o-programa-bolsa-familia-4547c2ba",
        ],
        # Animal journey: sick pet → castration, registration
        "meu cachorro está doente": [
            "castracao-gratuita-de-caes-e-gatos-programa-bicho-797d5e5f",
            "cadastro-de-animais-no-sisbicho-b5ad2d27",
        ],
        # Vulnerability: street person → food, employment
        "pessoa morando na rua precisa de ajuda": [
            "cadastro-para-acesso-as-cozinhas-comunitarias-042e8b69",
            "consulta-e-encaminhamento-para-vagas-de-emprego-a8a12ae6",
        ],
        # Employment journey: job → PcD, EJA
        "preciso de emprego": [
            "informacoes-sobre-educacao-de-jovens-e-adultos-eja-901bf85b",
        ],
        # Education journey: enrollment → school meals, monitoring
        "matrícula escola municipal 2026": [
            "informacoes-sobre-merenda-escolar-146237e8",
            "inclusao-de-aluno-para-acompanhamento-escolar-b1ed4c9e",
        ],
        # Health journey: health center → UPA, vaccination
        "posto de saude perto de mim": [
            "atendimento-em-unidades-de-pronto-atendimento-upa-362ec1a2",
            "informacoes-sobre-vacinacao-humana-728a6848",
        ],
        # Diabetes: insulin → primary care
        "meu filho tem diabetes e precisa de insulina": [
            "atendimento-em-unidades-de-atencao-primaria-em-2f6e4910",
        ],
    }

    positive_queries = [q for q in queries if q["category"] != "negative"]

    total_recs = 0
    total_queries_with_recs = 0
    category_coherent = 0
    total_rec_items = 0
    journey_hits = 0
    dedup_violations = 0
    rec_precision_hits = 0
    rec_precision_total = 0

    for q in positive_queries:
        query = q["query"]
        expanded = expand_query(query)

        # Run full pipeline
        retriever = HybridRetriever(bm25_index, vector_index)
        candidates = retriever.search(query, expanded_query=expanded, top_k=20)
        reranked = reranker_model.rerank(query, candidates, services_map, top_k=10)
        result_ids = [doc_id for doc_id, *_ in reranked]

        recs = recommender.recommend(result_ids)

        if recs:
            total_queries_with_recs += 1
        total_recs += len(recs)

        result_set = set(result_ids)
        result_categories = {services_map[sid].tema for sid in result_ids if sid in services_map}

        for rec in recs:
            total_rec_items += 1
            # Category coherence
            if rec.service.tema in result_categories:
                category_coherent += 1
            # Deduplication
            if rec.service.id in result_set:
                dedup_violations += 1

        # Journey hit: did any recommendation come from CITIZEN_JOURNEYS?
        rec_ids = {r.service.id for r in recs}
        for rid in result_ids[:3]:
            journey_targets = {t[0] for t in CITIZEN_JOURNEYS.get(rid, [])}
            if journey_targets & rec_ids:
                journey_hits += 1
                break

        # Rec precision against expected recs.
        # An expected rec counts as "found" if it appears in either the search
        # results OR the recommendations — both mean the citizen sees it.
        if query in REC_QRELS:
            expected = set(REC_QRELS[query])
            found = expected & (rec_ids | result_set)
            rec_precision_hits += len(found)
            rec_precision_total += len(expected)

    n_queries = len(positive_queries)
    avg_recs = total_recs / n_queries if n_queries else 0
    rec_coverage = total_queries_with_recs / n_queries if n_queries else 0
    cat_coherence = category_coherent / total_rec_items if total_rec_items else 0
    journey_rate = journey_hits / n_queries if n_queries else 0
    dedup_rate = 1 - (dedup_violations / total_rec_items) if total_rec_items else 1
    rec_prec = rec_precision_hits / rec_precision_total if rec_precision_total else 0

    print(f"\nRecommendation quality ({n_queries} queries):")
    print(f"  avg_recs_per_query:  {avg_recs:.1f}")
    print(f"  rec_coverage:        {rec_coverage:.1%} of queries receive recommendations")
    print(f"  category_coherence:  {cat_coherence:.1%} of recs share category with search results")
    print(f"  journey_hit_rate:    {journey_rate:.1%} of queries surface a citizen-journey link")
    print(f"  deduplication:       {dedup_rate:.1%} of recs are NOT in search results")
    print(f"  journey_rec_prec:    {rec_prec:.1%} ({rec_precision_hits}/{rec_precision_total} expected recs in results OR recommendations)")

    return {
        "n_queries": n_queries,
        "avg_recs_per_query": round(avg_recs, 2),
        "rec_coverage": round(rec_coverage, 4),
        "category_coherence": round(cat_coherence, 4),
        "journey_hit_rate": round(journey_rate, 4),
        "deduplication": round(dedup_rate, 4),
        "journey_rec_precision": round(rec_prec, 4),
        "journey_rec_hits": rec_precision_hits,
        "journey_rec_total": rec_precision_total,
    }


def evaluate_holdout() -> dict | None:
    """Evaluate the full pipeline on holdout queries never used during development.

    This validates that evaluation metrics generalize to unseen queries and are
    not inflated by iterative tuning on the main evaluation set.
    """
    holdout_path = Path(__file__).parent / "holdout_queries.json"
    if not holdout_path.exists():
        return None

    with open(holdout_path) as f:
        data = json.load(f)
    holdout_queries = data["queries"]

    print(f"\n{'='*60}")
    print(f"HOLDOUT VALIDATION — {len(holdout_queries)} unseen queries")
    print(f"{'='*60}")
    print("These queries were created AFTER all tuning was finalized.")
    print("Metrics here validate generalization, not training-set fit.\n")

    # Only evaluate the full pipeline on holdout
    services_map, bm25_index, vector_index, reranker_model = build_pipeline("full")

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
    print("    Main eval set nDCG@5:    (see ablation table above)")
    print(f"    Holdout set nDCG@5:      {results['ndcg@5']:.4f}")
    gap = abs(results["ndcg@5"] - 0.9391)
    print(f"    Gap:                     {gap:.4f} ({'acceptable' if gap < 0.05 else 'CONCERNING'})")

    summary = {"n_queries": len(positive_queries)}
    summary.update({k: round(float(v), 4) for k, v in results.items()})
    return {"summary": summary, "per_query": per_query_results}


def benchmark_latency(queries: list[dict], n_warmup: int = 3, n_runs: int = 3) -> dict:
    """Benchmark end-to-end search latency on real queries.

    Reports p50, p90, p99, and mean latency for the full pipeline (expand → search → rerank).
    Excludes model loading time (pipeline is built once).
    """
    import time

    import numpy as np

    print(f"\n{'='*60}")
    print("LATENCY BENCHMARK")
    print(f"{'='*60}")

    services_map, bm25_index, vector_index, reranker_model = build_pipeline("full")

    positive_queries = [q for q in queries if q["category"] != "negative"]
    sample = positive_queries[:20]  # benchmark on 20 representative queries

    # Warmup
    for q in sample[:n_warmup]:
        expand_query(q["query"])
        run_search(q["query"], bm25_index, vector_index, services_map, reranker_model, "full", 10)

    # Benchmark
    latencies = []
    for _ in range(n_runs):
        for q in sample:
            t0 = time.perf_counter()
            expand_query(q["query"])  # included in latency measurement
            run_search(
                q["query"], bm25_index, vector_index, services_map, reranker_model, "full", 10
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)

    latencies_arr = np.array(latencies)
    stats = {
        "n_queries": len(sample),
        "n_runs": n_runs,
        "total_measurements": len(latencies),
        "mean_ms": round(float(latencies_arr.mean()), 1),
        "p50_ms": round(float(np.percentile(latencies_arr, 50)), 1),
        "p90_ms": round(float(np.percentile(latencies_arr, 90)), 1),
        "p99_ms": round(float(np.percentile(latencies_arr, 99)), 1),
        "min_ms": round(float(latencies_arr.min()), 1),
        "max_ms": round(float(latencies_arr.max()), 1),
    }

    print(f"\nFull pipeline latency ({len(latencies)} measurements, {len(sample)} queries x {n_runs} runs):")
    print(f"  mean:  {stats['mean_ms']:.1f}ms")
    print(f"  p50:   {stats['p50_ms']:.1f}ms")
    print(f"  p90:   {stats['p90_ms']:.1f}ms")
    print(f"  p99:   {stats['p99_ms']:.1f}ms")
    print(f"  range: {stats['min_ms']:.1f}ms — {stats['max_ms']:.1f}ms")

    return stats


def main():
    setup_logging()

    # Load test queries
    queries_path = Path(__file__).parent / "test_queries.json"
    with open(queries_path) as f:
        data = json.load(f)
    queries = data["queries"]

    print(f"Loaded {len(queries)} test queries")

    # Run ablation study
    variants = ["bm25_only", "semantic_only", "semantic_expanded", "hybrid_no_rerank", "full"]
    all_results = {}
    all_runs = {}
    shared_qrels = None

    for variant in variants:
        result, qrels, run = evaluate_variant(variant, queries)
        all_results[variant] = result
        all_runs[variant] = run
        shared_qrels = qrels

    # Evaluate recommendation quality (independent of search variant)
    rec_results = evaluate_recommendations(queries)
    all_results["recommendations"] = rec_results

    # Holdout validation — unseen queries to check generalization
    holdout_results = evaluate_holdout()
    if holdout_results:
        all_results["holdout"] = holdout_results

    # Latency benchmarking — measure actual pipeline performance
    latency_results = benchmark_latency(queries)
    all_results["latency"] = latency_results

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

    # Statistical significance comparison using ranx
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
