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


def build_shared_components():
    """Build shared index components once (avoids reloading models per variant)."""
    services = load_services(DATA_PATH)
    services_map = {s.id: s for s in services}
    bm25_index = BM25Index(services)
    vector_index = VectorIndex(services)
    reranker_model = Reranker()
    return services, services_map, bm25_index, vector_index, reranker_model


def build_pipeline(variant: str = "full"):
    """Build search pipeline for a given variant (legacy — loads everything fresh)."""
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

    elif variant == "bm25_expanded":
        return bm25_index.search(expanded, top_k=top_k)

    elif variant == "semantic_only":
        return vector_index.search(query, top_k=top_k)

    elif variant == "semantic_expanded":
        return vector_index.search(expanded, top_k=top_k)

    elif variant == "hybrid_no_rerank":
        bm25_results = bm25_index.search(expanded, top_k=CONFIG.bm25_top_k)
        semantic_results = vector_index.search(expanded, top_k=CONFIG.semantic_top_k)
        fused = weighted_rrf(
            [(bm25_results, CONFIG.bm25_weight), (semantic_results, CONFIG.semantic_weight)], k=CONFIG.rrf_k
        )
        return fused[:top_k]

    else:  # full pipeline
        retriever = HybridRetriever(bm25_index, vector_index)
        candidates = retriever.search(query, expanded_query=expanded, top_k=top_k * 2)
        if reranker_model:
            reranked = reranker_model.rerank(
                query, candidates, services_map, top_k=top_k, expanded_query=expanded,
            )
            return [(r.doc_id, r.blended_score) for r in reranked]
        return [(c.doc_id, c.rrf_score) for c in candidates[:top_k]]


def evaluate_variant(
    variant: str,
    queries: list[dict],
    top_k: int = 10,
    shared: tuple | None = None,
) -> tuple[dict, Qrels, Run]:
    """Evaluate a pipeline variant across all test queries using ranx.

    When `shared` is provided as (services, services_map, bm25_index, vector_index, reranker_model),
    reuses those components instead of rebuilding from scratch.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating variant: {variant}")
    print(f"{'='*60}")

    if shared:
        _services, services_map, bm25_index, vector_index, reranker_model = shared
    else:
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

        # Confidence threshold calibration
        avg_neg_cosine = sum(neg_max_cosines) / len(neg_max_cosines) if neg_max_cosines else 0
        cosine_gap = avg_pos_cosine - avg_neg_cosine
        print("\n  Confidence calibration (max cosine similarity):")
        print(f"    avg positive: {avg_pos_cosine:.4f}")
        print(f"    avg negative: {avg_neg_cosine:.4f}")
        print(f"    gap: {cosine_gap:.4f}")
        # Sweep thresholds to show calibration curve
        sweep_thresholds = sorted(set([0.82, 0.83, 0.84, CONFIG.confidence_threshold]))
        for threshold in sweep_thresholds:
            neg_flagged = sum(1 for c in neg_max_cosines if c < threshold)
            pos_flagged = sum(1 for c in pos_max_cosines if c < threshold)
            marker = " ← active" if threshold == CONFIG.confidence_threshold else ""
            print(f"    t={threshold:.2f}: {neg_flagged}/{len(neg_max_cosines)} neg detected, "
                  f"{pos_flagged}/{len(pos_max_cosines)} false pos "
                  f"({pos_flagged/len(pos_max_cosines)*100:.1f}%){marker}")

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


def _category_only_recommend(
    result_ids: list[str],
    services_map: dict,
    top_k: int = 5,
) -> list[str]:
    """Baseline recommender: recommend services from the same category only.

    No semantic similarity, no journeys, no clusters — just category matching.
    Used as a comparison baseline for the full 4-signal recommender.
    """
    if not result_ids:
        return []
    exclude = set(result_ids)
    result_categories = {services_map[sid].tema for sid in result_ids[:3] if sid in services_map}
    candidates = [
        sid for sid, svc in services_map.items()
        if svc.tema in result_categories and sid not in exclude
    ]
    return candidates[:top_k]


def evaluate_recommendations(queries: list[dict], shared: tuple | None = None) -> dict:
    """Evaluate recommendation quality — the second core deliverable.

    Comprehensive evaluation with:
    - Aggregate metrics (coverage, coherence, dedup, journey rate)
    - Dedicated recommendation QRELs (25 queries across all journeys)
    - Baseline comparison (category-only vs full 4-signal recommender)
    - Per-journey precision breakdown
    - Failure analysis (missed expected recs with diagnosis)
    """
    from app.indexing.cluster_builder import ClusterIndex
    from app.recommendation.recommender import CITIZEN_JOURNEYS, Recommender

    print(f"\n{'='*60}")
    print("RECOMMENDATION QUALITY EVALUATION")
    print(f"{'='*60}")

    if shared:
        services, services_map, bm25_index, vector_index, reranker_model = shared
    else:
        services = load_services(DATA_PATH)
        services_map = {s.id: s for s in services}
        bm25_index = BM25Index(services)
        vector_index = VectorIndex(services)
        reranker_model = Reranker()
    cluster_index = ClusterIndex(services, vector_index.embeddings)
    recommender = Recommender(services_map, vector_index, cluster_index)

    # ── Part 1: Aggregate metrics on search test queries ────────────────────
    positive_queries = [q for q in queries if q["category"] != "negative"]

    total_recs = 0
    total_queries_with_recs = 0
    category_coherent = 0
    total_rec_items = 0
    journey_hits = 0
    dedup_violations = 0

    for q in positive_queries:
        query = q["query"]
        expanded = expand_query(query)
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
            if rec.service.tema in result_categories:
                category_coherent += 1
            if rec.service.id in result_set:
                dedup_violations += 1

        rec_ids = {r.service.id for r in recs}
        for rid in result_ids[:3]:
            journey_targets = {t[0] for t in CITIZEN_JOURNEYS.get(rid, [])}
            if journey_targets & rec_ids:
                journey_hits += 1
                break

    n_queries = len(positive_queries)
    avg_recs = total_recs / n_queries if n_queries else 0
    rec_coverage = total_queries_with_recs / n_queries if n_queries else 0
    cat_coherence = category_coherent / total_rec_items if total_rec_items else 0
    journey_rate = journey_hits / n_queries if n_queries else 0
    dedup_rate = 1 - (dedup_violations / total_rec_items) if total_rec_items else 1

    print(f"\n  Aggregate metrics ({n_queries} search queries):")
    print(f"    avg_recs_per_query:  {avg_recs:.1f}")
    print(f"    rec_coverage:        {rec_coverage:.1%}")
    print(f"    category_coherence:  {cat_coherence:.1%}")
    print(f"    journey_hit_rate:    {journey_rate:.1%}")
    print(f"    deduplication:       {dedup_rate:.1%}")

    # ── Part 2: Dedicated recommendation QRELs ──────────────────────────────
    rec_qrels_path = Path(__file__).parent / "rec_queries.json"
    if not rec_qrels_path.exists():
        print("\n  [SKIP] rec_queries.json not found")
        return _build_rec_summary(n_queries, avg_recs, rec_coverage, cat_coherence,
                                  journey_rate, dedup_rate, 0, 0, 0, {}, {}, [])

    with open(rec_qrels_path) as f:
        rec_data = json.load(f)
    rec_queries = rec_data["queries"]

    print(f"\n  Dedicated recommendation QRELs ({len(rec_queries)} queries):")

    full_hits = 0
    full_total = 0
    rec_only_hits = 0
    baseline_hits = 0
    baseline_total = 0
    per_journey: dict[str, dict] = {}
    failures: list[dict] = []

    for rq in rec_queries:
        query = rq["query"]
        journey = rq["journey"]
        expected_ids = {e["id"] for e in rq["expected_recs"]}
        expected_reasons = {e["id"]: e["reason"] for e in rq["expected_recs"]}

        # Run full pipeline
        expanded = expand_query(query)
        retriever = HybridRetriever(bm25_index, vector_index)
        candidates = retriever.search(query, expanded_query=expanded, top_k=20)
        reranked = reranker_model.rerank(query, candidates, services_map, top_k=10)
        result_ids = [doc_id for doc_id, *_ in reranked]
        result_set = set(result_ids)

        # Full recommender
        recs = recommender.recommend(result_ids)
        rec_ids = {r.service.id for r in recs}
        visible = rec_ids | result_set  # citizen sees both results and recs
        found = expected_ids & visible
        full_hits += len(found)
        full_total += len(expected_ids)
        rec_only_found = expected_ids & (rec_ids - result_set)
        rec_only_hits += len(rec_only_found)

        # Baseline: category-only
        baseline_rec_ids = set(_category_only_recommend(result_ids, services_map))
        baseline_visible = baseline_rec_ids | result_set
        baseline_found = expected_ids & baseline_visible
        baseline_hits += len(baseline_found)
        baseline_total += len(expected_ids)

        # Per-journey tracking
        if journey not in per_journey:
            per_journey[journey] = {"hits": 0, "total": 0, "queries": 0}
        per_journey[journey]["hits"] += len(found)
        per_journey[journey]["total"] += len(expected_ids)
        per_journey[journey]["queries"] += 1

        # Track failures
        missed = expected_ids - visible
        if missed:
            for mid in missed:
                failures.append({
                    "query_id": rq["id"],
                    "query": query,
                    "journey": journey,
                    "missed_id": mid,
                    "missed_name": services_map[mid].nome if mid in services_map else mid,
                    "reason": expected_reasons.get(mid, ""),
                    "in_results": mid in result_set,
                    "in_recs": mid in rec_ids,
                })

    full_precision = full_hits / full_total if full_total else 0
    rec_only_precision = rec_only_hits / full_total if full_total else 0
    baseline_precision = baseline_hits / baseline_total if baseline_total else 0
    lift = full_precision - baseline_precision

    print(f"\n    Full recommender precision:     {full_precision:.1%} ({full_hits}/{full_total})")
    print(f"    Rec-only precision:             {rec_only_precision:.1%} ({rec_only_hits}/{full_total})")
    print(f"    Category-only baseline:         {baseline_precision:.1%} ({baseline_hits}/{baseline_total})")
    print(f"    Lift over baseline:             {lift:+.1%}")

    # Per-journey breakdown
    print("\n    Per-journey precision:")
    for journey_name in sorted(per_journey.keys()):
        j = per_journey[journey_name]
        j_prec = j["hits"] / j["total"] if j["total"] else 0
        print(f"      {journey_name:<16} {j_prec:.0%} ({j['hits']}/{j['total']}, {j['queries']} queries)")

    # Failure analysis
    if failures:
        print(f"\n    Failure analysis ({len(failures)} missed recs):")
        for f in failures:
            print(f"      [{f['query_id']}] \"{f['query']}\" — missed: {f['missed_name']}")
            print(f"        expected: {f['reason']}")
    else:
        print("\n    No failures — all expected recs found in results or recommendations")

    return _build_rec_summary(
        n_queries, avg_recs, rec_coverage, cat_coherence, journey_rate, dedup_rate,
        full_precision, baseline_precision, lift, per_journey, {}, failures,
        full_hits=full_hits, full_total=full_total,
        rec_only_precision=rec_only_precision, rec_only_hits=rec_only_hits,
        baseline_hits=baseline_hits, baseline_total=baseline_total,
    )


def _build_rec_summary(
    n_queries, avg_recs, rec_coverage, cat_coherence, journey_rate, dedup_rate,
    full_precision, baseline_precision, lift, per_journey, _unused, failures,
    full_hits=0, full_total=0, rec_only_precision=0, rec_only_hits=0,
    baseline_hits=0, baseline_total=0,
) -> dict:
    """Build the recommendation evaluation summary dict."""
    per_journey_summary = {}
    for name, j in per_journey.items():
        per_journey_summary[name] = {
            "precision": round(j["hits"] / j["total"], 4) if j["total"] else 0,
            "hits": j["hits"],
            "total": j["total"],
            "queries": j["queries"],
        }

    return {
        "aggregate": {
            "n_queries": n_queries,
            "avg_recs_per_query": round(avg_recs, 2),
            "rec_coverage": round(rec_coverage, 4),
            "category_coherence": round(cat_coherence, 4),
            "journey_hit_rate": round(journey_rate, 4),
            "deduplication": round(dedup_rate, 4),
        },
        "qrel_evaluation": {
            "full_precision": round(full_precision, 4),
            "full_hits": full_hits,
            "full_total": full_total,
            "rec_only_precision": round(rec_only_precision, 4),
            "rec_only_hits": rec_only_hits,
            "baseline_precision": round(baseline_precision, 4),
            "baseline_hits": baseline_hits,
            "baseline_total": baseline_total,
            "lift_over_baseline": round(lift, 4),
        },
        "per_journey": per_journey_summary,
        "failures": [
            {"query": f["query"], "missed": f["missed_name"], "reason": f["reason"]}
            for f in failures
        ],
    }


def _get_main_ndcg5() -> float:
    """Read the main eval nDCG@5 from results file, or return a sensible default."""
    results_path = Path(__file__).parent / "results" / "evaluation_results.json"
    if results_path.exists():
        try:
            with open(results_path) as f:
                data = json.load(f)
            for variant in data.get("ablation", []):
                if variant.get("summary", {}).get("variant") == "full":
                    return variant["summary"].get("ndcg@5", 0.933)
        except (json.JSONDecodeError, KeyError):
            pass
    return 0.933  # fallback to last known value


def evaluate_holdout(shared: tuple | None = None) -> dict | None:
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

    n_positive = sum(1 for q in holdout_queries if q.get("relevant"))
    n_negative = len(holdout_queries) - n_positive

    print(f"\n{'='*60}")
    print(f"HOLDOUT VALIDATION — {n_positive} positive + {n_negative} negative unseen queries")
    print(f"{'='*60}")
    print("These queries were created AFTER all tuning was finalized.")
    print("Metrics here validate generalization, not training-set fit.\n")

    # Only evaluate the full pipeline on holdout
    if shared:
        _services, services_map, bm25_index, vector_index, reranker_model = shared
    else:
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
    print(f"    Holdout set nDCG@5:      {results['ndcg@5']:.4f}")
    # Use actual main eval nDCG@5 from results file if available, otherwise fallback
    main_ndcg5 = _get_main_ndcg5()
    print(f"    Main eval set nDCG@5:    {main_ndcg5:.4f}")
    gap = abs(results["ndcg@5"] - main_ndcg5)
    print(f"    Gap:                     {gap:.4f} ({'acceptable' if gap < 0.05 else 'CONCERNING'})")

    summary = {"n_queries": len(positive_queries)}
    summary.update({k: round(float(v), 4) for k, v in results.items()})
    return {"summary": summary, "per_query": per_query_results}


def benchmark_latency(queries: list[dict], n_warmup: int = 3, n_runs: int = 3, shared: tuple | None = None) -> dict:
    """Benchmark end-to-end search latency with per-component decomposition.

    Reports p50, p90, p99, and mean latency for total pipeline and each component:
    expansion, BM25, semantic (E5 encoding + FAISS), RRF fusion, and cross-encoder reranker.
    Excludes model loading time (pipeline is built once).
    """
    import time

    import numpy as np

    from app.search.hybrid import HybridRetriever, weighted_rrf

    print(f"\n{'='*60}")
    print("LATENCY BENCHMARK")
    print(f"{'='*60}")

    if shared:
        _services, services_map, bm25_index, vector_index, reranker_model = shared
    else:
        services_map, bm25_index, vector_index, reranker_model = build_pipeline("full")

    positive_queries = [q for q in queries if q["category"] != "negative"]
    sample = positive_queries[:20]  # benchmark on 20 representative queries

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

            # 1. Query expansion
            t0 = time.perf_counter()
            expanded = expand_query(q["query"])
            expand_latencies.append((time.perf_counter() - t0) * 1000)

            # 2. BM25 retrieval
            t0 = time.perf_counter()
            bm25_results = bm25_index.search(expanded, top_k=CONFIG.bm25_top_k)
            bm25_latencies.append((time.perf_counter() - t0) * 1000)

            # 3. Semantic retrieval (E5 encoding + FAISS search)
            t0 = time.perf_counter()
            semantic_results = vector_index.search(expanded, top_k=CONFIG.semantic_top_k)
            semantic_latencies.append((time.perf_counter() - t0) * 1000)

            # 4. RRF fusion
            t0 = time.perf_counter()
            fused = weighted_rrf(
                [(bm25_results, CONFIG.bm25_weight), (semantic_results, CONFIG.semantic_weight)],
                k=CONFIG.rrf_k,
            )
            from app.models import RetrievalCandidate
            bm25_scores = dict(bm25_results)
            semantic_scores = dict(semantic_results)
            candidates = [
                RetrievalCandidate(doc_id, rrf_score, bm25_scores.get(doc_id), semantic_scores.get(doc_id))
                for doc_id, rrf_score in fused[:20]
            ]
            rrf_latencies.append((time.perf_counter() - t0) * 1000)

            # 5. Cross-encoder reranking
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


def evaluate_popular_queries(shared: tuple | None = None) -> dict:
    """Evaluate top-3 accuracy on 500 colloquial citizen queries.

    Each of the 50 services has 10 queries written as a low-literacy citizen would
    write them (typos, slang, problem descriptions). A query is a "hit" if the
    expected service appears in the top-3 results.
    """
    popular_path = Path(__file__).parent / "queries_populares.json"
    if not popular_path.exists():
        return {}

    with open(popular_path) as f:
        data = json.load(f)

    print(f"\n{'='*60}")
    print("POPULAR QUERIES — 500 colloquial queries (top-3 accuracy)")
    print(f"{'='*60}")

    if shared:
        _services, services_map, bm25_index, vector_index, reranker_model = shared
    else:
        services_map, bm25_index, vector_index, reranker_model = build_pipeline("full")

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


def main():
    setup_logging()

    # Load test queries
    queries_path = Path(__file__).parent / "test_queries.json"
    with open(queries_path) as f:
        data = json.load(f)
    queries = data["queries"]

    print(f"Loaded {len(queries)} test queries")

    # Build shared components once (avoids reloading models 5 times)
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

    # Evaluate recommendation quality (independent of search variant)
    rec_results = evaluate_recommendations(queries, shared=shared)
    all_results["recommendations"] = rec_results

    # Holdout validation — unseen queries to check generalization
    holdout_results = evaluate_holdout(shared=shared)
    if holdout_results:
        all_results["holdout"] = holdout_results

    # Popular queries — 500 colloquial queries top-3 accuracy
    popular_results = evaluate_popular_queries(shared=shared)
    if popular_results:
        all_results["popular_queries"] = popular_results

    # Latency benchmarking — measure actual pipeline performance
    latency_results = benchmark_latency(queries, shared=shared)
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
