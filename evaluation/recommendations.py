"""Recommendation quality — aggregate metrics, QRELs, and baseline comparison."""

from __future__ import annotations

import json
from pathlib import Path

from app.indexing.cluster_builder import ClusterIndex
from app.recommendation.recommender import CITIZEN_JOURNEYS, Recommender
from app.search.hybrid import HybridRetriever
from app.search.query_processor import expand_query

from .shared import SharedComponents


def evaluate_recommendations(queries: list[dict], *, shared: SharedComponents) -> dict:
    """Evaluate recommendation quality with aggregate metrics, QRELs, and baseline."""
    print(f"\n{'='*60}")
    print("RECOMMENDATION QUALITY EVALUATION")
    print(f"{'='*60}")

    services = shared.services
    services_map = shared.services_map
    bm25_index = shared.bm25_index
    vector_index = shared.vector_index
    reranker_model = shared.reranker

    cluster_index = ClusterIndex(services, vector_index.embeddings)
    recommender = Recommender(services_map, vector_index, cluster_index)

    # ── Part 1: Aggregate metrics on search test queries ────────────────────
    positive_queries = [q for q in queries if q["category"] != "negative"]
    agg = _compute_aggregate_metrics(positive_queries, bm25_index, vector_index,
                                      services_map, reranker_model, recommender)

    # ── Part 2: Recommendation ablation ───────────────────────────────────
    ablation = _recommendation_ablation(
        positive_queries, services, services_map,
        bm25_index, vector_index, reranker_model,
    )

    # ── Part 3: Dedicated recommendation QRELs ──────────────────────────────
    rec_qrels_path = Path(__file__).parent / "rec_queries.json"
    if not rec_qrels_path.exists():
        print("\n  [SKIP] rec_queries.json not found")
        return _build_rec_summary(agg, ablation=ablation)

    with open(rec_qrels_path) as f:
        rec_data = json.load(f)

    qrel_results = _evaluate_rec_qrels(
        rec_data["queries"], bm25_index, vector_index,
        services_map, reranker_model, recommender,
    )

    return _build_rec_summary(agg, ablation=ablation, **qrel_results)


def _compute_aggregate_metrics(positive_queries, bm25_index, vector_index,
                                services_map, reranker_model, recommender):
    """Compute aggregate recommendation metrics across search queries."""
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

    return {
        "n_queries": n_queries,
        "avg_recs_per_query": round(avg_recs, 2),
        "rec_coverage": round(rec_coverage, 4),
        "category_coherence": round(cat_coherence, 4),
        "journey_hit_rate": round(journey_rate, 4),
        "deduplication": round(dedup_rate, 4),
    }


def _recommendation_ablation(
    positive_queries, services, services_map,
    bm25_index, vector_index, reranker_model,
):
    """Compare recommendation variants: category-only → semantic → full (with journeys)."""
    import app.recommendation.recommender as rec_mod

    print(f"\n  Recommendation ablation ({len(positive_queries)} queries):")

    cluster_index = ClusterIndex(services, vector_index.embeddings)

    # Pre-compute search results once (shared across all variants)
    query_results = []
    for q in positive_queries:
        query = q["query"]
        expanded = expand_query(query)
        retriever = HybridRetriever(bm25_index, vector_index)
        candidates = retriever.search(query, expanded_query=expanded, top_k=20)
        reranked = reranker_model.rerank(query, candidates, services_map, top_k=10)
        result_ids = [doc_id for doc_id, *_ in reranked]
        query_results.append(result_ids)

    variants = [
        ("category_only", False, False),
        ("semantic", False, True),
        ("full (with journeys)", True, True),
    ]

    variant_results = {}
    for label, use_journeys, use_semantic in variants:
        rec_count = 0
        coverage = 0
        cat_coherent = 0
        rec_items = 0
        journey_hits_count = 0

        # Toggle journeys by swapping the module-level dict
        saved_journeys = rec_mod.CITIZEN_JOURNEYS
        if not use_journeys:
            rec_mod.CITIZEN_JOURNEYS = {}

        recommender = Recommender(services_map, vector_index, cluster_index) if use_semantic else None

        for result_ids in query_results:
            recs_ids = (
                [r.service.id for r in recommender.recommend(result_ids)]
                if recommender
                else _category_only_recommend(result_ids, services_map)
            )

            if recs_ids:
                coverage += 1
            rec_count += len(recs_ids)

            result_categories = {services_map[sid].tema for sid in result_ids if sid in services_map}
            for rid in recs_ids:
                rec_items += 1
                svc = services_map.get(rid)
                if svc and svc.tema in result_categories:
                    cat_coherent += 1

            rec_id_set = set(recs_ids)
            for rid in result_ids[:3]:
                journey_targets = {t[0] for t in saved_journeys.get(rid, [])}
                if journey_targets & rec_id_set:
                    journey_hits_count += 1
                    break

        rec_mod.CITIZEN_JOURNEYS = saved_journeys

        n = len(positive_queries)
        variant_results[label] = {
            "avg_recs": round(rec_count / n, 1) if n else 0,
            "coverage": round(coverage / n, 4) if n else 0,
            "cat_coherence": round(cat_coherent / rec_items, 4) if rec_items else 0,
            "journey_rate": round(journey_hits_count / n, 4) if n else 0,
        }

    print(f"\n    {'Variant':<25} {'Avg Recs':>8} {'Coverage':>10} {'Cat Coher':>10} {'Journey':>10}")
    print("    " + "-" * 65)
    for label, stats in variant_results.items():
        print(f"    {label:<25} {stats['avg_recs']:>8.1f} {stats['coverage']:>9.1%} "
              f"{stats['cat_coherence']:>9.1%} {stats['journey_rate']:>9.1%}")

    return variant_results


def _category_only_recommend(result_ids, services_map, top_k=5):
    """Baseline recommender: recommend services from the same category only."""
    if not result_ids:
        return []
    exclude = set(result_ids)
    result_categories = {services_map[sid].tema for sid in result_ids[:3] if sid in services_map}
    candidates = [
        sid for sid, svc in services_map.items()
        if svc.tema in result_categories and sid not in exclude
    ]
    return candidates[:top_k]


def _evaluate_rec_qrels(rec_queries, bm25_index, vector_index,
                         services_map, reranker_model, recommender):
    """Evaluate recommendation precision against dedicated QRELs."""
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

        expanded = expand_query(query)
        retriever = HybridRetriever(bm25_index, vector_index)
        candidates = retriever.search(query, expanded_query=expanded, top_k=20)
        reranked = reranker_model.rerank(query, candidates, services_map, top_k=10)
        result_ids = [doc_id for doc_id, *_ in reranked]
        result_set = set(result_ids)

        recs = recommender.recommend(result_ids)
        rec_ids = {r.service.id for r in recs}
        visible = rec_ids | result_set
        found = expected_ids & visible
        full_hits += len(found)
        full_total += len(expected_ids)
        rec_only_found = expected_ids & (rec_ids - result_set)
        rec_only_hits += len(rec_only_found)

        baseline_rec_ids = set(_category_only_recommend(result_ids, services_map))
        baseline_visible = baseline_rec_ids | result_set
        baseline_found = expected_ids & baseline_visible
        baseline_hits += len(baseline_found)
        baseline_total += len(expected_ids)

        if journey not in per_journey:
            per_journey[journey] = {"hits": 0, "total": 0, "queries": 0}
        per_journey[journey]["hits"] += len(found)
        per_journey[journey]["total"] += len(expected_ids)
        per_journey[journey]["queries"] += 1

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

    search_hits = full_hits - rec_only_hits
    print(f"\n    Expected recs found anywhere:   {full_precision:.1%} ({full_hits}/{full_total})")
    print(f"      ├─ found in search results:   {search_hits}/{full_total} ({search_hits/full_total:.1%})")
    print(f"      └─ found ONLY via recs:       {rec_only_hits}/{full_total} ({rec_only_precision:.1%}) ← true rec value")
    print(f"    Category-only baseline:         {baseline_precision:.1%} ({baseline_hits}/{baseline_total})")
    print(f"    Lift over baseline:             {lift:+.1%}")

    print("\n    Per-journey precision:")
    for journey_name in sorted(per_journey.keys()):
        j = per_journey[journey_name]
        j_prec = j["hits"] / j["total"] if j["total"] else 0
        print(f"      {journey_name:<16} {j_prec:.0%} ({j['hits']}/{j['total']}, {j['queries']} queries)")

    if failures:
        print(f"\n    Failure analysis ({len(failures)} missed recs):")
        for f in failures:
            print(f"      [{f['query_id']}] \"{f['query']}\" — missed: {f['missed_name']}")
            print(f"        expected: {f['reason']}")
    else:
        print("\n    No failures — all expected recs found in results or recommendations")

    return {
        "full_precision": full_precision,
        "baseline_precision": baseline_precision,
        "lift": lift,
        "per_journey": per_journey,
        "failures": failures,
        "full_hits": full_hits,
        "full_total": full_total,
        "rec_only_precision": rec_only_precision,
        "rec_only_hits": rec_only_hits,
        "baseline_hits": baseline_hits,
        "baseline_total": baseline_total,
    }


def _build_rec_summary(agg: dict, ablation: dict | None = None, **qrel_results) -> dict:
    """Build the recommendation evaluation summary dict."""
    per_journey = qrel_results.pop("per_journey", {}) or {}
    failures = qrel_results.pop("failures", []) or []

    result: dict = {"aggregate": agg}

    if qrel_results:
        result["qrel_evaluation"] = {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in qrel_results.items()
        }

    if per_journey:
        result["per_journey"] = {
            name: {
                "precision": round(j["hits"] / j["total"], 4) if j["total"] else 0,
                **j,
            }
            for name, j in per_journey.items()
        }

    if failures:
        result["failures"] = [
            {"query": f["query"], "missed": f["missed_name"], "reason": f["reason"]}
            for f in failures
        ]

    if ablation:
        result["ablation"] = ablation

    return result
