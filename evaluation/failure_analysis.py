"""Failure pattern analysis — diagnoses WHY queries fail, not just which ones."""

from __future__ import annotations

import json
from pathlib import Path

from app.search.query_processor import expand_query

from .shared import SharedComponents
from .variants import run_search


def analyze_failures(*, shared: SharedComponents) -> dict:
    """Analyze failure patterns across tuning and holdout query sets.

    For each non-trivial result (expected service not at rank 1), diagnoses the
    failure pattern: vocabulary gap, sibling confusion, ambiguity dilution, or
    cross-signal conflict.
    """
    print(f"\n{'='*60}")
    print("FAILURE PATTERN ANALYSIS")
    print(f"{'='*60}")

    # Load both query sets
    query_sets = {}
    for name, filename in [("tuning", "test_queries.json"), ("holdout", "holdout_queries.json")]:
        path = Path(__file__).parent / filename
        if path.exists():
            with open(path) as f:
                query_sets[name] = json.load(f)["queries"]

    bm25_index = shared.bm25_index
    vector_index = shared.vector_index
    services_map = shared.services_map
    reranker = shared.reranker

    all_diagnoses: list[dict] = []
    pattern_counts: dict[str, int] = {}

    for set_name, queries in query_sets.items():
        positive = [q for q in queries if q["category"] != "negative" and q.get("relevant")]

        for q in positive:
            query = q["query"]
            expected = q["relevant"]
            best_id = max(expected, key=lambda k: expected[k])
            best_grade = expected[best_id]

            # Per-component analysis
            expanded = expand_query(query)
            bm25_results = bm25_index.search(expanded, top_k=10)
            sem_results = vector_index.search(expanded, top_k=10)

            pipeline = run_search(
                query, bm25_index, vector_index, services_map, reranker, "full", 10,
            )
            pipeline_ids = [did for did, _ in pipeline[:5]]

            # Ranks in each engine
            rank_pipeline = _find_rank(pipeline, best_id)
            rank_bm25 = _find_rank(bm25_results, best_id)
            rank_sem = _find_rank(sem_results, best_id)
            max_cosine = sem_results[0][1] if sem_results else 0.0

            # Skip perfect results
            if rank_pipeline == 1:
                continue

            # Diagnose pattern
            pattern = _diagnose(
                rank_pipeline, rank_bm25, rank_sem,
                best_grade, len(expected), pipeline_ids, best_id, services_map,
            )
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

            diag = {
                "query_id": q["id"],
                "set": set_name,
                "category": q["category"],
                "query": query,
                "expected": services_map[best_id].nome if best_id in services_map else best_id,
                "expected_grade": best_grade,
                "rank_pipeline": rank_pipeline,
                "rank_bm25": rank_bm25,
                "rank_sem": rank_sem,
                "max_cosine": round(max_cosine, 4),
                "expanded": expanded != query,
                "pattern": pattern,
            }

            # What outranked it?
            if rank_pipeline and rank_pipeline > 1:
                outrankers = []
                for i in range(min(rank_pipeline - 1, 3)):
                    svc = services_map.get(pipeline_ids[i])
                    if svc:
                        outrankers.append({"name": svc.nome, "tema": svc.tema})
                diag["outranked_by"] = outrankers

            all_diagnoses.append(diag)

    # Print summary
    total_imperfect = len(all_diagnoses)
    total_queries = sum(
        len([q for q in qs if q["category"] != "negative" and q.get("relevant")])
        for qs in query_sets.values()
    )
    print(f"\n  {total_imperfect}/{total_queries} queries with imperfect ranking (expected not at #1)")

    print("\n  Failure pattern distribution:")
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        print(f"    {pattern:<30} {count:>3} ({count/total_imperfect*100:.0f}%)")

    # Detailed examples per pattern
    for pattern in sorted(pattern_counts.keys()):
        examples = [d for d in all_diagnoses if d["pattern"] == pattern]
        print(f"\n  --- {pattern} ({len(examples)} cases) ---")
        _print_pattern_explanation(pattern)
        for d in examples[:3]:
            rank_str = f"#{d['rank_pipeline']}" if d["rank_pipeline"] else ">10"
            print(f"    [{d['query_id']}] \"{d['query']}\" → rank {rank_str}")
            if d.get("outranked_by"):
                top = d["outranked_by"][0]
                print(f"      outranked by: {top['name']} ({top['tema']})")
        if len(examples) > 3:
            print(f"    ... and {len(examples) - 3} more")

    # Actionable insights
    print(f"\n  {'='*50}")
    print("  ACTIONABLE INSIGHTS")
    print(f"  {'='*50}")
    _print_insights(pattern_counts, all_diagnoses)

    return {
        "total_queries": total_queries,
        "imperfect_count": total_imperfect,
        "pattern_distribution": pattern_counts,
        "diagnoses": all_diagnoses,
    }


def _find_rank(results: list[tuple[str, float]], target_id: str) -> int | None:
    """Find 1-based rank of target_id in results, or None if absent."""
    for i, (did, _) in enumerate(results[:10]):
        if did == target_id:
            return i + 1
    return None


def _diagnose(
    rank_pipeline, rank_bm25, rank_sem,
    best_grade, n_relevant, pipeline_ids, best_id, services_map,
) -> str:
    """Classify the failure pattern."""
    # Both engines miss entirely
    if rank_bm25 is None and rank_sem is None:
        return "both_engines_miss"

    # BM25 misses, semantic finds it
    if rank_bm25 is None and rank_sem is not None:
        return "vocabulary_gap"

    # Semantic misses, BM25 finds it
    if rank_sem is None and rank_bm25 is not None:
        return "semantic_blind_spot"

    # Multiple valid services, none strongly preferred
    if best_grade <= 2 and n_relevant >= 3:
        return "ambiguity_dilution"

    # Correct service found but a closely related sibling outranks it
    if rank_pipeline and rank_pipeline <= 5:
        top_svc = services_map.get(pipeline_ids[0])
        best_svc = services_map.get(best_id)
        if top_svc and best_svc and top_svc.tema == best_svc.tema:
            return "sibling_confusion"

    # Cross-category result outranks
    if rank_pipeline and rank_pipeline <= 5:
        return "cross_category_pollution"

    # Found but ranked low
    if rank_pipeline and rank_pipeline > 5:
        return "low_ranking"

    return "not_in_top10"


def _print_pattern_explanation(pattern: str) -> None:
    """Print a one-line explanation of the failure pattern."""
    explanations = {
        "sibling_confusion": "  A closely related service in the same category outranks the expected one.",
        "ambiguity_dilution": "  Query is genuinely ambiguous — multiple valid services dilute the ranking.",
        "vocabulary_gap": "  BM25 misses entirely due to zero word overlap; semantic finds it but gets diluted.",
        "semantic_blind_spot": "  Semantic search misses; BM25 finds it by keyword match.",
        "cross_category_pollution": "  A service from a different category outranks due to partial term overlap.",
        "both_engines_miss": "  Neither BM25 nor semantic search finds the expected service in top 10.",
        "low_ranking": "  Found by both engines but ranked below position 5.",
        "not_in_top10": "  Expected service not in top 10 of full pipeline.",
    }
    print(explanations.get(pattern, ""))


def _print_insights(pattern_counts: dict, diagnoses: list[dict]) -> None:
    """Print actionable insights based on failure patterns."""
    total = sum(pattern_counts.values())
    if not total:
        print("  No failures to analyze.")
        return

    sibling = pattern_counts.get("sibling_confusion", 0)
    ambiguity = pattern_counts.get("ambiguity_dilution", 0)
    vocab = pattern_counts.get("vocabulary_gap", 0)
    cross = pattern_counts.get("cross_category_pollution", 0)

    if sibling + ambiguity > total * 0.5:
        print(f"  1. {sibling + ambiguity}/{total} failures are sibling confusion or ambiguity.")
        print("     These are ranking precision issues, not retrieval failures.")
        print("     Fix: increase cross-encoder weight to improve top-1 precision.")
        print("     At 50 services CE has low discrimination; at 500+ this self-corrects.")

    if vocab > 0:
        vocab_examples = [d for d in diagnoses if d["pattern"] == "vocabulary_gap" and not d["expanded"]]
        if vocab_examples:
            print(f"  2. {len(vocab_examples)} vocabulary-gap failures lack synonym coverage:")
            for d in vocab_examples[:3]:
                print(f"     \"{d['query']}\" → needs synonym for {d['expected']}")

    if cross > 0:
        print(f"  3. {cross} cross-category pollution cases — partial keyword overlap")
        print("     pulls irrelevant categories into top results.")
        print("     Fix: add category-aware reranking or boost same-category results.")

    holdout_ambig = [d for d in diagnoses if d["set"] == "holdout" and d["category"] == "ambiguous"]
    if holdout_ambig:
        print(f"  4. Holdout ambiguous queries ({len(holdout_ambig)} imperfect):")
        print("     These are the hardest case — generic queries with many valid answers.")
        print("     nDCG penalizes because relevance grades are spread across services.")
        print("     This is a labeling challenge, not a retrieval failure.")
