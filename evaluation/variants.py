"""Search variant evaluation — ablation study across pipeline configurations."""

from __future__ import annotations

from ranx import Qrels, Run, evaluate

from app.config import CONFIG
from app.indexing.bm25_index import BM25Index
from app.indexing.vector_index import VectorIndex
from app.search.hybrid import HybridRetriever, weighted_rrf
from app.search.query_processor import expand_query
from app.search.reranker import Reranker

from .shared import SharedComponents

METRICS = ["ndcg@5", "ndcg@10", "precision@5", "mrr@10", "recall@10"]


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
    *,
    shared: SharedComponents,
) -> tuple[dict, Qrels, Run]:
    """Evaluate a pipeline variant across all test queries using ranx."""
    print(f"\n{'='*60}")
    print(f"Evaluating variant: {variant}")
    print(f"{'='*60}")

    services_map = shared.services_map
    bm25_index = shared.bm25_index
    vector_index = shared.vector_index
    reranker_model = shared.reranker

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

    print(f"\nResults ({variant}) — {len(positive_queries)} positive queries:")
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

    # Negative query analysis
    if negative_queries:
        _analyze_negatives(
            negative_queries, per_query_results,
            bm25_index, vector_index, services_map, reranker_model, variant, top_k,
        )

    # Failure analysis
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


def _analyze_negatives(
    negative_queries, per_query_results,
    bm25_index, vector_index, services_map, reranker_model, variant, top_k,
):
    """Analyze negative (out-of-scope) queries for false positive rate and confidence."""
    print(f"\n  Negative queries ({len(negative_queries)}) — out-of-scope robustness:")

    # Collect positive-query cosine baselines
    pos_max_cosines = []
    for pq in per_query_results:
        sem_results = vector_index.search(pq["query"], top_k=1)
        if sem_results:
            pos_max_cosines.append(sem_results[0][1])
    avg_pos_cosine = sum(pos_max_cosines) / len(pos_max_cosines) if pos_max_cosines else 0

    # Evaluate each negative query
    neg_max_cosines = []
    for q in negative_queries:
        bm25_hits = len(bm25_index.search(q["query"], top_k=top_k))
        sem_results = vector_index.search(q["query"], top_k=1)
        max_cosine = sem_results[0][1] if sem_results else 0.0
        neg_max_cosines.append(max_cosine)

        neg_search = run_search(
            q["query"], bm25_index, vector_index, services_map, reranker_model, variant, top_k
        )
        n_returned = len(neg_search)
        top_score = neg_search[0][1] if neg_search else 0.0
        per_query_results.append({
            "query_id": q["id"],
            "query": q["query"],
            "category": "negative",
            "top_results": [doc_id for doc_id, _ in neg_search[:5]],
            "n_returned": n_returned,
            "top_score": top_score,
            "max_cosine": max_cosine,
            "bm25_hits": bm25_hits,
        })

    # Confidence threshold calibration
    from app.config import CONFIG

    avg_neg_cosine = sum(neg_max_cosines) / len(neg_max_cosines) if neg_max_cosines else 0
    cosine_gap = avg_pos_cosine - avg_neg_cosine
    print(f"    cosine gap (pos-neg): {cosine_gap:.4f} (pos={avg_pos_cosine:.4f}, neg={avg_neg_cosine:.4f})")

    sweep_thresholds = sorted(set([0.80, 0.81, 0.82, 0.83, 0.84, 0.85, CONFIG.confidence_threshold]))
    for threshold in sweep_thresholds:
        neg_flagged = sum(1 for c in neg_max_cosines if c < threshold)
        pos_flagged = sum(1 for c in pos_max_cosines if c < threshold)
        marker = " ← active" if threshold == CONFIG.confidence_threshold else ""
        print(f"    t={threshold:.2f}: {neg_flagged}/{len(neg_max_cosines)} neg detected, "
              f"{pos_flagged}/{len(pos_max_cosines)} false pos "
              f"({pos_flagged/len(pos_max_cosines)*100:.1f}%){marker}")
