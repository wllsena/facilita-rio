"""Prometheus metrics for search observability."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

SEARCH_REQUESTS = Counter(
    "search_requests_total",
    "Total number of search requests",
)

SEARCH_LATENCY = Histogram(
    "search_latency_seconds",
    "End-to-end search latency",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

RERANKER_LATENCY = Histogram(
    "reranker_latency_seconds",
    "Cross-encoder reranking latency",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

SEARCH_RESULT_COUNT = Histogram(
    "search_result_count",
    "Number of results returned per search",
    buckets=[0, 1, 3, 5, 10, 20],
)

INDEX_SIZE = Gauge(
    "index_size_services",
    "Number of services in the index",
)

CACHE_HITS = Counter(
    "cache_hits_total",
    "Search cache hits",
)
