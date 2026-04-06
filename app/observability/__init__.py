"""Structured logging and Prometheus metrics."""

from __future__ import annotations

import structlog
from prometheus_client import Counter, Gauge, Histogram


def setup_logging(json_format: bool = False) -> None:
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer() if json_format else structlog.dev.ConsoleRenderer(),
    ]
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


SEARCH_REQUESTS = Counter("search_requests_total", "Total search requests")
SEARCH_LATENCY = Histogram("search_latency_seconds", "End-to-end search latency",
                           buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0])
RERANKER_LATENCY = Histogram("reranker_latency_seconds", "Cross-encoder reranking latency",
                             buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5])
SEARCH_RESULT_COUNT = Histogram("search_result_count", "Results returned per search",
                                buckets=[0, 1, 3, 5, 10, 20])
INDEX_SIZE = Gauge("index_size_services", "Number of services in the index")
CACHE_HITS = Counter("cache_hits_total", "Search cache hits")
