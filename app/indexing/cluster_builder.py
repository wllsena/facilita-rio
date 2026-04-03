"""Pre-compute semantic clusters over service embeddings."""

from __future__ import annotations

import numpy as np
import structlog
from sklearn.cluster import AgglomerativeClustering

from app.config import CONFIG
from app.models import Service

logger = structlog.get_logger()


class ClusterIndex:
    """Agglomerative clustering of services in embedding space."""

    def __init__(self, services: list[Service], embeddings: np.ndarray) -> None:
        self._services = services
        n_clusters = min(CONFIG.n_clusters, len(services))

        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        self._labels = clustering.fit_predict(embeddings)

        # Build cluster → service_ids mapping
        self._cluster_to_ids: dict[int, list[str]] = {}
        for i, label in enumerate(self._labels):
            label = int(label)
            self._cluster_to_ids.setdefault(label, []).append(services[i].id)

        # Build service_id → cluster mapping
        self._id_to_cluster = {services[i].id: int(self._labels[i]) for i in range(len(services))}

        logger.info(
            "clusters_built",
            n_clusters=n_clusters,
            cluster_sizes=[len(v) for v in self._cluster_to_ids.values()],
        )

    def same_cluster(self, id_a: str, id_b: str) -> bool:
        """Check if two services belong to the same semantic cluster."""
        return self._id_to_cluster.get(id_a) == self._id_to_cluster.get(id_b)

    def get_cluster_members(self, service_id: str) -> list[str]:
        """Get all service IDs in the same cluster as the given service."""
        cluster = self._id_to_cluster.get(service_id)
        if cluster is None:
            return []
        return [sid for sid in self._cluster_to_ids.get(cluster, []) if sid != service_id]
