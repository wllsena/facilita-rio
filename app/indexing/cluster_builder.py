"""Agglomerative clustering over service embeddings."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from app.config import CONFIG
from app.models import Service


class ClusterIndex:

    def __init__(self, services: list[Service], embeddings: np.ndarray) -> None:
        n_clusters = min(CONFIG.n_clusters, len(services))
        labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(embeddings)
        self._id_to_cluster = {services[i].id: int(labels[i]) for i in range(len(services))}

    def same_cluster(self, id_a: str, id_b: str) -> bool:
        return self._id_to_cluster.get(id_a) == self._id_to_cluster.get(id_b)
