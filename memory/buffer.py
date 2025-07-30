import numpy as np

class MemoryBuffer:
    def __init__(self, clustering_threshold=10):
        self.snapshots = []
        self.clustering_threshold = clustering_threshold
        self.clusters = {}     # cluster_id: list of indices into self.snapshots
        self.centroids = {}    # cluster_id: np.ndarray centroid vector

    def add_snapshot(self, snapshot):
        self.snapshots.append(snapshot)

    def get_embeddings(self):
        return np.array([s.embedding for s in self.snapshots])

    def get_by_field(self, field):
        grouped = {}
        for snap in self.snapshots:
            key = getattr(snap, field, "Other") or "Other"
            grouped.setdefault(key, []).append(snap)
        return grouped

    def needs_clustering(self):
        # Only true after 10 prompts, and after every new prompt past that
        return len(self.snapshots) >= self.clustering_threshold and \
            (len(self.snapshots) == self.clustering_threshold or len(self.snapshots) > self.clustering_threshold)

    def update_clusters(self, clusters_dict):
        """clusters_dict: {cluster_id: [indices]}"""
        self.clusters = clusters_dict
        self._update_centroids()

    def _update_centroids(self):
        embeddings = self.get_embeddings()
        self.centroids = {}
        for cid, indices in self.clusters.items():
            if indices:
                self.centroids[cid] = np.mean(embeddings[indices], axis=0)
            else:
                self.centroids[cid] = np.zeros(embeddings.shape[1])  # fallback

    def get_centroid(self, cluster_idx):
        return self.centroids.get(cluster_idx, None)

    def get_cluster_snapshots(self, cluster_idx):
        indices = self.clusters.get(cluster_idx, [])
        return [self.snapshots[i] for i in indices]
