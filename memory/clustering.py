from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

def adaptive_kmeans(embeddings, min_k=2, max_k=10):
    if len(embeddings) < min_k:
        # Not enough for clustering
        return {}
    best_k = min_k
    best_score = -1
    for k in range(min_k, min(max_k + 1, len(embeddings))):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(embeddings)
        if len(set(kmeans.labels_)) == 1:
            # All assigned to one cluster, silhouette_score will fail
            continue
        score = silhouette_score(embeddings, kmeans.labels_)
        if score > best_score:
            best_score = score
            best_k = k
    # Fit final model
    final_kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit(embeddings)
    labels = final_kmeans.labels_
    clusters = {i: [] for i in range(best_k)}
    for idx, label in enumerate(labels):
        clusters[label].append(idx)
    return clusters