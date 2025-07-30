import numpy as np
from typing import List, Dict, Any

# Priority Mapping: Converts textual priority to numerical scores.
PRIORITY_MAP = {"high": 2, "medium": 1, "low": 0}

def normalize(val: float, minv: float, maxv: float) -> float:
    """
    Min-max normalization scales a value to the [0, 1] range.

    Args:
        val (float): Value to normalize.
        minv (float): Minimum value in data range.
        maxv (float): Maximum value in data range.

    Returns:
        float: Normalized value.
    """
    if maxv - minv == 0:
        return 0.0
    normalized_value = (val - minv) / (maxv - minv)
    return normalized_value

def aggregate_priority(snapshots: List[Any]) -> float:
    """
    Calculates the average priority for a set of snapshots.

    Args:
        snapshots (List[Any]): List of snapshot objects, each with metadata.

    Returns:
        float: Mean numeric priority.
    """
    priorities = []
    for snapshot in snapshots:
        priority_str = (snapshot.metadata or {}).get("priority", "medium").lower()
        numeric_priority = PRIORITY_MAP.get(priority_str, 1)
        priorities.append(numeric_priority)
    
    mean_priority = np.mean(priorities) if priorities else 1.0
    return mean_priority

def weighted_score(cluster_dict: Dict[str, float], weights: List[float]) -> float:
    """
    Computes a weighted aggregate score for clustering based on priority, size, and similarity.

    Args:
        cluster_dict (Dict[str, float]): Cluster attributes, normalized [0, 1]:
            - 'priority': normalized priority value.
            - 'size': normalized cluster size.
            - 'avg_similarity': average similarity within cluster.
        weights (List[float]): Weight factors [w_priority, w_size, w_similarity], summing to 1.

    Returns:
        float: Weighted cluster score.
    """
    factors = ['priority', 'size', 'avg_similarity']
    weighted_sum = sum(weights[i] * cluster_dict[factor] for i, factor in enumerate(factors))
    return weighted_sum

# Example usage (for clarity, testing, and debugging):
if __name__ == "__main__":
    class MockSnapshot:
        def __init__(self, priority):
            self.metadata = {'priority': priority}

    # Example snapshot priorities
    snapshots = [
        MockSnapshot('high'),
        MockSnapshot('low'),
        MockSnapshot('medium'),
        MockSnapshot('medium'),
    ]

    # Calculate aggregated priority
    agg_priority = aggregate_priority(snapshots)
    print(f"Aggregated Priority: {agg_priority}")

    # Example cluster dict
    cluster_example = {
        'priority': normalize(agg_priority, 0, 2),  # priorities range 0 (low) to 2 (high)
        'size': normalize(8, 1, 10),  # example cluster size: 8 out of [1,10]
        'avg_similarity': 0.75  # assumed calculated similarity
    }

    # Define weights
    weights = [0.6, 0.3, 0.1]  # priority most important, then size, then similarity

    # Compute weighted score
    score = weighted_score(cluster_example, weights)
    print(f"Weighted Score: {score:.3f}")