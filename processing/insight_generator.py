from models.gemma_interface import generate_insight

def draft_insights(clusters, buffer):
    insights = []
    for cluster_id, indices in clusters.items():
        topics = [buffer.snapshots[i].content for i in indices[:10]]
        prompt = f"Summarize themes: {topics}"
        insight = generate_insight(prompt)
        insights.append({
            "cluster_id": cluster_id,
            "avg_similarity": 0.8,  # Calculate actual similarity if needed
            "snapshots": [buffer.snapshots[i].__dict__ for i in indices],
            "insight": insight
        })
    return insights
