import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def check_hallucination(generated_text, buffer, threshold=0.8):
    embedding = buffer.model.encode(generated_text, normalize_embeddings=True)
    similarities = [cosine_similarity(embedding, s.embedding) for s in buffer.snapshots]
    max_sim = max(similarities) if similarities else 0
    return max_sim >= threshold, max_sim
