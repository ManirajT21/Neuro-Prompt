from sentence_transformers import SentenceTransformer
from models.embedding import get_embedding
import time

class Snapshot:
    def __init__(self, content, type_, project=None, topic=None, metadata=None):
        self.content = content
        self.type = type_
        self.project = project
        self.topic = topic
        self.timestamp = time.time()
        self.metadata = metadata or {}
        self.embedding = get_embedding(content)

class SnapshotBuffer:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(embedding_model_name)
        self.snapshots = []

    def add(self, content, type_, project=None, topic=None, metadata=None):
        snap = Snapshot(content, type_, project, topic, metadata)
        snap.embedding = self.model.encode(content, normalize_embeddings=True)
        self.snapshots.append(snap)
        return snap
    
    def clear(self):
        self.snapshots.clear()
