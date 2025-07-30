from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    emb = model.encode(text)
    return emb.tolist()
