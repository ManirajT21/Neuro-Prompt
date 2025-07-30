from app.ollama_client import query_gemma

def generate_insight(prompt):
    return query_gemma(prompt)
