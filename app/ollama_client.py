import requests

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"

def query_gemma(prompt: str, model="gemma"):
    response = requests.post(
        OLLAMA_ENDPOINT,
        json={
            "prompt": prompt,
            "model": model,
            "stream": False    # <--- This is the fix!
        }
    )
    if response.ok:
        # Ollama returns {'response': 'text...'}
        return response.json()["response"]
    else:
        raise Exception(f"Ollama error: {response.text}")
