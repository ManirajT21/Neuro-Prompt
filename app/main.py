import uvicorn
from app.api import app
from memory.snapshot import SnapshotBuffer
from memory.buffer import MemoryBuffer
from memory.clustering import adaptive_kmeans
from processing.insight_generator import draft_insights
from processing.hallucination_filter import check_hallucination
from app.ollama_client import query_gemma
buffer = SnapshotBuffer()

if __name__ == "__main__":
    import uvicorn
    # This form is safe for both script and CLI use
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
