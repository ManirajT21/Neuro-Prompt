from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Literal, List, Dict, Any, Optional
import numpy as np
import time
import os
import re

from memory.snapshot import SnapshotBuffer
from memory.buffer import MemoryBuffer
from memory.clustering import adaptive_kmeans
from processing.insight_generator import draft_insights
from app.ollama_client import query_gemma
from models.embedding import get_embedding

EMBEDDING_DIM = 384
EMBEDDING_MODES = ("real", "fake", "random")
DEFAULT_EMBEDDING_MODE = os.environ.get("EMBEDDING_MODE", "real").lower()
global_embedding_mode = DEFAULT_EMBEDDING_MODE

CLUSTERING_THRESHOLD = 10
SNAPSHOT_RESET_THRESHOLD = 50

app = FastAPI()
buffer = SnapshotBuffer()
memory = MemoryBuffer(clustering_threshold=CLUSTERING_THRESHOLD)

global_state = {
    "prompt_count": 0,
    "last_clusters": [],
    "last_snapshots": [],
    "last_cluster_objs": [],
    "last_snapshot_lists": [],
    "last_insights": [],
    "memory_augmented_insights": [],
    "can_generate_insights": False,
    "insights_generated_for_this_clustering": False,
    "correction_log": [],
    "cluster_match_log": [],
    "similarity_debug_log": [],
}

def reset_memory():
    buffer.clear()
    memory.snapshots.clear()
    if hasattr(memory, "clusters"):
        memory.clusters = {}
    if hasattr(memory, "centroids"):
        memory.centroids = {}
    global_state.update({
        "prompt_count": 0,
        "last_clusters": [],
        "last_snapshots": [],
        "last_cluster_objs": [],
        "last_snapshot_lists": [],
        "last_insights": [],
        "memory_augmented_insights": [],
        "can_generate_insights": False,
        "insights_generated_for_this_clustering": False,
        "correction_log": [],
        "cluster_match_log": [],
        "similarity_debug_log": [],
    })

class GenerateRequest(BaseModel):
    prompt: str
    priority: Literal['high', 'medium', 'low'] = 'medium'

def readable_time(ts):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b + 1e-8))

def to_jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {k: to_jsonable(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    return obj

def extract_keywords_from_insight(insight_text):
    pattern = r"(key( repeated)? keywords?|keywords?)\s*[:\-]\s*(.+)"
    for line in insight_text.splitlines():
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            keywords = match.group(3)
            keywords = re.split(r"[.;]", keywords)[0]
            keywords = keywords.strip(" .:-").lower()
            return keywords
    if ":" in insight_text:
        keywords = insight_text.split(":")[-1]
        return keywords.strip(" .:-").lower()
    return insight_text.lower()

def find_best_insight(prompt: str, insights: List[Dict], cluster_centroids=None, threshold: float = 0.5):
    prompt_emb = get_embedding(prompt)
    best_idx, best_sim = None, 0.0
    sims_debug = []
    for idx, cluster in enumerate(insights):
        insight_text = cluster.get("insight", "")
        if not insight_text:
            continue
        # Use both keywords and cluster centroid
        keywords = extract_keywords_from_insight(insight_text)
        insight_emb = get_embedding(keywords)
        sim_kw = cosine_similarity(prompt_emb, insight_emb)
        sim_cen = 0.0
        if cluster_centroids and idx in cluster_centroids:
            sim_cen = cosine_similarity(prompt_emb, cluster_centroids[idx])
        sim = max(sim_kw, sim_cen)
        sims_debug.append((keywords, sim_kw, sim_cen, sim))
        if sim > best_sim:
            best_sim = sim
            best_idx = idx
    # Debug print similarities
    print("\n[SIMILARITY DEBUG] Prompt:", prompt)
    for i, (text, sim_kw, sim_cen, sim) in enumerate(sims_debug):
        print(f"  Insight {i}: sim_kw={sim_kw:.3f} sim_centroid={sim_cen:.3f} FINAL={sim:.3f} | insight_keywords={text!r}")
    global_state["similarity_debug_log"].append({
        "prompt": prompt,
        "sims": [{ "idx": i, "sim_kw": float(sim_kw), "sim_centroid": float(sim_cen), "sim": float(sim), "insight_keywords": text} for i, (text, sim_kw, sim_cen, sim) in enumerate(sims_debug)]
    })
    return (best_idx, best_sim) if best_sim >= threshold else (None, best_sim)

@app.post("/generate/")
async def generate(item: GenerateRequest, embedding_mode: Optional[str] = Query(None)):
    global global_embedding_mode

    if global_state["prompt_count"] >= SNAPSHOT_RESET_THRESHOLD:
        reset_memory()
        return {"reset": f"Memory reset after {SNAPSHOT_RESET_THRESHOLD} snapshots."}

    if embedding_mode and embedding_mode in EMBEDDING_MODES:
        global_embedding_mode = embedding_mode

    snap = buffer.add(
        content=item.prompt,
        type_="text",
        project="current",
        metadata={"priority": item.priority}
    )
    memory.add_snapshot(snap)
    global_state["prompt_count"] += 1

    # --- CLUSTERING LOGIC ---
    do_clustering = memory.needs_clustering()
    if do_clustering:
        embeddings = memory.get_embeddings()
        clusters = adaptive_kmeans(embeddings)
        memory.update_clusters(clusters)

        # Each cluster with explicit cluster_id and organized prompt contents
        clusters_for_insight = []
        for cluster_id, indices in clusters.items():
            clusters_for_insight.append({
                "cluster_id": cluster_id,
                "snapshots": [
                    {
                        "content": memory.snapshots[i].content,
                        "time": readable_time(memory.snapshots[i].timestamp)
                    }
                    for i in indices
                ],
            })

        print("CLUSTER DEBUG:", {k: [memory.snapshots[i].content for i in idxs] for k, idxs in clusters.items()})
        global_state["last_clusters"] = clusters_for_insight
        global_state["last_snapshots"] = memory.snapshots[-CLUSTERING_THRESHOLD:]
        global_state["last_cluster_objs"] = clusters_for_insight
        global_state["last_snapshot_lists"] = [c["snapshots"] for c in clusters_for_insight]
        global_state["can_generate_insights"] = True
        global_state["insights_generated_for_this_clustering"] = False

    # --- MEMORY AUGMENTATION LOGIC ---
    output = None
    match_log_entry = {}
    memory_layer = global_state.get("memory_augmented_insights", [])
    cluster_centroids = getattr(memory, 'centroids', None)
    best_idx, best_sim = None, 0.0
    best_insight = None

    # Only allow augmentation if clustering happened (after 10 prompts and insights are in memory)
    if memory_layer and len(memory.snapshots) >= CLUSTERING_THRESHOLD:
        best_idx, best_sim = find_best_insight(item.prompt, memory_layer, cluster_centroids)
        if best_idx is not None:
            best_insight = memory_layer[best_idx]["insight"]

    if best_insight:
        aug_prompt = f"{best_insight}\nUser: {item.prompt}\nAnswer:"
        output = query_gemma(aug_prompt)
        match_log_entry = {
            "memory_augmented": True,
            "cluster_idx": best_idx,
            "similarity": best_sim,
            "used_insight": best_insight,
            "output": output,
            "timestamp": time.time(),
        }
    else:
        output = query_gemma(item.prompt)
        match_log_entry = {
            "memory_augmented": False,
            "reason": "No memory match or memory layer empty or not after 10th prompt",
            "output": output,
            "timestamp": time.time(),
        }

    global_state["cluster_match_log"].append(match_log_entry)

    display_snaps = memory.snapshots[-CLUSTERING_THRESHOLD:]
    snapshot_list = [
        {
            "type": s.type,
            "content": s.content,
            "project": s.project,
            "time": readable_time(s.timestamp),
            "metadata": s.metadata
        }
        for s in display_snaps
    ]

    return to_jsonable({
        "response": output,
        "last_10_snapshots": snapshot_list,
        "clusters": global_state["last_clusters"],
        "can_generate_insights": global_state["can_generate_insights"],
        "insights_generated": global_state["insights_generated_for_this_clustering"],
        "insights": global_state.get("last_insights", []),
        "match_log": match_log_entry,
        "embedding_mode": global_embedding_mode
    })

@app.post("/generate_insights/")
async def generate_insights():
    if not global_state.get("can_generate_insights") or global_state.get("insights_generated_for_this_clustering"):
        return {"error": "Insights cannot be generated right now. Complete clustering and don't generate twice."}
    clusters = global_state.get("last_cluster_objs", [])
    if not clusters:
        return {"error": "No clusters available. Generate more prompts."}
    insights = []
    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        snaps = cluster["snapshots"]
        topics = [snap["content"] for snap in snaps][:10]
        prompt = (
            f"Given these user queries: {topics}\n"
            "Summarize the main theme of these queries in one short sentence.\n"
            "Then, in a line starting with 'Keywords:', list all key repeated words or concepts (comma-separated)."
        )
        insight = query_gemma(prompt)
        insights.append({"cluster": cluster_id, "insight": insight, "prompts": topics})
    global_state["insights_generated_for_this_clustering"] = True
    global_state["can_generate_insights"] = False
    global_state["last_insights"] = insights
    print("[DEBUG] New cluster insights:", [i["insight"] for i in insights])
    return to_jsonable({"cluster_insights": insights})

@app.post("/augment_memory/")
async def augment_memory():
    # Only after 10th prompt
    if len(memory.snapshots) < CLUSTERING_THRESHOLD:
        return {"error": f"Memory augmentation only allowed after {CLUSTERING_THRESHOLD} prompts."}
    latest = global_state.get("last_insights", [])
    if not latest:
        return {"error": "No insights to augment. Generate insights first."}
    global_state["memory_augmented_insights"] = latest.copy()
    print("[DEBUG] memory_augmented_insights updated:", [i["insight"] for i in latest])
    return {"augmented_count": len(latest)}

@app.get("/correction_log/")
async def get_correction_log():
    return to_jsonable({"correction_log": global_state["correction_log"]})

@app.get("/match_log/")
async def get_match_log():
    return to_jsonable({"cluster_match_log": global_state["cluster_match_log"]})

@app.get("/similarity_debug_log/")
async def get_similarity_debug_log():
    return to_jsonable({"similarity_debug_log": global_state["similarity_debug_log"]})

@app.get("/embedding_mode/")
async def get_embedding_mode(mode: Optional[str] = None):
    global global_embedding_mode
    if mode and mode in EMBEDDING_MODES:
        global_embedding_mode = mode
    return {"embedding_mode": global_embedding_mode, "valid_modes": EMBEDDING_MODES}

@app.post("/reset/")
async def manual_reset():
    reset_memory()
    return {"reset": f"Memory manually reset."}