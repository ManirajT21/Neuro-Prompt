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

# --- PATCHED CLUSTER MATCH FUNCTION: Tight threshold, prioritize keyword, allow prompt only if >0.95 ---
def find_best_insight(
    prompt: str,
    insights: List[Dict],
    clusters: List[Dict],
    get_embedding,
    threshold: float = 0.8,   # << tighten the threshold
):
    prompt_emb = get_embedding(prompt)
    best_sim = 0.0
    best_insight_obj = None
    best_cluster_id = None
    best_prompt_list = []
    match_reason = None

    # Map cluster_id to cluster prompts for quick lookup
    cluster_prompts_by_id = {int(c["cluster_id"]): [s["content"] for s in c["snapshots"]] for c in clusters}

    sims_debug = []

    for insight in insights:
        cluster_id = int(insight["cluster"])
        cluster_prompts = cluster_prompts_by_id.get(cluster_id, [])
        # Compute similarity to all prompts in this cluster
        sim_scores = []
        for p in cluster_prompts:
            sim = cosine_similarity(prompt_emb, get_embedding(p))
            sim_scores.append(sim)
        # Compute similarity to cluster keywords (from insight)
        keywords = extract_keywords_from_insight(insight["insight"])
        kw_emb = get_embedding(keywords)
        kw_sim = cosine_similarity(prompt_emb, kw_emb)
        sims_debug.append({
            "cluster_id": cluster_id,
            "sim_scores": sim_scores,
            "keywords": keywords,
            "kw_sim": kw_sim,
            "max_sim": max(sim_scores) if sim_scores else 0.0,
            "prompts": cluster_prompts,
        })
        # Prefer keywords
        if kw_sim >= threshold and kw_sim > best_sim:
            best_sim = kw_sim
            best_insight_obj = insight
            best_cluster_id = cluster_id
            best_prompt_list = cluster_prompts
            match_reason = "keyword"
        # Only allow prompt match if extremely high, and not overshadowed by a good keyword match
        elif sim_scores and max(sim_scores) > 0.95 and max(sim_scores) > best_sim:
            best_sim = max(sim_scores)
            best_insight_obj = insight
            best_cluster_id = cluster_id
            best_prompt_list = cluster_prompts
            match_reason = "prompt"

    print("\n[SIMILARITY DEBUG] Prompt:", prompt)
    for d in sims_debug:
        print(
            f"  Cluster {d['cluster_id']}: "
            f"max_sim={d['max_sim']:.3f} | "
            f"keywords={d['keywords']!r} | "
            f"prompt_sims={[round(x,3) for x in d['sim_scores']]} | "
            f"kw_sim={d['kw_sim']:.3f}"
        )

    global_state["similarity_debug_log"].append({
        "prompt": prompt,
        "clusters": sims_debug
    })

    if best_insight_obj is not None:
        return best_insight_obj, best_sim, best_cluster_id, best_prompt_list, match_reason
    return None, best_sim, None, [], None

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
    clusters = global_state.get("last_clusters", [])
    best_insight_obj, best_sim, best_cluster_id, best_prompt_list, match_reason = None, 0.0, None, [], None

    # Only allow augmentation if clustering happened (after 10 prompts and insights are in memory)
    if memory_layer and len(memory.snapshots) >= CLUSTERING_THRESHOLD:
        best_insight_obj, best_sim, best_cluster_id, best_prompt_list, match_reason = find_best_insight(
            item.prompt, memory_layer, clusters, get_embedding, threshold=0.8
        )

    if best_insight_obj:
        # Provide the matched cluster's insight and up to 5 prompt examples for better LLM context
        cluster_examples = "\n".join(f"- {p}" for p in best_prompt_list[:5])
        aug_prompt = (
            f"{best_insight_obj['insight']}\n"
            f"Here are related user queries for context:\n{cluster_examples}\n"
            f"User: {item.prompt}\nAnswer:"
        )
        output = query_gemma(aug_prompt)
        match_log_entry = {
            "memory_augmented": True,
            "cluster_idx": best_cluster_id,
            "similarity": best_sim,
            "used_insight": best_insight_obj["insight"],
            "cluster_examples": best_prompt_list[:5],
            "match_reason": match_reason,
            "output": output,
            "timestamp": time.time(),
        }
    else:
        output = query_gemma(item.prompt)
        match_log_entry = {
            "memory_augmented": False,
            "reason": "No suitable cluster match or memory layer empty or not after 10th prompt",
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