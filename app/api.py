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
    "cluster_centroids": {},
    "next_cluster_id": 0,
    "cluster_history": [],
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
        "cluster_centroids": {},
        "next_cluster_id": 0,
        "cluster_history": [],
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

def compute_cluster_centroid(snapshot_indices, memory_snapshots):
    if not snapshot_indices:
        return None
    embeddings = []
    for idx in snapshot_indices:
        if idx < len(memory_snapshots):
            embedding = get_embedding(memory_snapshots[idx].content)
            embeddings.append(embedding)
    if not embeddings:
        return None
    return np.mean(embeddings, axis=0)

def match_clusters_to_stable_ids(new_clusters, memory_snapshots, stored_centroids, threshold=0.7):
    if not stored_centroids:
        stable_mapping = {}
        new_centroids = {}
        for new_id, indices in new_clusters.items():
            stable_id = global_state["next_cluster_id"]
            global_state["next_cluster_id"] += 1
            stable_mapping[new_id] = stable_id
            centroid = compute_cluster_centroid(indices, memory_snapshots)
            if centroid is not None:
                new_centroids[stable_id] = centroid
        global_state["cluster_centroids"] = new_centroids
        return stable_mapping

    stable_mapping = {}
    new_centroids = {}
    used_stable_ids = set()
    new_cluster_centroids = {}
    for new_id, indices in new_clusters.items():
        centroid = compute_cluster_centroid(indices, memory_snapshots)
        if centroid is not None:
            new_cluster_centroids[new_id] = centroid

    for new_id, new_centroid in new_cluster_centroids.items():
        best_match_id = None
        best_similarity = 0.0
        for stable_id, stored_centroid in stored_centroids.items():
            if stable_id in used_stable_ids:
                continue
            similarity = cosine_similarity(new_centroid, stored_centroid)
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match_id = stable_id
        if best_match_id is not None:
            stable_mapping[new_id] = best_match_id
            used_stable_ids.add(best_match_id)
            new_centroids[best_match_id] = new_centroid
        else:
            new_stable_id = global_state["next_cluster_id"]
            global_state["next_cluster_id"] += 1
            stable_mapping[new_id] = new_stable_id
            new_centroids[new_stable_id] = new_centroid
    global_state["cluster_centroids"] = new_centroids
    return stable_mapping

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

def find_best_insight(
    prompt: str,
    insights: List[Dict],
    clusters: List[Dict],
    get_embedding,
    threshold: float = 0.8,
):
    if not insights or not clusters:
        return None, 0.0, None, [], None

    prompt_emb = get_embedding(prompt)
    best_sim = 0.0
    best_insight_obj = None
    best_cluster_id = None
    best_prompt_list = []
    match_reason = None

    cluster_prompts_by_id = {}
    for c in clusters:
        cluster_id = c.get("cluster_id")
        if cluster_id is not None:
            cluster_id_str = str(cluster_id)
            cluster_prompts_by_id[cluster_id_str] = [s["content"] for s in c.get("snapshots", [])]

    for insight in insights:
        insight_cluster_id = insight.get("cluster")
        if insight_cluster_id is None:
            continue
        insight_cluster_id_str = str(insight_cluster_id)
        cluster_prompts = cluster_prompts_by_id.get(insight_cluster_id_str, [])
        sim_scores = []
        for p in cluster_prompts:
            try:
                p_emb = get_embedding(p)
                sim = cosine_similarity(prompt_emb, p_emb)
                sim_scores.append(sim)
            except Exception:
                continue
        insight_text = insight.get("insight", "")
        if not insight_text:
            continue
        keywords = extract_keywords_from_insight(insight_text)
        try:
            kw_emb = get_embedding(keywords)
            kw_sim = cosine_similarity(prompt_emb, kw_emb)
        except Exception:
            kw_sim = 0.0

        max_prompt_sim = max(sim_scores) if sim_scores else 0.0

        # Prefer keyword match, then prompt match, only if above threshold
        if kw_sim >= threshold and kw_sim > best_sim:
            best_sim = kw_sim
            best_insight_obj = insight
            best_cluster_id = insight_cluster_id_str
            best_prompt_list = cluster_prompts
            match_reason = "keyword"
        elif sim_scores and max_prompt_sim >= threshold and max_prompt_sim > best_sim:
            best_sim = max_prompt_sim
            best_insight_obj = insight
            best_cluster_id = insight_cluster_id_str
            best_prompt_list = cluster_prompts
            match_reason = "prompt"

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

    do_clustering = memory.needs_clustering()
    if do_clustering:
        embeddings = memory.get_embeddings()
        raw_clusters = adaptive_kmeans(embeddings)
        stable_mapping = match_clusters_to_stable_ids(
            raw_clusters, 
            memory.snapshots, 
            global_state["cluster_centroids"]
        )
        stable_clusters = {}
        for raw_id, indices in raw_clusters.items():
            stable_id = stable_mapping.get(raw_id, raw_id)
            stable_clusters[stable_id] = indices
        memory.update_clusters(stable_clusters)
        clusters_for_insight = []
        for stable_cluster_id, indices in stable_clusters.items():
            cluster_data = {
                "cluster_id": str(stable_cluster_id),
                "snapshots": [
                    {
                        "content": memory.snapshots[i].content,
                        "time": readable_time(memory.snapshots[i].timestamp)
                    }
                    for i in indices if i < len(memory.snapshots)
                ],
            }
            clusters_for_insight.append(cluster_data)
        history_entry = {
            "timestamp": time.time(),
            "prompt_count": global_state["prompt_count"],
            "raw_clusters": {str(k): [memory.snapshots[i].content for i in v if i < len(memory.snapshots)] for k, v in raw_clusters.items()},
            "stable_mapping": stable_mapping,
            "stable_clusters": {str(k): [memory.snapshots[i].content for i in v if i < len(memory.snapshots)] for k, v in stable_clusters.items()}
        }
        global_state["cluster_history"].append(history_entry)
        global_state["last_clusters"] = clusters_for_insight
        global_state["last_snapshots"] = memory.snapshots[-CLUSTERING_THRESHOLD:]
        global_state["last_cluster_objs"] = clusters_for_insight
        global_state["last_snapshot_lists"] = [c["snapshots"] for c in clusters_for_insight]
        global_state["can_generate_insights"] = True
        global_state["insights_generated_for_this_clustering"] = False
        insights = []
        for cluster in clusters_for_insight:
            cluster_id = cluster["cluster_id"]
            snaps = cluster["snapshots"]
            topics = [snap["content"] for snap in snaps][:10]
            prompt_for_insight = (
                f"Given these user queries: {topics}\n"
                "Summarize the main theme of these queries in one short sentence.\n"
                "Then, in a line starting with 'Keywords:', list all key repeated words or concepts (comma-separated)."
            )
            insight = query_gemma(prompt_for_insight)
            insight_obj = {
                "cluster": str(cluster_id),
                "insight": insight, 
                "prompts": topics
            }
            insights.append(insight_obj)
        global_state["last_insights"] = insights
        global_state["memory_augmented_insights"] = insights.copy()
        global_state["insights_generated_for_this_clustering"] = True
        global_state["can_generate_insights"] = False

    output = None
    match_log_entry = {}
    memory_layer = global_state.get("memory_augmented_insights", [])
    clusters = global_state.get("last_clusters", [])
    best_insight_obj, best_sim, best_cluster_id, best_prompt_list, match_reason = None, 0.0, None, [], None

    # --- The ONLY change here: prompt formatting if best match found (sim >= 0.8) ---
    if memory_layer and len(memory.snapshots) >= CLUSTERING_THRESHOLD:
        try:
            best_insight_obj, best_sim, best_cluster_id, best_prompt_list, match_reason = find_best_insight(
                item.prompt, memory_layer, clusters, get_embedding, threshold=0.8
            )
        except Exception as e:
            print(f"[ERROR] Failed to find best insight: {e}")
            best_insight_obj = None

    if best_insight_obj:
        aug_prompt = (
            f"{item.prompt}\n\n"
            f"Matched Insight:\n{best_insight_obj['insight']}\n"
            f"Answer:"
        )
        output = query_gemma(aug_prompt)
        match_log_entry = {
            "memory_augmented": True,
            "cluster_idx": best_cluster_id,
            "similarity": best_sim,
            "used_insight": best_insight_obj["insight"],
            "match_reason": match_reason,
            "output": output,
            "timestamp": time.time(),
            "prompt": item.prompt,
        }
    else:
        output = query_gemma(item.prompt)
        match_log_entry = {
            "memory_augmented": False,
            "reason": "No suitable cluster match or memory layer empty or not after 10th prompt",
            "output": output,
            "timestamp": time.time(),
            "prompt": item.prompt,
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
        "embedding_mode": global_embedding_mode,
        "stable_cluster_count": len(global_state["cluster_centroids"])
    })

@app.post("/generate_insights/")
async def generate_insights():
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
        insight_obj = {
            "cluster": str(cluster_id),
            "insight": insight,
            "prompts": topics
        }
        insights.append(insight_obj)

    global_state["insights_generated_for_this_clustering"] = True
    global_state["can_generate_insights"] = False
    global_state["last_insights"] = insights
    global_state["memory_augmented_insights"] = insights.copy()
    return to_jsonable({"cluster_insights": insights})

@app.post("/augment_memory/")
async def augment_memory():
    if len(memory.snapshots) < CLUSTERING_THRESHOLD:
        return {"error": f"Memory augmentation only allowed after {CLUSTERING_THRESHOLD} prompts."}
    latest = global_state.get("last_insights", [])
    if not latest:
        return {"error": "No insights to augment. Generate insights first."}
    global_state["memory_augmented_insights"] = latest.copy()
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

@app.get("/cluster_history/")
async def get_cluster_history():
    return to_jsonable({"cluster_history": global_state["cluster_history"]})

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

@app.get("/validate_clusters/")
async def validate_clusters():
    clusters = global_state.get("last_clusters", [])
    insights = global_state.get("memory_augmented_insights", [])
    centroids = global_state.get("cluster_centroids", {})

    validation_report = {
        "clusters_count": len(clusters),
        "insights_count": len(insights),
        "stable_centroids_count": len(centroids),
        "next_cluster_id": global_state.get("next_cluster_id", 0),
        "cluster_details": [],
        "insight_details": [],
        "centroid_details": [],
        "mismatches": []
    }

    for cluster in clusters:
        cluster_id = str(cluster.get("cluster_id", ""))
        prompts = [s["content"] for s in cluster.get("snapshots", [])]
        validation_report["cluster_details"].append({
            "cluster_id": cluster_id,
            "prompts": prompts,
            "prompt_count": len(prompts)
        })

    for insight in insights:
        insight_cluster_id = str(insight.get("cluster", ""))
        insight_text = insight.get("insight", "")
        stored_prompts = insight.get("prompts", [])
        validation_report["insight_details"].append({
            "cluster_id": insight_cluster_id,
            "insight": insight_text[:100] + "..." if len(insight_text) > 100 else insight_text,
            "stored_prompts": stored_prompts,
            "stored_prompt_count": len(stored_prompts)
        })

    for centroid_id, centroid in centroids.items():
        validation_report["centroid_details"].append({
            "cluster_id": str(centroid_id),
            "centroid_exists": centroid is not None,
            "centroid_shape": np.array(centroid).shape if centroid is not None else None
        })

    cluster_lookup = {str(c["cluster_id"]): [s["content"] for s in c["snapshots"]] for c in clusters}
    for insight in insights:
        insight_cluster_id = str(insight.get("cluster", ""))
        actual_prompts = set(cluster_lookup.get(insight_cluster_id, []))
        stored_prompts = set(insight.get("prompts", []))

        if actual_prompts != stored_prompts:
            validation_report["mismatches"].append({
                "cluster_id": insight_cluster_id,
                "actual_prompts": list(actual_prompts),
                "stored_prompts": list(stored_prompts),
                "missing_from_insight": list(actual_prompts - stored_prompts),
                "extra_in_insight": list(stored_prompts - actual_prompts)
            })

    return to_jsonable(validation_report)

@app.post("/generate_no_layer/")
async def generate_no_layer(item: GenerateRequest, embedding_mode: Optional[str] = Query(None)):
    """
    Bypasses all memory/insight logic, simply calls the LLM directly.
    """
    global global_embedding_mode
    if embedding_mode and embedding_mode in EMBEDDING_MODES:
        global_embedding_mode = embedding_mode

    output = query_gemma(item.prompt)  # Direct call, no memory logic

    # Log the response if you want to analyze later
    match_log_entry = {
        "memory_augmented": False,
        "reason": "Direct LLM call, no memory/insight used",
        "output": output,
        "timestamp": time.time(),
        "prompt": item.prompt,
    }
    global_state["cluster_match_log"].append(match_log_entry)

    return to_jsonable({
        "response": output,
        "last_10_snapshots": [],
        "clusters": [],
        "can_generate_insights": False,
        "insights_generated": False,
        "insights": [],
        "match_log": match_log_entry,
        "embedding_mode": global_embedding_mode,
        "stable_cluster_count": 0
    })
