"""
Batch embeddings generator for local graph JSONs.

Features:
- Reads config from config.yaml (paths, embeddings params).
- For each paper.json, generates embeddings for nodes/edges.
- Saves embeddings in paper_dir/embeddings/<10-char-id>.json
- Deterministic IDs: sha1 hash of key fields (first 10 hex chars).
- Resumable: skips already processed files.
- Logs errors to SETTINGS.paths.output_dir / "embedding_errors".
"""

import os
import json
import random
import time
import traceback
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from config import load_settings
from intervention_graph_creation.src.utils import short_id

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

SETTINGS = load_settings()
MODEL = SETTINGS.embeddings.model
BATCH_SIZE = SETTINGS.embeddings.batch_size
OUTPUT_DIR = SETTINGS.paths.output_dir

ERRORS_DIR = OUTPUT_DIR / "embedding_errors"
ERRORS_DIR.mkdir(parents=True, exist_ok=True)


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")
client = OpenAI(api_key=api_key)

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------


def atomic_write_json(path: Path, data: dict) -> None:
    """Safe atomic write to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{random.randint(0, 1_000_000)}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)



def log_error(obj_id: str, obj_type: str, text: str, error: Exception):
    """Save error info to embedding_errors/<id>.json"""
    error_path = ERRORS_DIR / f"{obj_id}.json"
    payload = {
        "id": obj_id,
        "type": obj_type,
        "text": text,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_json(error_path, payload)


# -----------------------------------------------------------------------------
# Text builders
# -----------------------------------------------------------------------------

def node_text(node: dict) -> str:
    parts = []
    if node.get("name"):
        parts.append(f"Name: {node['name']}")
    if node.get("description"):
        parts.append(f"Description: {node['description']}")
    if node.get("aliases"):
        parts.append(f"Aliases: {', '.join(node['aliases'])}")
    if node.get("concept_category"):
        parts.append(f"Category: {node['concept_category']}")
    return " | ".join(parts)


def edge_text(edge: dict, logical_chain_title: Optional[str]) -> str:
    parts = []
    if edge.get("type"):
        parts.append(f"Type: {edge['type']}")
    if edge.get("description"):
        parts.append(f"Description: {edge['description']}")
    if logical_chain_title:
        parts.append(f"Concept: {logical_chain_title}")
    if edge.get("source_node"):
        parts.append(f"From: {edge['source_node']}")
    if edge.get("target_node"):
        parts.append(f"To: {edge['target_node']}")
    return " | ".join(parts)


# -----------------------------------------------------------------------------
# Task enumeration
# -----------------------------------------------------------------------------

class EmbTask:
    def __init__(self, embeddings_dir: Path, obj_type: str, obj_id: str, text: str):
        self.embeddings_dir = embeddings_dir
        self.obj_type = obj_type
        self.obj_id = obj_id
        self.text = text

    @property
    def out_path(self) -> Path:
        return self.embeddings_dir / f"{self.obj_id}.json"


def enumerate_tasks(paper_json: Path) -> List[EmbTask]:
    """Build embedding tasks for one paper.json"""
    data = json.loads(paper_json.read_text(encoding="utf-8"))
    embeddings_dir = paper_json.parent / "embeddings"
    tasks: List[EmbTask] = []

    # Nodes
    for node in data.get("nodes", []):
        nid = short_id(f"{node.get('name','')}|{node.get('type','')}")
        tasks.append(EmbTask(embeddings_dir, "node", nid, node_text(node)))

    # Edges
    for chain in data.get("logical_chains", []):
        title = chain.get("title")
        for edge in chain.get("edges", []):
            eid = short_id(f"{edge.get('type','')}|{edge.get('source_node','')}|{edge.get('target_node','')}")
            tasks.append(EmbTask(embeddings_dir, "edge", eid, edge_text(edge, title)))

    return tasks


# -----------------------------------------------------------------------------
# Embeddings client
# -----------------------------------------------------------------------------

def embed_batch(texts: List[str]) -> List[List[float]]:
    """Call OpenAI API for a batch of texts."""
    resp = client.embeddings.create(model=MODEL, input=texts)
    return [item.embedding for item in resp.data]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    json_files = list(OUTPUT_DIR.rglob("*.json"))
    json_files = [p for p in json_files if p.parent.name not in ("embedding_errors", "issues", "error")]
    if not json_files:
        print("No JSON files found.")
        return

    all_tasks: List[EmbTask] = []
    for jf in json_files:
        try:
            if (jf.parent / "embeddings").exists():
                print(f"[SKIP] {jf} -> embeddings folder already exists")
                continue

            all_tasks.extend(enumerate_tasks(jf))
        except Exception as e:
            print(f"[WARN] Failed to enumerate {jf}: {e}")

    unique: dict[str, EmbTask] = {t.obj_id: t for t in all_tasks}
    tasks = list(unique.values())
    print(f"Total tasks: {len(all_tasks)}, Pending: {len(tasks)}")

    if not tasks:
        print("✅ All embeddings already computed. Nothing to do.")
        return

    batch_size = BATCH_SIZE
    i = 0
    while i < len(tasks):
        batch = tasks[i:i + batch_size]
        i += batch_size
        try:
            embs = embed_batch([t.text for t in batch])
            for t, emb in zip(batch, embs):
                atomic_write_json(t.out_path, {
                    "id": t.obj_id,
                    "type": t.obj_type,
                    "text": t.text,
                    "embedding": emb,
                })
        except Exception as e:
            for t in batch:
                log_error(t.obj_id, t.obj_type, t.text, e)
            print(f"[ERROR] Batch failed with {len(batch)} items: {e}")
            if len(batch) > 1:
                i -= len(batch)
                batch_size = max(1, batch_size // 2)
                print(f"[INFO] Reducing batch size to {batch_size} and retrying...")

    print("Done.")


if __name__ == "__main__":
    main()
