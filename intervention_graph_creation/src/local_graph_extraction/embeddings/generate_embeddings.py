"""
Batch embeddings generator for local graph JSONs (async version).

Features:
- Reads config from config.yaml (paths, embeddings params).
- For each paper.json, generates embeddings for nodes/edges.
- Saves embeddings in paper_dir/embeddings/<10-char-id>.json
- Deterministic IDs: sha1 hash of key fields (first 10 hex chars).
- Resumable: skips already processed files.
- Errors:
  - Writes per-object error JSONs to SETTINGS.paths.embeddings_error_dir.
  - On fatal error for a paper, moves the entire article folder to SETTINGS.paths.embeddings_error_dir and writes error.txt inside (via write_failure).
- Async processing with configurable concurrency.
- Automatic retries (max_retries=3) with exponential backoff (2–5s).
- Logging to ./logs/embeddings_MM-DD_HH-MM.log
"""

import os
import json
import random
import time
import traceback
import asyncio
from asyncio import Semaphore
from pathlib import Path
from typing import List
import logging
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

from config import load_settings
from intervention_graph_creation.src.utils import short_id_node, short_id_edge
from intervention_graph_creation.src.local_graph_extraction.core.node import Node
from intervention_graph_creation.src.local_graph_extraction.core.edge import Edge
from intervention_graph_creation.src.local_graph_extraction.extract.utilities import write_failure

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

SETTINGS = load_settings()
MODEL = SETTINGS.embeddings.model
BATCH_SIZE = SETTINGS.embeddings.batch_size
OUTPUT_DIR: Path = SETTINGS.paths.output_dir

# Single errors folder for both per-object error JSONs and moved article folders
EMB_ERRORS_DIR: Path = SETTINGS.paths.embeddings_error_dir
EMB_ERRORS_DIR.mkdir(parents=True, exist_ok=True)

MAX_CONCURRENT_BATCHES = SETTINGS.embeddings.max_cuncurrent_batches

LOGS_DIR: Path = SETTINGS.paths.logs_dir
LOGS_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOGS_DIR / f"embeddings_{datetime.now().strftime('%m-%d_%H-%M')}.log"

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")
client = OpenAI(api_key=api_key)

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{random.randint(0, 1_000_000)}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

def log_error(obj_id: str, obj_type: str, text: str, error: Exception):
    """Per-object error json (node/edge) stays in embeddings_error_dir."""
    error_path = EMB_ERRORS_DIR / f"{obj_id}.json"
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
    # if node.get("description"):
    #     parts.append(f"Description: {node['description']}")
    # if node.get("aliases"):
    #     parts.append(f"Aliases: {', '.join(node['aliases'])}")
    # if node.get("concept_category"):
    #     parts.append(f"Category: {node['concept_category']}")
    return " | ".join(parts)

def edge_text(edge: dict) -> str:
    parts = []
    if edge.get("type"):
        parts.append(f"Type: {edge['type']}")
    if edge.get("description"):
        parts.append(f"Description: {edge['description']}")
    if edge.get("source_node"):
        parts.append(f"From: {edge['source_node']}")
    if edge.get("target_node"):
        parts.append(f"To: {edge['target_node']}")
    return " | ".join(parts)

# -----------------------------------------------------------------------------
# Task enumeration
# -----------------------------------------------------------------------------

class EmbTask:
    def __init__(self, embeddings_dir: Path, obj_type: str, obj_id: str, text: str, paper_dir: Path):
        self.embeddings_dir = embeddings_dir
        self.obj_type = obj_type
        self.obj_id = obj_id
        self.text = text
        self.paper_dir = paper_dir

    @property
    def out_path(self) -> Path:
        return self.embeddings_dir / f"{self.obj_id}.json"

def enumerate_tasks(paper_json: Path) -> List[EmbTask]:
    data = json.loads(paper_json.read_text(encoding="utf-8"))
    embeddings_dir = paper_json.parent / "embeddings"
    paper_dir = paper_json.parent
    tasks: List[EmbTask] = []

    for node in data.get("nodes", []):
        nid = short_id_node(Node(**node))
        tasks.append(EmbTask(embeddings_dir, "node", nid, node_text(node), paper_dir))

    for edge in data.get("edges", []):
        eid = short_id_edge(Edge(**edge))
        tasks.append(EmbTask(embeddings_dir, "edge", eid, edge_text(edge), paper_dir))

    return tasks

# -----------------------------------------------------------------------------
# Async Embeddings client with retries
# -----------------------------------------------------------------------------

async def embed_batch_async(
    batch: List[EmbTask],
    sem: Semaphore,
    stats: dict,
    max_retries: int = 3
):
    async with sem:
        delay = 2.0
        for attempt in range(1, max_retries + 1):
            try:
                texts = [t.text for t in batch]
                resp = await asyncio.to_thread(
                    client.embeddings.create,
                    model=MODEL,
                    input=texts,
                )
                embs = [item.embedding for item in resp.data]

                if len(embs) != len(batch):
                    raise ValueError(
                        f"Embedding count mismatch: expected {len(batch)}, got {len(embs)}"
                    )

                for t, emb in zip(batch, embs):
                    atomic_write_json(
                        t.out_path,
                        {
                            "id": t.obj_id,
                            "type": t.obj_type,
                            "text": t.text,
                            "embedding": emb,
                        },
                    )

                stats["done"] += len(batch)
                logger.info(
                    "Batch done (%d items, %.2fs). Progress: %d/%d, Remaining: %d",
                    len(batch),
                    delay,
                    stats["done"],
                    stats["total"],
                    stats["total"] - stats["done"],
                )
                return

            except Exception as e:
                if attempt >= max_retries:
                    tb = traceback.format_exc()
                    # Per-object error files
                    for t in batch:
                        log_error(t.obj_id, t.obj_type, t.text, e)
                    # Paper-level fatal: move whole folder using shared write_failure
                    unique_dirs = {t.paper_dir for t in batch}
                    for d in unique_dirs:
                        if d.exists():
                            write_failure(
                                output_root=OUTPUT_DIR,
                                error_root=EMB_ERRORS_DIR,
                                paper_id=d.name,
                                err=e,
                                error_type="embeddings_failed",
                                context={"traceback": tb, "batch_size": len(batch)},
                                logger=logger,
                            )
                    stats["errors"] += len(batch)
                    logger.error(
                        "Batch failed after %d attempts (%d items): %s",
                        attempt,
                        len(batch),
                        e,
                    )
                    return

                sleep_for = delay + random.uniform(0, 3.0)
                logger.warning(
                    "Batch failed (attempt %d/%d, %d items). Error: %s. Sleeping %.1fs ...",
                    attempt,
                    max_retries,
                    len(batch),
                    e,
                    sleep_for,
                )
                await asyncio.sleep(sleep_for)
                delay = min(delay * 2.0, 10.0)

# -----------------------------------------------------------------------------
# Async main
# -----------------------------------------------------------------------------

async def async_main(max_concurrent_batches: int = MAX_CONCURRENT_BATCHES):
    json_files = [
        p for p in OUTPUT_DIR.rglob("*.json")
        if not any(part in {"embeddings_error", "issues", "error", "embeddings"} for part in p.parts)
    ]
    if not json_files:
        logger.info("No JSON files found.")
        return

    all_tasks: List[EmbTask] = []
    skipped_files = 0
    processed_files = 0

    for jf in json_files:
        try:
            if (jf.parent / "embeddings").exists():
                skipped_files += 1
                continue
            tasks = enumerate_tasks(jf)
            if tasks:
                all_tasks.extend(tasks)
                processed_files += 1
        except Exception as e:
            logger.warning(
                "Failed to enumerate %s: %s\n%s",
                jf, str(e), traceback.format_exc()
            )
            # Fatal for this paper: move folder using shared write_failure
            paper_dir = jf.parent
            if paper_dir.exists():
                write_failure(
                    output_root=OUTPUT_DIR,
                    error_root=EMB_ERRORS_DIR,
                    paper_id=paper_dir.name,
                    err=e,
                    error_type="enumeration_failed",
                    context={"json_file": str(jf), "traceback": traceback.format_exc()},
                    logger=logger,
                )

    unique: dict[str, EmbTask] = {t.obj_id: t for t in all_tasks}
    tasks = list(unique.values())

    logger.info(
        "JSON files: %d, Skipped: %d, Processed files: %d, Tasks: %d",
        len(json_files), skipped_files, processed_files, len(tasks)
    )

    if not tasks:
        logger.info("All embeddings already computed. Nothing to do.")
        return

    batches = [tasks[i:i + BATCH_SIZE] for i in range(0, len(tasks), BATCH_SIZE)]
    logger.info(
        "Planned batches: %d (batch_size=%d, concurrency=%d)",
        len(batches), BATCH_SIZE, max_concurrent_batches
    )

    sem = Semaphore(max_concurrent_batches)
    stats = {"done": 0, "errors": 0, "total": len(tasks)}

    start_time = time.time()
    await asyncio.gather(*(embed_batch_async(batch, sem, stats) for batch in batches))
    duration = time.time() - start_time

    logger.info(
        "Done. Success: %d, Errors: %d, Duration: %.1fs",
        stats["done"], stats["errors"], duration
    )

def main():
    asyncio.run(async_main(MAX_CONCURRENT_BATCHES))

if __name__ == "__main__":
    main()
