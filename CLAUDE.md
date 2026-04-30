# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project extracts structured knowledge graphs from AI safety literature. PDFs and JSONL files are processed via OpenAI's API (`o3` model) to extract `concept` and `intervention` nodes connected by logical chain edges. The resulting graph is stored in FalkorDB; duplicate nodes are merged via embedding similarity.

## Commands

```bash
# Install dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install

# Lint and format (ruff)
uv run pre-commit run --all-files

# Run extraction (async batch mode) on inputs in config.yaml input_dir
python intervention_graph_creation/src/local_graph_extraction/extract/extractor.py

# Start FalkorDB with empty graph (extraction pipeline use)
docker run -p 6379:6379 -p 3000:3000 -it --rm falkordb/falkordb:latest

# Start FalkorDB and load existing dump.rdb (graph_analysis use; ~15 min indices)
# - Use -it (foreground TTY) — detached mode (-d) silently exits with code 255
#   during index construction with no logs (no TTY → child processes die silently).
# - Mount target MUST be /var/lib/falkordb/data (not /data) for the :edge image.
# - Place dump.rdb (the wide-embeddings ARD variant per issue #132) in <data_dir>.
# - On Windows + WSL: run inside `wsl` shell directly (Docker is in WSL only).
docker run -p 6379:6379 -p 3000:3000 -it --rm \
  --volume ./data:/var/lib/falkordb/data \
  falkordb/falkordb falkordb:edge

# Ingest extracted JSONs into FalkorDB
python intervention_graph_creation/src/local_graph_extraction/db/ai_safety_graph.py

# Run unit tests
uv run python intervention_graph_creation/unit_tests/test_data_interfaces_loaders.py
uv run python intervention_graph_creation/unit_tests/test_data_interfaces_pdf_extractor.py

# View graph at http://localhost:3000/graph
```

## Configuration

All paths and DB settings are in `config.yaml` (loaded via `config.py:load_settings()`). Key sections:

- `paths.input_dir` — where PDFs and `arxiv.jsonl` are read from
- `paths.output_dir` — where extracted per-paper directories are written
- `falkordb.host/port/graph` — FalkorDB connection and graph name

## Architecture

### Pipeline stages

**1. Data loading** (`intervention_graph_creation/src/data_interfaces/`)  
Loaders for PDFs from local folder, Hugging Face ARD dataset, or arXiv IDs by URL/ID list. Returns `Publication` Pydantic objects.

**2. Extraction** (`src/local_graph_extraction/extract/extractor.py`)  
`Extractor` uploads PDFs/JSONL text to OpenAI and calls `o3` with `PROMPT_EXTRACT`. Supports:
- `process_dir()` — sequential per-file calls
- `process_dir_batch_async()` — concurrent OpenAI Batch API (50k req/batch cap, auto-retry on errors)

Each paper outputs to a subdirectory under `output_dir/{paper_stem}/`:
- `{stem}.json` — parsed nodes/edges/meta (feeds into ingestion)
- `{stem}_raw_response.txt` — full API response
- `{stem}_summary.txt` — reasoning narrative from the model

**3. Core schemas** (`src/local_graph_extraction/core/`)  
- `node.py` — `Node` (Pydantic) + `GraphNode` (adds `np.ndarray` embedding). Nodes are `concept|intervention`. Intervention nodes require `intervention_lifecycle` (1–6) and `intervention_maturity` (1–4); concept nodes require `concept_category` and forbid those fields.
- `edge.py` — `Edge` + `GraphEdge`. Fields: `type`, `source_node`, `target_node`, `description`, `edge_confidence` (1–5). Self-loops are rejected by validator.
- `paper_schema.py` — `PaperSchema` wraps `List[GraphNode]`, `List[GraphEdge]`, `List[Meta]`.
- `local_graph.py` — `LocalGraph` validates node/edge references, then adds `BAAI/bge-large-en-v1.5` embeddings (1024-dim) to nodes and edges via SentenceTransformers lazy-loaded on first use.

**4. FalkorDB ingestion** (`src/local_graph_extraction/db/ai_safety_graph.py`)  
`AISafetyGraph` upserts nodes as `:{Concept|Intervention}:NODE` and edges as `:EDGE {etype}`. Also creates `:Source` and `:Rationale` nodes. After ingestion, creates VECTOR indexes (1024-dim cosine) on `(n:NODE).embedding` and `[r:EDGE].embedding`. Problem papers are moved to `output_dir/issues/`.

### Key prompts

- `src/prompt/final_primary_prompt.py` (`PROMPT_EXTRACT`) — primary extraction prompt; instructs `o3` to trace logical chains, assign confidence/lifecycle/maturity.
- Multiple versioned variants exist (`_v2short`, `_updated6`, etc.); `final_primary_prompt.py` is the active one.

### Graph analysis

`graph_analysis/` contains standalone scripts for clustering, UMAP, pathway analysis, and report generation. These are research/analysis scripts, not part of the ingestion pipeline. They read from FalkorDB or the exported `graph.json`.
