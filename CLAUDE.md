# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project extracts structured knowledge graphs from AI safety literature. PDFs are processed via OpenAI's API to extract `concept` and `intervention` nodes connected by logical chain edges. The resulting graph is stored in FalkorDB and duplicate nodes are merged via embedding similarity.

## Commands

```bash
# Install dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install

# Lint and format (ruff)
uv run pre-commit run --all-files

# Run extraction on PDFs in intervention_graph_creation/data/raw/pdfs_local/
uv run intervention_graph_creation/src/local_graph_extraction/extract.py

# Start FalkorDB (requires Docker)
docker run -p 6379:6379 -p 3000:3000 -it --rm falkordb/falkordb:latest

# Ingest extracted JSONs into FalkorDB
uv run intervention_graph_creation/src/local_graph_extraction/db.py

# Merge duplicate nodes via embedding similarity
uv run intervention_graph_creation/src/local_graph_extraction/merge.py

# View graph at http://localhost:3000/graph
```

## Architecture

The pipeline has three stages:

### 1. Extraction (`src/local_graph_extraction/extract.py`)
Uploads PDFs to OpenAI Files API and calls a reasoning model (`gpt-5`) with `PROMPT_EXTRACT`. The model returns a structured JSON validated against `PaperSchema` (Pydantic). Each paper produces two output files:
- `{paper_stem}.json` — parsed structured output
- `{paper_stem}_raw_response.json` — full API response

Input: `intervention_graph_creation/data/raw/pdfs_local/`
Output: `intervention_graph_creation/data/processed/`

### 2. Graph Schemas
Two separate schemas exist for different pipeline stages:

- **`src/local_graph_extraction/core.py`** — `PaperSchema` used by the new extraction pipeline. Nodes are typed `concept|intervention` with attributes like `intervention_lifecycle` (1–6), `intervention_maturity` (1–4), and `concept_category`. Edges live inside `LogicalChain` objects.

- **`src/local_graph_extraction/prompts.py`** — `OutputSchema` used by the older FalkorDB ingestion pipeline. Uses flat `NODE_TYPES`/`EDGE_TYPES` enumerations with `confidence` floats and `canonical_name` normalization.

### 3. FalkorDB Ingestion & Merging (`db.py`, `merge.py`)
`db.py` upserts papers, nodes, and edges into FalkorDB. `merge.py` creates OpenAI batch embeddings for all graph nodes, uses USearch vector index to find near-duplicate pairs by cosine similarity, and merges them via `AISafetyGraph.merge_nodes()`.

## Environment

Requires `OPENAI_API_KEY` set in `.env` (see `.env.example`).

## Key Prompts

- **`src/prompt/final_primary_prompt.py`** (`PROMPT_EXTRACT`) — Primary extraction prompt. Instructs the model to trace logical chains from problems → concepts → interventions, assign confidence (1–5), lifecycle (1–6), and maturity (1–4) scores.
- **`src/local_graph_extraction/prompts.py`** (`EXTRACTION_PROMPT_TEMPLATE`) — Older prompt used with the flat `OutputSchema`. Defines node/edge ontology and normalization rules.
- `PROMPT_RESPONSE_EVAL` in `final_primary_prompt.py` — Rubric for evaluating extraction quality.
