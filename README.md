# AISafetyIntervention_LiteratureExtraction

This repository contains all outcomes created in the 2025 Eleuther AI Summer of Open AI Research program for project #5, Scientific Literature Knowledge Extraction Tool.

## Development Setup

### Prerequisites

- [UV](https://github.com/astral-sh/uv) (Python package installer and resolver)
- [pre-commit](https://pre-commit.com/) (Git hooks manager)

### Installation

1. Create and activate a virtual environment using UV:

   ```bash
   uv venv
   source .venv/bin/activate  # On Unix/macOS
   # or
   .\.venv\Scripts\activate  # On Windows
   ```

2. Install development dependencies:

   ```bash
   uv sync
   ```

3. Install pre-commit hooks:

   ```bash
   uv run pre-commit install
   ```

### Development Workflow

Pre-commit hooks are configured to run on each commit to ensure code quality.

To manually run all pre-commit checks:

```bash
uv run pre-commit run --all-files
```

### FalkorDB

You can put extracted files into FalkorDB to visualize. Start an instance with 

```bash
docker run -p 6379:6379 -p 3000:3000 -it --rm falkordb/falkordb:latest
```

then fill it out with

`uv run src/db.py` and merge with `uv run src/merge.py`

#### Loading the full ARD dump.rdb

To load the full ARD wide-embeddings dump (per issue
[#132](https://github.com/MartinLeitgab/AISafetyIntervention_LiteratureExtraction/issues/132))
for graph analysis (`graph_analysis/` scripts), use the `:edge` image and mount
the dump directory at `/var/lib/falkordb/data`:

```bash
# Foreground (-it) — required. ~15 min for indices to build before DB is ready.
docker run -p 6379:6379 -p 3000:3000 -it --rm \
  --volume ./data:/var/lib/falkordb/data \
  falkordb/falkordb falkordb:edge
```

Notes:
- **Use `-it` (foreground TTY), not `-d` (detached).** Detached mode silently
  exits with code 255 during index construction with no logs — the container's
  child processes die silently when there's no TTY. Run in a dedicated terminal
  and leave open.
- Mount target MUST be `/var/lib/falkordb/data` (the `:edge` image's data
  directory). The earlier `:latest` command in this section is for an empty
  in-container graph and does not work for loading external dumps.
- The dump file must be named `dump.rdb` and placed in the mounted directory.
- On Windows, run the command inside a WSL shell directly (Docker is typically
  only available inside WSL on Windows).
- The container will print
  `<module> Graph 'AISafetyIntervention' processing virtual key: N/23` lines
  for each of 23 virtual keys, then `Constructing indexes.` After ~15 min
  indices finish and the DB is ready to accept queries on port 6379.

and you can see the result at <http://localhost:3000/graph>

## Possible follow up projects (given a Knowledge Graph)

- Automated Discovery of Emerging Research Fronts and "Invisible Colleges": identify clusters of papers that are conceptually related, even if they don't cite each other directly
- Knowledge Gap Detection and Research Opportunity Mapping: identify weakly connected or isolated concepts, then prioritize them as potential research frontiers
- Consensus / Controversy Atlas (claim-level meta-analysis): aggregate by concept and time to generate consensus scores and timelines

Also other projects that should already be possible with a citation graph:

- Cross-Disciplinary Concept Bridging Tool: take the most influential concepts in one subgraph and combine them with concepts from another subgraph (transformers in protein folding)
- Concept Evolution and Trend Forecasting
