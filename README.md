# AISafetyIntervention_LiteratureExtraction

This repository contains all outcomes created in the 2025 Eleuther AI Summer of Open AI Research program for project #5, Scientific Literature Knowledge Extraction Tool.

## Development Setup

### Prerequisites

- [UV](https://github.com/astral-sh/uv) (Python package installer and resolver)
- [pre-commit](https://pre-commit.com/) (Git hooks manager)
- docker (to start local FalkorDB instance) 

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

and you can see the result at <http://localhost:3000/graph>

## Possible follow up projects (given a Knowledge Graph)

- Automated Discovery of Emerging Research Fronts and "Invisible Colleges": identify clusters of papers that are conceptually related, even if they don't cite each other directly
- Knowledge Gap Detection and Research Opportunity Mapping: identify weakly connected or isolated concepts, then prioritize them as potential research frontiers
- Consensus / Controversy Atlas (claim-level meta-analysis): aggregate by concept and time to generate consensus scores and timelines

Also other projects that should already be possible with a citation graph:

- Cross-Disciplinary Concept Bridging Tool: take the most influential concepts in one subgraph and combine them with concepts from another subgraph (transformers in protein folding)
- Concept Evolution and Trend Forecasting
