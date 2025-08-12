## AISafety Intervention – Local Graph Extraction (Story 3)

### What this does
- **Goal**: Extract concept-intervention graphs from markdown papers, render quick PNGs, and store structured graphs in **FalkorDB**.
- **Pipeline**:
  1. Read papers from `development_paper_set/md/`
  2. Build prompts using `src/prompts/prompt.py`
  3. Run OpenAI batch inference to produce a structured `ConceptGraph` (Pydantic model)
  4. Store nodes/edges in FalkorDB graph `concept_graphs`
  5. Render a PNG per paper to `development_paper_set/graphs/`
  6. Save OpenAI batch manifests in `openai_batch_manifests/`

### Prerequisites
- **Python**: 3.10+ recommended
- **Docker**: for FalkorDB
- **OpenAI API key**: set via environment variable

### Quickstart
1) Install Python deps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt
```

2) Run FalkorDB via Docker (Redis API on 6379, UI on 3000)

```bash
docker run --rm --name falkordb -p 6379:6379 -p 3000:3000 falkordb/falkordb
```

3) Set environment variables (create a `.env` or export in shell)

```bash
# required
export OPENAI_API_KEY="sk-..."

# optional (defaults shown)
export FALKORDB_HOST="localhost"
export FALKORDB_PORT="6379"
export FALKORDB_PASSWORD=""  # set if your FalkorDB requires auth
```

4) Put input markdown files under `development_paper_set/md/` (sample set included)

5) Run the extractor

```bash
# From repo root
python -m src.create_local_graphs
# or
python src/create_local_graphs.py
```

Notes:
- By default, `src/create_local_graphs.py` uses a hardcoded `md_dir` in `main()`. Update it to your absolute path to `development_paper_set/md/` if needed, and adjust `max_files`.
- Output PNGs go to `development_paper_set/graphs/`. Graph data is stored in FalkorDB graph `concept_graphs`.
- Batch manifests are written to `openai_batch_manifests/` for resuming/debugging.

### Directory overview
- `development_paper_set/md/`: input markdown papers
- `development_paper_set/graphs/`: per-paper graph PNGs (auto-created)
- `inputdata_development_paper_set/directory.txt`: curated paper list
- `openai_batch_manifests/`: OpenAI batch job manifests
- `src/create_local_graphs.py`: entrypoint for Story 3 local graph extraction
- `src/models/`: Pydantic models for `ConceptGraph`, nodes, edges
- `src/prompts/prompt.py`: prompt used to drive structured extraction
- `src/model_provider/openai_provider.py`: OpenAI batch integration and parsing
- `src/db/falkordb_client.py`: minimal FalkorDB client utilities
- `src/db/concept_graph_store.py`: mapping from `ConceptGraph` to FalkorDB nodes/edges
- `src/utils.py`: NetworkX conversion and PNG rendering helpers

### FalkorDB
- The extractor selects/creates a graph named `concept_graphs` and writes:
  - **Nodes**: labeled `Concept` or `Intervention` with `paper_id` and other properties
  - **Relationships**: types from `RelationshipType` with `paper_id`
- Web UI: `http://localhost:3000` (when using the Docker command above)

### Troubleshooting
- **Cannot connect to FalkorDB**: ensure Docker container is running and `FALKORDB_HOST/FALKORDB_PORT` are correct.
- **OpenAI batch latency**: batch jobs poll until completion; manifests are saved in `openai_batch_manifests/`.
- **PNG rendering**: ensure `matplotlib` and `networkx` are installed (included in `src/requirements.txt`).


