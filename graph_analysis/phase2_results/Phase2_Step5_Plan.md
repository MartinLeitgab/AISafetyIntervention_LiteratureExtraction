# Phase 2 Step 5 Plan
## LLM Cluster Naming, Judge Review, and Workshop Pathway Examples

**Created:** 2026-03-29
**Branch:** martin/main
**Inputs:** Step 4 outputs (cluster PKLs, path files, connectivity CSVs, KMeans model)
**Goal:** Produce workshop-ready named cluster taxonomy and human-readable pathway examples for the most prevalent and least-covered areas of the ARD corpus.

---

## Status Entering Step 5

All Step 4 analyses complete ✅. All sampling artifacts removed ✅.

| Prerequisite | Status |
|---|---|
| risk_clusters.csv, intervention_clusters.csv (40 each) | ✅ Step 4 |
| optionA_chainbody_clusters.csv (40 chain clusters) | ✅ Step 4 |
| optionA_kmeans_model.pkl + optionA_cluster_labels.pkl | ✅ Step 4 |
| representative_pathways_consim2.jsonl (432,776 VarB paths) | ✅ Step 4 |
| representative_pathways_edgeonly.jsonl (3,473 EDGE-only) | ✅ Step 4 |
| risk_to_intervention_edges.csv (1,289 rows) | ✅ Step 4 |
| Algorithmic cluster names (centroid-top-1 only) | ✅ Step 4 — replaced by LLM names in Step 5 |

---

## Why LLM Naming (Not Algorithmic)

The Step 4 algorithmic names (top-1 centroid-sim node name) work as placeholders but have two weaknesses:
1. **Single-node bias**: the top-1 node name reflects one specimen, not the cluster's shared theme
2. **Near-duplicate crowding**: x-risk clusters (10, 21, 25, 26) all produce near-identical names from their near-duplicate hub nodes

LLM naming from top-10 representative nodes produces synthesized labels that capture the shared causal logic across the full cluster, not just the most central node. This is the established approach in BERTopic, TopicGPT, and related work (see Step 4 plan LLM-as-judge rationale section).

---

## Section A: Three-Pass LLM Cluster Naming

**Script:** `graph_analysis/phase2_step5_naming.py`
**Output dir:** `phase2_results/step5_naming/`

### Cluster scope: ~120 total
- 40 risk clusters (edge_config=0.9, mode=unconstrained, algo=agglomerative)
- 40 intervention clusters (same config)
- 40 Option A chain body clusters (path-body mean embedding KMeans k=40)

### Node selection per cluster
For each cluster, retrieve the **top-10 nodes by cosine similarity to cluster centroid** (from node_attrs embeddings). Include per node:
- `name`
- `description` (truncated to 150 chars)
- `concept_category` or `type` (as category label)

**Rationale for top-10 (not top-5):** Large heterogeneous clusters (100+ nodes) contain sub-themes that top-5 misses. Top-10 gives the LLM enough breadth to detect multi-theme clusters and flag them as split candidates. Cost is minimal (~120 extra tokens per call).

### Pass 1 — LLM Name Generation

**Model:** `gpt-4o-mini` (sufficient for labeling; cost-effective at 240 calls)

**Prompt structure:**
```
You are naming clusters in an AI safety knowledge graph.
Cluster type: {risk | intervention | chain}
Representative nodes (ordered by semantic centrality):
1. [category] node_name: description (truncated)
...10 nodes...

Task: Generate a concise cluster name and description.
- Name: 5-10 words, captures the shared theme across ALL nodes listed
- Description: 1 sentence (max 30 words), captures the causal logic or shared concept
- For risk clusters: what failure mode or danger does this cluster represent?
- For intervention clusters: what action or approach does this cluster represent?
- For chain clusters: what conceptual bridge (intermediate reasoning) connects risk to
  intervention? Frame as: "the body of thinking about [X] connecting [risk type] to
  [intervention type]"

Respond as JSON: {"name": "...", "description": "..."}
```

### Pass 2 — LLM-as-Judge Review

**Model:** `gpt-4o-mini`

For each cluster: pass the proposed name + description back alongside the same top-10 nodes. Judge evaluates:
1. Does the name accurately capture the shared theme across MOST nodes?
2. Do nodes span >2 clearly distinct themes? → **split candidate** flag
3. Is the name specific enough to distinguish from adjacent clusters?
4. Overall confidence: high / medium / low

If confidence is medium/low OR split_candidate=True → add to human review list.
If `suggested_revision` provided → use as `final_name` instead of Pass 1 output.

**Prompt structure:**
```
You are reviewing a cluster name for an AI safety knowledge graph.
Cluster type: {cluster_type}
Proposed name: "{name}"
Proposed description: "{description}"

Representative nodes:
1. [category] node_name: description
...

Review criteria:
1. Does the name accurately capture the shared theme? (accurate: true/false)
2. Do nodes clearly span >2 distinct themes? (split_candidate: true/false)
3. Is the name specific and distinctive? (confidence: high/medium/low)
4. If revisions needed, suggest an improved name (suggested_revision: "..." or null)

Respond as JSON: {"accurate": bool, "issues": "..." or null, "split_candidate": bool,
                  "confidence": "high/medium/low", "suggested_revision": "..." or null}
```

### Pass 3 — Human Review Checklist

Mandatory human review (regardless of judge confidence):
- All 40 risk clusters — confirm major families match workshop paper claims
- All clusters cited in main paper findings or figures

Auto-flagged for human review (from Pass 2):
- `judge_accurate = false`
- `judge_confidence = medium or low`
- `judge_split_candidate = true`
- `n_sources = 1` (single-paper extraction artifact risk)

**Output:** `step5_naming/human_review_checklist.csv` — one row per cluster needing review, with proposed name, judge issues, and top-5 node names for quick reference.

### Output Files

| File | Description |
|------|-------------|
| `step5_naming/risk_cluster_names_llm.csv` | 40 rows: cluster_id, final_name, description, judge_confidence, split_candidate |
| `step5_naming/intervention_cluster_names_llm.csv` | 40 rows, same schema |
| `step5_naming/chain_cluster_names_llm.csv` | 40 rows, same schema |
| `step5_naming/all_clusters_naming_detail.csv` | All 120 rows with full Pass 1 + Pass 2 outputs |
| `step5_naming/human_review_checklist.csv` | Mandatory + flagged clusters for human review |

---

## Section B: Prevalent Pathway Examples

**Script:** `graph_analysis/phase2_step5_examples.py`
**Output dir:** `phase2_results/step5_examples/`

### Goal
For the workshop paper: human-readable pathway chains showing how the AI safety literature most commonly links risks to interventions. Each example shows the full node-by-node chain with category labels and source paper URL.

### Example format (per pathway)
```json
{
  "risk_cluster": {"id": 10, "name": "Existential catastrophe from misaligned AI"},
  "chain_cluster": {"id": 2, "name": "Alignment progress lagging behind capability advances"},
  "intervention_cluster": {"id": 8, "name": "Fund and expand AI safety research teams"},
  "n_paths_in_combination": 4521,
  "examples": [
    {
      "type": "EDGE-only",
      "source_url": "...",
      "path_length": 6,
      "chain": [
        {"category": "risk", "name": "..."},
        {"category": "problem analysis", "name": "..."},
        {"category": "theoretical insight", "name": "..."},
        {"category": "implementation mechanism", "name": "..."},
        {"category": "validation evidence", "name": "..."},
        {"category": "intervention", "name": "..."}
      ]
    },
    ...up to 3 examples
  ]
}
```

### Selection criteria for examples per combination
Priority ordering:
1. **EDGE-only paths** (max_consec_SIM = 0) — strongest single-paper causal chain evidence
2. **Shortest paths** (path_length ≤ 5) — most readable for paper
3. **Diverse source URLs** — not more than 1 example from the same paper

**Coverage targets:**
- **Top 15 risk→intervention connections** by n_paths (from risk_to_intervention_edges.csv)
- For each: identify the dominant chain cluster (most paths through it), select 3 examples
- **Top 10 Option A chain clusters** by n_paths: 2 examples each
- **Top 10 Option B co-occurrence families** by n_paths: 2 examples each with decoded signature

### Implementation approach
1. Load KMeans model + cluster memberships → build `node_to_risk` and `node_to_interv` dicts
2. Stream `representative_pathways_consim2.jsonl` — for each path:
   - Look up risk cluster: `node_to_risk[path[0]]`
   - Look up intervention cluster: `node_to_interv[path[-1]]`
   - Predict chain cluster: `kmeans.predict(body_mean_emb)`
   - If EDGE-only candidate: also check `representative_pathways_edgeonly.jsonl`
3. Collect path metadata grouped by (risk_cid, chain_cid, interv_cid) triple
4. For top combinations: select best 3 examples per priority criteria above
5. After naming step: replace cluster IDs with LLM-generated names in output

**Note:** Path files already contain `node_names` and `categories` — no need to re-look up node_attrs for the pathway text itself. Descriptions can optionally be added for richer examples.

---

## Section C: Gap Pathway Examples

### Gap definition
Since all 6 structural gap types = 0 (every cluster has at least one cross-level connection), "gaps" for the workshop paper mean **thin coverage areas** — risk categories the literature discusses but with few documented solution pathways. These are scientifically meaningful: they identify where AI safety research has identified a problem but hasn't yet developed a rich intervention literature.

### Gap identification criteria

**Primary: path count thinness**
| Risk cluster | n_paths (total to any intervention) | n_nodes | Cluster name |
|---|---|---|---|
| 36 | 3 | 3 | Slow RL-based interferometer alignment |
| 13 | 8 | 4 | Unverified rationality in HCH-based alignment |
| 39 | 151 | 10 | Biased AI safety research trend assessment |
| 23 | 440 | 20 | Societal manipulation by engagement-maximizing algorithms |
| 29 | 493 | 15 | Extreme wealth concentration from advanced AI |

**Secondary: narrow intervention connectivity**
- Cluster 13: only 2 distinct intervention clusters connected
- Cluster 36: only 3 distinct intervention clusters connected
- These risks have a very narrow proposed solution space

**Tertiary: EDGE-only path absence**
- Risk clusters with zero EDGE-only paths — all connections are via similarity bridges, no single-paper complete causal chain

### Output per gap cluster
For each of the top-10 gap clusters:
- All available pathways (may be as few as 3 for cluster 36)
- Annotation: n_paths total, n_distinct_interventions, has_edge_only_path (true/false)
- Interpretation note: "Low path count indicates limited intervention literature for this risk category in the ARD corpus. May reflect corpus coverage limits, not genuine research absence."

---

## Execution Order

1. Run `phase2_step5_naming.py` — ~10 min (120 clusters × 2 API calls, gpt-4o-mini)
2. Run `phase2_step5_examples.py` — ~15 min (streaming 432K paths + KMeans predict)
3. Human review of `step5_naming/human_review_checklist.csv`
4. Update any revised names in the naming CSVs

---

## Output Summary

| File | Description |
|------|-------------|
| `step5_naming/risk_cluster_names_llm.csv` | LLM-synthesized names for 40 risk clusters |
| `step5_naming/intervention_cluster_names_llm.csv` | LLM-synthesized names for 40 intervention clusters |
| `step5_naming/chain_cluster_names_llm.csv` | LLM-synthesized names for 40 chain clusters |
| `step5_naming/all_clusters_naming_detail.csv` | Full Pass 1 + Pass 2 detail for all 120 clusters |
| `step5_naming/human_review_checklist.csv` | Clusters flagged for human review (mandatory + auto-flagged) |
| `step5_examples/pathway_examples_prevalent.json` | Top-15 connections + top-10 chain/Option-B examples with 3 paths each |
| `step5_examples/pathway_examples_gaps.json` | Top-10 thin-coverage risk clusters with all available pathway examples |
| `step5_examples/pathway_examples_edgeonly.json` | Best EDGE-only examples across top combinations (single-paper ground truth) |

---

## Paper Disclosure Text

*"Cluster names were generated by GPT-4o-mini from the top-10 most representative node names and descriptions per cluster (centroid-similarity ranked). A second GPT-4o-mini pass reviewed each name for accuracy and flagged split candidates. All 40 risk clusters and all clusters cited in main findings were reviewed manually. Cluster quality was validated by silhouette score, EDGE purity, ARI stability, and path-filtered betweenness centrality — not by LLM judgment."*
