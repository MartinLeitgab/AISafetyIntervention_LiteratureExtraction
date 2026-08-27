# Source pipeline: the January 2026 analysis run

The scripts and receipts of an earlier analysis pass over the AI safety knowledge graph.
It ran in four stages: assemble the per-paper graphs into one global graph, merge
near-duplicate nodes and normalise edge types, cluster the node embeddings into topical
groups, then add a cross-document similarity layer and compute traversal metrics over it.

This is a **separate run on a different graph** from the one in `graph_analysis/`. It starts
from 202,446 nodes and produces a merged 200,061-node graph; the released substrate is
un-merged and holds 200,525. Counts from the two are not interchangeable.

## What each stage did

### Stage 0 — assemble the global graph

Reads the per-document extractions (11,790 document directories, one JSON graph each) and
concatenates them into a single graph, attaching the stored 1536-dimensional embedding to
every node that has one.

`stage0_collect_local_graphs.py` · `analyze_edge_types.py` surveys the raw edge vocabulary
before normalisation.

### Stage 1 — near-duplicate merge and edge normalisation

Merges nodes that name the same concept in different words, then collapses the raw edge
vocabulary to a canonical set.

Candidates require **both** cosine similarity >= 0.88 on embeddings **and** Jaccard overlap
>= 0.05 over the node's name and aliases. Search is blocked by node type and concept
category, uses a FAISS `IndexIVFFlat` capped at 50 neighbours per node, and groups the
surviving pairs transitively with Union-Find. The canonical node of each group is the one
with the longest description; its embedding is the mean of the group's.

The deployed configuration produced **4,411 candidate pairs**, which Union-Find grouped into
**2,385 removals (1.2%)**, the largest group holding 695 nodes. Edge normalisation unified
bidirectional pairs and naming variants, removing **4,665 edges (2.3%)** and leaving twelve
canonical types.

Three further threshold configurations were run for comparison and are in
`results/full/`: 0.65/0.30 (416 pairs), 0.85 with no Jaccard filter (105,390 pairs, 11.9%
removal), and 0.95 with no Jaccard filter (503 pairs).

`stage1_merge_graph.py` · logs in `results/full/` · summary in `STAGE1_RESULTS.md`

### Stage 2.1 — semantic clustering

Clusters node embeddings into navigable topical groups. Clustering in the original
1536-dimensional space finds no structure (silhouette ~0.02), so the deployed run reduces to
150 dimensions with UMAP (cosine metric, `n_neighbors=50`, `min_dist=0.0`) and runs K-means
at k=40 over the reduced coordinates.

Of the 200,061 nodes, **198,790 carry an embedding** and were clustered; 1,271 were skipped.
The 40 clusters range from **2,065 to 8,098 nodes**. Quality on the reduced coordinates:
silhouette **0.2983**, Calinski-Harabasz **55,410**, Davies-Bouldin **1.15**.

`stage2_semantic_clustering_umap.py` (deployed) · `stage2_semantic_clustering_full.py`
(1536-D comparison) · `validate_clustering.py` (per-cluster quality) ·
`compute_cluster_metrics.py` (cluster-level network metrics) · `extract_representatives.py`
(twenty representative nodes per cluster, ranked by distance to the 1536-D centroid)

### Stage 2.2 — similarity layer and traversal metrics

The concatenated graph is fragmented by construction: every structural edge comes from a
single paper, so before anything links them the graph is 18,424 components with the largest
holding 55 nodes. This stage adds cosine-similarity edges **within** a concept category at
tau >= 0.80 to connect content across documents, then walks the augmented graph.

**169,083 similarity edges** were added, bringing the graph to **366,609 edges** and
collapsing it to 6,522 components with the largest holding 142,772 nodes (71.4%).
Risk-to-risk similarity edges are deliberately excluded, so that clusters of semantically
similar risks cannot amplify each other in the traversal metrics.

Traversal is breadth-first and hybrid: the first hop from any start node must follow a
structural edge, preserving the causal structure the source paper asserted, after which up to
two similarity hops are allowed.

`1_add_similarity_edges.py` · `2_compute_metrics.py` · `sensitivity_analysis.py` and
`tau_sensitivity_fast.py` (parameter sweeps) · `race_framing_validation.py` and
`race_framing_recall_sample.py` (keyword classifier and its annotated samples)

## The graph this run produced

| | |
|---|---|
| Nodes before merge | 202,446 |
| Nodes after merge | 200,061 |
| Nodes carrying an embedding | 198,790 |
| Structural edges after normalisation | 197,542 |
| Similarity edges at tau >= 0.80 | 169,083 |
| Total edges | 366,609 |
| Risk nodes | 17,903 |
| Canonical edge types | 12 |
| Topical clusters | 40 |

Node composition among those carrying an embedding: interventions 36,928, implementation
mechanism 34,244, validation evidence 28,717, design rationale 27,762, problem analysis
27,317, theoretical insight 26,888, risk 16,934.

Canonical edge types by frequency: `motivates` 40,014, `implemented_by` 38,785,
`validated_by` 35,659, `caused_by` 31,759, `mitigated_by` 27,760, `enabled_by` 10,245,
`addressed_by` 5,577, `refined_by` 4,681, `specified_by` 1,575, `required_by` 1,206,
`preceded_by` 216, `related_to` 65.

Similarity edges by category: interventions 57,633, problem analysis 49,574, implementation
mechanism 21,271, design rationale 20,272, theoretical insight 10,653, validation evidence
9,680.

## Layout

```
stage0_graph_processing/     assembly of the global graph
stage1_merging_plan/         near-duplicate merge and edge normalisation
  results/full/              one log per threshold configuration
  STAGE1_RESULTS.md          comparison of the four configurations
stage2_clustering/
  stage2.1/                  UMAP reduction and K-means clustering
    results/                 cluster labels, representatives, quality metrics
  stage2.2/                  similarity layer, traversal metrics, race classifier
    results/                 metric outputs
    *_annotated.csv          the annotated classifier samples
```

## Data

The intermediate graphs each run to several gigabytes and are not committed: the assembled
graph, the four merged variants, the clustered graph and the metric-bearing graph come to
roughly 68 GB together. They exist and are kept locally — ask and they can be uploaded
wherever the release ends up. Everything here is reproducible from them by running the
stages in order.
