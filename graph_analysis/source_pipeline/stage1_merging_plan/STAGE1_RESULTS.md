# Stage 1 Deduplication & Edge Normalization

## Summary

Testing optimal thresholds for graph deduplication using cosine similarity (embeddings) and Jaccard overlap (names/aliases).

**Input:** 202,446 nodes (165,291 concepts + 37,155 interventions), 202,207 edges from 11,787 papers

**Problems addressed:**
1. Node duplication: same concepts/interventions extracted multiple times with different phrasings
2. Edge type inconsistencies: bidirectional pairs (`mitigated_by` vs `mitigates`), naming variants (`design_rationale` vs `design rationale`)

## Experiments

| Experiment | Cosine | Jaccard | Candidates | Dedup % | Max Group | Status |
|------------|--------|---------|------------|---------|-----------|--------|
| exp3_relaxed | 0.65 | 0.3 | 416 | 0.2% | 9 nodes | ✅ Too conservative |
| exp_no_jaccard | 0.85 | 0.0 | 105,390 | 11.9% | 6,611 nodes | ✅ Too aggressive |
| exp_0.88_0.05 | 0.88 | 0.05 | 4,411 | 1.2% | 695 nodes | ✅ **OPTIMAL** |
| exp_0.95_0.0 | 0.95 | 0.0 | 503 | 0.2% | 63 nodes | ✅ Too conservative |

## Key Findings

### exp_no_jaccard (0.85 + 0.0)
- **Problem:** Without Jaccard filter, created massive false clusters
- 6,611-node group merged different concepts: "superintelligence emergence" + "AI without governance"
- Median Jaccard: 0.0 (most pairs share NO words)
- **Conclusion:** Name overlap check is essential

### exp_0.88_0.05 (0.88 + 0.05) ⭐ CHOOSEN
- **Results:** 2,385 nodes removed (1.2%), reasonable deduplication rate
- 4,411 candidates (including interventions with different lifecycle/maturity)
- Max group: 695 nodes (all "AI existential risk" variants - legitimate)
- All 10 boundary examples are valid duplicates:
  - "US semiconductor export controls" ↔ "AI chip export control regulations"
  - "Misalignment of AI objectives" ↔ "objective misalignment in smarter-than-human AI"
  - 8/10 examples: different phrasings of "AI existential risk"
- Median Jaccard: 0.20 (practical name overlap requirement)

**Jaccard 0.05 = ~1 shared word in 6-7 unique words (minimum protection against false merges)**

### exp_0.95_0.0 (0.95 + 0.0)
- **Results:** 419 nodes removed (0.2%), very conservative - same as 0.65+0.3
- 503 candidates (8.6x fewer than 0.88+0.05)
- Max group: 63 nodes
- All 10 boundary examples are perfect semantic matches with Jaccard=0.0:
  - "Existential catastrophe from misaligned/uncontrolled AGI" ↔ "Existential catastrophe from poorly aligned AGI"
  - "Insufficient AI safety research capacity" ↔ "Insufficient AI alignment research capacity"
  - "university curricula lacking AI safety coursework" ↔ "Lack of dedicated university AI safety curricula"
- Median Jaccard: 0.0 (high cosine alone catches only near-identical texts)
- **Conclusion:** Very high cosine threshold without Jaccard misses many legitimate duplicates with different wording

## Merging Algorithm

### 1. Candidate Search
- **Blocking:** Group nodes by type and category to reduce comparisons
  - Concepts: by `(type, concept_category)`
  - Interventions: by `(type,)` only - enables merging across different lifecycle/maturity stages
- **FAISS IndexIVFFlat:** Fast similarity search using normalized embeddings (METRIC_INNER_PRODUCT = cosine similarity)
  - 1000 clusters for quantization, 50 nearest neighbors per query
  - Only pairs above **cosine threshold** considered as candidates
- **Jaccard Filter:** Secondary filter - calculate name/alias overlap, reject candidates below **Jaccard threshold**
- **Result:** Pairs passing both thresholds become merge candidates

### 2. Node Merging
- **Union-Find:** Group candidates into merge clusters (transitive closure)
- **Canonical Selection:** Choose node with longest description, most aliases, or lexicographically first URL
- **Attribute Merging:**
  - `name`: from canonical node
  - `aliases`: union of all names + aliases (excluding canonical name)
  - `description`, `node_rationale`: longest from group
  - `embedding`: mean of all embeddings
  - **Temporal evolution (interventions):** store `lifecycle_history`/`maturity_history` as arrays with `{value, source_urls, count}`
  - `merge_count`: number of merged nodes

### 3. Edge Updates
- **Remapping:** Update source/target to canonical node IDs
- **Self-loop Removal:** Delete edges where source == target after merging
- **Type Normalization:** Apply `edge_type_mapping.json` (reverse bidirectional pairs, rename variants)
- **Deduplication:** Merge edges with same `(source, type, target, confidence)` tuple
  - Keep longest description and rationale
  - Average embeddings
  - Track `merge_count`

## Statistics (exp_0.88_0.05)

**Cosine Similarity:** Min=0.88, Max=0.98, Mean=0.91, Median=0.91  
**Jaccard Overlap:** Min=0.17, Max=0.50, Mean=0.21, Median=0.20

**Merge Distribution:**
- 199,223 groups: 1 node (no merge)
- 631 groups: 2 nodes
- 90 groups: 3 nodes
- 1 group: 695 nodes (all "AI existential risk")

**Output:** 200,061 nodes, 197,542 edges

## Edge Type Normalization

**Resolved inconsistencies from Stage 0:**
- Bidirectional pairs unified: `mitigates` → `mitigated_by`, `implements` → `implemented_by`
- Naming variants standardized: `design_rationale`/`design rationale`/`design_rationale_for` → single form
- Low-frequency duplicates merged with main types

**Final edge types (12):** motivates, implemented_by, validated_by, caused_by, mitigated_by, enabled_by, addressed_by, refined_by, specified_by, required_by, preceded_by, related_to
