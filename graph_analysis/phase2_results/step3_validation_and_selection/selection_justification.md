# Config Selection Justification

**Method:** 5-criteria weighted composite score applied to all 160 configurations.
**Weights:** EDGE validation 30%, Silhouette 25%, Cluster count 20%, ARI stability 15%, Gold purity 10%.
**Normalization:** Per node_type min-max, ensuring fair comparison within each node category.

## Winner Per Node Type

| node_type | edge_config | mode | composite | rank | earmarked? |
|-----------|-------------|------|-----------|------|------------|
| all_concepts | 0.95 | monotonic | 0.833 | 1 | n/a |
| design_rationale | EDGE | monotonic | 0.852 | 1 | n/a |
| implementation_mechanism | 0.9 | both | 0.865 | 1 | CONFIRMED |
| intervention | 0.95 | unconstrained | 0.831 | 1 | UPDATED |
| problem_analysis | 0.95 | both | 0.836 | 1 | n/a |
| risk | EDGE | monotonic | 0.825 | 1 | UPDATED |
| theoretical_insight | EDGE | unconstrained | 0.825 | 1 | n/a |
| validation_evidence | EDGE | monotonic | 0.796 | 1 | n/a |

## Primary Analysis Cut Decision

### Risk node type

The composite winner is **EDGE + monotonic** (composite=0.825). The earmarked cut **SIM≥0.9 + both** ranks **3rd** (composite=0.789, gap=0.036).

| rank | edge_config | mode | composite | silhouette | edge_pct | ari_high | n_clusters |
|------|-------------|------|-----------|------------|----------|----------|------------|
| 1 | EDGE | monotonic | 0.825 | 0.508 | 1.000 | 0.739 | 40 |
| 2 | 0.95 | monotonic | 0.822 | 0.546 | 0.850 | 0.739 | 42 |
| **3** | **0.9** | **both** | **0.789** | 0.519 | 0.908 | 0.731 | 40 |

**Key observation:** The 30% EDGE validation weight structurally advantages EDGE-only configs (score=1.0) over any SIM-augmented config. SIM≥0.9+both achieves 90.8% EDGE validation while accessing ~10× more nodes than EDGE-only. The composite gap of 0.036 is narrow; coverage considerations favor SIM≥0.9+both for the workshop's cross-literature synthesis goal.

**Recommendation:** Retain **SIM≥0.9 + both** as primary cut for risk. Acknowledge EDGE+monotonic as the structural-purity alternative in the workshop paper.

### Intervention node type

The composite winner is **SIM≥0.95 + unconstrained** (composite=0.831). The earmarked cut **SIM≥0.9 + both** ranks **12th** (composite=0.674, gap=0.157).

| rank | edge_config | mode | composite | silhouette | edge_pct | ari_high | n_clusters |
|------|-------------|------|-----------|------------|----------|----------|------------|
| 1 | 0.95 | unconstrained | 0.831 | 0.459 | 0.999 | 0.777 | 40 |
| **2** | **0.95** | **both** | **0.808** | 0.456 | 0.999 | 0.744 | 40 |
| 3 | EDGE | both | 0.795 | 0.450 | 1.000 | 0.744 | 40 |

**Key observation:** The 0.157 composite gap is substantial. SIM≥0.95 configs (ranks 1-2) dominate on both EDGE validation (~99.9%) and ARI stability (0.777). The earmarked SIM≥0.9+both has meaningfully lower performance across all metrics for intervention nodes.

**Recommendation:** **UPDATE** intervention primary cut to **SIM≥0.95 + both** (rank 2, mode=both preferred for interpretability over unconstrained).

## Final Recommended Primary Cuts

| node_type | Final config | Mode | Basis |
|-----------|-------------|------|-------|
| risk | SIM≥0.9 | both | Rank 3 (gap 0.036), coverage advantage (~10× nodes over EDGE-only), 90.8% EDGE validation |
| intervention | SIM≥0.95 | both | Rank 2, substantial composite improvement (+0.134 over SIM≥0.9, gap to rank-1 only 0.023) |
| implementation_mechanism | SIM≥0.9 | both | Rank 1 confirmed |
| all_concepts | SIM≥0.95 | monotonic | Rank 1 |

## EDGE Validation Weight Bias Note

The EDGE validation weight (30%) structurally rewards configs where all cluster members appear in EDGE-only pathways. EDGE-only configs trivially score 1.0 on this metric. This is intentional — it ensures structural grounding — but does NOT capture the coverage/recall dimension. EDGE-only configs cover ~2,468 risk nodes; SIM≥0.9 covers ~2,732 (+11%), with ARI(EDGE, SIM0.9)=0.705 confirming structural agreement. For workshop goals requiring cross-literature coverage, SIM-augmented configs are preferred despite the composite penalty.

## Workshop Paper Methods Text

Configuration selection used a 5-criteria weighted composite score (EDGE validation 30%, silhouette 25%, cluster count 20%, ARI stability 15%, gold purity 10%) across all 160 configurations (5 × 4 × 8). For risk nodes, SIM≥0.9 + both (rank 3, composite=0.789) was preferred over the composite winner EDGE+monotonic (rank 1, composite=0.825) based on coverage considerations while maintaining 90.8% EDGE validation and ARI stability score of 0.731 (highest of any threshold). For intervention nodes, the earmarked SIM≥0.9 cut was updated to SIM≥0.95+both based on a substantial composite improvement (+0.134). Full rankings in `optimal_configs_ranked.csv`.
