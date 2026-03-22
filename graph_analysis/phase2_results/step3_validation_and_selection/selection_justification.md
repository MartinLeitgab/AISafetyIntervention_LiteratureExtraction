# Config Selection Justification

**Method:** 5-criteria weighted composite score applied to all 160 configurations.
**Weights:** EDGE validation 30%, Silhouette 25%, Cluster count 20%, ARI stability 15%, Gold purity (via interpretability proxy) 10%.
**Normalization:** Per node_type min-max (not global), ensuring fair comparison within each node category.

## Winner Per Node Type

| node_type | edge_config | mode | composite | rank | earmarked? |
|-----------|-------------|------|-----------|------|------------|
| all_concepts | 0.95 | monotonic | 0.833 | 1 | n/a |
| design_rationale | EDGE | monotonic | 0.852 | 1 | n/a |
| implementation_mechanism | 0.9 | both | 0.865 | 1 | n/a |
| intervention | 0.95 | unconstrained | 0.831 | 1 | UPDATED |
| problem_analysis | 0.95 | both | 0.836 | 1 | n/a |
| risk | EDGE | monotonic | 0.825 | 1 | UPDATED |
| theoretical_insight | EDGE | unconstrained | 0.825 | 1 | n/a |
| validation_evidence | EDGE | monotonic | 0.796 | 1 | n/a |

## Primary Analysis Cut Decision

The earmarked primary cut (SIM≥0.9, mode=both) is **UPDATED** for one or more node types.
Risk winner: edge_config=EDGE, mode=monotonic (composite=0.825)
Intervention winner: edge_config=0.95, mode=unconstrained (composite=0.831)

## Top-3 Per Node Type (risk and intervention)

| node_type | rank | edge_config | mode | composite | silhouette | edge_pct | ari_high |
|-----------|------|-------------|------|-----------|------------|----------|----------|
| risk | 1 | EDGE | monotonic | 0.825 | 0.508 | 1.000 | 0.739 |
| risk | 2 | 0.95 | monotonic | 0.822 | 0.546 | 0.850 | 0.739 |
| risk | 3 | 0.9 | both | 0.789 | 0.519 | 0.908 | 0.731 |
| intervention | 1 | 0.95 | unconstrained | 0.831 | 0.459 | 0.999 | 0.777 |
| intervention | 2 | 0.95 | both | 0.808 | 0.456 | 0.999 | 0.744 |
| intervention | 3 | EDGE | both | 0.795 | 0.450 | 1.000 | 0.744 |

## Workshop Paper Methods Text

Configuration selection was determined by a 5-criteria weighted composite score (EDGE validation 30%, silhouette 25%, cluster count 20%, ARI stability 15%, gold purity 10%) applied to all 160 configurations (5 edge thresholds × 4 modes × 8 node types). See `optimal_configs_ranked.csv` for full rankings and `multi_criteria_parallel.png` for visualization.