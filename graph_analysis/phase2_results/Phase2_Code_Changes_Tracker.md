# Phase 2 Code Changes & Improvements Tracker

**Document Version:** 1.0  
**Last Updated:** February 2026  
**Status:** Active

---

## CHANGE #1: ARI Cross-Threshold Stability Plot Redesign

**Priority:** HIGH  
**Affects:** Step 2, Substep #7  
**Current Status:** ❌ Not Implemented

### Current Implementation Issues

**File:** `phase2_step2_metrics_stability.py`  
**Current plot:** Symmetric 2D heatmap showing redundant data
- Upper and lower triangles are mirror images (ARI is symmetric)
- Diagonal always 1.0 (self-comparison)
- Difficult to compare modes (requires separate plots per mode)
- Threshold ordering suboptimal (EDGE first, should be last)

### Proposed Changes

**New visualization:** 1D line plot with multiple traces

**Design specifications:**
```python
# X-axis: Threshold pair labels (ordered for continuity)
threshold_pairs = [
    '0.8→0.85',
    '0.85→0.9', 
    '0.9→0.95',
    '0.95→EDGE',
    '0.8→0.9',      # Skip-1 pairs
    '0.85→0.95',
    '0.9→EDGE',
    '0.8→0.95',     # Skip-2 pairs
    '0.85→EDGE',
    '0.8→EDGE'      # Max distance
]

# Y-axis: ARI value (0 to 1.0)
# Add horizontal line at 0.7 (target threshold)

# Multiple traces:
# - One line per mode (4 modes × different colors/markers)
# - One subplot per node type (8 node types in grid)
# OR
# - One line per node type (8 node types × different colors)
# - One subplot per mode (4 modes in 2×2 grid)
```

### Implementation Code

**Function location:** `phase2_step2_metrics_stability.py`

**Replace function:**
```python
def plot_cross_threshold_stability(df_ari: pd.DataFrame):
    """
    Plot 5: Cross-threshold stability (ARI)
    UPDATED: 1D line plot showing ARI degradation across threshold distances
    """
    print("\n" + "="*80)
    print("GENERATING PLOT 5: Cross-Threshold Stability (ARI)")
    print("="*80)
    
    # Define threshold pairs in order (adjacent → distant)
    threshold_order = ['EDGE', '0.95', '0.9', '0.85', '0.8']
    
    # Generate all unique pairs (lower triangle only)
    pairs = []
    for i in range(len(threshold_order)):
        for j in range(i+1, len(threshold_order)):
            t1, t2 = threshold_order[j], threshold_order[i]  # Reverse for ascending
            pairs.append({
                'label': f'{t1}→{t2}',
                't1': t1,
                't2': t2,
                'distance': j - i  # 1 = adjacent, 2 = skip-1, etc.
            })
    
    # Sort by distance (adjacent first)
    pairs_sorted = sorted(pairs, key=lambda x: (x['distance'], x['label']))
    pair_labels = [p['label'] for p in pairs_sorted]
    
    # Get unique node types and modes
    node_types = df_ari['node_type'].unique()
    modes = ['unconstrained', 'single_risk', 'monotonic', 'both']
    
    # OPTION A: One subplot per node type, multiple mode lines
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = {'unconstrained': 'blue', 'single_risk': 'green', 
              'monotonic': 'orange', 'both': 'red'}
    markers = {'unconstrained': 'o', 'single_risk': 's', 
               'monotonic': '^', 'both': 'd'}
    
    for idx, node_type in enumerate(node_types):
        ax = axes[idx]
        
        for mode in modes:
            # Get ARI data for this node_type/mode
            subset = df_ari[(df_ari['node_type'] == node_type) & 
                           (df_ari['mode'] == mode)]
            
            if len(subset) == 0:
                continue
            
            # Extract ARI values for each pair
            # NOTE: Need to reconstruct pairs from CSV data
            # CSV has: node_type, mode, ari_mean, ari_median, ari_min, ari_max
            # This is AGGREGATE across all pairs - need pairwise data!
            
            # ISSUE: Current CSV only has summary statistics
            # Need to modify data collection to store pairwise ARI values
            
            # Placeholder for now - using mean as proxy
            ari_values = [subset['ari_mean'].values[0]] * len(pair_labels)
            
            ax.plot(range(len(pair_labels)), ari_values, 
                   marker=markers[mode], color=colors[mode], 
                   label=mode, linewidth=2, markersize=6)
        
        # Formatting
        ax.axhline(0.7, color='red', linestyle='--', linewidth=1.5, 
                  alpha=0.7, label='Target (0.7)')
        ax.set_xticks(range(len(pair_labels)))
        ax.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('ARI', fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.set_title(f'{node_type.replace("_", " ").title()}', 
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    
    plt.tight_layout()
    plt.savefig(PLOT_CROSS_THRESHOLD_ARI, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {PLOT_CROSS_THRESHOLD_ARI}")
```

### Data Collection Changes Required

**Current data structure (stability_ari_matrix.csv):**
```csv
node_type,mode,ari_mean,ari_median,ari_min,ari_max,ari_std,n_comparisons
risk,unconstrained,0.597,0.598,0.458,0.719,0.094,10
```

**Required data structure (stability_ari_pairwise.csv):**
```csv
node_type,mode,threshold_1,threshold_2,ari
risk,unconstrained,EDGE,0.8,0.458
risk,unconstrained,EDGE,0.85,0.512
risk,unconstrained,EDGE,0.9,0.623
risk,unconstrained,EDGE,0.95,0.719
risk,unconstrained,0.8,0.85,0.598
...
```

**Function to modify:** Step 2 data loading/ARI calculation section

```python
def calculate_cross_threshold_ari(cluster_data: Dict) -> pd.DataFrame:
    """
    Calculate pairwise ARI between all threshold configurations
    Returns: DataFrame with pairwise ARI values (not just summary stats)
    """
    from sklearn.metrics import adjusted_rand_score
    
    results = []
    thresholds = ['EDGE', '0.8', '0.85', '0.9', '0.95']
    
    for node_type in NODE_TYPES:
        for mode in MODES:
            # Get cluster assignments for all thresholds
            cluster_assignments = {}
            for thresh in thresholds:
                key = f"{thresh}_{mode}_{node_type}"
                if key in cluster_data:
                    cluster_assignments[thresh] = cluster_data[key]['labels']
            
            # Calculate pairwise ARI for all threshold pairs
            for i, t1 in enumerate(thresholds):
                for j, t2 in enumerate(thresholds):
                    if i >= j:  # Skip diagonal and upper triangle
                        continue
                    
                    if t1 not in cluster_assignments or t2 not in cluster_assignments:
                        continue
                    
                    # Must align nodes (use intersection of node IDs)
                    labels1 = cluster_assignments[t1]
                    labels2 = cluster_assignments[t2]
                    
                    # Calculate ARI
                    ari = adjusted_rand_score(labels1, labels2)
                    
                    results.append({
                        'node_type': node_type,
                        'mode': mode,
                        'threshold_1': t1,
                        'threshold_2': t2,
                        'ari': ari
                    })
    
    return pd.DataFrame(results)
```

### Testing Requirements

1. Verify pairwise ARI calculation produces same summary stats as current implementation
2. Check that all 10 pairs per node_type/mode are present
3. Validate threshold ordering produces intuitive plot (gradual degradation visible)
4. Confirm multiple modes are distinguishable (different colors/markers)

### Expected Output

**Visual improvements:**
- Clear view of ARI degradation pattern (adjacent pairs → distant pairs)
- Easy mode comparison within each node type
- No redundant data (only lower triangle shown)
- Target line (0.7) for reference

**Insight improvements:**
- Immediately see which threshold pairs meet target
- Identify high-stability clusters (0.9-0.95-EDGE)
- Quantify degradation rate across threshold distance

---

## CHANGE #2: Source Diversity Data Generation Fix

**Priority:** CRITICAL (blocking Substep #6)  
**Affects:** Step 2, Substep #6  
**Current Status:** ❌ Data shows all zeros

### Problem Description

`cluster_source_diversity.csv` shows all `n_sources = 0` and `nodes_with_sources = 0`

### Root Cause Analysis Required

Check in Step 1 checkpoint generation:
1. Do graph nodes have source attributes?
   - Expected: `source_file`, `source_file_list`, or `first_published_source`
2. Are attributes populated or None/null?
3. Is analysis script looking for correct attribute names?

### Investigation Steps

```python
# In Step 1 checkpoint script - add diagnostic
def diagnose_source_attributes(graph):
    """Check what source information is available"""
    sample_nodes = graph.query("MATCH (n:NODE) RETURN n LIMIT 100")
    
    attribute_counts = {
        'source_file': 0,
        'source_file_list': 0,
        'first_published_source': 0,
        'source_paper': 0,
        'none_or_unknown': 0
    }
    
    for node in sample_nodes:
        # Check each possible attribute
        for attr in ['source_file', 'source_file_list', 
                     'first_published_source', 'source_paper']:
            if hasattr(node, attr) and getattr(node, attr) not in [None, 'unknown', '']:
                attribute_counts[attr] += 1
        
        # Count nodes with no source info
        if all(getattr(node, attr, None) in [None, 'unknown', ''] 
               for attr in ['source_file', 'source_file_list']):
            attribute_counts['none_or_unknown'] += 1
    
    print("Source Attribute Availability (n=100 sample):")
    for attr, count in attribute_counts.items():
        print(f"  {attr}: {count}")
```

### Fix Implementation (Pending Diagnosis)

Will update once root cause identified.

---

## CHANGE #3: Hub Quality Metrics Generation

**Priority:** CRITICAL (blocking Substep #14)  
**Affects:** Step 2, Substep #14  
**Current Status:** ❌ **NOT IMPLEMENTED - CODE COMPLETELY MISSING**

### Root Cause

**File:** `phase2_step2_metrics_stability.py`  
**Issue:** Hub quality analysis was planned but never coded

**Evidence:**
```bash
# No hub-related code in entire file
grep -i "hub" phase2_step2_metrics_stability.py
# Returns: (empty)

# Missing outputs:
PLOT_HUB_QUALITY = OUTPUT_DIR / "hub_quality_scatter.png"
OUT_HUB_METRICS = OUTPUT_DIR / "hub_quality_metrics.csv"

# Missing function:
def analyze_hub_quality(cluster_data, graph_edges, node_attrs): ...
```

### Required Data Collection

**Output file:** `hub_quality_metrics.csv`

**Columns:**
- `edge_config`, `mode`, `node_type`
- `hub_node_id`, `hub_name`
- `degree_total`, `degree_in`, `degree_out`
- `degree_edge_only` (edges using only EDGE, not similarity)
- `n_sources` (unique source documents)
- `n_risk_categories` (distinct risk categories connected)
- `hub_category` (Convergence / Framework / Artifact - requires manual annotation)

### Implementation

**Step 1: Add output variables** (after line 83 in phase2_step2_metrics_stability.py)

```python
# Add to output files section
OUT_HUB_METRICS = OUTPUT_DIR / "hub_quality_metrics.csv"

# Add to visualization outputs section (after line 92)
PLOT_HUB_QUALITY = OUTPUT_DIR / "hub_quality_scatter.png"
```

**Step 2: Add hub analysis function** (after analyze_betweenness function)

```python
def analyze_hub_quality(
    df_metrics: pd.DataFrame,
    node_attrs: Dict,
    edge_data: List[Dict],
    cluster_files_dir: Path
) -> pd.DataFrame:
    """
    Extract top-20 intervention hubs per configuration
    Calculate EDGE%, source diversity, risk diversity
    """
    print("\n" + "="*80)
    print("ANALYZING INTERVENTION HUB QUALITY")
    print("="*80)
    
    results = []
    
    for _, row in tqdm(df_metrics.iterrows(), total=len(df_metrics), desc="Hub analysis"):
        edge_config = row['edge_config']
        mode = row['mode']
        node_type = row['node_type']
        
        # Only analyze intervention nodes
        if node_type != 'intervention':
            continue
        
        # Load cluster file to get intervention node IDs
        cluster_file = cluster_files_dir / row['cluster_filepath']
        if not cluster_file.exists():
            continue
        
        cluster_data = load_cluster_file(cluster_file)
        assignments = get_cluster_assignments(cluster_data)
        intervention_nodes = list(assignments.keys())
        
        # Build edge lookup for this configuration
        # Filter edges by edge_config and mode constraints
        config_edges = []
        for e in edge_data:
            # Check if edge belongs to this config
            if edge_config == 'EDGE':
                if e.get('edge_type') == 'EDGE':
                    config_edges.append(e)
            else:
                # Similarity threshold
                sim_val = float(edge_config)
                if e.get('edge_type') == 'EDGE' or e.get('similarity', 0) >= sim_val:
                    config_edges.append(e)
        
        # Calculate degrees for each intervention node
        node_degrees = {}
        for node_id in intervention_nodes:
            # Total degree (all edges in this config)
            edges = [e for e in config_edges 
                    if e['source'] == node_id or e['target'] == node_id]
            degree_total = len(edges)
            
            # EDGE-only degree
            edge_only = [e for e in edges if e.get('edge_type') == 'EDGE']
            degree_edge_only = len(edge_only)
            
            # Degree in/out
            degree_in = len([e for e in edges if e['target'] == node_id])
            degree_out = len([e for e in edges if e['source'] == node_id])
            
            # Risk diversity (count unique risk categories connected)
            connected_risks = set()
            for e in edges:
                other_node = e['target'] if e['source'] == node_id else e['source']
                if other_node in node_attrs:
                    other_cat = node_attrs[other_node].get('category', '')
                    if other_cat == 'risk':
                        # Get risk category/subcategory
                        risk_cat = node_attrs[other_node].get('risk_category', 'unknown')
                        connected_risks.add(risk_cat)
            
            # Source diversity
            sources = set()
            if node_id in node_attrs:
                # Check multiple possible attribute names
                source_list = node_attrs[node_id].get('source_file_list', [])
                if source_list and isinstance(source_list, list):
                    sources = set(source_list)
                else:
                    # Try single source_file attribute
                    source_file = node_attrs[node_id].get('source_file', None)
                    if source_file and source_file not in [None, 'unknown', '']:
                        sources = {source_file}
            
            node_degrees[node_id] = {
                'degree_total': degree_total,
                'degree_in': degree_in,
                'degree_out': degree_out,
                'degree_edge_only': degree_edge_only,
                'n_sources': len(sources),
                'n_risk_categories': len(connected_risks)
            }
        
        # Get top-20 by total degree
        top20 = sorted(node_degrees.items(), 
                      key=lambda x: x[1]['degree_total'], 
                      reverse=True)[:20]
        
        for node_id, metrics in top20:
            node_name = node_attrs.get(node_id, {}).get('name', f'Node_{node_id}')
            
            edge_pct = (metrics['degree_edge_only'] / metrics['degree_total'] * 100 
                       if metrics['degree_total'] > 0 else 0)
            
            results.append({
                'edge_config': edge_config,
                'mode': mode,
                'node_type': 'intervention',
                'hub_node_id': node_id,
                'hub_name': node_name,
                'degree_total': metrics['degree_total'],
                'degree_in': metrics['degree_in'],
                'degree_out': metrics['degree_out'],
                'degree_edge_only': metrics['degree_edge_only'],
                'edge_percentage': edge_pct,
                'n_sources': metrics['n_sources'],
                'n_risk_categories': metrics['n_risk_categories'],
                'hub_category': 'unknown'  # To be filled by manual annotation
            })
    
    df = pd.DataFrame(results)
    print(f"\n✓ Analyzed {len(df)} hub nodes across {len(df) // 20} configurations")
    return df
```

**Step 3: Add hub visualization function** (after plot_betweenness function)

```python
def plot_hub_quality_scatter(df_hubs: pd.DataFrame):
    """
    Plot 7: Hub quality scatter
    X-axis: EDGE% (percentage of degree from EDGE-only connections)
    Y-axis: Source diversity (number of unique sources)
    Point size: Total degree
    Color: Risk diversity (number of risk categories)
    """
    print("\n" + "="*80)
    print("GENERATING PLOT 7: Hub Quality Scatter")
    print("="*80)
    
    if len(df_hubs) == 0:
        print("⚠ No hub data available - skipping plot")
        return
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create scatter plot
    scatter = ax.scatter(
        df_hubs['edge_percentage'],
        df_hubs['n_sources'],
        s=df_hubs['degree_total'] * 3,  # Scale point size by degree
        c=df_hubs['n_risk_categories'],  # Color by risk diversity
        alpha=0.6,
        cmap='viridis',
        edgecolors='black',
        linewidth=0.5
    )
    
    # Reference lines
    ax.axhline(3, color='red', linestyle='--', alpha=0.5, linewidth=2,
              label='Min sources target (3)')
    ax.axvline(60, color='orange', linestyle='--', alpha=0.5, linewidth=2,
              label='EDGE% target (60)')
    
    # Formatting
    ax.set_xlabel('EDGE-only % of Total Degree', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Unique Sources', fontsize=13, fontweight='bold')
    ax.set_title('Intervention Hub Quality Assessment\n' + 
                'Top-20 Hubs per Configuration (160 configs × 20 = 3200 hubs)',
                fontsize=14, fontweight='bold')
    
    # Colorbar for risk diversity
    cbar = plt.colorbar(scatter, ax=ax, label='Number of Risk Categories')
    cbar.ax.tick_params(labelsize=10)
    
    # Legend
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Add quadrant labels
    ax.text(0.95, 0.95, 'High Quality\n(EDGE% + Sources)', 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    ax.text(0.05, 0.05, 'Low Quality\n(Few EDGE% + Sources)', 
           transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
           horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(PLOT_HUB_QUALITY, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {PLOT_HUB_QUALITY}")
    
    # Print summary statistics
    print("\n### Hub Quality Summary")
    print(f"Total hubs analyzed: {len(df_hubs):,}")
    print(f"Hubs meeting EDGE% >60%: {(df_hubs['edge_percentage'] > 60).sum()}")
    print(f"Hubs meeting sources ≥3: {(df_hubs['n_sources'] >= 3).sum()}")
    print(f"Hubs meeting both criteria: {((df_hubs['edge_percentage'] > 60) & (df_hubs['n_sources'] >= 3)).sum()}")
    print(f"Mean EDGE%: {df_hubs['edge_percentage'].mean():.1f}%")
    print(f"Mean sources: {df_hubs['n_sources'].mean():.1f}")
    print(f"Mean risk categories: {df_hubs['n_risk_categories'].mean():.1f}")
```

**Step 4: Integrate into main() function**

```python
def main():
    """Main execution function"""
    # ... existing code ...
    
    # After betweenness analysis, add:
    
    # ========================================================================
    # HUB QUALITY ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("SECTION: HUB QUALITY ANALYSIS")
    print("="*80)
    
    df_hubs = analyze_hub_quality(
        df_metrics=df_metrics,
        node_attrs=node_attrs,
        edge_data=edge_data,
        cluster_files_dir=CLUSTER_FILES_DIR
    )
    
    # Save results
    df_hubs.to_csv(OUT_HUB_METRICS, index=False)
    print(f"\n✓ Saved: {OUT_HUB_METRICS}")
    
    # Generate visualization
    plot_hub_quality_scatter(df_hubs)
    
    # ... rest of existing code ...
```

### Expected Output Files

**hub_quality_metrics.csv** (sample):
```csv
edge_config,mode,node_type,hub_node_id,hub_name,degree_total,degree_in,degree_out,degree_edge_only,edge_percentage,n_sources,n_risk_categories,hub_category
EDGE,unconstrained,intervention,12345,Constitutional AI,187,134,53,187,100.0,8,5,unknown
0.9,both,intervention,12345,Constitutional AI,243,178,65,189,77.8,8,5,unknown
```

**hub_quality_scatter.png:** Scatter plot with ~3200 points (160 configs × 20 hubs)

### Manual Inspection Component

After generating hub_quality_metrics.csv:

1. **Select 5 sample hubs** for detailed inspection:
   - 2 high EDGE% + high sources (likely Convergence)
   - 2 low EDGE% + high degree (likely Framework or Artifact)
   - 1 moderate across all metrics

2. **For each hub, examine 3-5 neighbors:**
   - Extract connected nodes from graph
   - Check if neighbors share EDGE connections or only similarity
   - Categorize as: Convergence / Framework / Artifact

3. **Update hub_category column** in CSV with findings

### Testing Requirements

1. Verify hub counts: 160 configs with interventions × 20 hubs = ~3200 rows
2. Check degree calculations match expectations
3. Validate source diversity extraction (not all zeros)
4. Confirm risk category diversity makes sense (1-10 range expected)

### Expected Insights

From preliminary cluster size data, we expect:
- High-threshold hubs (0.9, 0.95) show high EDGE% (>70%)
- Low-threshold hubs (0.8) show low EDGE% (<40%)
- Hub stability across thresholds validates they're not artifacts

---

## CHANGE #6: EDGE Validation Breakdown Per-Mode Visualization

**Priority:** MEDIUM  
**Affects:** Step 2, Substep #4  
**Current Status:** ❌ Not Implemented

### Current Implementation Issues

**File:** `phase2_step2_metrics_stability.py`  
**Current plot:** Single stacked bar chart aggregating across all modes
- Cannot see mode-specific validation patterns
- Hides important differences (unconstrained vs "both" at same threshold)
- Difficult to assess which mode optimizes EDGE validation

### Proposed Changes

**New visualization:** 2×2 grid showing EDGE validation per mode

**Design specifications:**
```python
# 2×2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
modes = ['unconstrained', 'single_risk', 'monotonic', 'both']

# Each subplot: One mode, stacked bars by edge config
# X-axis: Edge config (EDGE, 0.8, 0.85, 0.9, 0.95)
# Y-axis: Number of configurations
# Stack colors: Validation rate bins (<60%, 60-80%, 80-90%, 90-100%, 100%)
```

### Implementation Code

**Function location:** `phase2_step2_metrics_stability.py`

**Replace function:**
```python
def plot_edge_validation_breakdown(df_metrics: pd.DataFrame):
    """
    Plot 6: EDGE validation rate distribution
    UPDATED: 2x2 grid showing breakdown per mode
    """
    print("\n" + "="*80)
    print("GENERATING PLOT 6: EDGE Validation Breakdown (Per Mode)")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    modes = ['unconstrained', 'single_risk', 'monotonic', 'both']
    edge_configs = ['EDGE', '0.8', '0.85', '0.9', '0.95']
    
    # Define validation bins
    bins = [
        (0, 0.6, '<60%', 'red'),
        (0.6, 0.8, '60-80%', 'orange'),
        (0.8, 0.9, '80-90%', 'yellow'),
        (0.9, 1.0, '90-100%', 'lightgreen'),
        (1.0, 1.01, '100%', 'darkgreen')  # Exactly 100%
    ]
    
    for idx, mode in enumerate(modes):
        ax = axes[idx]
        
        # Count configs in each validation bin per edge config
        bin_counts = {edge: [0]*len(bins) for edge in edge_configs}
        
        for edge in edge_configs:
            subset = df_metrics[
                (df_metrics['edge_config'] == edge) & 
                (df_metrics['mode'] == mode)
            ]
            
            for _, row in subset.iterrows():
                edge_val = row['edge_validation_mean']
                
                # Assign to bin
                for bin_idx, (low, high, label, color) in enumerate(bins):
                    if low <= edge_val < high:
                        bin_counts[edge][bin_idx] += 1
                        break
        
        # Create stacked bar chart
        x_pos = np.arange(len(edge_configs))
        bottom = np.zeros(len(edge_configs))
        
        for bin_idx, (low, high, label, color) in enumerate(bins):
            heights = [bin_counts[edge][bin_idx] for edge in edge_configs]
            ax.bar(x_pos, heights, bottom=bottom, 
                  label=label, color=color, alpha=0.8)
            bottom += heights
        
        # Formatting
        ax.set_xticks(x_pos)
        ax.set_xticklabels(edge_configs, fontsize=10)
        ax.set_ylabel('Number of Configurations', fontsize=11)
        ax.set_xlabel('Edge Config', fontsize=11)
        ax.set_title(f'Mode: {mode.replace("_", " ").title()}', 
                    fontsize=12, fontweight='bold')
        ax.legend(title='Validation Rate', fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add target line annotation
        ax.axhline(0, color='black', linewidth=0.5)
        ax.text(0.02, 0.98, 'Target: >60%', 
               transform=ax.transAxes, fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('EDGE-only Complete Pathway Validation Rate Distribution\n' + 
                'Breakdown by Mode',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(PLOT_EDGE_VALIDATION, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {PLOT_EDGE_VALIDATION}")
```

### Expected Output

**Visual improvements:**
- Clear comparison of mode impact on EDGE validation
- Immediate identification of which modes achieve >60% target at each threshold
- Pattern visibility: "both" mode concentrates in green (90-100%), unconstrained in red/orange (<80%)

**Insight improvements:**
- Quantify mode effect on literature grounding
- Support recommendation for "both" mode as optimal
- Identify discovery (unconstrained) vs precision (both) trade-off

### Testing Requirements

1. Verify all 160 configs properly assigned to bins
2. Check stacked bar heights sum to expected config count per edge/mode
3. Validate color coding matches validation rate ranges
4. Ensure subplot titles clearly distinguish modes

---

## CHANGE #7: Silhouette Plot Improvements & Algorithm Comparison

**Priority:** HIGH  
**Affects:** Step 2, Substep #1  
**Current Status:** ❌ Multiple issues identified

### Issues in Current Implementation

**File:** `phase2_step2_metrics_stability.py`, function `plot_silhouette_by_nodetype()`

1. **Y-axis label ambiguous:** Says "Silhouette Score" - should specify "Mean Silhouette Score"
2. **No algorithm label:** Plot shows Agglomerative results only, no indication which algorithm
3. **Missing algorithms:** Louvain and HDBSCAN results not computed or plotted
4. **Legend marker mismatch:** Code uses circles/squares, legend shows only circles
5. **Missing silhouette definition:** Document doesn't explain what silhouette measures

### Implementation Changes

**Change 1: Fix Y-axis label** (line 933)

```python
# OLD
ax.set_ylabel("Silhouette Score", fontsize=10)

# NEW
ax.set_ylabel("Mean Silhouette Score\n(Intra-cluster tightness vs Inter-cluster separation)", 
              fontsize=9)
```

**Change 2: Add algorithm indicator to subplot titles** (line 931)

```python
# OLD
ax.set_title(f"{node_type.replace('_', ' ').title()}", fontsize=12, fontweight='bold')

# NEW
algorithm = df_quality['algorithms'].iloc[0] if 'algorithms' in df_quality.columns else 'agglomerative'
ax.set_title(f"{node_type.replace('_', ' ').title()}\nAlgorithm: {algorithm.title()}", 
            fontsize=11, fontweight='bold')
```

**Change 3: Fix legend marker shapes** (lines 941-947)

```python
# OLD - only shows circles
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], 
               markersize=10, label=str(e))
    for i, e in enumerate(EDGE_CONFIGS)
]

# NEW - shows correct shapes
legend_elements = []
for i, e in enumerate(EDGE_CONFIGS):
    marker = 'o' if e == 'EDGE' else 's'
    legend_elements.append(
        plt.Line2D([0], [0], marker=marker, color='w', 
                  markerfacecolor=colors[i], markersize=10, 
                  label=f"{e} ({'circle' if marker=='o' else 'square'})")
    )
```

**Change 4: Add algorithm comparison overlay**

Create new function `plot_algorithm_comparison()`:

```python
def plot_algorithm_comparison(df_metrics: pd.DataFrame):
    """
    Plot: Algorithm comparison overlay
    Compare Agglomerative vs Louvain vs HDBSCAN silhouette scores
    """
    # Check if multiple algorithms present
    if 'algorithms' not in df_metrics.columns:
        print("⚠ No algorithm data - skipping comparison plot")
        return
    
    algorithms = df_metrics['algorithms'].unique()
    if len(algorithms) < 2:
        print("⚠ Only one algorithm found - skipping comparison")
        return
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.flatten()
    
    # Colors per algorithm
    alg_colors = {'agglomerative': 'blue', 'louvain': 'green', 'hdbscan': 'red'}
    
    for idx, node_type in enumerate(NODE_TYPES):
        ax = axes[idx]
        
        # For each edge config and mode, plot all algorithms
        x_positions = []
        for i, edge_config in enumerate(EDGE_CONFIGS):
            for j, mode in enumerate(MODES):
                x_pos = i * 5 + j  # Spacing
                
                for alg in algorithms:
                    mask = (
                        (df_metrics['node_type'] == node_type) &
                        (df_metrics['edge_config'] == str(edge_config)) &
                        (df_metrics['mode'] == mode) &
                        (df_metrics['algorithms'] == alg)
                    )
                    
                    if mask.sum() > 0:
                        sil = df_metrics[mask]['silhouette_mean'].values[0]
                        ax.scatter(x_pos, sil, color=alg_colors.get(alg, 'gray'),
                                 s=60, alpha=0.7, marker='o')
                
                x_positions.append(x_pos)
        
        # Format
        ax.set_title(f"{node_type.replace('_', ' ').title()}", 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel("Mean Silhouette Score", fontsize=10)
        ax.axhline(0.3, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 0.7)
        
        # Simplified x-axis (just edge configs)
        ax.set_xticks([i * 5 + 1.5 for i in range(len(EDGE_CONFIGS))])
        ax.set_xticklabels([str(e) for e in EDGE_CONFIGS], fontsize=9)
    
    # Algorithm legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor=alg_colors.get(alg, 'gray'),
                  markersize=10, label=alg.title())
        for alg in algorithms
    ]
    fig.legend(handles=legend_elements, loc='lower right', 
              bbox_to_anchor=(0.98, 0.02), title="Algorithm", fontsize=10)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(OUTPUT_DIR / "algorithm_comparison_silhouette.png", 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: algorithm_comparison_silhouette.png")
```

**Change 5: Add algorithm execution**

Current Step 1 only runs Agglomerative. Need to add:

```python
# In step1 clustering execution
for algorithm in ['agglomerative', 'louvain', 'hdbscan']:
    # Run clustering with each algorithm
    # Store results with algorithm tag in metrics
```

### Expected Outputs

1. **Corrected silhouette_by_nodetype.png:** Proper labels, legend, axes
2. **New algorithm_comparison_silhouette.png:** Shows all 3 algorithms overlaid
3. **Updated all_cluster_metrics.csv:** Include `algorithms` column

### Testing Requirements

1. Verify legend matches actual markers
2. Check algorithm comparison shows meaningful differences
3. Validate silhouette formula explanation added to documentation

---

## CHANGE #8: Cluster Cohesion Metrics Implementation

**Priority:** MEDIUM  
**Affects:** Step 2, Substep #3  
**Current Status:** ❌ **NOT IMPLEMENTED - COMPLETELY MISSING**

### Root Cause

**File:** `phase2_step2_metrics_stability.py`  
**Issue:** No cohesion analysis code exists

**Evidence:**
```bash
grep -i "cohesion\|intra\|inter.*cluster" phase2_step2_metrics_stability.py
# Returns: (empty)

# No output file defined
grep "OUT_" phase2_step2_metrics_stability.py | grep -i cohesion
# Returns: (empty)
```

### Implementation

**Step 1: Add output variable** (after line 83)

```python
OUT_COHESION_ANALYSIS = OUTPUT_DIR / "cohesion_analysis.csv"
```

**Step 2: Add cohesion analysis function**

```python
def analyze_cluster_cohesion(
    df_metrics: pd.DataFrame,
    node_attrs: Dict,
    cluster_files_dir: Path
) -> pd.DataFrame:
    """
    Calculate intra-cluster compactness and inter-cluster separation
    """
    print("\n" + "="*80)
    print("ANALYZING CLUSTER COHESION METRICS")
    print("="*80)
    
    from scipy.spatial.distance import cosine
    
    results = []
    
    for _, row in tqdm(df_metrics.iterrows(), total=len(df_metrics), desc="Cohesion analysis"):
        edge_config = row['edge_config']
        mode = row['mode']
        node_type = row['node_type']
        
        if not row['cluster_file_found']:
            continue
        
        cluster_file = cluster_files_dir / row['cluster_filepath']
        if not cluster_file.exists():
            continue
        
        cluster_data = load_cluster_file(cluster_file)
        if not cluster_data:
            continue
        
        assignments = get_cluster_assignments(cluster_data, 'agglomerative')
        
        # Group nodes by cluster
        clusters = {}
        for node_id, cluster_id in assignments.items():
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(node_id)
        
        # Get embeddings for all nodes
        node_embeddings = {}
        for node_id in assignments.keys():
            if node_id in node_attrs and 'embedding' in node_attrs[node_id]:
                node_embeddings[node_id] = node_attrs[node_id]['embedding']
        
        if len(node_embeddings) == 0:
            continue
        
        # Calculate cluster centroids
        centroids = {}
        for cluster_id, node_ids in clusters.items():
            embeddings = [node_embeddings[nid] for nid in node_ids 
                         if nid in node_embeddings]
            if len(embeddings) > 0:
                centroids[cluster_id] = np.mean(embeddings, axis=0)
        
        # Calculate intra-cluster distances (average within each cluster)
        intra_distances = []
        for cluster_id, node_ids in clusters.items():
            embeddings = [node_embeddings[nid] for nid in node_ids 
                         if nid in node_embeddings]
            
            if len(embeddings) < 2:
                continue
            
            # Average pairwise distance within cluster
            dists = []
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    dist = cosine(embeddings[i], embeddings[j])
                    dists.append(dist)
            
            if len(dists) > 0:
                intra_distances.append(np.mean(dists))
        
        # Calculate inter-cluster distances (minimum between centroids)
        inter_distances = []
        cluster_ids = list(centroids.keys())
        for i in range(len(cluster_ids)):
            for j in range(i+1, len(cluster_ids)):
                c1 = centroids[cluster_ids[i]]
                c2 = centroids[cluster_ids[j]]
                dist = cosine(c1, c2)
                inter_distances.append(dist)
        
        # Compute statistics
        intra_mean = np.mean(intra_distances) if intra_distances else 0
        inter_mean = np.mean(inter_distances) if inter_distances else 0
        inter_min = np.min(inter_distances) if inter_distances else 0
        
        separation_ratio = inter_mean / intra_mean if intra_mean > 0 else 0
        
        results.append({
            'edge_config': edge_config,
            'mode': mode,
            'node_type': node_type,
            'intra_cluster_mean': intra_mean,
            'intra_cluster_std': np.std(intra_distances) if intra_distances else 0,
            'inter_cluster_mean': inter_mean,
            'inter_cluster_min': inter_min,
            'inter_cluster_std': np.std(inter_distances) if inter_distances else 0,
            'separation_ratio': separation_ratio,
            'n_clusters_analyzed': len(clusters)
        })
    
    df = pd.DataFrame(results)
    print(f"\n✓ Analyzed cohesion for {len(df)} configurations")
    return df
```

**Step 3: Integrate into main()**

```python
# After stability analysis, add:
df_cohesion = analyze_cluster_cohesion(
    df_metrics=df_metrics,
    node_attrs=node_attrs,
    cluster_files_dir=CLUSTER_FILES_DIR
)

# Save results
df_cohesion.to_csv(OUT_COHESION_ANALYSIS, index=False)
print(f"\n✓ Saved: {OUT_COHESION_ANALYSIS}")
```

### Expected Output

**cohesion_analysis.csv** (sample):
```csv
edge_config,mode,node_type,intra_cluster_mean,intra_cluster_std,inter_cluster_mean,inter_cluster_min,inter_cluster_std,separation_ratio,n_clusters_analyzed
EDGE,unconstrained,risk,0.35,0.12,0.72,0.45,0.18,2.06,40
0.8,unconstrained,risk,0.28,0.10,0.68,0.42,0.15,2.43,80
```

### Interpretation Guidance

**Separation ratio interpretation:**
- **>2.5:** Excellent separation
- **2.0-2.5:** Good separation
- **1.5-2.0:** Acceptable separation
- **<1.5:** Poor separation (clusters overlap)

**Expected patterns:**
- 0.8 configs: Higher inter-cluster separation (semantic grouping)
- EDGE configs: Lower inter-cluster separation (literature grouping tight in embedding space)
- Should align with silhouette paradox findings

---

## CHANGE #9: Cluster Centroid Similarity Analysis (Semantic Stability)

**Priority:** HIGH - Critical for understanding actual threshold stability  
**Affects:** Step 2, Substep #8  
**Current Status:** ❌ **NOT IMPLEMENTED**

### Problem with Current Migration Analysis

**Migration rate (as implemented) is NOT useful:**
- Measures cluster ID changes without matching clusters
- 100% migration could mean: identical clustering with relabeled IDs (artifact) OR complete reorganization (real)
- Cannot distinguish between the two

**ARI is incomplete:**
- Measures if nodes A,B stay together/apart (co-membership structure)
- Misses semantic stability: cluster {A,B,C} could shift from "RLHF methods" centroid to "broad alignment" centroid
- ARI stays high (pairs preserved) but cluster meaning changed

### What We Need: Centroid Similarity

**For each node at each threshold transition:**
1. Get centroid embedding of node's cluster at threshold T1
2. Get centroid embedding of node's cluster at threshold T2
3. Calculate cosine similarity between centroids
4. High similarity (>0.8): node's cluster semantically stable
5. Low similarity (<0.5): cluster context reorganized

**This measures:** Do mechanism clusters maintain semantic coherence across thresholds, or do boundaries/meanings reorganize even when member pairs stay together?

### Implementation

**Step 1: Add output files**

```python
OUT_CENTROID_SIMILARITY = OUTPUT_DIR / "cluster_centroid_similarity.csv"
PLOT_CENTROID_SIMILARITY = OUTPUT_DIR / "centroid_similarity_heatmap.png"
```

**Step 2: Add analysis function**

```python
def analyze_cluster_centroid_similarity(
    df_metrics: pd.DataFrame,
    node_attrs: Dict,
    cluster_files_dir: Path
) -> pd.DataFrame:
    """
    Calculate semantic stability via cluster centroid similarity
    For each node: compare centroid of its cluster at T1 vs T2
    """
    print("\n" + "="*80)
    print("ANALYZING CLUSTER CENTROID SIMILARITY (Semantic Stability)")
    print("="*80)
    
    from scipy.spatial.distance import cosine
    
    results = []
    
    for node_type in tqdm(NODE_TYPES, desc="Node types"):
        for mode in MODES:
            # Adjacent threshold pairs
            for i in range(len(EDGE_CONFIGS) - 1):
                edge1 = str(EDGE_CONFIGS[i])
                edge2 = str(EDGE_CONFIGS[i + 1])
                
                # Load both clustering results
                mask1 = (
                    (df_metrics['node_type'] == node_type) &
                    (df_metrics['edge_config'] == edge1) &
                    (df_metrics['mode'] == mode) &
                    (df_metrics['cluster_file_found'] == True)
                )
                mask2 = (
                    (df_metrics['node_type'] == node_type) &
                    (df_metrics['edge_config'] == edge2) &
                    (df_metrics['mode'] == mode) &
                    (df_metrics['cluster_file_found'] == True)
                )
                
                if mask1.sum() == 0 or mask2.sum() == 0:
                    continue
                
                filepath1 = cluster_files_dir / df_metrics[mask1].iloc[0]['cluster_filepath']
                filepath2 = cluster_files_dir / df_metrics[mask2].iloc[0]['cluster_filepath']
                
                if not filepath1.exists() or not filepath2.exists():
                    continue
                
                # Load cluster assignments
                cluster1 = load_cluster_file(filepath1)
                cluster2 = load_cluster_file(filepath2)
                
                if not cluster1 or not cluster2:
                    continue
                
                assign1 = get_cluster_assignments(cluster1, 'agglomerative')
                assign2 = get_cluster_assignments(cluster2, 'agglomerative')
                
                # Build cluster→nodes mapping
                clusters1 = {}
                for node_id, cid in assign1.items():
                    if cid not in clusters1:
                        clusters1[cid] = []
                    clusters1[cid].append(node_id)
                
                clusters2 = {}
                for node_id, cid in assign2.items():
                    if cid not in clusters2:
                        clusters2[cid] = []
                    clusters2[cid].append(node_id)
                
                # Calculate centroids for each cluster
                centroids1 = {}
                for cid, nodes in clusters1.items():
                    embeddings = [node_attrs[nid]['embedding'] for nid in nodes 
                                 if nid in node_attrs and 'embedding' in node_attrs[nid]]
                    if embeddings:
                        centroids1[cid] = np.mean(embeddings, axis=0)
                
                centroids2 = {}
                for cid, nodes in clusters2.items():
                    embeddings = [node_attrs[nid]['embedding'] for nid in nodes 
                                 if nid in node_attrs and 'embedding' in node_attrs[nid]]
                    if embeddings:
                        centroids2[cid] = np.mean(embeddings, axis=0)
                
                # For each node: compare its T1 centroid vs T2 centroid
                common_nodes = set(assign1.keys()) & set(assign2.keys())
                centroid_sims = []
                
                for node_id in common_nodes:
                    cid1 = assign1[node_id]
                    cid2 = assign2[node_id]
                    
                    if cid1 in centroids1 and cid2 in centroids2:
                        sim = 1 - cosine(centroids1[cid1], centroids2[cid2])
                        centroid_sims.append(sim)
                
                if len(centroid_sims) == 0:
                    continue
                
                # Calculate statistics
                mean_sim = np.mean(centroid_sims)
                median_sim = np.median(centroid_sims)
                
                # Categorize stability
                high_stable = sum(1 for s in centroid_sims if s > 0.8)
                moderate = sum(1 for s in centroid_sims if 0.5 <= s <= 0.8)
                low_stable = sum(1 for s in centroid_sims if s < 0.5)
                
                results.append({
                    'node_type': node_type,
                    'mode': mode,
                    'threshold_from': edge1,
                    'threshold_to': edge2,
                    'n_nodes': len(centroid_sims),
                    'centroid_sim_mean': mean_sim,
                    'centroid_sim_median': median_sim,
                    'centroid_sim_std': np.std(centroid_sims),
                    'centroid_sim_min': np.min(centroid_sims),
                    'centroid_sim_max': np.max(centroid_sims),
                    'high_stable_pct': high_stable / len(centroid_sims),  # >0.8 sim
                    'moderate_pct': moderate / len(centroid_sims),        # 0.5-0.8
                    'low_stable_pct': low_stable / len(centroid_sims)     # <0.5
                })
    
    df = pd.DataFrame(results)
    print(f"\n✓ Analyzed centroid similarity for {len(df)} transitions")
    return df
```

**Step 3: Add visualization**

```python
def plot_centroid_similarity_heatmap(df_sim: pd.DataFrame):
    """
    Heatmap showing mean centroid similarity across transitions
    High values (green) = clusters semantically stable
    Low values (red) = cluster meanings reorganize
    """
    print("\n" + "="*80)
    print("GENERATING PLOT: Centroid Similarity Heatmap")
    print("="*80)
    
    if len(df_sim) == 0:
        print("⚠ No similarity data - skipping plot")
        return
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.flatten()
    
    transitions = []
    for i in range(len(EDGE_CONFIGS) - 1):
        transitions.append(f"{EDGE_CONFIGS[i]}→{EDGE_CONFIGS[i+1]}")
    
    for idx, node_type in enumerate(NODE_TYPES):
        ax = axes[idx]
        
        # Build matrix: modes × transitions
        mat = np.zeros((len(MODES), len(transitions)))
        
        for i, mode in enumerate(MODES):
            for j, (edge1, edge2) in enumerate(zip(EDGE_CONFIGS[:-1], EDGE_CONFIGS[1:])):
                mask = (
                    (df_sim['node_type'] == node_type) &
                    (df_sim['mode'] == mode) &
                    (df_sim['threshold_from'] == str(edge1)) &
                    (df_sim['threshold_to'] == str(edge2))
                )
                
                if mask.sum() > 0:
                    mat[i, j] = df_sim[mask]['centroid_sim_mean'].values[0]
        
        # Plot
        im = ax.imshow(mat, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(len(transitions)))
        ax.set_xticklabels(transitions, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(np.arange(len(MODES)))
        ax.set_yticklabels([m.replace('_', '\n') for m in MODES], fontsize=9)
        ax.set_title(f"{node_type.replace('_', ' ').title()}", 
                    fontsize=11, fontweight='bold')
        
        # Add values
        for i in range(len(MODES)):
            for j in range(len(transitions)):
                if mat[i, j] > 0:
                    ax.text(j, i, f'{mat[i,j]:.2f}',
                           ha="center", va="center",
                           color="black" if mat[i,j] < 0.5 else "white",
                           fontsize=8)
        
        plt.colorbar(im, ax=ax, label='Centroid Similarity')
    
    plt.suptitle('Cluster Centroid Similarity Across Thresholds\n' +
                '(Semantic Stability: High=Green, Low=Red)',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(PLOT_CENTROID_SIMILARITY, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {PLOT_CENTROID_SIMILARITY}")
```

**Step 4: Integrate into main()**

```python
# Replace analyze_node_migration() section with:
df_centroid_sim = analyze_cluster_centroid_similarity(
    df_metrics=df_metrics,
    node_attrs=node_attrs,
    cluster_files_dir=CLUSTER_FILES_DIR
)

df_centroid_sim.to_csv(OUT_CENTROID_SIMILARITY, index=False)
print(f"\n✓ Saved: {OUT_CENTROID_SIMILARITY}")

plot_centroid_similarity_heatmap(df_centroid_sim)
```

### Expected Outputs

**cluster_centroid_similarity.csv:**
```csv
node_type,mode,threshold_from,threshold_to,n_nodes,centroid_sim_mean,centroid_sim_median,centroid_sim_std,high_stable_pct,moderate_pct,low_stable_pct
risk,unconstrained,EDGE,0.8,2639,0.72,0.75,0.18,0.45,0.42,0.13
risk,unconstrained,0.8,0.85,3471,0.68,0.71,0.20,0.38,0.48,0.14
```

**centroid_similarity_heatmap.png:** 8 subplots showing semantic stability across transitions

### Interpretation Guide

**Centroid similarity values:**
- **>0.8 (dark green):** High semantic stability - cluster maintains same conceptual region
- **0.5-0.8 (yellow):** Moderate stability - cluster shifts but related concepts
- **<0.5 (red):** Low stability - cluster reorganizes to different semantic region

**Combined with ARI:**
- **High ARI + High centroid sim:** Stable mechanisms (pairs stay together in same semantic space)
- **High ARI + Low centroid sim:** Structure preserved but meanings reorganize (concerning)
- **Low ARI + Low centroid sim:** Complete reorganization (expected at distant thresholds)

**Workshop use:** Report both metrics to demonstrate threshold robustness of mechanism clusters.

---

## CHANGE #10: EDGE Purity Per Cluster Analysis

**Priority:** HIGH - Critical for taxonomy confidence assignment  
**Affects:** Step 2, Substep #5  
**Current Status:** ❌ **NOT IMPLEMENTED**

### Root Cause

**File:** `phase2_step2_metrics_stability.py`  
**Issue:** No per-cluster EDGE purity analysis exists

**Evidence:**
```bash
grep -i "edge.*purity\|purity.*edge" phase2_step2_metrics_stability.py
# Returns: (empty)
```

**Current:** Config-level EDGE% in `quality_metrics_summary.csv`  
**Missing:** Per-cluster EDGE% distribution

### Conceptual Foundation

**EDGE purity = validation metric, not clustering input:**
- Clustering uses node embeddings (no edges)
- EDGE purity calculated post-hoc to measure literature grounding
- High purity = cluster validated by single-source pathways
- Low purity = cluster driven by similarity aggregation

### Implementation

**Step 1: Add output file**

```python
OUT_EDGE_PURITY = OUTPUT_DIR / "cluster_edge_purity.csv"
PLOT_EDGE_PURITY = OUTPUT_DIR / "edge_purity_histograms.png"
```

**Step 2: Add analysis function**

```python
def analyze_edge_purity_per_cluster(
    df_metrics: pd.DataFrame,
    node_attrs: Dict,
    cluster_files_dir: Path,
    edge_data: List[Dict]
) -> pd.DataFrame:
    """
    Calculate % of nodes in each cluster that appear in EDGE-only complete pathways
    """
    print("\n" + "="*80)
    print("ANALYZING EDGE PURITY PER CLUSTER")
    print("="*80)
    
    results = []
    
    # First, identify nodes in EDGE-only complete pathways
    # (Requires pathway data from Step 1 or FalkorDB query)
    edge_only_pathway_nodes = set()
    
    # Query pathways with all EDGE edges, maturity>=3, >=4 categories
    # This is simplified - actual implementation needs pathway extraction
    for node_id, attrs in node_attrs.items():
        # Check if node appears in any EDGE-only complete pathway
        # This requires pathway membership data from Step 1 checkpoint
        if attrs.get('in_edge_only_pathway', False):  # Placeholder
            edge_only_pathway_nodes.add(node_id)
    
    # For each cluster, calculate EDGE purity
    for _, row in tqdm(df_metrics.iterrows(), total=len(df_metrics), desc="EDGE purity"):
        if not row['cluster_file_found']:
            continue
        
        cluster_file = cluster_files_dir / row['cluster_filepath']
        if not cluster_file.exists():
            continue
        
        cluster_data = load_cluster_file(cluster_file)
        if not cluster_data:
            continue
        
        assignments = get_cluster_assignments(cluster_data, 'agglomerative')
        
        # Group by cluster ID
        clusters = {}
        for node_id, cid in assignments.items():
            if cid not in clusters:
                clusters[cid] = []
            clusters[cid].append(node_id)
        
        # Calculate EDGE purity per cluster
        for cluster_id, node_ids in clusters.items():
            edge_nodes = [nid for nid in node_ids if nid in edge_only_pathway_nodes]
            edge_purity = len(edge_nodes) / len(node_ids) if node_ids else 0
            
            results.append({
                'edge_config': row['edge_config'],
                'mode': row['mode'],
                'node_type': row['node_type'],
                'cluster_id': cluster_id,
                'cluster_size': len(node_ids),
                'n_edge_nodes': len(edge_nodes),
                'edge_purity': edge_purity,
                'is_gold_standard': edge_purity > 0.8
            })
    
    df = pd.DataFrame(results)
    print(f"\n✓ Analyzed EDGE purity for {len(df)} clusters")
    return df
```

**Step 3: Add visualization**

```python
def plot_edge_purity_histograms(df_purity: pd.DataFrame):
    """
    Histograms showing EDGE purity distribution per config
    """
    print("\n" + "="*80)
    print("GENERATING PLOT: EDGE Purity Histograms")
    print("="*80)
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.flatten()
    
    purity_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    
    for idx, node_type in enumerate(NODE_TYPES):
        ax = axes[idx]
        
        subset = df_purity[df_purity['node_type'] == node_type]
        
        if len(subset) == 0:
            continue
        
        # Histogram per edge config
        for edge_config in EDGE_CONFIGS:
            config_subset = subset[subset['edge_config'] == str(edge_config)]
            if len(config_subset) == 0:
                continue
            
            counts, _ = np.histogram(config_subset['edge_purity'], bins=purity_bins)
            ax.bar(range(len(counts)), counts, alpha=0.6, label=str(edge_config))
        
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, rotation=45, ha='right')
        ax.set_ylabel('Cluster Count', fontsize=10)
        ax.set_xlabel('EDGE Purity', fontsize=10)
        ax.set_title(f"{node_type.replace('_', ' ').title()}", 
                    fontsize=11, fontweight='bold')
        ax.axvline(3.5, color='red', linestyle='--', alpha=0.5, 
                  label='Gold Standard (>80%)')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('EDGE Purity Distribution Across Clusters\n' +
                '(% nodes from EDGE-only complete pathways)',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(PLOT_EDGE_PURITY, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {PLOT_EDGE_PURITY}")
```

### Critical Dependency

**Requires pathway membership data:**
- Need to know which nodes appear in EDGE-only complete pathways
- This information must be extracted in Step 1 or queried from FalkorDB
- Current Step 1 checkpoints may not include this metadata

**Solution:** Add to Step 1 checkpoint generation:
```python
# For each node, check if it appears in any EDGE-only complete pathway
for node_id in all_nodes:
    node_attrs[node_id]['in_edge_only_pathway'] = check_edge_pathway_membership(node_id)
```

### Expected Outputs

**cluster_edge_purity.csv:**
```csv
edge_config,mode,node_type,cluster_id,cluster_size,n_edge_nodes,edge_purity,is_gold_standard
EDGE,unconstrained,risk,0,81,81,1.0,True
0.9,both,risk,5,65,59,0.908,True
0.8,unconstrained,risk,12,120,35,0.292,False
```

**Summary statistics expected:**
- 0.9 both risk: 25/40 clusters (62.5%) are gold standard (>80% purity)
- 0.8 unconstrained risk: 8/76 clusters (10.5%) are gold standard

### Workshop Use

**Taxonomy confidence assignment:**
- Gold standard clusters (>80% purity): Auto-label, low validation burden
- Mixed clusters (40-80%): Moderate validation
- Similarity-driven (<40%): Heavy manual validation required

**Results reporting:**
- "62% of final mechanism clusters achieved >80% EDGE purity"
- "Literature grounding validated through single-source pathway membership"

---

## CHANGE #11: Source Diversity Data Fix (Step 1 Checkpoint)

**Priority:** MEDIUM - Blocks Substep #6  
**Affects:** Step 1 checkpoint generation, impacts Step 2 Substep #6  
**Current Status:** ⚠️ **DATA PIPELINE ISSUE**

### Root Cause

**Analysis code exists and works correctly** (lines 625-656 in phase2_step2_metrics_stability.py)  
**Problem:** Node attributes lack source information

**Evidence:**
- `cluster_source_diversity.csv` generated with 6,402 rows
- All `n_sources = 0`, all `nodes_with_sources = 0`
- Code checks for `source_file_list` and `source_file` attributes - finds none

**Issue location:** Step 1 checkpoint generation (`graph_node_attributes.pkl`)

### Fix Required in Step 1

**File:** Step 1 load and parse script (checkpoint generation)

**Current:** Node attributes extracted without source information  
**Needed:** Extract source attribution from FalkorDB nodes

**Add to checkpoint generation:**

```python
def extract_node_attributes_with_sources(graph):
    """
    Extract node attributes INCLUDING source information
    """
    node_attrs = {}
    
    # Query all nodes with source information
    query = """
    MATCH (n:NODE)
    RETURN n.id AS node_id,
           n.name AS name,
           n.embedding AS embedding,
           n.category AS category,
           n.source_file AS source_file,
           n.first_published AS first_published
    """
    
    results = graph.query(query)
    
    for record in results:
        node_id = record['node_id']
        node_attrs[node_id] = {
            'name': record['name'],
            'embedding': record['embedding'],
            'category': record['category'],
            'source_file': record['source_file'],  # Single source
            'first_published': record['first_published']
        }
        
        # If multiple sources exist, query separately
        source_query = f"""
        MATCH (n:NODE {{id: {node_id}}})-[:EXTRACTED_FROM]->(s:SOURCE)
        RETURN s.filename AS source
        """
        source_results = graph.query(source_query)
        
        if len(source_results) > 0:
            node_attrs[node_id]['source_file_list'] = [
                r['source'] for r in source_results
            ]
    
    return node_attrs
```

**Alternative if sources not in FalkorDB:**

Extract from pathway data (nodes inherit sources from pathways they appear in):

```python
def infer_sources_from_pathways(node_attrs, pathway_data):
    """
    If source not stored on nodes, infer from pathway membership
    """
    for node_id in node_attrs.keys():
        sources = set()
        
        # Find all pathways containing this node
        for pathway in pathway_data:
            if node_id in pathway['nodes']:
                # Pathway source is the paper it was extracted from
                if 'source' in pathway:
                    sources.add(pathway['source'])
        
        if len(sources) > 0:
            node_attrs[node_id]['source_file_list'] = list(sources)
```

### Testing

After fix, verify:
```python
# Check sample nodes have source info
sample_nodes = random.sample(list(node_attrs.keys()), 100)
with_sources = sum(1 for nid in sample_nodes 
                  if 'source_file' in node_attrs[nid] or 
                     'source_file_list' in node_attrs[nid])
print(f"Nodes with sources: {with_sources}/100")
# Expected: >90/100
```

### Expected Impact

Once fixed, `cluster_source_diversity.csv` should show:
```csv
edge_config,mode,node_type,cluster_id,n_sources,cluster_size,nodes_with_sources
EDGE,unconstrained,risk,3,5,81,81
EDGE,unconstrained,risk,14,3,66,66
```

Typical ranges:
- EDGE-only clusters: 1-3 sources (single pathway origin)
- SIM≥0.9 clusters: 3-8 sources (cross-paper aggregation)

---

## CHANGE #12: Cluster Size Distribution Plot Y-Axis Fix

**Priority:** LOW (visual only, CSV analysis sufficient)  
**Affects:** Step 2, Substep #2 visualization  
**Current Status:** ⚠️ **Y-AXIS TRUNCATED**

### Problem

**Plot:** `cluster_size_distributions.png`  
**Issue:** Y-axis limits cut off distributions on low end (possibly high end too)

**Impact:** Cannot rely on visual analysis - CSV used instead

### Fix Required

**File:** `phase2_step2_metrics_stability.py`, function `plot_cluster_size_distributions()`

**Add proper y-axis limit checking:**

```python
def plot_cluster_size_distributions(df_quality: pd.DataFrame):
    """
    Plot: Cluster size distributions
    FIXED: Proper y-axis limits to avoid truncation
    """
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.flatten()
    
    for idx, node_type in enumerate(NODE_TYPES):
        ax = axes[idx]
        
        # ... existing plotting code ...
        
        # FIX: Calculate proper y-axis limits
        all_cluster_counts = []
        for edge_config in EDGE_CONFIGS:
            for mode in MODES:
                mask = (
                    (df_quality['node_type'] == node_type) &
                    (df_quality['edge_config'] == str(edge_config)) &
                    (df_quality['mode'] == mode)
                )
                if mask.sum() > 0:
                    all_cluster_counts.append(df_quality[mask]['n_clusters'].values[0])
        
        if len(all_cluster_counts) > 0:
            y_min = min(all_cluster_counts)
            y_max = max(all_cluster_counts)
            
            # Add 5% padding
            y_range = y_max - y_min
            y_padding = y_range * 0.05 if y_range > 0 else 5
            
            # BEFORE (likely causing issue):
            # ax.set_ylim(35, 70)  # Hard-coded limits
            
            # AFTER (proper limits):
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
        
        # ... rest of formatting ...
```

**Alternative - auto-scale:**
```python
# Let matplotlib auto-scale based on data
# Remove any manual ylim() calls
```

### Expected Result

All cluster count distributions visible without truncation.

---

## CHANGE #13: Source Diversity Data - Simple Bug Fix

**Priority:** MEDIUM  
**Affects:** Step 2, Substep #6  
**Current Status:** ⚠️ **NEEDS DIAGNOSIS**

### Investigation Required

**Current code** (lines 625-656) looks correct but produces all zeros.

**Diagnostic already in code** (lines 554-574):
```python
print(f"Sample node attributes: {list(sample_node.keys())}")
print(f"Nodes with non-empty source_file_list: {n_with_source_list:,}")
print(f"Nodes with non-empty source_file: {n_with_source_file:,}")
```

**Action:** Run Step 2 script with diagnostics enabled, check output for:
1. What attributes are actually present in `node_attrs`?
2. Are `source_file` or `source_file_list` attributes present but empty/None?
3. Or completely missing from node_attrs dictionary?

**Possible simple fixes:**

**If attributes present but wrong format:**
```python
# Current code expects:
attrs['source_file_list']  # as list

# May actually be:
attrs['source_files']  # different key name
attrs['sources']  # different key name
attrs['source_file_list']  # but as string, not list
```

**If in different location:**
```python
# May be nested:
attrs['metadata']['source_file']
attrs['extraction_info']['sources']
```

**Next steps:**
1. Check actual Step 1 checkpoint (`graph_node_attributes.pkl`) structure
2. Print sample node to see exact attribute names
3. Update code to match actual data structure OR
4. Fix Step 1 checkpoint generation (CHANGE #11)

---

## CHANGE #14: Path Length Sensitivity Plot Generation

**Priority:** LOW (weak signal, minimal actionability)  
**Affects:** Step 2, Substep #29  
**Current Status:** ❌ **NOT IMPLEMENTED**

### Root Cause

**No plot variable or function exists:**
```bash
grep "PLOT_PATH\|path.*sensitivity" phase2_step2_metrics_stability.py
# Returns: (empty)
```

**Data available:** `path_length_mean` column in CSV, but no visualization generated

### Analysis Finding

**Weak dependence detected:**
- Path length varies 2.12x (6.74→14.32 hops)
- Silhouette changes only ~10% (0.414→0.455)
- Correlation r=0.233 (weak, r²=0.054)
- Signal likely noise from other factors (node type, config)

### Implementation

**File:** `phase2_step2_metrics_stability.py`

**Step 1: Add output variable**
```python
PLOT_PATH_SENSITIVITY = OUTPUT_DIR / "path_length_sensitivity.png"
```

**Step 2: Add plotting function**
```python
def plot_path_length_sensitivity(df_metrics: pd.DataFrame):
    """
    Plot: Path length vs silhouette score
    NOTE: Weak relationship detected (r=0.233)
    """
    print("\n" + "="*80)
    print("GENERATING PLOT: Path Length Sensitivity")
    print("="*80)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot with different colors per edge config
    colors = {'EDGE': 'black', '0.8': 'blue', '0.85': 'green', 
              '0.9': 'orange', '0.95': 'red'}
    
    for edge in EDGE_CONFIGS:
        subset = df_metrics[df_metrics['edge_config'] == str(edge)]
        
        ax.scatter(subset['path_length_mean'], 
                  subset['silhouette_mean'],
                  c=colors.get(str(edge), 'gray'),
                  s=50, alpha=0.6,
                  label=str(edge))
    
    # Add correlation line
    valid = df_metrics[df_metrics['path_length_mean'].notna()]
    if len(valid) > 0:
        from scipy.stats import pearsonr
        corr, p_value = pearsonr(valid['path_length_mean'], 
                                 valid['silhouette_mean'])
        
        # Linear fit
        z = np.polyfit(valid['path_length_mean'], 
                      valid['silhouette_mean'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid['path_length_mean'].min(),
                           valid['path_length_mean'].max(), 100)
        ax.plot(x_line, p(x_line), 'k--', alpha=0.3, 
               label=f'r={corr:.3f}, p={p_value:.3f}')
    
    ax.set_xlabel('Mean Path Length (hops)', fontsize=11)
    ax.set_ylabel('Mean Silhouette Score', fontsize=11)
    ax.set_title('Path Length vs Cluster Quality\n(Weak Correlation: r=0.233)',
                fontsize=12, fontweight='bold')
    ax.legend(title='Edge Config', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_PATH_SENSITIVITY, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {PLOT_PATH_SENSITIVITY}")
    print(f"  Correlation: {corr:.3f} (weak)")
```

**Step 3: Call in main()**
```python
# After other plots:
plot_path_length_sensitivity(df_metrics)
```

### Expected Output

Scatter plot showing weak positive correlation, high variance.

### Workshop Use

**Limited value - report as:**
- "Path length weakly correlated with silhouette (r=0.233, p<0.05)"
- "Cluster quality primarily determined by node type and edge configuration"
- Do NOT claim causality or use for optimization

---

## CHANGE #15: Enrichment Substep Implementations

**Priority:** MEDIUM - Required for workshop completeness  
**Affects:** Step 2, Substeps #9-11, #13, #15-16, #31

### Substep #10: Multi-Risk Cluster Characterization

**Missing outputs:**
- `multi_risk_clusters.csv`
- Manual inspection notes

**Implementation:**
```python
def analyze_multi_risk_clusters(df_metrics, cluster_files_dir):
    """Identify clusters containing >1 unique risk"""
    results = []
    for _, row in df_metrics.iterrows():
        clusters = load_cluster_file(Path(row['cluster_filepath']))
        for cid, members in clusters.items():
            risks = [m for m in members if node_attrs[m]['category'] == 'risk']
            unique_risks = len(set(node_attrs[r]['name'] for r in risks))
            if unique_risks > 1:
                results.append({
                    'edge_config': row['edge_config'],
                    'cluster_id': cid,
                    'n_risks': unique_risks,
                    'risk_names': list(set(...)),
                    'cluster_size': len(members)
                })
    return pd.DataFrame(results)
```

**Manual inspection:** Sample 10-20 multi-risk clusters, categorize as:
- Legitimate aggregation (related risks)
- Over-aggregation (unrelated risks clustered)
- Extract patterns for paper

---

### Substep #11: Risk Diversity Per Configuration

**Missing outputs:**
- `umap_risks.png` (UMAP projection colored by risk cluster)
- `risk_diversity_stats.csv` (frequency, cluster centers, distribution skew)

**Implementation:**
```python
def analyze_risk_diversity(embeddings, risk_labels):
    """UMAP visualization + diversity statistics"""
    # UMAP projection
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    # Plot colored by cluster
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=risk_labels, cmap='tab20')
    
    # Statistics
    stats = {
        'risk_name': [...],
        'frequency': [...],  # Count per risk type
        'is_cluster_center': [...],  # Closest to centroid
        'gini_coefficient': ...,  # Distribution uniformity
    }
    return stats

# Expected: Power-law distribution (few frequent risks, many rare)
```

---

### Substep #13: Intervention Maturity Distribution (+ Substep #31 merged)

**Root cause:** Line 1202 uses global counts, not per-cluster
```python
# WRONG (current):
mode_data[stage].append(lifecycle_counts[stage]['all'] / len(MODES))

# CORRECT:
# Count interventions in THIS cluster for THIS config/mode
```

**Missing outputs:**
- `maturity_distribution_heatmap.png` (per cluster, not global)

**Clarifications:**
- Lifecycle from intervention node attributes (local graph metadata)
- Plot should show variation across clusters (currently all identical)
- **Algorithm:** Louvain only (same as rest of CSV)
- **Question:** Maturity distribution per cluster, not integrated over all

**Fixed implementation:**
```python
def plot_lifecycle_distribution_per_cluster(df_metrics, cluster_files_dir, node_attrs):
    """Plot lifecycle for each CLUSTER, not overall"""
    
    for edge_config in EDGE_CONFIGS:
        for mode in MODES:
            # Get clusters for this config
            row = df_metrics[...filter...]
            clusters = load_cluster_file(row['cluster_filepath'])
            
            # Count lifecycle per cluster
            cluster_lifecycles = []
            for cid, members in clusters.items():
                interventions = [m for m in members 
                               if node_attrs[m]['category'] == 'intervention']
                lifecycle_dist = count_by_lifecycle(interventions, node_attrs)
                cluster_lifecycles.append(lifecycle_dist)
            
            # Plot heatmap: clusters × lifecycle stages
            # Now shows variation across clusters
```

**Merge from Substep #31:** Remove duplicate, consolidate unique items to #13.

---

### Substep #15: Category-Specific Mechanism Families

**Missing outputs:**
- `umap_concepts.png` (UMAP for each concept category)
- `category_mechanism_families.csv` (cluster characterization per category)

**Implementation:**
```python
def analyze_category_mechanisms(df_metrics, cluster_files_dir):
    """UMAP + families for each concept category"""
    
    for category in ['problem_analysis', 'theoretical_insight', 
                     'design_rationale', 'implementation_mechanism',
                     'validation_evidence']:
        # Filter to this category
        subset = df_metrics[df_metrics['node_type'] == category]
        
        # UMAP visualization
        plot_umap_category(subset, category)
        
        # Extract mechanism families (top clusters by size/centrality)
        families = extract_mechanism_families(subset, top_n=20)
        # columns: cluster_id, size, exemplar_name, top_terms, description
```

---

### Substep #16: Mechanism Transfer Betweenness

**Issues identified:**

**1. Plot order reversed:**
```python
# Sort descending before plotting
df_betweenness = df_betweenness.sort_values('betweenness', ascending=False)
```

**2. Decimal precision excessive:**
```python
# Format betweenness in millions with 3 significant digits
betweenness_millions = betweenness / 1e6
ax.text(..., f'{betweenness_millions:.3g}M')  # Not .4f
```

**3. Category matching check:**
Validate examples:
- "AI governance standards" → design_rationale ✓
- "Competitive funding for AI safety" → implementation_mechanism ✓
Need systematic validation of all top-20 nodes.

**4. Missing 6th category plot:**
**Answer:** Only 5 concept categories + risk/intervention = 7 total. No 6th category exists.
Plot 6 empty because indexing goes beyond available categories.

**5. Terminal validation nodes:**
**Root cause:** Extraction allowed incomplete pathways (validation→∅ without reaching intervention).
**Fix:** Filter nodes to only those appearing in COMPLETE risk→intervention pathways.
```python
# Only include nodes from complete pathways
complete_pathway_nodes = extract_nodes_from_complete_pathways()
df_betweenness = df_betweenness[df_betweenness['node_id'].isin(complete_pathway_nodes)]
```

**Actionable insight:**
- Identify trivial/general nodes (high betweenness but low specificity)
- Example: "Safety standards" applies to many paths → not distinctive mechanism
- Flag nodes with betweenness >> mean as potential "hub artifacts" for manual review

---

### Substep #9: Mode Impact on Clustering Quality

**Missing outputs:**
- `edge_density_heatmap.png`
- `mode_stability_heatmap.png`  
- `mode_comparison_stats.csv`

**Updated interpretation:**
- **Weak silhouette dependency:** Mode changes silhouette <0.05 (noise level)
- **Strong EDGE% signal:** Mode affects validation 27%→91% (actionable)
- **Risk hubs drive difference:** Unconstrained includes multi-risk hubs → lower EDGE%

**Implementation:**
```python
def analyze_mode_impact(df_metrics):
    """Edge density + stability + comparison stats"""
    
    # Edge density: Count edges per mode/threshold
    density = calculate_edge_density_by_mode(df_metrics)
    
    # Stability: ARI between modes at same threshold
    stability = calculate_mode_stability_ari(df_metrics)
    
    # Comparison stats
    stats = df_metrics.groupby('mode').agg({
        'silhouette_mean': ['mean', 'std'],
        'edge_validation_mean': ['mean', 'std'],
        'n_clusters': ['mean', 'std']
    })
    
    return density, stability, stats

# Heatmaps: modes × edge configs
# Stats CSV: Direct comparison table
```

**Recommendation update:**
"Mode constraints have minimal impact on cluster quality (silhouette Δ<0.05) but major impact on literature grounding (EDGE% Δ64%). Select mode based on EDGE validation, not silhouette."

---

## CHANGE #4: Additional Missing Plots

**Priority:** MEDIUM  
**Status:** ❌ Not Generated

### Required Plots

1. `node_migration_heatmap.png` (Plot 20)
2. `path_length_sensitivity.png` (Plot 4)
3. `maturity_distribution_heatmap.png` (Plot 19)
4. `edge_density_heatmap.png` (Plot 21)
5. `mode_stability_heatmap.png` (Plot 22)

Will add implementation details as these are prioritized.

---

## CHANGE #5: UMAP Visualizations

**Priority:** LOW (Enrichment substeps)  
**Status:** ❌ Not Generated

### Required Plots

1. `umap_risks.png` (Plot 15)
2. `umap_interventions.png` (Plot 16)
3. `umap_concepts.png` (Plot 17)

Will add implementation details if needed.

---

## Implementation Priority Order

1. **ARI plot redesign** (CHANGE #1) - Improves key workshop figure
2. **Source diversity fix** (CHANGE #2) - Unblocks critical substep
3. **Hub quality metrics** (CHANGE #3) - Unblocks critical substep
4. **Missing plots** (CHANGE #4) - Completes Step 2 deliverables
5. **UMAP plots** (CHANGE #5) - Optional enrichment

---

**END OF CODE CHANGES TRACKER**
