#!/usr/bin/env bash
# phase2_step4_F3_sweep_thresholds.sh [rev8 — Task #7 cross-threshold sweep]
#
# Runs F3 body recluster (k-scan + Pareto frontier) on the canonical path
# files for each of the 5 thresholds. No sampling; full Agglomerative.
# Records wall time per threshold.
#
# Hybrid path-source choice per threshold (matches §19.2 of Step 4 findings):
#   EDGE-only : DFS hop-wise   paths_hopwise_v4_edge_only.jsonl
#   sim=0.95  : DFS hop-wise   paths_hopwise_v4_sim0.95.jsonl
#   sim=0.9   : DFS hop-wise   paths_hopwise_v4_sim0.9.jsonl   (canonical)
#   sim=0.85  : DFS hop-wise   paths_hopwise_v4_sim0.85.jsonl  (max=12 cap)
#   sim=0.8   : BFS-shortest   paths_custom_sim0.8.jsonl       (DFS intractable)
#
# Output suffix per threshold: edge_only, sim0.95, sim0.9, sim0.85, sim0.8.
# Outputs land in graph_analysis/phase2_results/step4_finalanalysis/step4_cluster_tables/.
#
# Pass --order 'sim0.95 sim0.9 ...' to override default ordering.
# Default order runs cheapest-first so failures surface early before sim=0.8
# (potentially many hours of clustering).

set -e

PROJ=/c/Users/malei/0_project_work/eleutherAI_SOAR_step1knowledgegraphcreation/AISafetyIntervention_LiteratureExtraction
LOGDIR="$PROJ/graph_analysis/logfiles/phase4_logs"
mkdir -p "$LOGDIR"

declare -A PATH_FILE
PATH_FILE[edge_only]=graph_analysis/phase1_rawpathsfiles/paths_hopwise_v4_edge_only.jsonl
PATH_FILE[sim0.95]=graph_analysis/phase1_rawpathsfiles/paths_hopwise_v4_sim0.95.jsonl
PATH_FILE[sim0.9]=graph_analysis/phase1_rawpathsfiles/paths_hopwise_v4_sim0.9.jsonl
PATH_FILE[sim0.85]=graph_analysis/phase1_rawpathsfiles/paths_hopwise_v4_sim0.85.jsonl
PATH_FILE[sim0.8]=graph_analysis/phase1_rawpathsfiles/paths_custom_sim0.8.jsonl

declare -A SIM_THRESH
SIM_THRESH[edge_only]=0.9   # consim1 SIM-edge set unused for EDGE-only paths; arg required by F3 but inert
SIM_THRESH[sim0.95]=0.95
SIM_THRESH[sim0.9]=0.9
SIM_THRESH[sim0.85]=0.85
SIM_THRESH[sim0.8]=0.8

# Default ordering: smallest-input-first. sim=0.8 last (largest, may run many hours).
ORDER=(edge_only sim0.95 sim0.9 sim0.85 sim0.8)
if [ "$1" = "--order" ]; then
  shift
  ORDER=("$@")
fi

cd "$PROJ"
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_LOG="$LOGDIR/F3_sweep_thresholds_${TS}.log"
echo "Sweep log: $SWEEP_LOG"

{
  echo "================================================================================"
  echo "F3 cross-threshold body recluster sweep — $(date)"
  echo "================================================================================"
  echo "Order: ${ORDER[*]}"
  echo

  for SUFFIX in "${ORDER[@]}"; do
    PATHS="${PATH_FILE[$SUFFIX]}"
    SIM="${SIM_THRESH[$SUFFIX]}"
    if [ ! -f "$PATHS" ]; then
      echo "[$SUFFIX] SKIP: input file missing: $PATHS"
      continue
    fi
    echo "--------------------------------------------------------------------------------"
    echo "[$SUFFIX] starting at $(date +%H:%M:%S)"
    echo "  paths_file = $PATHS"
    echo "  sim_thresh = $SIM"
    echo "--------------------------------------------------------------------------------"
    START=$(date +%s)
    PYTHONUNBUFFERED=1 /c/Users/malei/anaconda3/python.exe -u \
        graph_analysis/phase2_step4_F3_body_recluster.py \
        --paths-file "$PATHS" \
        --sim-threshold "$SIM" \
        --output-suffix "$SUFFIX" \
        --hdbscan-mcs 5,10,20,50,100 \
        --umap-n-components 15 \
        --umap-n-neighbors 15 \
        --umap-min-dist 0.0 \
        --sil-threshold 0.25 \
        --cov-threshold 0.50 \
        --z-threshold 2.0 \
        --centroid-sim-cutoff 0.77 \
        --cutoff-sweep 0.70,0.72,0.74,0.76,0.77,0.78,0.80,0.82 \
        --iter-coverage-target 0.999 \
        --iter-max 50 \
        --final-min-cluster-size 5 \
        --resid-method hdbscan \
        --resid-mcs 5 2>&1

    # Run sensitivity analysis to emit vpn_strict_RIbody_<suffix>.jsonl
    /c/Users/malei/anaconda3/python.exe -u \
        graph_analysis/phase2_step4_F3a_vpn_coverage_sensitivity.py \
        --paths-file "$PATHS" \
        --memberships-pkl graph_analysis/phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/cluster_memberships_rev8_${SUFFIX}.pkl \
        --output-suffix "$SUFFIX" \
        --sim-threshold "$SIM" \
        --coverage-thresholds 1.00,0.80,0.60,0.40,0.20,0.0 2>&1

    END=$(date +%s)
    ELAPSED=$((END - START))
    echo "[$SUFFIX] complete at $(date +%H:%M:%S) — wall=${ELAPSED}s"
    echo
  done

  echo "================================================================================"
  echo "Sweep complete — $(date)"
  echo "================================================================================"
} | tee -a "$SWEEP_LOG"
