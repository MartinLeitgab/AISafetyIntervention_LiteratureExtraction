#!/usr/bin/env bash
# phase2_step4_F3_neighbor_cutoff_runs.sh [rev8 — Task #7 robustness sweep]
#
# After the canonical F3 run at cutoff=0.77 has completed for a given
# threshold suffix (e.g. edge_only), this wrapper:
#   1. Copies the canonical PKL to a cutoff-tagged filename (cluster_memberships_rev8_<suffix>_cutoff0.77.pkl)
#   2. Runs F3 at cutoffs 0.73, 0.75, 0.79, 0.81 with cutoff-tagged output suffixes
#   3. Runs F3b to compare ARI / cluster counts across all 5 cutoffs
#
# Usage:
#   bash phase2_step4_F3_neighbor_cutoff_runs.sh <threshold_suffix> <paths_file> <sim_threshold>
# Example:
#   bash phase2_step4_F3_neighbor_cutoff_runs.sh edge_only \
#     graph_analysis/phase1_rawpathsfiles/paths_hopwise_v4_edge_only.jsonl 0.9
#
# Prerequisites:
#   - Canonical F3 run at cutoff=0.77 has already produced
#     cluster_memberships_rev8_<suffix>.pkl

set -e

if [ $# -lt 3 ]; then
  echo "Usage: $0 <threshold_suffix> <paths_file> <sim_threshold>"
  exit 1
fi

SUFFIX=$1
PATHS=$2
SIM=$3

PROJ=/c/Users/malei/0_project_work/eleutherAI_SOAR_step1knowledgegraphcreation/AISafetyIntervention_LiteratureExtraction
PKL_DIR="$PROJ/graph_analysis/phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
LOGDIR="$PROJ/graph_analysis/logfiles/phase4_logs"
mkdir -p "$LOGDIR"

cd "$PROJ"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOGDIR/F3_neighbor_cutoff_${SUFFIX}_${TS}.log"
echo "Log: $LOG"

# Step 1: copy canonical PKL to cutoff-tagged name
canonical_pkl="$PKL_DIR/cluster_memberships_rev8_${SUFFIX}.pkl"
cutoff077_pkl="$PKL_DIR/cluster_memberships_rev8_${SUFFIX}_cutoff0.77.pkl"
if [ ! -f "$canonical_pkl" ]; then
  echo "ERROR: canonical PKL missing: $canonical_pkl"
  exit 2
fi
cp "$canonical_pkl" "$cutoff077_pkl"
echo "Copied canonical 0.77 PKL → $cutoff077_pkl"

{
  echo "================================================================================"
  echo "F3 neighbor-cutoff sweep — $SUFFIX — $(date)"
  echo "================================================================================"

  for CUTOFF in 0.73 0.75 0.79 0.81; do
    OUT="${SUFFIX}_cutoff${CUTOFF}"
    echo "--------------------------------------------------------------------------------"
    echo "[$CUTOFF] starting at $(date +%H:%M:%S)"
    echo "--------------------------------------------------------------------------------"
    START=$(date +%s)
    PYTHONUNBUFFERED=1 /c/Users/malei/anaconda3/python.exe -u \
        graph_analysis/phase2_step4_F3_body_recluster.py \
        --paths-file "$PATHS" \
        --sim-threshold "$SIM" \
        --output-suffix "$OUT" \
        --hdbscan-mcs 5,10,20,50,100 \
        --umap-n-components 15 \
        --umap-n-neighbors 15 \
        --umap-min-dist 0.0 \
        --sil-threshold 0.25 \
        --cov-threshold 0.50 \
        --z-threshold 2.0 \
        --centroid-sim-cutoff "$CUTOFF" \
        --cutoff-sweep "$CUTOFF" \
        --iter-coverage-target 0.999 \
        --iter-max 50 \
        --final-min-cluster-size 5 \
        --resid-method hdbscan \
        --resid-mcs 5 2>&1
    END=$(date +%s)
    ELAPSED=$((END - START))
    echo "[$CUTOFF] complete at $(date +%H:%M:%S) — wall=${ELAPSED}s"
    echo
  done

  echo "--------------------------------------------------------------------------------"
  echo "Comparison: ARI / cluster counts across cutoffs vs 0.77"
  echo "--------------------------------------------------------------------------------"
  /c/Users/malei/anaconda3/python.exe -u \
      graph_analysis/phase2_step4_F3b_full_pipeline_cutoff_compare.py \
      --pkl-pattern "${PKL_DIR}/cluster_memberships_rev8_${SUFFIX}_cutoff{cutoff}.pkl" \
      --cutoffs 0.73,0.75,0.77,0.79,0.81 \
      --reference-cutoff 0.77 \
      --output-suffix "$SUFFIX" 2>&1

  echo "================================================================================"
  echo "Sweep complete — $(date)"
  echo "================================================================================"
} | tee -a "$LOG"
