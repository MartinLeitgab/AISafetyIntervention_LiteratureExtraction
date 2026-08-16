"""
phase2_step4_apply_hybrid_metafamily_names.py

Applies hybrid naming strategy to all PathbuildB meta-family name fields:
  - For 12 meta-families whose dominant B-family is in top-40 (cluster_id 0-39):
    use the dominant B-family's v3 LLM name (causal, high-confidence)
  - For the remaining 20 meta-families:
    use the second-pass LLM meta-family name from pathbuildB_metafamily_names_llm.csv

Updates:
  step4_cluster_tables/pathbuildB_metafamily_summary_consim1.csv  — adds final_name column
  step4_connectivity/ri_meta_triplets_consim1.csv                 — patches meta_family_name
  step4_connectivity/ri_meta_triplets_top20_consim1.csv           — patches meta_family_name
"""

import pandas as pd
from pathlib import Path

BASE = Path("graph_analysis/phase2_results/step4_finalanalysis")
NAMING_DIR = Path("graph_analysis/phase2_results/step5_naming")

# ── Load inputs ─────────────────────────────────────────────────────────────

summ = pd.read_csv(
    BASE / "step4_cluster_tables/pathbuildB_metafamily_summary_consim1.csv"
)
v3 = pd.read_csv(NAMING_DIR / "pathbuildB_chain_names_llm_v3.csv")
meta_llm = pd.read_csv(NAMING_DIR / "pathbuildB_metafamily_names_llm.csv")

# v3 map: cluster_id (0-39) → final_name
v3_map = dict(zip(v3["cluster_id"].astype(int), v3["final_name"].astype(str)))

# second-pass LLM map: meta_family_id → final_name
meta_llm_map = dict(
    zip(meta_llm["meta_family_id"].astype(int), meta_llm["final_name"].astype(str))
)

# ── Build hybrid name map ────────────────────────────────────────────────────

hybrid_map = {}  # meta_family_id → final_name
source_map = {}  # meta_family_id → source label

for _, row in summ.iterrows():
    mf_id = int(row["meta_family_id"])
    dom_id = int(row["dominant_family_id"])

    if dom_id in v3_map:
        # dominant B-family is top-40 — use v3 name
        hybrid_map[mf_id] = v3_map[dom_id]
        source_map[mf_id] = f"v3_dominant_B{dom_id}"
    else:
        # use second-pass LLM meta-family name
        hybrid_map[mf_id] = meta_llm_map.get(mf_id, str(row["dominant_family_name"]))
        source_map[mf_id] = "second_pass_llm"

print("Hybrid name map:")
for mf_id in sorted(hybrid_map):
    src = source_map[mf_id]
    print(f"  MF{mf_id:2d} [{src}]: {hybrid_map[mf_id]}")

# ── Update summary table ─────────────────────────────────────────────────────

summ["final_name"] = summ["meta_family_id"].map(lambda x: hybrid_map.get(int(x), ""))
summ["final_name_source"] = summ["meta_family_id"].map(
    lambda x: source_map.get(int(x), "")
)
out_summ = BASE / "step4_cluster_tables/pathbuildB_metafamily_summary_consim1.csv"
summ.to_csv(out_summ, index=False)
print(f"\n✅ Updated: {out_summ}")

# ── Patch triplet files ──────────────────────────────────────────────────────

for fname in ["ri_meta_triplets_consim1.csv", "ri_meta_triplets_top20_consim1.csv"]:
    fpath = BASE / "step4_connectivity" / fname
    df = pd.read_csv(fpath)
    before = df["meta_family_name"].tolist()
    df["meta_family_name"] = df["meta_family_id"].map(
        lambda x: hybrid_map.get(int(x), "MISSING")
    )

    # Sanity check: count any MISSING
    n_missing = (df["meta_family_name"] == "MISSING").sum()
    n_changed = sum(a != b for a, b in zip(before, df["meta_family_name"].tolist()))

    df.to_csv(fpath, index=False)
    print(f"\n✅ Patched: {fname}")
    print(f"   Rows: {len(df)} | Changed: {n_changed} | Missing: {n_missing}")

# ── Print summary table sorted by path count ────────────────────────────────

print("\n═══ FINAL META-FAMILY NAMES (by path count) ═══")
for _, r in summ.sort_values("n_paths_total", ascending=False).iterrows():
    src_flag = "▸v3" if "v3" in r["final_name_source"] else "▸llm"
    print(
        f"  MF{int(r['meta_family_id']):2d} {src_flag} ({int(r['n_paths_total']):6d} paths): {r['final_name']}"
    )

# ── Verify no leftover placeholder-style names ────────────────────────────────

tri = pd.read_csv(BASE / "step4_connectivity/ri_meta_triplets_consim1.csv")
b_prefix = tri[tri["meta_family_name"].str.startswith("B", na=False)]
via_names = tri[tri["meta_family_name"].str.lower().str.startswith("via", na=False)]
print("\n✅ Verification — ri_meta_triplets_consim1.csv:")
print(f"   B-prefix names remaining: {len(b_prefix)}")
print(f"   'via'-prefix names remaining: {len(via_names)}")
