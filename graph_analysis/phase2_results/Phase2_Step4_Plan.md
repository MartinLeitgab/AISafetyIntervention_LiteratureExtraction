# Phase 2 Step 4/5 — VPN Fix Rerun Status

**Last updated:** 2026-04-05  
**Branch:** martin/main  
**Root fix:** VPN rebuilt from maturity>=3 endpoint paths only + sim_edge_set restricted to VPN-pair edges.

---

## Code fix status (all scripts)

All code fixes are in the working tree (dirty, not yet committed):
- VPN maturity>=3 filter: applied in all 9 Category B scripts
- sim_edge_set VPN restriction: applied in all 4 scripts that build it

---

## Rerun status

### ✅ Complete (reran with fixes, Apr 5 ~20:00–20:23)
| Script | Key outputs |
|--------|-------------|
| `phase2_step4b_paths_and_plots.py` | `optionA_kmeans_model.pkl`, `optionB_cooccurrence_families.csv`, cluster tables |
| `phase2_step4_cluster_naming.py` | `*_cluster_names.csv` in step4_cluster_tables/ |
| `phase2_step4_connectivity.py` | `risk_clusters_09.pkl`, `within_cluster_edge_density.png`, `intervention_clusters.csv`, `risk_clusters.csv` |

### ❌ Still needs rerun (pre-fix outputs, code already fixed)
| Script | Last output timestamp | Depends on |
|--------|----------------------|------------|
| `phase2_step4_pathbuildB_connectivity.py` | Apr 5 18:37–18:40 | step4b ✅ |
| `phase2_step4_umap_plots.py` | Apr 5 18:40–18:42 | step4b ✅ |
| `phase2_step4_phase_c_reruns.py` | Apr 4 15:05 | step4b ✅ |
| `phase2_step5_naming.py` | Apr 5 08:27–12:21 | connectivity ✅ |
| `phase2_step5_triplet_simreach.py` | Apr 5 08:31 | connectivity ✅ |
| `phase2_step5_examples.py` | Apr 5 08:27–11:53 | connectivity ✅ |

---

## Run order

All dependencies are satisfied. Run sequentially (each loads ~2.5 GB — parallel runs caused the prior WSL crash):

```
1. phase2_step4_pathbuildB_connectivity.py   (~5 min)
2. phase2_step4_umap_plots.py                (~10 min, UMAP is memory-heavy)
3. phase2_step4_phase_c_reruns.py            (~5 min)
4. phase2_step5_naming.py                    (~5 min, calls OpenAI API)
5. phase2_step5_triplet_simreach.py          (~3 min)
6. phase2_step5_examples.py                  (~3 min)
```

After all reruns complete:
- Update `Step4_Findings_Report.md` with corrected VPN definition and new node counts
- Update `step5_naming/Step5_Review_Summary.md`
- Commit all code changes + output file changes and push to martin/main

---

## Memory note

FalkorDB docker container NOT needed for any of the above scripts (all load from PKL files).  
Stop it before running to free ~1 CPU core: `docker stop falkordb`
