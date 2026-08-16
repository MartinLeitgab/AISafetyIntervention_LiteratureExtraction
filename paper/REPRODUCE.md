# Reproducing every number in the Paper A draft

Each quantitative claim in `paper/paperA_draft_v2.tex` carries a `% SRC:` comment naming the receipt
JSON it came from. This file maps claim → script → receipt so anyone can re-derive them.

All scripts are **Class B** (no LLM calls, no API keys, no network) and fail fast on missing inputs.
Run them from `graph_analysis/`.

## Inputs you need

| Input | Where | In this repo? |
|---|---|---|
| `paths_hopwise_v4_edge_only.jsonl` (8,954 raw chains) | `graph_analysis/phase1_rawpathsfiles/` | ✅ committed |
| `paths_hopwise_v4_edge_only_deduped.jsonl` (2,772 chains) | same | ✅ committed |
| `graph_node_attributes.pkl` (200,525 nodes) | `graph_analysis/phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/` | ❌ too large — rebuild with `phase2_step1_loadandparse.py` against the FalkorDB dump, or ask the corresponding author |
| `graph_edge_data.pkl` (1,767,833 edges) | same | ❌ same |
| 100 judge reports | branch `anthropic_judge_test`, `extraction_validator/extend_try_1/` | ✅ in git, different branch |
| Meta-grader archive | Drive folder `judge_material` | ❌ Drive only |
| Judge recovery bundle | Drive `judge_material/judge_recovery_bundle_item3.zip` | ❌ Drive only |

The **receipts are committed**, so every number in the paper is checkable without the two PKLs or the
Drive archives. The PKLs are only needed to re-derive the receipts from scratch.

## Claim → script → receipt

| Paper section | Claims | Script | Receipt |
|---|---|---|---|
| Methods, structural diagnostics + SIM layer | 15,123 components; largest 61; degree 2.02; clustering 0.013; 1,435,806 SIM edges → 4,124 components, largest 152,753 | inline in `experiment_paper_claim_audit.py` (union-find) | — |
| §Pathway Dataset | 2,772 chains; 87.4% all-five; length distribution; 1,868 papers; 2,643 R→I pairs; maturity 27.6/57.3/12.7/2.5 | `experiment_dataset_strength_descriptives.py`, `experiment_paper_claim_audit.py` | `experiment_dataset_strength_report.json`, `experiment_paper_claim_audit.json` |
| §Mechanism-Level Retrieval | the three worked chains | `experiment_query_demo.py` | `experiment_query_demo_report.json` |
| §Guidance — merge artefact | EC 90.1× / 33.6× / 31× flattening; 4,066 members, 7,777 edges | `experiment_merge_vs_simexcl_ec.py` | `experiment_merge_vs_simexcl_ec_report.json` |
| §Guidance — merge recall | 54,282 vs 4,411 candidate pairs; 1,140 residual exact-name dups | `experiment_J6_merge_approx_v2_ticketspec.py` | `experiment_J6_merge_approx_v2_report.json` |
| §Guidance — race prevalence | 2.21% of PA nodes; 1.65% of risks; 0.50% of chains | `experiment_race_prevalence_scan.py` | `experiment_race_prevalence_report.json` |
| §Guidance — race odds ratios | 3.82 / 2.73 / 5.17 across units | `experiment_race_dedup_robustness.py` | `experiment_race_dedup_robustness_report.json` |
| §Guidance — the 88% reproduction failure | 38 single-path in top-100; 2.6% race-framed; 2.0× gradient | `experiment_race_top100_rederive.py` | `experiment_race_top100_rederive_report.json` |
| §Methods, Validation | judge audit, meta-grader table, Fleiss κ, error profile, 23/441 recovery | `experiment_judge_full_receipt.py` | `experiment_judge_full_report.json` |
| **Whole draft** | **regression check of every number** | `experiment_paper_claim_audit.py` | `experiment_paper_claim_audit.json` |

> 🔴 **This copy is stale and is kept for history.** The canonical claim-to-script-to-receipt
> map is `REPRODUCE.md` in `../AISafetyIntervention_PaperA_shared`, which travels with the
> manuscript. Read that one.

## The one command that checks everything

```bash
cd graph_analysis
python -u experiment_paper_claim_audit.py
# -> 257/257 PASS, 0 FAIL   (2026-08-16)
```

It re-derives each claim from the **raw** path files and node PKL (not from the other receipts) and
compares against the value printed in the draft. If someone edits a number in the `.tex`, update the
matching `check(...)` line so this stays a live regression test.

## Rebuilding the judge receipt

```bash
git archive origin/anthropic_judge_test extraction_validator/extend_try_1 | tar -x -C /tmp/judge
cd graph_analysis
python -u experiment_judge_full_receipt.py \
    --judge-reports /tmp/judge/extraction_validator/extend_try_1 \
    --grader-archive <unzipped meta-grader archive> \
    --recovery      <unzipped judge_recovery_bundle>/data
```

Sanity check: the script must reproduce the three aggregates in
`extraction_validator/results.md` on branch `judge_handoff_workshop_items_2_3` exactly
(69.65/73.13, 62.69/95.77, 66.96/80.18). If it does not, an input is wrong.

## Known discrepancies with the frozen Overleaf analysis

Documented in `paper/paperA_v2_AUDIT.md` §P1 and §P5. In short: the structural counts differ because
the earlier numbers were measured on the merged graph with a sparser similarity layer, and the 88%
race-framing figure **does not reproduce** (2.6% re-derived). The draft prints our re-derived values.
