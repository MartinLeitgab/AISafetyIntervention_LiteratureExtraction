# Review response — what was implemented, 2026-08-14

Against `paper/REVIEW_neurips_paperA_altstyle_2026-08-14.md` and
`paper/REVIEW_workshop_paperA_altstyle_2026-08-14.md`.

**Manuscript:** `paperA_altstyle.tex` in `../AISafetyIntervention_PaperA_shared`, commits
`a851caf` + `4bed4d6`, pushed. **Analysis:** branch `paper/receipts-clean`, commit
`11a00c4`, pushed. Claim audit **112/112 PASS** (was 42/42). asciify + texlint clean.

## New measurements (all Class B, all with receipts)

| Reviewer item | What we now measure | Number |
|---|---|---|
| W-1 / Q1 | judged 100 vs the 1,868 chain-yielding docs | **12 papers overlap, 17 of 2,772 chains (0.6%)**; per-type yield 3.1% (Arbital) to 40.5% (arXiv) |
| W-3 / Q4 | chain set under all nine gate settings | yield **15.9% -> 99.4%**; all-five stays **82.0–89.8%**; arXiv share **38.1% -> 15.0%** |
| Q-W9 / W-11 | is the containment step lossless? | **no** — 78.3% of dropped paths carry a novel node; **6.13%** of chain-set nodes (1,169 of 19,073, 695 papers) survive in no kept chain; thresholds 0.60/0.70/0.80/0.90 -> 2,658/2,772/3,356/5,460 |
| W-4 / Q-W7 | agreement on raw grader scores | ICC(2,1) **0.921 -> 0.151**, Krippendorff alpha **0.917 -> 0.043**; kappa collapses under two of three binnings, so it is not a binning artifact |
| W-6 | silhouette scored in one space | UMAP k=40: **0.281 in UMAP space, 0.004 in the original space**, below direct k=40's 0.014. The "13–17x improvement" was an artifact |
| W-8 / Q-W11 | intake | 13,632 records in, **86.4%** reach the graph; 1,667 / 128 / 58 failure buckets |
| S-W4 | recency | **no document dated after 2023**; median year 2021 |

Verification that licenses the above: the in-memory re-enumeration reproduces the released
8,954-path and 2,772-chain files **exactly** before any relaxed row is reported; the script
exits non-zero otherwise.

## Text and structure

- Abstract: "only 7 of 100" removed, 5.02 counterweight added in the same sentence; 282 words.
- Figure 3 (data-reduction funnel, every arrow marked APPLIED / MEASURED-ONLY) and Figure 4
  (pipeline + schema) added. Figure 1 panel A no longer asserts the retracted template claim.
- Appendix A now carries the extraction prompt (condensed, ASCII, elisions marked).
- Methods: full model configuration (o3, reasoning effort medium, **no** structured-output
  constraint, retries, `text-embedding-3-small`); released dump identified against the
  earlier merged/sparser substrate; release manifest.
- Related Work: argument mining (Teufel & Moens; Lippi & Torroni; Lauscher SciArg),
  evidence synthesis (Nye EBM-NLP; Marshall & Wallace), judge bias (Panickssery; Wang).
  Both `[CITE:]` placeholders resolved (AI Safety Atlas = Grey & Segerie 2025).
- Limitations: population mismatch; gate as selection mechanism; pre/post grader confound
  + judge-family overlap; no stability run; no retrieval evaluation; corpus ends 2023.
- Impact Statement: misattribution (with three mitigations), licensing/redistribution,
  dual use, personal data.
- Arithmetic: EC1/EC2 1.03x, flattening 30.7x, grader table at two decimals, the two
  unrelated "7"s disambiguated, Fig. 1 caption softened.
- Style: "rather than" 44 -> 18 occurrences.

## Found while working — not a reviewer item

- The manuscript said every grader "records an improvement on every paper it scored".
  False: Opus improves on 91.6% of its 95. Fixed; unanimity holds only on the 13 common papers.
- The deployed clustering is **agglomerative** (cosine, average linkage) at k=40, not
  k-means. Appendix D now reports both, and agglomerative fitted in 1536D degenerates
  (92.4% of nodes in one cluster).

## Still open (nothing here can be closed without new resources or a team decision)

1. **Human audit of >=30 chains** (Q2) — the single highest-value item; human-only.
   Highlighted in Limitations as not performed.
2. **Manual 50-instance error taxonomy** with adjudication — same.
3. **Baseline run** (Q5: abstract-only / non-reasoning model on ~200 docs) and
   **repeat-extraction stability** (Q-W13) — need extraction budget.
4. **Sham-repair control** (Q6) — needs grader re-runs; the confound is now named.
5. **Retrieval evaluation** against an embedding baseline (S-W1/S-W3) — stated as a limitation.
6. **Team gates**: release URL, licence name, author list, compute-donor consent,
   AI-assistance scope, cluster-representative publication. All still rendered as
   `\OPEN{}` highlights — 9 in the built PDF.
7. **MIT-anchor of a chain subset** (S-W5) — Paper B material, deliberately out of scope.
