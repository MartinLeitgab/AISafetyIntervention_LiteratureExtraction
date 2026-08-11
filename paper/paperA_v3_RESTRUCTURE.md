# Paper A v2 → v3: restructure against ML-paper-writing best practice

**Done 2026-08-11.** Sources worked through:

| Guide | Used for |
|---|---|
| Nanda, *Highly Opinionated Advice on How to Write ML Papers* (AlignmentForum) | claims-first narrative, evidence standards, baselines, red-teaming, figure policy, time allocation |
| ICML 2022 *Best Practices* | claims↔evidence matching, reproducibility, compute disclosure, dataset/licence, metric definition |
| Foerster, *How to ML Paper* | abstract X/Y/Z/1 formula, contribution statement, filler deletion, tense, `\citet`/`\citep`, hyperref/cleveref |
| Perez, *Easy Paper Writing Tips* | sentence-level prose rules, figure legibility, first-page figure |
| Nanda/Perez, *Tips for Empirical Alignment Research* | prioritisation, "every experiment is a win", plotting, honest write-up |

**Verdict: substantial restructure, not a rewrite.** The evidence base, the receipt
discipline and the honesty of v2 were already strong. What v2 lacked was a
claims-first spine, any figure at all, a stated position on baselines, and several
ICML hygiene items. Section content mostly survived; the frame around it changed.

---

## 1. The narrative change (the big one)

**v2** presented five unranked "contributions" — pipeline, dataset, descriptives,
retrieval demo, practical guidance. A reader could not say what the paper claimed.
Nanda: *"a paper is a narrative built on 1–3 specific concrete claims."*

**v3** states a one-sentence thesis and three claims, each owning one section:

> For a corpus of this kind, **the per-paper reasoning chain is a reliable unit of
> analysis and the aggregate graph is not.**

- **C1 — the chains carry argument structure** (§C1): 87.4% all-five-stages, 4–16
  length spread, judge study showing omission ≫ fabrication.
- **C2 — chains are a usable unit, and reading at chain level exposes what the
  corpus holds** (§C2): retrieval demo, maturity profile, corpus yield.
- **C3 — the aggregate graph does not support graph-analytic inference** (§C3):
  four measured artefacts + controls, two failed reproductions.

This unifies what were previously two stapled-together halves (a resource paper
and a negative-results paper) into one argument where each half is evidence for
the same thesis. Title changed to express it:
*"Reasoning Chains, Not Knowledge Graphs: …"*.

---

## 2. New content

### 2.1 Figure 1 — was a `[FIG:]` marker, now exists

`graph_analysis/experiment_figure1.py` → `plots/figure1_dataset.{png,pdf}`,
three panels spanning both columns on page 1 (Foerster + Perez both require a
page-1 figure; Nanda says spend as much time on figures as on the whole rest of
the paper).

- **A** chain-length distribution, canonical 7-node chain highlighted
- **B** intervention maturity, 2.5% deployed annotated
- **C** corpus yield, 11,779 documents -> 1,868 with a complete chain (the honesty panel)

Plots **only from committed receipts**, so it regenerates without the 3.2 GB PKLs.
Okabe-Ito palette (colour-blind safe), white background, axis text at body size.

### 2.2 Scope composition — drafted, then WITHDRAWN

I added a scope-composition result (11.6% of chains fitting no harm or mechanism
family; 30.5% whose risk is an ML capability gap rather than a human harm),
aggregated by `graph_analysis/experiment_scope_composition.py` from
`phase2_routing_assignments.jsonl`.

**Withdrawn on Martin's rule that Paper B material enters Paper A only if the
analysis code is committed, reproducible from GitHub, and relevant.** It fails
the first two:

| File | Git status |
|---|---|
| `graph_analysis/phase2_step5_opus_routing.py` | TRACKED |
| `phase2_routing_assignments.jsonl` | **UNTRACKED** |
| `phase2_routing_active_catalog.json` | **UNTRACKED** |

Neither the assignment file nor the catalogue it depends on is committed, and
re-deriving them costs on the order of a full Opus routing run against a
catalogue that is a separate paper's contribution. A reader could not check the
number. Removed from the abstract, the C2 section, the conclusion, Figure 1
panel C, Appendix G's cross-reference, and the claim audit; the audit is back to
**42/42**. The script and its receipt stay on disk for the Paper-B line, and both
the `.tex` and the audit script carry a comment saying what was withdrawn and the
condition for reinstating it.

Figure 1 panel C now shows **corpus yield** instead — 11,779 documents → 1,868
(15.9%) yielding a complete chain → the 2,772 chains in panel A — computed
entirely from tracked path files plus the claim-audit receipt.

### 2.3 §Reproducibility and Release — new section (ICML)

Claim→script→receipt discipline, release + licence, rebuildable checkpoints,
and a **Compute** paragraph naming APIs and pinned model identifiers. v2 had all
of this in `REPRODUCE.md` and none of it in the paper.

### 2.4 Baselines — stated position where v2 was silent

Nanda: *"It's not enough to just have baselines; you must strive to have the
strongest possible baselines."* This paper has **none**, and v2 never said so.
v3 adds:

- §Limitations "No comparative baseline" — names the three comparisons not run
  (flat triple extraction, abstract-only, non-reasoning model) and no ablation,
  and tells the reader to read C1 as evidence of structure rather than of
  necessity.
- §C1 now separates two failure modes and names the controls that test the right
  one. **Corrected 2026-08-11 after Martin's challenge**, which was right:

  My first draft proposed running the pipeline over documents that are *not*
  about AI safety. That is not a control for C1. Any literature that names a
  problem, analyses it and proposes a remedy contains genuine
  risk-to-intervention arguments, so a chain extracted from a materials-science
  paper is *correct*, not spurious. A high emission rate out of domain would show
  the schema is domain-general — a **scope** property, arguably a feature — while
  C1 is a **fidelity** claim about whether the structure is read from the source
  or imposed on it. The two were conflated.

  The controls that actually test fidelity hold the domain fixed:
  1. **Schema ablation** — re-extract with a prompt that does not name the five
     stages, then measure how often the emergent chain maps onto them. Structure
     surviving un-prompted is read, not imposed. This doubles as the missing
     ablation.
  2. **Degraded-source control** — re-extract from sentence-shuffled text,
     abstract-only, or reference-list-only versions of the *same* safety
     documents. Same vocabulary, no intact argument. A similar rate of complete
     chains would show confabulation from topical vocabulary.

  §C1 also now states **what C1 does not claim**: nothing about whether the
  extracted argument is correct, whether the intervention works, or whether the
  paper is good safety research. A faithful extraction from a weak paper is a
  success.

  Appendix G is re-framed accordingly: the Euclid chain is a *fidelity* failure
  (the model invented a "prime scarcity" risk the source never asserts), not a
  scope failure, and it is an existence proof rather than a rate.

🔴 This is a real research gap, not a writing gap. See §5 below.

### 2.5 Post-hoc disclosure (Nanda)

§C3 opens by declaring its analyses post-hoc re-derivations of an earlier pass,
and tells the reader to discount accordingly.

### 2.6 Dual-use paragraph

Short, concrete: the likely harm is someone quoting cluster sizes or centrality
rankings off the released graph as facts about the field. Points at §C3.

---

## 3. Prose and mechanics

| Fix | Detail |
|---|---|
| **Abstract rewritten** | v2 contained a **broken duplicated clause** ("a keyword pattern occurring in 2.2% of the corpus appears in 2.2% of the corpus…"). New abstract follows Foerster's X/Y/Z/1: field → why it is hard → what we did → the three claims with numbers. |
| Contributions | 5 vague items → 3 labelled claims (**C1/C2/C3**) with the evidence named inline. |
| `hyperref` + `cleveref` | added; every `§Foo`-style manual cross-reference replaced with `\cref{}`. 22 labels / 47 references, all resolving. |
| `booktabs` | every table switched from `\hline` to `\toprule/\midrule/\bottomrule`. |
| Passive voice | removed throughout ("The model is instructed to" → "The model traces"; "Extraction quality is assessed" → "We assess extraction quality"). |
| Filler | deleted "It is worth stating", "Note that", "however", "in order to", hedges. |
| Long sentences | split; semicolon-chains broken into one-idea sentences. |
| Terminology | one term per concept (chain / stage / path set), no synonym drift. |
| Reproducibility count | stays **42/42** — three scope checks were added then removed with the withdrawn claims (§2.2). |

---

## 4. Verification run this session

```
graph_analysis/experiment_figure1.py               → figure1_dataset.{png,pdf}
graph_analysis/experiment_paper_claim_audit.py     → 42/42 PASS, 0 FAIL
scratchpad/texlint.py paper/paperA_draft_v2.tex    → PASSED, 0 blocking issues
git ls-files on the routing inputs                 → assignments + catalogue UNTRACKED
```

The audit is back to its v2 count of 42 after the three scope checks were removed
with the claims they backed. Figure 1 depends on no untracked file.

🔴 **The manuscript has not been compiled.** No LaTeX toolchain is installed on
this machine. `texlint.py` checks environment balance, label/reference
resolution, citation keys against `refs.bib`, brace balance, single
`\bibliography`, and graphics presence — the errors a compiler hits first — but it
is not a compile. **First action on Overleaf: pull and build**, and check that
`figure1_dataset.png` lands where `\includegraphics` expects it.

---

## 5. What the guides say is still missing

Ordered by how much a reviewer would care.

1. 🔴 **The two fidelity controls** (§2.4): schema ablation, and the
   degraded-source control. Both hold the domain fixed and both are cheap
   (~100–200 re-extractions each). The schema ablation is the higher-value of the
   two because it doubles as the missing ablation. Needs extraction budget +
   authorisation. *Not* the out-of-domain run I first proposed — see §2.4 for why
   that tests scope rather than fidelity.
2. 🔴 **Human adjudication** (Sai, #150). The validation chain is LLM-internal end
   to end; every guide flags this shape of claim.
3. **Appendix A** — the extraction prompt is still a `[GAP:]`. ICML explicitly
   requires the prompt/algorithm be reproducible from the paper.
4. **Related Work** — five strands drafted and all citations resolve; needs one
   read for flow, plus the two open `[CITE:]` markers (AI Safety Atlas, PICO).
5. **A second worked failure case** with a missing stage (Appendix G note).
6. **Author list + contribution statement** (gate G15) and **compute-donor
   consent** (gate G14) — both unchanged, both still blocking submission.

---

## 6. Files touched

```
paper/paperA_draft_v2.tex                        rewritten (v3)
paper/figure1_dataset.{png,pdf}                  new
paper/paperA_v3_RESTRUCTURE.md                   this file
graph_analysis/experiment_figure1.py             new
graph_analysis/plots/figure1_dataset.{png,pdf}   new
graph_analysis/experiment_paper_claim_audit.py   unchanged count (42/42); carries a
                                                 comment on the withdrawn checks
graph_analysis/experiment_scope_composition.py   written, NOT used by Paper A
phase2_results/experiment_scope_composition_report.json   receipt, NOT used by Paper A
../AISafetyIntervention_PaperA/{main.tex,figure1_dataset.*,NEXT_STEPS.md,
                                paperA_v3_RESTRUCTURE.md}     synced
```

`paperA_v2_AUDIT.md` is unchanged and still valid: it audits the *numbers*, and no
number changed in this restructure.
