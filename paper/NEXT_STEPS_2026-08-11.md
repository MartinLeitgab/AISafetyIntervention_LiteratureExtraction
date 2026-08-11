# Paper A — where we are and what's next

**State as of 2026-08-11 (draft v3).** This is the pick-up-here document. Update the checkboxes as
things land; if you're a fresh session, read this first, then `paperA_v3_RESTRUCTURE.md` for what
changed and why, `paperA_v2_GAPS.md` for gap detail and `paperA_v2_AUDIT.md` for what was verified.

**v3 (2026-08-11)** restructured the draft against five ML-paper-writing guides (Nanda, ICML 2022,
Foerster, Perez, empirical-alignment tips). Claims-first spine (C1/C2/C3), Figure 1 built,
reproducibility + baselines sections added, abstract rewritten (it had a broken duplicated clause),
and the fidelity-control argument corrected. No number changed. A scope-composition result was
drafted and withdrawn because its input file is untracked. Full account: `paperA_v3_RESTRUCTURE.md`.

---

## Where things stand

| Artifact | State |
|---|---|
| `paper/paperA_draft_v2.tex` | **Complete draft, v3.** Conference-free two-column `article`. Every number carries a `% SRC:` comment. 42/42 claims re-derive |
| `paper/figure1_dataset.{png,pdf}` | **Figure 1 built** — chain length / maturity / corpus yield, from committed receipts only |
| `paper/paperA_v3_RESTRUCTURE.md` | what v3 changed, mapped guide-by-guide, + what is still missing |
| `paper/refs.bib` | 16 entries, all verified against source; every cited key resolves |
| `paper/REPRODUCE.md` | claim → script → receipt map + one-command check |
| `paper/paperA_v2_AUDIT.md` | audit of every claim against raw data (42/42 numeric PASS) |
| `paper/paperA_v2_GAPS.md` | gap list incl. gates G14 (compute donor) + G15 (authorship) |
| `paper/gleb_message_2026-08-11.md` | drafted, **not sent** — needs the Overleaf link |
| `paper/ticket_147_rewrite_proposal.md` | superseded; issue rewrite already executed |
| PR **#149** | open, **not merged** — the reproducibility chain |
| Issue **#147** | closed as superseded |
| Issue **#150** | open, assigned to Sai — 5 human-only items |
| Repo `AISafetyIntervention_PaperA` | created, private, Overleaf-synced (`main.tex`, `refs.bib`) |
| Authorship working notes | 🔒 `C:\Users\malei\paperA_private\AUTHORSHIP_WORKING_NOTES_DO_NOT_COMMIT.md` — never commit |

**Critical path: Sai's two items (#150) are the only genuine blockers.** Everything else below is
Martin's and totals under two days.

---

## 1. Immediate

- [ ] 🔴 **Pull in Overleaf and compile v3.** The manuscript has NOT been compiled — no LaTeX
      toolchain on this machine. A static lint (environments, labels/refs, citation keys, braces,
      single `\bibliography`, graphics present) passes with 0 blocking issues, but that is not a
      build. New in the preamble: `hyperref` + `cleveref` + `booktabs` rules. New on page 1:
      `\includegraphics{figure1_dataset.png}` inside a `figure*` — confirm the file uploads and the
      float does not collide with the abstract.
- [ ] **Decide on the two fidelity controls** (item 0 below). They are the one thing a reviewer will
      ask for that no amount of rewriting supplies.

## 2. Martin's read-through (in progress)

Sections worth specific attention:

- [ ] **§Practical Guidance** — Gleb's reframed section; the part he will react to. Check nothing
      mis-states his work.
- [ ] **The 88% paragraph** — rewritten around the reproduction failure. Confirm the framing before
      Gleb sees it.
- [ ] **Appendix G** — the infinite-primes failure case. Deliberately unflattering to the pipeline;
      confirm it stays.
- [ ] **§Pathway Dataset "What this is not"** — states only 15.9% of the corpus yields a complete
      chain. Honest, and the number a reviewer will quote back.

## 3. Outreach — after the read-through

- [ ] **Send the Gleb message** (`gleb_message_2026-08-11.md`); fill `<OVERLEAF LINK>`. Locks the
      Aug 17–28 sprint and points him at PR #149 to verify numbers himself.
- [ ] **Ask Axel one question:** was `--local` him? It closes his authorship tier. Same message can
      offer him the Related Work section (it's built from his literature collection).
- [ ] **Reach out to Sai** on #150 — the 50-instance error taxonomy and the ~20-paper human-anchored
      spot-check. These are the blockers.
- [ ] **Merge PR #149** once satisfied.

## 4. Martin's work items (~1 day)

- [ ] 🔴 **Item 0 — the two fidelity controls.** C1 claims the chain structure is *read from* each
      paper rather than *imposed by* the schema. Nothing in the draft tests that. Two cheap
      experiments do, and both hold the domain fixed:
      **(a) schema ablation** — re-extract ~100–200 papers with a prompt that does not name the five
      stages, then measure how often the emergent chain maps onto them (this doubles as the missing
      ablation); **(b) degraded-source control** — re-extract from sentence-shuffled / abstract-only /
      reference-list-only versions of the same papers and see whether complete chains still appear.
      Needs extraction budget + an explicit go-ahead.
      🔴 Do **not** run the out-of-domain version I first proposed (extract from non-safety papers):
      any problem→solution literature yields genuine chains, so that measures corpus scope, not
      fidelity. See `paperA_v3_RESTRUCTURE.md` §2.4.
- [x] **Figure 1** — built: `graph_analysis/experiment_figure1.py` → `paper/figure1_dataset.png`,
      three panels (chain length / maturity / corpus yield), plots only from committed receipts.
- [ ] **Appendix A** — paste `PROMPT_EXTRACT` from
      `intervention_graph_creation/src/prompt/final_primary_prompt.py`, trim to one page. ICML lists
      the prompt/algorithm as a reproducibility requirement, so this is no longer cosmetic.
- [ ] **Related Work final pass** — GraphRAG/LLM+KG strand written with verified citations; ARD, MIT
      Risk Repository, SciERC and LLM-as-a-judge now cited. Needs a read, not a rewrite.
- [ ] **Dataset/code release artifact** — the abstract promises one and none exists. Needs a hosted
      dump (graph + path set + extraction code) and a stable link.

## 5. 🔴 Gates — decisions only Martin can make

- [ ] **G14 — compute-donor consent.** Credits came from a private acquaintance and may have been
      issued under terms that didn't contemplate transfer. Ask: named / anonymous / omitted.
      Default if unreachable: anonymous. Gate text sits in the `.tex` §Acknowledgments — do not
      delete that comment until closed.
- [ ] **G15 — freeze the author list.** Working notes are private (path above). Open sub-items:
      confirm Axel's `--local`, decide Tier 2 invitations, draft the CRediT contribution statement.
      Decided already: Martin last author (+ recommend corresponding author); Gleb + Mike shared
      first.
- [ ] **Venue.** Nothing committed. The draft is deliberately venue-neutral two-column so it can go
      to a workshop or straight to arXiv without rework. Ten minutes on the first sprint call.

## 6. Data still owed to the authorship analysis

- [ ] Full SOAR Discord history + the subthreads (on-disk logs are partial).
- [ ] Month-1 SOAR cohort list vs later joiners (Mike joined after month 1).
- [ ] Overleaf revision history — the only record of who actually wrote.

---

## Fallback if Sai slips

Submit with the validation section scoped to what the judge data already supports. The receipt
(`experiment_judge_full_report.json`) already carries the judge audit, the meta-grader table with
honest per-grader *n*, Fleiss κ, the auto-derived error profile and the recovery result. What would
be missing is the manual taxonomy and the human anchor — both would be stated as limitations, which
the draft already does. This fallback is written into the Gleb message.

## Things a fresh session must not re-derive wrongly

- 🔴 `git blame` is useless on this repo (2026-03-08 bulk commit `6e1632f` rewrote the pipeline).
- 🔴 The 88% race figure **does not reproduce** (2.6% re-derived) — do not reinstate it.
- 🔴 "~60 recovered of ~400" is **wrong** (disjoint populations); the figure is 23/441 = 5.2%.
- 🔴 The reporting unit is the **2,772 de-duplicated** chain set, not the raw 8,954.
- 🔴 Intervention maturity is **LLM-assigned and un-adjudicated** — the judge study does not score
  it. Do not promote it to a measured rate.
- 🔴 **Paper B material stays out of Paper A** unless the producing code *and* its data are committed
  and a reader can re-derive the number. The scope-composition result (11.6% / 30.5%) was drafted
  into v3 and withdrawn on exactly this test: `phase2_routing_assignments.jsonl` is untracked.
- 🔴 The paper's thesis is **"the chain is the reliable unit, the aggregate graph is not."** C3 is a
  contribution, not a caveat — do not soften it back into a limitations paragraph.
- 🔴 Nothing private goes anywhere under `0_project_work` — that tree is a git repo pointing at
  `github.com/AI-Plans/FairCoder`.
