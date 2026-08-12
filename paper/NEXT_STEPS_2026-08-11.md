# Paper A — where we are and what's next

**State as of 2026-08-12 (draft v3, pushed).** This is the pick-up-here document. Update the
checkboxes as things land. Fresh session: read this first, then `paperA_v3_RESTRUCTURE.md` for what
changed and why, `paperA_v2_GAPS.md` for gap detail, `paperA_v2_AUDIT.md` for what was verified.

**v3 restructured the draft** against five ML-paper-writing guides (Nanda, ICML 2022, Foerster,
Perez, empirical-alignment tips): claims-first spine (C1/C2/C3), Figure 1 built, reproducibility +
baselines sections added, AI-assistance disclosure added, abstract rewritten then trimmed
285 → 231 words, and the fidelity-control argument corrected. A scope-composition result was drafted
and withdrawn (untracked input). **No number in the paper changed.** Full account:
`paperA_v3_RESTRUCTURE.md`.

---

## Where things stand

| Artifact | State |
|---|---|
| `paper/paperA_draft_v2.tex` | **Complete draft, v3.** Venue-neutral two-column `article`. Every number carries a `% SRC:` comment. 42/42 claims re-derive. Pure ASCII |
| `paper/paperA_altstyle.tex` | **Alternative-style edition** of the same paper, presentation modelled on arXiv:2506.03053v2 (MAEBE): Research-Goals subsection, findings-as-headings, Related Work after Results, combined Limitations/Outlook/Conclusions, Impact Statement, appendices A--I. Same claims; the main body was then thinned from 159 to **75 distinct result numbers** (benchmarked against arXiv:2602.06941v3, which carries ~28), everything cut moving to Appendix B (composition), G (race) or H (judge protocol). Nothing deleted but raw counts printed beside their own percentages. A presentation choice for Martin, not a replacement — pick one before submission |
| `paper/figure1_dataset.{png,pdf}` | **Figure 1 built** — chain length / maturity / corpus yield, from committed receipts only |
| `paper/texlint.py` | static manuscript check (no LaTeX toolchain here) — run before every commit |
| `paper/asciify_tex.py` | forces the `.tex` to pure ASCII; fails loudly on unmapped characters |
| `paper/paperA_v3_RESTRUCTURE.md` | what v3 changed, guide by guide, + what is still missing |
| `paper/refs.bib` | 16 entries, all verified against source; every cited key resolves; pure ASCII |
| `paper/REPRODUCE.md` | claim → script → receipt map + one-command check |
| `paper/paperA_v2_AUDIT.md` | audit of every claim against raw data |
| `paper/paperA_v2_GAPS.md` | gap list incl. gates G14 (compute donor) + G15 (authorship) |
| `paper/gleb_message_2026-08-11.md` | drafted, **not sent** — needs the Overleaf link |
| Repo `AISafetyIntervention_PaperA` | `origin/master` @ `9e6f4fc`, one branch only, Overleaf syncing cleanly |
| Analysis repo | branch `paper/reproducible-claim-chain`, well ahead of `origin/main`, PR #149 open and unmerged |
| Issue **#147** | **closed** — rewritten as #150. `paper/ticket_147_rewrite_proposal.md` is the historical record |
| Issue **#150** | open, assigned to Sai — 5 human-only items |
| Authorship working notes | 🔒 `C:\Users\malei\paperA_private\AUTHORSHIP_WORKING_NOTES_DO_NOT_COMMIT.md` — never commit |

**Critical path: Sai's two items (#150), plus item 0 below.** Everything else is Martin's and totals
about a day.

---

## 1. Immediate

- [x] ~~Overleaf shows `main.tex` as binary~~ — fixed 2026-08-12. Cause + permanent guard in
      "Operational knowledge" below.
- [x] ~~Compile on Overleaf and read the built PDF.~~ **Done 2026-08-12 — compiles and renders
      correctly** (figure float, bibliography, `cleveref` references, the AI-assistance section).
- [ ] **Decide on item 0** (the two fidelity controls). The one thing a reviewer will ask for that
      no amount of rewriting supplies.

## 2. Martin's read-through

- [ ] **§C3 Practical Guidance** — Gleb's reframed material; the part he will react to.
- [ ] **The 88% paragraph** — rewritten around the reproduction failure. Confirm before Gleb sees it.
- [ ] **Appendix G** — the infinite-primes failure case, now framed as a *fidelity* failure and an
      existence proof. Deliberately unflattering; confirm it stays.
- [ ] **§C1 "What the length spread does and does not settle"** — the corrected control argument.
- [ ] **§Reproducibility → Use of AI assistance** — the disclosure, and its `[GAP:]` on scope.

## 3. Outreach — after the read-through

- [ ] **Send the Gleb message** (`gleb_message_2026-08-11.md`); fill `<OVERLEAF LINK>`. Locks the
      Aug 17–28 sprint and points him at PR #149 to verify numbers himself.
- [ ] **Ask Axel one question:** was `--local` him? Closes his authorship tier. Same message can
      offer him Related Work (built from his literature collection).
- [ ] **Reach out to Sai** on #150 — the 50-instance error taxonomy and the ~20-paper human anchor.
- [ ] **Merge PR #149** once satisfied. Not merged; deliberately left to Martin.

## 4. Martin's work items

- [ ] 🔴 **Item 0 — the two fidelity controls.** C1 claims the chain structure is *read from* each
      paper rather than *imposed by* the schema, and nothing in the draft tests that. Two cheap
      experiments do, both holding the domain fixed:
      **(a) schema ablation** — re-extract ~100–200 papers with a prompt that does not name the five
      stages, then measure how often the emergent chain maps onto them (doubles as the missing
      ablation); **(b) degraded-source control** — re-extract from sentence-shuffled /
      abstract-only / reference-list-only versions of the same papers and see whether complete
      chains still appear. Needs extraction budget + an explicit go-ahead.
      🔴 Do **not** run the out-of-domain version first proposed (extract from non-safety papers):
      any problem→solution literature yields genuine chains, so that measures corpus *scope*, not
      *fidelity*. Reasoning in `paperA_v3_RESTRUCTURE.md` §2.4.
- [x] **Figure 1** — `graph_analysis/experiment_figure1.py` → `paper/figure1_dataset.png`.
- [x] **AI-assistance disclosure** — §Reproducibility, with a `[GAP:]` on the drafting scope.
- [ ] **Appendix A** — paste `PROMPT_EXTRACT` from
      `intervention_graph_creation/src/prompt/final_primary_prompt.py`, trim to one page. ICML lists
      the prompt as a reproducibility requirement, so this is not cosmetic.
- [ ] **Related Work final pass** — five strands drafted, all citations resolve. Needs a read, not a
      rewrite, plus two open `[CITE:]` markers (AI Safety Atlas, PICO).
- [ ] **Dataset/code release artefact** — the abstract promises one and none exists. Needs a hosted
      dump (graph + path set + extraction code) and a stable link.
- [ ] **A second worked failure case** with a missing stage (Appendix G note). Optional.

## 5. 🔴 Gates — decisions only Martin can make

- [ ] **G14 — compute-donor consent.** Credits came from a private acquaintance, possibly under terms
      that did not contemplate transfer. Ask: named / anonymous / omitted. Default if unreachable:
      anonymous. Gate text sits in the `.tex` Acknowledgments — do not delete that comment until closed.
- [ ] **G15 — freeze the author list.** Working notes are private (path above). Open: confirm Axel's
      `--local`, decide Tier 2 invitations, draft the CRediT statement. Decided: Martin last author
      (+ corresponding); Gleb + Mike shared first. **Settle the AI-disclosure scope in the same
      conversation** — the venue policies that require disclosure also bar an LLM from authorship.
- [ ] **Venue.** Nothing committed. Draft is venue-neutral two-column so it can go to a workshop or
      straight to arXiv. Note: ICLR 2026 desk-rejects undisclosed LLM use; ICML 2026 permits
      assistance but forbids crediting an LLM. Re-check the disclosure wording against the choice.

## 6. Data still owed to the authorship analysis

- [ ] Full SOAR Discord history + the subthreads (on-disk logs are partial).
- [ ] Month-1 SOAR cohort list vs later joiners (Mike joined after month 1).
- [ ] Overleaf revision history — the only record of who actually wrote.

---

## Fallback if Sai slips

Submit with the validation section scoped to what the judge data already supports. The receipt
(`experiment_judge_full_report.json`) carries the judge audit, the meta-grader table with honest
per-grader *n*, Fleiss κ, the auto-derived error profile and the recovery result. Missing would be
the manual taxonomy and the human anchor — both already stated as limitations. This fallback is
written into the Gleb message.

---

## Operational knowledge (learned the hard way, 2026-08-11/12)

**Two repos, one manuscript.** `paper/paperA_draft_v2.tex` in the analysis repo and `main.tex` in the
PaperA repo are byte-identical by convention. Edit one, copy to the other, `diff` before committing.
Analysis scripts and receipts go to the analysis repo; the manuscript and its assets go to PaperA.

**🔴 Always push to the PaperA repo.** Martin's Overleaf syncs from that remote — unpushed work is
invisible to him. Figures too, not just the `.tex`.

**🔴 Keep `main.tex` pure ASCII.** Overleaf types a file as text-or-binary when it *first enters the
project* and caches that verdict. One `U+1F534` in a LaTeX comment got the file registered as binary
and locked against editing; cleaning the bytes afterwards did **not** release it. Recovery needed
both halves: pure-ASCII bytes **and** deleting the file inside Overleaf so it was re-typed on the
next pull. Guard: header comment in the `.tex` + `python paper/asciify_tex.py` + `paper/texlint.py`.

**Overleaf branch behaviour.** Each sync creates an `overleaf-YYYY-MM-DD-HHMM` branch and deletes it
seconds later — normal, ignore. A branch that *lingers* means a sync hit a divergence and parked its
state; inspect it before doing anything. The one from this session contained only the deletion of
`main.tex` and was safely removed after confirming
`git diff --diff-filter=DM <branch> origin/master` was empty. **Never merge an Overleaf branch into
master without that check.**

**No LaTeX toolchain on this machine.** `paper/texlint.py` stands in: environment balance,
label/reference resolution, citation keys vs `refs.bib`, brace balance, a single `\bibliography`,
graphics present, and a `%` inside a BibTeX entry. It is not a compile — say so when reporting.

**BibTeX has no in-entry comment syntax.** A `%` between two fields is literal text and aborts the
parse ("expecting a `,` or a `}`"). Keep remarks outside the braces. `texlint.py` now catches this.

**This shell mangles backslashes in heredocs.** A `\times` in a `python - <<'PY'` block became a
literal tab inside the manuscript. Use a script file for anything containing LaTeX, never a heredoc.

**The analysis repo runs a `ruff-format` pre-commit hook** that rewrites new Python files and aborts
the commit. Re-run the affected scripts to confirm they still work, re-stage, commit again.

**Verification loop after any manuscript change:**
`python paper/asciify_tex.py …` → `python paper/texlint.py …` →
`python graph_analysis/experiment_paper_claim_audit.py` (expect 42/42 PASS) → copy to both repos →
`diff` → commit → push.

---

## Things a fresh session must not re-derive wrongly

- 🔴 `git blame` is useless on the analysis repo (2026-03-08 bulk commit `6e1632f` rewrote the pipeline).
- 🔴 The 88% race figure **does not reproduce** (2.6% re-derived) — do not reinstate it.
- 🔴 "~60 recovered of ~400" is **wrong** (disjoint populations); the figure is 23/441 = 5.2%.
- 🔴 The reporting unit is the **2,772 de-duplicated** chain set, not the raw 8,954.
- 🔴 Intervention maturity is **LLM-assigned and un-adjudicated** — the judge study does not score it.
  Do not promote it to a measured rate.
- 🔴 **Paper B material stays out of Paper A** unless the producing code *and* its data are committed
  and a reader can re-derive the number. The scope-composition result (11.6% / 30.5%) was drafted
  into v3 and withdrawn on exactly this test: `phase2_routing_assignments.jsonl` is untracked.
- 🔴 The thesis is **"the chain is the reliable unit, the aggregate graph is not."** C3 is a
  contribution, not a caveat — do not soften it into a limitations paragraph.
- 🔴 An **out-of-domain extraction run is not a control for C1.** It tests scope, not fidelity. See
  item 0.
- 🔴 The paper claims **fidelity to what a paper argued** — not that the argument is correct, the
  intervention works, or the research is good. A faithful extraction from a weak paper is a success.
