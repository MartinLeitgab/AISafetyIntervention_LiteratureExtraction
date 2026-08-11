# Message to Gleb — sprint kickoff + draft + reproduction findings

**Drafted 2026-08-11. NOT yet sent.** Replies to his 2026-08-07 mail proposing Aug 17–28.

**Before sending, fill in:** the Overleaf project link (you've linked the GitHub repo already).

**Tone note:** this carries one genuinely awkward item — a figure of his does not reproduce.
It is placed in the middle, framed as a question to him rather than a verdict, and surrounded by
the things that *did* hold up. That is deliberate: he is a volunteer being asked to keep
contributing, and the failure is at least partly explained by a substrate difference we can't
close from our side.

---

Hi Gleb,

**Aug 17–28 works — let's lock it.** I'll be available across those two weeks.

I've got the outline done, and rather more than an outline: there's now a full draft, plus the
analysis chain committed so you can check every number in it yourself.

**The draft** — Overleaf: `<LINK>` (synced from
`github.com/MartinLeitgab/AISafetyIntervention_PaperA`, so you can edit in either place).

**The analysis** — PR #149 on the main repo:
`github.com/MartinLeitgab/AISafetyIntervention_LiteratureExtraction/pull/149`.
That PR exists mostly so you don't have to take my word for anything. `paper/REPRODUCE.md` maps
every claim in the paper to the script and the receipt JSON it came from, and there's one command
that re-derives all of them from raw data:

```
cd graph_analysis && python -u experiment_paper_claim_audit.py     # 42/42 PASS
```

Previously the path builders were in git but their outputs and the receipts weren't, so nobody
but me could check a figure. That's fixed.

**Where the paper landed.** The thesis is the one we agreed: the pathway dataset, mechanism-level
retrieval, and clustering-free descriptives, with your graph-metrics work as §Practical Guidance —
four measured failure modes each with the control that removes it. That section is written and it
is one of the stronger parts of the paper. Your §Appendix on the race classifier validation is
kept and attributed to you as a manual validation.

**Two things I have to flag, because they concern your section.**

*First, the structural numbers differ on our substrate.* Components 18,424 → 15,123, largest
component 55 → 61, clustering coefficient 0.007 → 0.013, within-category similarity edges 169,083 →
1,435,806. I think this is fully explained: your counts were on the merged 200,061-node graph, and
the similarity layer in the dump I'm using is about 8.5× denser than yours. Same qualitative
conclusion every time. The paper prints the numbers for the graph it actually describes, with the
difference noted.

*Second — and this is the one I'd like your read on — I could not reproduce the 88%.* Using your
definitions (importance = eigenvector centrality, path diversity = distinct first-hop structural
problem-analysis neighbours, race-framed = sole PA neighbour matching "competitive|race"), on the
merged + full-SIM graph:

| | your figure | re-derived |
|---|---:|---:|
| single-path risks in the top-100 | 41 | **38** |
| of those, race-framed | 88% | **2.6%** |
| head-vs-population gradient | 44× | **2.0×** |

The structural half reproduces well — 38 vs 41 is the same finding. The framing half doesn't, by a
lot. My best guess is that the denser similarity layer changes the centrality ranking enough that
we're looking at a different top-100 — but I can't close that from my side without your
Iteration-B pipeline. **Do you still have it, or the intermediate files?** If you can point me at
the graph you ran on, I'd like to settle this properly rather than leave it as a discrepancy.

For now the 88% is out of the abstract, and the guidance section reports the reproduction failure
instead. Honestly I think that's a *stronger* result: it says a selection-conditioned statistic on
this kind of graph isn't merely inflated, it's unstable under pipeline parameters — which is
exactly the warning the section exists to give. Script is
`graph_analysis/experiment_race_top100_rederive.py` in the PR if you want to poke at it.

**What's still open for the minimum workshop bar** — this is the whole list:

| # | Item | Owner | Size |
|---|---|---|---|
| 1 | Related Work — ~3 paragraphs. GraphRAG/LLM+KG strand is drafted with 6 verified citations; still need ARD, Stampy/Atlas, MIT Risk Repository, relation-extraction, LLM-as-judge | Axel + me | ~half a day |
| 2 | Figure 1 — chain-length histogram + maturity profile, plots straight from a receipt | me | ~1 h |
| 3 | Manual 50-instance error taxonomy | Sai (#150) | the real one |
| 4 | Human-anchored spot-check, ~20 papers — no human has adjudicated anything yet | Sai (#150) | ~half a day |
| 5 | Author list, affiliations, acknowledgements | me, needs your input | minutes |
| 6 | Dataset/code release artefact — the abstract promises one and there isn't one yet | me | ~half a day |
| 7 | Read-through of §Practical Guidance for anything I've mis-stated about your work | **you** | — |

Item 7 is the one I actually need from you. Items 3 and 4 are the only genuine blockers, and
they're on Sai's ticket (#150 — I closed #147 and rewrote it; his six judge.py bug fixes are done
and merged-ready, and it turns out those bugs don't affect any number we report).

**Plan for the two weeks:** you read the draft and §Practical Guidance in week 1 and tell me what's
wrong; I finish Related Work, the figure and the release artefact in parallel; Sai's two items land
whenever they land, and if they slip we submit with the validation section scoped to what the
judge data already supports. Nothing else is on the critical path.

No venue committed yet — I've deliberately kept the draft in a plain two-column style rather than a
conference class, so we can point it at a workshop or arXiv without rework. Happy to discuss where
to aim it.

Thanks for sticking with this — the guidance section only exists because you did the exploration
that mapped those failure modes.

Martin

---

## Follow-ups after he replies

- If he still has the Iteration-B pipeline → re-run the 88% properly; that either restores the
  figure or confirms the artefact. Either is publishable and both beat a discrepancy note.
- Ask him to publish `cluster_representatives_20.json` (Appendix D auditability).
- Confirm authorship order.
