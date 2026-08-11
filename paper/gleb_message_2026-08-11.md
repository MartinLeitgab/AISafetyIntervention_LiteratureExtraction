# Reply to Gleb — delivering the promised outline + sprint confirmation

**Drafted 2026-08-11. NOT yet sent.** Reply in the existing "Project wrap-up" thread.

**Thread position:** Gleb proposed Aug 17–28 (Aug 7). Martin replied the same day promising
"an updated paper outline to you next week with a list of open work", and mentioned Jeff Parks.
This message delivers exactly that, on time, and confirms the dates.

**Fill in before sending:** `<OVERLEAF LINK>`.

**Written to fold into the thread:** no re-litigating the plan (settled in June/July), no repeated
thanks, no re-introduction of context he already has. It opens by delivering what was promised,
confirms his dates, and picks up the ticket references he's already seen (#147/PR 148 → #150/PR 149).

**On the awkward item:** the 88% non-reproduction sits in the middle, framed as a question to him
rather than a verdict, with the half that *did* reproduce stated first and a plausible innocent
explanation offered. It is not softened away — it's out of the abstract and he needs to know — but
he is a volunteer being asked to keep contributing.

---

Hi Gleb,

**August 17–28 works — let's lock those dates.** I'll keep that fortnight clear.

As promised, here's the outline with the list of open work. It turned into rather more than an
outline: there's a full draft, plus the whole analysis chain committed so you can check any number
in it yourself rather than take my word for it.

**Draft (Overleaf):** `<OVERLEAF LINK>` — synced from
`github.com/MartinLeitgab/AISafetyIntervention_PaperA`, so edit in whichever you prefer.

**Analysis (PR #149):**
`github.com/MartinLeitgab/AISafetyIntervention_LiteratureExtraction/pull/149`
`paper/REPRODUCE.md` maps every claim in the paper to the script and receipt it came from, and one
command re-derives all of them from the raw data:

```
cd graph_analysis && python -u experiment_paper_claim_audit.py     # 42/42 PASS
```

Until now the path builders were in git but their outputs and the receipts weren't, so nobody but
me could verify a figure. That's fixed.

The structure is what we agreed: the pathway dataset, mechanism-level retrieval and clustering-free
descriptives up front, and your graph-metrics work as the practical-guidance section — four
measured failure modes, each with the control that removes it. It reads as one of the stronger
parts of the paper. Your race-classifier validation is kept as an appendix and attributed to you.

**Two things I need to flag, because both touch your section.**

The structural numbers come out differently on my substrate — components 18,424 → 15,123, largest
component 55 → 61, clustering coefficient 0.007 → 0.013, within-category similarity edges 169,083 →
1,435,806. I think that's fully explained: your counts were on the merged 200,061-node graph, and
the similarity layer in the dump I'm using is roughly 8.5× denser than yours. Same qualitative
conclusion in every case; the paper just prints the numbers for the graph it actually describes.

The second one I'd genuinely like your read on: **I couldn't reproduce the 88%.** Using your
definitions — importance by eigenvector centrality, path diversity as distinct first-hop structural
problem-analysis neighbours, race-framed by the same keyword rule — on the merged + full-similarity
graph:

| | your figure | re-derived |
|---|---:|---:|
| single-path risks in the top-100 | 41 | **38** |
| of those, race-framed | 88% | **2.6%** |
| head-vs-population gradient | 44× | **2.0×** |

The structural half reproduces well — 38 against 41 is the same finding. The framing half doesn't,
by a wide margin. My best guess is that the denser similarity layer shifts the centrality ranking
enough that we're looking at a different top-100, but I can't settle that from my side.
**Do you still have the Iteration-B pipeline, or the intermediate graph you ran on?** If you can
point me at it I'd rather resolve this properly than leave a discrepancy in the paper.

For now the 88% is out of the abstract, and the guidance section reports the reproduction failure
instead. I actually think that's the stronger result: it says a selection-conditioned statistic on
this kind of graph isn't merely inflated, it's unstable under pipeline parameters — which is
precisely the warning that section exists to give. The script is
`graph_analysis/experiment_race_top100_rederive.py` in the PR if you want to dig in.

**Open work for the minimum workshop bar — this is the whole list:**

| # | Item | Owner | Size |
|---|---|---|---|
| 1 | Related Work — GraphRAG/LLM+KG strand drafted with verified citations; needs a final pass | me + Axel/Jeff | ~half a day |
| 2 | Figure 1 — chain-length histogram + maturity profile, plots straight from a receipt | me | ~1 h |
| 3 | Manual 50-instance error taxonomy | Sai (#150) | the real one |
| 4 | Human-anchored spot-check, ~20 papers — nothing has been human-adjudicated yet | Sai (#150) | ~half a day |
| 5 | Author list, affiliations, acknowledgements | me — needs your input | minutes |
| 6 | Dataset/code release artefact — the abstract promises one and there isn't one | me | ~half a day |
| 7 | Read through the practical-guidance section and tell me what I've mis-stated about your work | **you** | — |

Item 7 is what I actually need from you. Items 3 and 4 are the only real blockers and both sit with
Sai — I closed #147 and rewrote it as #150, scoped down to five items that genuinely need a human.
His six judge.py bug fixes are done and merge-ready, and it turned out those bugs don't affect any
number we report, so they were never blocking us.

Jeff has offered to review drafts, so I'll send him the Overleaf link too once you've had first
pass.

**Plan for the fortnight:** you review the draft and the guidance section in week 1; I finish
Related Work, the figure and the release artefact in parallel; Sai's two items land when they land,
and if they slip we submit with the validation section scoped to what the judge data already
supports. Nothing else is on the critical path.

No venue committed yet — I kept the draft in a plain two-column style rather than a conference
class so we can point it at a workshop or straight at arXiv without rework. Worth ten minutes of
the first sprint call.

Thanks,
Martin

---

## Follow-ups once he replies

- If he still has Iteration-B → re-run the 88% properly. Either it restores the figure or it
  confirms the artefact; both beat a discrepancy note.
- Ask him to publish `cluster_representatives_20.json` (makes Appendix D auditable).
- Settle authorship order.
- Send Jeff the Overleaf link after Gleb's first pass.
