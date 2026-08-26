# The null-repair grader arm: design, numbers, and why the stage was cut

**Status: CUT from the manuscript 2026-08-16 (decision D2). Not deleted — documented here so
the work is recoverable if anyone resumes it.** The manuscript keeps one paragraph naming
the confound and the control; everything else it used to carry is below.

## What was run

Three meta-graders (`claude-opus-4-5`, `gemini-3-pro`, `gpt-5.1`) scored each of the judged
extractions 0–100 **before** and **after** the judge's proposed repairs, in a single pass,
with the repair list visible in the prompt. The intent was to measure whether the judge's
repairs improve an extraction.

## Why it cannot measure that

Each grader saw the repair list it was asked to score. There was no blinding, no order
randomisation and no null-repair arm. A positive movement is therefore equally consistent
with two explanations that the data cannot separate:

1. the repairs genuinely improved the extraction, and
2. the grader rated the extraction higher *because it was shown a list of fixes*.

All three graders duly recorded an improvement on almost every paper, which is what both
explanations predict.

## The design that would separate them

**One batch call.** Same three graders, same papers, same rubric, but with an **empty or
fabricated repair list** substituted for the judge's. Compare the pre/post movement against
the real arm.

- A materially smaller movement on the sham arm rescues the original result.
- A similar movement confirms a presentation effect — which is itself publishable, and is
  the more likely outcome given the agreement collapse recorded below.

Fix the rubric prompt once for all three graders while doing it; that also retires the
95/95/13 denominator problem, since the uneven denominators come from schema drift within
each grader's agent session rather than from anything about the papers.

## The numbers that were cut from the manuscript

All in `phase2_results/experiment_review_grader_agreement_report.json` and
`phase2_results/experiment_judge_full_report.json` (`item2_meta_graders`). Re-derive with
`experiment_review_grader_agreement.py --grader-archive <dir>`.

| Statistic | Pre-repair | Post-repair |
|---|---|---|
| ICC(2,1) | 0.921 | 0.151 |
| ICC(2,k) | 0.972 | 0.348 |
| Krippendorff's α (interval) | 0.917 | 0.043 |
| Fleiss' κ, four a-priori bands (60/75/85) | 0.54 | 0.09 |
| Fleiss' κ, three equal thirds | 0.811 | 0.318 |
| Fleiss' κ, median split | — | vanishes |

n = 13 throughout: the papers all three graders scored. Improvement rates were 91.6% /
98.9% / 100% of each grader's own scored set; Gemini's post-repair mean saturated at
95.77 ± 1.80.

**The diagnosis those numbers supported**, which the manuscript no longer makes: graders who
agree closely about an extraction (ICC 0.92) and then disagree about its repair (ICC 0.15)
behave as a presentation effect predicts and not as a shared perception of improvement
would. The reversal survives two independent binnings, so it is not an artifact of
cut-points.

## Why it was cut rather than run

Two reasons, and the second is the one that decides it.

1. **Reachability.** Two of the three graders are Gemini and GPT-5.1. They are not reachable
   on subscription auth, and a Claude-only arm is not comparable to the three-grader design
   it would have to be scored against.
2. **It is second-order.** The judge is not claimed as a validated instrument anywhere in
   the paper. It is a diagnostic pass whose outputs are reported as its own un-adjudicated
   opinions, with the denominator attached to every one. This stage *was* the attempt to
   validate the judge, and it failed by construction. Validating the validator is a layer
   below what the paper claims; if human time is ever spent on validation, it belongs on the
   extraction, not on the judge's opinion of the extraction.

Six of six external reviewers, at both the conference and workshop bars, independently
recommended cutting the stage rather than rescuing it.

## If you resume this

Inputs are recoverable. The grader prompt and bundler are on
`origin/judge_handoff_workshop_items_2_3`
(`extraction_validator/prompts/judge_and_extraction_evaluation_prompt.md`,
`combine_judge_and_extraction_with_original_text.py`); the 100 judge reports are on
`origin/anthropic_judge_test` (`extraction_validator/extend_try_1/`); the meta-grader archive
is `Final-archive-from-Mike.zip`. Source text comes from the ARD HuggingFace dataset.

Cost at the time of writing: ~11.5k input tokens per call, ~1.2M input and ~50k output per
grader over 100 papers, so roughly USD 4–8 per grader and USD 12–25 for all three.

**If the arm is run and the stage becomes a result, restore the manuscript paragraph and the
appendix table, and restore the fourteen `check(...)` lines removed from
`experiment_paper_claim_audit.py` on 2026-08-16.**
