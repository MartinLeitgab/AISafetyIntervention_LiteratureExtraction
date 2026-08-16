#!/usr/bin/env python
"""Meta-grader agreement on the RAW 0-100 scores (reviewer item W-4 / Q-W7).

Both simulated reviews object that Fleiss' kappa is the wrong instrument here: the rubric
scores are continuous, the manuscript bins them into four ordinal bands whose cut-points
it never states, and kappa is then computed on n=13 with one grader saturated near the top
of the scale. They ask for ICC or Krippendorff's alpha on the raw scores, the cut-points
stated, and evidence on whether the post-repair "collapse" is a binning artifact.

This script computes, over the papers all three graders scored:
  - ICC(2,1) and ICC(2,k): two-way random effects, absolute agreement
  - Krippendorff's alpha with the interval difference function
  - Fleiss' kappa under the deployed bands AND under two alternative binnings, to show
    how much of the reported number is a choice of cut-points

Grader rows are loaded with the same loader the main judge receipt uses, so the two
receipts cannot drift apart.

Class B (no LLM, no network). Run from graph_analysis/:
    python -u experiment_review_grader_agreement.py --grader-archive <dir>

Output: graph_analysis/phase2_results/experiment_review_grader_agreement_report.json
"""

import argparse
import json
import statistics as st
import sys
from itertools import combinations
from pathlib import Path

from experiment_judge_full_receipt import BANDS, fleiss_kappa, load_grader, spearman

ROOT = Path(__file__).parent
OUT = ROOT / "phase2_results/experiment_review_grader_agreement_report.json"

ALT_BANDS = {
    "deployed_quartile_ish": BANDS,
    "even_thirds_of_the_scale": [(0, 34, "low"), (34, 67, "mid"), (67, 101, "high")],
    "median_split_of_observed_scores": None,  # filled at runtime
}


def icc(matrix):
    """matrix: list of rows (subjects), each a list of k rater scores. Two-way random,
    absolute agreement. Returns ICC(2,1) and ICC(2,k)."""
    n = len(matrix)
    k = len(matrix[0])
    grand = st.mean(x for row in matrix for x in row)
    row_means = [st.mean(row) for row in matrix]
    col_means = [st.mean(row[j] for row in matrix) for j in range(k)]
    ss_rows = k * sum((m - grand) ** 2 for m in row_means)
    ss_cols = n * sum((m - grand) ** 2 for m in col_means)
    ss_total = sum((x - grand) ** 2 for row in matrix for x in row)
    ss_err = ss_total - ss_rows - ss_cols
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    icc21 = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    icc2k = (msr - mse) / (msr + (msc - mse) / n)
    return {
        "ICC_2_1": round(icc21, 3),
        "ICC_2_k": round(icc2k, 3),
        "MSR": round(msr, 2),
        "MSC": round(msc, 2),
        "MSE": round(mse, 2),
        "n_subjects": n,
        "k_raters": k,
    }


def krippendorff_alpha_interval(matrix):
    """Complete-data interval alpha over rows of equal-length rater vectors."""
    m = len(matrix[0])
    values = [x for row in matrix for x in row]
    N = len(values)
    do = st.mean(
        sum((a - b) ** 2 for a, b in combinations(row, 2)) * 2 / (m * (m - 1))
        for row in matrix
    )
    de = sum((a - b) ** 2 for a, b in combinations(values, 2)) * 2 / (N * (N - 1))
    return round(1 - do / de, 3) if de else None


def band_of(score, bands):
    for lo, hi, name in bands:
        if lo <= score < hi:
            return name
    return None


def kappa_under(matrix, bands):
    from collections import Counter

    rows = []
    for row in matrix:
        c = Counter()
        for x in row:
            c[band_of(x, bands)] += 1
        rows.append(c)
    return fleiss_kappa(rows, [b[2] for b in bands])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grader-archive", required=True)
    a = ap.parse_args()
    ma = Path(a.grader_archive)
    if not ma.is_dir():
        raise SystemExit(f"FATAL: grader archive not found: {ma}")

    opus, _ = load_grader(ma / "test_extend_all_evaluation_opus_4_5", "*.json")
    gem, _ = load_grader(ma / "test_extend_all_evaluation_gemini_pro_3", "*.json")
    third, _ = load_grader(
        ma / "extend_try_with_extration_and_judge_and_original_text",
        "*_evaluation.json",
    )
    graders = {"claude-opus-4-5": opus, "gemini-3-pro": gem, "gpt-5.1": third}
    names = list(graders)
    common = sorted(set.intersection(*[set(g) for g in graders.values()]))
    if len(common) < 3:
        raise SystemExit("FATAL: fewer than 3 papers scored by all three graders")

    pre = [[graders[g][p]["pre"] for g in names] for p in common]
    post = [[graders[g][p]["post"] for g in names] for p in common]

    med = st.median([x for row in pre + post for x in row])
    ALT_BANDS["median_split_of_observed_scores"] = [
        (0, med, "below_median"),
        (med, 101, "at_or_above_median"),
    ]

    out = {
        "experiment": "meta-grader agreement on raw 0-100 scores (W-4 / Q-W7)",
        "n_common_papers": len(common),
        "graders": names,
        "per_grader_n_scored": {g: len(rows) for g, rows in graders.items()},
        "raw_score_agreement": {
            "pre_repair": {
                **icc(pre),
                "krippendorff_alpha_interval": krippendorff_alpha_interval(pre),
            },
            "post_repair": {
                **icc(post),
                "krippendorff_alpha_interval": krippendorff_alpha_interval(post),
            },
        },
        "pairwise_spearman": {
            f"{a_} vs {b_}": {
                "pre": spearman(
                    [graders[a_][p]["pre"] for p in common],
                    [graders[b_][p]["pre"] for p in common],
                ),
                "post": spearman(
                    [graders[a_][p]["post"] for p in common],
                    [graders[b_][p]["post"] for p in common],
                ),
            }
            for a_, b_ in combinations(names, 2)
        },
        "fleiss_kappa_by_binning": {
            label: {
                "bands": [{"lo": b[0], "hi": b[1], "label": b[2]} for b in bands],
                "pre": kappa_under(pre, bands),
                "post": kappa_under(post, bands),
            }
            for label, bands in ALT_BANDS.items()
        },
        "score_dispersion": {
            g: {
                "pre_mean": round(st.mean([graders[g][p]["pre"] for p in common]), 1),
                "pre_sd": round(st.pstdev([graders[g][p]["pre"] for p in common]), 1),
                "post_mean": round(st.mean([graders[g][p]["post"] for p in common]), 1),
                "post_sd": round(st.pstdev([graders[g][p]["post"] for p in common]), 1),
            }
            for g in names
        },
        "READING": (
            "ICC and alpha are computed on the raw scores and need no cut-points, so they "
            "are the instruments to quote. The three kappa rows show how much of the "
            "reported kappa is a binning choice: if the pre/post pattern reverses or "
            "flattens across binnings, the kappa collapse is not a stable finding."
        ),
    }
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
