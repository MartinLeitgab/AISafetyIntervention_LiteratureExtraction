#!/usr/bin/env python3
"""Price a reduction of the #176 packet from 30 chains to 20. Class B: no LLM, no network.

WHY THIS EXISTS. The annotator cannot complete 30 chains in the time available, so the
sample size has to fall. A size change after annotation has begun is the kind of decision
that quietly destroys a pre-registered study, so it is priced here rather than argued, and
the drop list is fixed by a rule before any of the unjudged chains is read.

WHAT MAKES THE ESTIMAND WORK, and therefore what a cut must not break. The 30 speak for the
2,772-chain reporting unit only through post-stratification: each stratum's human rate is
reweighted by that stratum's population share, taken from #175's host-proportional 200-chain
sample. The property that licenses it is COVERAGE -- every stratum carrying population weight
must hold at least one judged chain. Coverage is a function of WHICH strata are occupied, not
of how many chains sit in them, so a cut that keeps one chain per stratum keeps the estimator
legitimate at any n. That is the central finding of this script.

This reproduces the weights, the STRATUM_TO_CODE mapping and the variance formula of
experiment_review_human_adjudication.py exactly; if that script changes, change this one.

    cd graph_analysis
    python -u experiment_review_packet_resize_assessment.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
PACKET = HERE / "phase2_results" / "human_review_packet"
MANIFEST = PACKET / "manifest.json"
SHEET = PACKET / "verdict_sheet.csv"
PRECISION = HERE / "phase2_results" / "experiment_review_chain_precision_report.json"
RELABEL = HERE / "phase2_results" / "confidence_relabel_raw" / "results.jsonl"
OUT = HERE / "phase2_results" / "experiment_review_packet_resize_assessment_report.json"

# Kept identical to experiment_review_human_adjudication.py.
STRATUM_TO_CODE = {
    "intervention_not_proposed": ("intervention_not_proposed", None),
    "faithful_both_agree": ("faithful", True),
    "model_disagreement_faithful_but_link_not_asserted": ("faithful", False),
    "risk_framing_invented_both_agree": ("risk_framing_invented", False),
    "model_disagreement_invented_but_link_asserted": ("risk_framing_invented", True),
    "intermediate_unsupported": ("intermediate_unsupported", None),
    "known_judge_false_positive": ("chain_belongs_to_a_different_document", None),
    "gate_rejected_intervention_not_proposed": ("intervention_not_proposed", None),
    "gate_rejected_faithful": ("faithful", None),
    "gate_rejected_invented": ("risk_framing_invented", None),
}

# The 20-chain design this script recommends. Fixed here so it is reviewable and so the
# choice cannot drift once verdicts accumulate.
TARGET_CELLS = {
    "intervention_not_proposed": 4,
    "faithful_both_agree": 4,
    "model_disagreement_faithful_but_link_not_asserted": 2,
    "risk_framing_invented_both_agree": 2,
    "model_disagreement_invented_but_link_asserted": 1,
    "intermediate_unsupported": 2,
    "known_judge_false_positive": 1,
    "gate_rejected_faithful": 2,
    "gate_rejected_intervention_not_proposed": 1,
    "gate_rejected_invented": 1,
}


def die(msg: str) -> None:
    raise SystemExit(
        f"FATAL: {msg}\n\n"
        f"  This script reads:\n    {MANIFEST}\n    {SHEET}\n    {PRECISION}\n"
        f"    {RELABEL}\n"
        f"  It builds nothing and judges nothing. The packet is built by\n"
        f"  experiment_build_human_review_packet.py, and must NOT be rebuilt now that\n"
        f"  verdicts exist.\n"
    )


def load_json(p: Path):
    if not p.is_file():
        die(f"missing input: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> int:
    manifest = load_json(MANIFEST)
    prec = load_json(PRECISION)

    weights: dict[str, dict[str, float]] = {}
    for key, arm in (("real_arm", "A"), ("gate_rejected_arm", "B")):
        a = prec[key]
        if sum(a["reason_codes"].values()) != a["n_parsed"]:
            die(f"{key}: reason codes do not sum to n_parsed; stage-1 report is wrong")
        weights[arm] = {k: v / a["n_parsed"] for k, v in a["reason_codes"].items()}

    if not RELABEL.is_file():
        die(f"missing #179 relabel results: {RELABEL}")
    seen: dict[str, list[bool]] = defaultdict(list)
    for line in RELABEL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "verdict" not in r:
            continue
        a = r["verdict"].get("is_this_link_asserted_by_the_document")
        if a is not None:
            seen[r.get("group") or r.get("reason_code")].append(bool(a))
    split = {
        c: {True: sum(v) / len(v), False: 1 - sum(v) / len(v)} for c, v in seen.items()
    }

    def w_of(stratum: str) -> float:
        code, asserted = STRATUM_TO_CODE[stratum]
        arm = "B" if stratum.startswith("gate_rejected") else "A"
        w = weights[arm].get(code, 0.0)
        if asserted is not None:
            w *= split.get(code, {}).get(asserted, 0.0)
        return w

    A = [s for s in STRATUM_TO_CODE if not s.startswith("gate_rejected")]
    B = [s for s in STRATUM_TO_CODE if s.startswith("gate_rejected")]
    by = {m["packet_id"]: m for m in manifest}
    everything = set(by)
    recall = {m["packet_id"] for m in manifest if m.get("recall_arm")}
    double = {m["packet_id"] for m in manifest if m["double_coded"]}

    judged = set()
    if SHEET.is_file():
        import csv

        with SHEET.open(encoding="utf-8-sig", newline="") as fh:
            judged = {
                r["packet_id"]
                for r in csv.DictReader(fh)
                if (r.get("verdict") or "").strip()
            }

    def evaluate(keep: set[str]) -> dict:
        cells = {s: sum(1 for p in keep if by[p]["stratum_code"] == s) for s in A + B}
        out = {"n": len(keep), "cells": cells, "n_recall": len(keep & recall)}
        for name, strata in (("A", A), ("B", B)):
            out[f"arm_{name}_n"] = sum(cells[s] for s in strata)
            out[f"arm_{name}_coverage"] = round(
                sum(w_of(s) for s in strata if cells[s]), 4
            )
            out[f"arm_{name}_uncovered"] = [s for s in strata if not cells[s]]
        out["arm_A_halfwidth_pp"] = {}
        for p in (0.10, 0.20, 0.50):
            v = sum(w_of(s) ** 2 * p * (1 - p) / cells[s] for s in A if cells[s])
            out["arm_A_halfwidth_pp"][str(p)] = round(1.96 * math.sqrt(v) * 100, 1)
        return out

    # Drop rule, applied in this order and recorded so it is auditable:
    #   1. every already-judged chain is retained -- dropping one would be selection on
    #      an outcome already observed;
    #   2. every stratum keeps at least one chain, so coverage cannot fall;
    #   3. within a stratum, recall-arm chains are kept before non-recall ones, and
    #      double-coded before single-coded, so the two secondary arms survive the cut;
    #   4. surplus comes off the strata the build over-sampled relative to their weight.
    def rank(pid: str) -> tuple:
        m = by[pid]
        return (0 if m.get("recall_arm") else 1, 0 if m["double_coded"] else 1, pid)

    keep = set(judged)
    for stratum, target in TARGET_CELLS.items():
        have = [p for p in keep if by[p]["stratum_code"] == stratum]
        pool = sorted(
            [
                p
                for p in everything
                if by[p]["stratum_code"] == stratum and p not in keep
            ],
            key=rank,
        )
        while len(have) < target and pool:
            p = pool.pop(0)
            keep.add(p)
            have.append(p)

    full, cut = evaluate(everything), evaluate(keep)

    if cut["arm_A_coverage"] < full["arm_A_coverage"]:
        die(
            f"the proposed cut drops arm-A coverage from {full['arm_A_coverage']} to "
            f"{cut['arm_A_coverage']}. That breaks the post-stratified estimand and no "
            "time saving justifies it. Re-allocate so every stratum keeps a chain."
        )

    report = {
        "study": "resizing the #176 packet from 30 chains to 20 -- feasibility",
        "reason": (
            "annotator time. Not motivated by any verdict observed: the three judged "
            "chains are 1 faithful and 2 inferred_but_reasonable, zero unsupported, so "
            "nothing seen so far bears on the rate the study reports."
        ),
        "judged_at_decision_time": sorted(judged),
        "stratum_weights": {s: round(w_of(s), 4) for s in A + B},
        "design_30": full,
        "design_20": cut,
        "chains_dropped": sorted(everything - keep),
        "what_survives": {
            "arm_A_population_coverage": cut["arm_A_coverage"],
            "note": (
                "Unchanged at 1.0. Coverage is what licenses post-stratification, and it "
                "depends on which strata are occupied rather than on n."
            ),
            "heaviest_stratum_untouched": (
                "faithful_both_agree carries 37.8% of the population weight and the packet "
                "only ever held 4 chains in it. All 4 survive, so the dominant variance "
                "term is identical at 30 and at 20 -- which is why the interval barely "
                "moves."
            ),
        },
        "what_degrades": {
            "arm_A_95pct_halfwidth_pp": {
                "at_p_0.10": [
                    full["arm_A_halfwidth_pp"]["0.1"],
                    cut["arm_A_halfwidth_pp"]["0.1"],
                ],
                "at_p_0.20": [
                    full["arm_A_halfwidth_pp"]["0.2"],
                    cut["arm_A_halfwidth_pp"]["0.2"],
                ],
                "at_p_0.50": [
                    full["arm_A_halfwidth_pp"]["0.5"],
                    cut["arm_A_halfwidth_pp"]["0.5"],
                ],
            },
            "zero_count_ceiling": {
                "explanation": (
                    "P1 predicts `unsupported` stays rare. If it is zero the study's "
                    "sharpest sentence is a ceiling, and the ceiling is set by n alone. "
                    "This is the largest single cost of the cut."
                ),
                "upper_bound_on_0_of_30_pct": round(100 * wilson(0, 30)[1], 1),
                "upper_bound_on_0_of_20_pct": round(100 * wilson(0, 20)[1], 1),
            },
            "recall_arm_documents": [full["n_recall"], cut["n_recall"]],
            "recall_arm_zero_miss_ceiling_pct": [
                round(100 * wilson(0, full["n_recall"])[1], 1),
                round(100 * wilson(0, cut["n_recall"])[1], 1),
            ],
        },
        "reviewer_comments_unaffected": {
            "R9": "a human validates something -- answered by any defensible sample",
            "R12": "source spans -- the rubric captures risk_quote and intervention_quote",
            "R21": "gates unvalidated -- arm B still gives a direction, as at 6",
            "R11": (
                "inter-annotator agreement -- already open with one annotator; "
                f"{len(keep & double)} of 8 double-coded rows survive, so a second "
                "annotator remains possible"
            ),
            "R13": (
                "asked for 50-100 chain-yielding documents. 30 already failed it and 20 "
                "fails it the same way, so the defence does not change: a stratified "
                "sample with full population coverage and a post-stratified estimator "
                "beats a simple random sample of the same size."
            ),
        },
        "LIMITS": (
            "The hours below rest on three measured chains (30, 60, 60 minutes) and the "
            "marginal cost of the recall enumeration is measured on exactly one document, "
            "so the range is wide on purpose. Re-check at chain 5 as the run record says."
        ),
    }

    n_rec, n_vo = cut["n_recall"], cut["n"] - cut["n_recall"]
    report["cost_hours"] = {
        "design_30": [
            round((20 * 45 + 10 * 60) / 60, 1),
            round((20 * 45 + 10 * 90) / 60, 1),
        ],
        "design_20": [
            round((n_vo * 45 + n_rec * 60) / 60, 1),
            round((n_vo * 45 + n_rec * 90) / 60, 1),
        ],
        "basis": "verdict-only 45 min (mean of C01 30 and C02 60); recall 60-90 min",
    }

    print(f"judged so far: {sorted(judged)}")
    print(
        f"\n30 as built : arm A n={full['arm_A_n']} cov {full['arm_A_coverage']:.3f} | "
        f"arm B n={full['arm_B_n']} cov {full['arm_B_coverage']:.3f} | recall {full['n_recall']}"
    )
    print(
        f"20 proposed : arm A n={cut['arm_A_n']} cov {cut['arm_A_coverage']:.3f} | "
        f"arm B n={cut['arm_B_n']} cov {cut['arm_B_coverage']:.3f} | recall {cut['n_recall']}"
    )
    print("\n95% half-width on the weighted share (arm A):")
    for p in ("0.1", "0.2", "0.5"):
        print(
            f"  p={p}:  30 -> {full['arm_A_halfwidth_pp'][p]:.1f} pp    "
            f"20 -> {cut['arm_A_halfwidth_pp'][p]:.1f} pp"
        )
    z = report["what_degrades"]["zero_count_ceiling"]
    print(
        f"\nzero-count ceiling: 0 of 30 -> {z['upper_bound_on_0_of_30_pct']}%   "
        f"0 of 20 -> {z['upper_bound_on_0_of_20_pct']}%"
    )
    print(
        f"\nhours: 30 -> {report['cost_hours']['design_30']}   20 -> {report['cost_hours']['design_20']}"
    )
    print(f"\ndrop these 10: {report['chains_dropped']}")

    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
