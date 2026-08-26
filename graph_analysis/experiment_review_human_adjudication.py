#!/usr/bin/env python3
"""Analyse the #176 human adjudication packet. PRE-REGISTERED: written before annotation.

This script exists BEFORE the verdict sheet is filled in, and that is the point. The 30
chains are a reason-code-stratified sample, not a random one, so several different rates
can be computed from the same verdicts and only one of them is the estimand. Fixing which
one now removes the freedom to pick the flattering rate later.

THE ESTIMAND, stated once
    Post-stratified human chain precision on the 2,772-chain reporting unit: the share of
    chains a domain reader judges `faithful`, `inferred_but_reasonable` or `unsupported`,
    with each stratum's human cell reweighted by that stratum's share of the population.

WHY REWEIGHTING IS LEGITIMATE HERE, and it is the whole reason a 30-chain sample can speak
for 2,772: #175's real arm is 200 chains stratified by the URL host of the risk node,
PROPORTIONAL to the chain set. Its reason-code shares are therefore population estimates,
not sample quirks. The human cells sit inside those codes, so
    rate = sum over strata of ( population share of stratum * human rate within stratum )
is a standard post-stratified estimator. Take away the proportional stage-1 sample and
this collapses into a rate for 30 hand-picked chains, which is worth nothing.

WHAT THIS SCRIPT WILL NOT DO, listed so a future session does not add them
  - No inter-annotator agreement. One annotator, decided 2026-08-26. Reviewer R11 stays
    open and the write-up must say so. Do not compute agreement from the annotator
    re-judging their own rows; that is test-retest, a different instrument.
  - No corpus omission or recall rate. This packet only shows chains that WERE emitted, so
    it cannot see what extraction missed. The 0.6% vs 28.8% discrepancy (R16) is untouched.
  - No maturity-label validation. Not asked in the rubric.
  - No confidence interval that ignores the design. Cells are 1-7 chains; the binomial
    interval on a cell of 3 is nearly the whole unit interval, and the reweighted interval
    is reported by the same stratified formula, never by pooling the 30 as if they were a
    simple random sample.
  - No human-anchored 17.8 pp. Arm B carries six observations. Directional only.

Class B: no LLM call, no network, no API key.

    cd graph_analysis
    python -u experiment_review_human_adjudication.py            # dry run until filled in
    python -u experiment_review_human_adjudication.py --strict    # crash on a partial sheet
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
PACKET = HERE / "phase2_results" / "human_review_packet"
SHEET = PACKET / "verdict_sheet.csv"
SHEET2 = PACKET / "verdict_sheet_annotator2.csv"
MANIFEST = PACKET / "manifest.json"
PRECISION = HERE / "phase2_results" / "experiment_review_chain_precision_report.json"
RELABEL = HERE / "phase2_results" / "confidence_relabel_raw" / "results.jsonl"
OUT = HERE / "phase2_results" / "experiment_review_human_adjudication_report.json"

VERDICTS = ("faithful", "inferred_but_reasonable", "unsupported")
YESNO = ("yes", "no")
LEVELS = ("0", "1", "2", "3")

LEVEL_FIELDS = (
    "risk_inference_level",
    "intervention_inference_level",
    "body_inference_level",
)

# Which stage-1 reason code each packet stratum sits inside. Two strata split one code by
# whether the second instrument (#179) agreed, so the split has to be named explicitly --
# the stratum label alone does not carry it.
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


def die(msg: str) -> None:
    raise SystemExit(
        f"FATAL: {msg}\n\n"
        f"  This script reads:\n"
        f"    {SHEET}\n"
        f"    {MANIFEST}\n"
        f"    {PRECISION}\n"
        f"    {RELABEL}\n"
        f"  The packet is built by: python -u experiment_build_human_review_packet.py\n"
        f"  This script does NOT build the packet, does NOT call an LLM, and does NOT\n"
        f"  invent a verdict for an unfilled row."
    )


def load_json(p: Path) -> dict:
    if not p.is_file():
        die(f"missing input: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def population_weights() -> tuple[dict, dict, dict]:
    """Population shares per reason code, for both arms, plus the #179 within-code split.

    Arm A shares come from #175's 200-chain host-proportional sample of the reporting
    unit. Arm B shares come from the 96-chain gate-rejected arm and describe that arm
    only -- arm B is NOT part of the released reporting unit and its rate is reported
    separately, never blended into arm A's.
    """
    rep = load_json(PRECISION)
    out = {}
    for arm_key, arm_name in (("real_arm", "A"), ("gate_rejected_arm", "B")):
        codes = rep[arm_key]["reason_codes"]
        n = rep[arm_key]["n_parsed"]
        if sum(codes.values()) != n:
            die(
                f"{arm_key}: reason codes sum to {sum(codes.values())} but n_parsed is {n}. "
                "The weights would be wrong; fix the stage-1 report before continuing."
            )
        out[arm_name] = {k: v / n for k, v in codes.items()}

    # #179 re-labelled 33 invented + 33 faithful first hops and asked whether the document
    # asserts the link. That gives the within-code agree/disagree split for exactly those
    # two codes. No other code has a second instrument, and none is invented for them.
    split: dict[str, dict[bool, float]] = {}
    if RELABEL.is_file():
        by_code: dict[str, list[bool]] = defaultdict(list)
        for line in RELABEL.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "verdict" not in r:
                continue
            a = r["verdict"].get("is_this_link_asserted_by_the_document")
            if a is None:
                continue
            by_code[r.get("group") or r.get("reason_code") or "unknown"].append(bool(a))
        for code, vals in by_code.items():
            if code in ("faithful", "risk_framing_invented") and vals:
                yes = sum(vals)
                split[code] = {True: yes / len(vals), False: 1 - yes / len(vals)}
    return out["A"], out["B"], split


def read_sheet(path: Path) -> list[dict]:
    if not path.is_file():
        die(f"missing verdict sheet: {path}")
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def validate(rows: list[dict], strict: bool) -> tuple[list[dict], list[str]]:
    """Return the filled rows and every complaint. Never repairs a row."""
    filled, problems = [], []
    for r in rows:
        pid = r.get("packet_id", "?")
        v = (r.get("verdict") or "").strip().lower()
        if not v:
            continue  # not yet judged; dry run reports this as progress, not as an error
        if v not in VERDICTS:
            problems.append(f"{pid}: verdict '{v}' is not one of {VERDICTS}")
            continue
        a = (r.get("risk_link_asserted") or "").strip().lower()
        if a not in YESNO:
            problems.append(f"{pid}: risk_link_asserted '{a}' is not yes/no")
        lv = {}
        for f in LEVEL_FIELDS:
            s = (r.get(f) or "").strip()
            if s not in LEVELS:
                problems.append(f"{pid}: {f} '{s}' is not 0-3")
            else:
                lv[f] = int(s)

        # The rubric says the verdict follows the levels. Where it does not, notes must
        # explain -- so an unexplained contradiction is surfaced, never silently accepted.
        if len(lv) == 3:
            worst = max(lv.values())
            implied = (
                "faithful"
                if lv["risk_inference_level"] == 0
                and lv["intervention_inference_level"] == 0
                and worst == 0
                else "unsupported"
                if worst == 3
                else "inferred_but_reasonable"
            )
            if implied != v and not (r.get("notes") or "").strip():
                problems.append(
                    f"{pid}: levels imply '{implied}' but verdict is '{v}', and notes is "
                    "empty. The rubric requires a reason when the two diverge."
                )
        filled.append({**r, "verdict": v, "risk_link_asserted": a, **lv})

    if problems and strict:
        die(
            "verdict sheet has "
            + str(len(problems))
            + " problem(s):\n  - "
            + "\n  - ".join(problems)
        )
    return filled, problems


def wilson(k: int, n: int) -> tuple[float, float]:
    """Wilson interval. Honest at the tiny cell sizes this study has; Wald is not."""
    if n == 0:
        return (0.0, 1.0)
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def post_stratify(cells: dict, weights: dict, split: dict) -> dict:
    """Reweight per-stratum human rates to the population.

    Returns the point estimate per verdict plus a stratified standard error. Strata with
    no filled row are reported as uncovered rather than imputed -- an imputed cell would
    make the coverage figure a lie.
    """
    est = {v: 0.0 for v in VERDICTS}
    var = 0.0
    covered_w, uncovered = 0.0, []
    for stratum, rows in sorted(cells.items()):
        code, asserted = STRATUM_TO_CODE[stratum]
        w = weights.get(code, 0.0)
        if asserted is not None:
            w *= split.get(code, {}).get(asserted, 0.0)
        if not rows:
            uncovered.append({"stratum": stratum, "weight": round(w, 4)})
            continue
        covered_w += w
        n = len(rows)
        for v in VERDICTS:
            p = sum(1 for r in rows if r["verdict"] == v) / n
            est[v] += w * p
            if v == "unsupported":
                var += (w**2) * p * (1 - p) / n
    return {
        "weighted": {v: round(est[v], 4) for v in VERDICTS},
        "population_coverage": round(covered_w, 4),
        "uncovered_strata": uncovered,
        "unsupported_se": round(math.sqrt(var), 4),
        "NOTE": (
            "The three weighted shares sum to population_coverage, not to 1.0, whenever a "
            "stratum is uncovered. Renormalising them to 1.0 would silently assume the "
            "uncovered strata behave like the covered ones. Do not renormalise; report "
            "the coverage."
        ),
    }


def main() -> int:
    strict = "--strict" in sys.argv

    manifest = load_json(MANIFEST)
    rows = read_sheet(SHEET)

    man_ids = [m["packet_id"] for m in manifest]
    sheet_ids = [r["packet_id"] for r in rows]
    if man_ids != sheet_ids:
        die(
            f"manifest and verdict sheet disagree on the packet ids.\n"
            f"  manifest: {len(man_ids)} ids, sheet: {len(sheet_ids)} ids\n"
            f"  This means the packet was rebuilt after the sheet was started, or the\n"
            f"  rebuild was interrupted. Verdicts recorded against the old ids point at\n"
            f"  different chains and CANNOT be salvaged by matching on order."
        )

    by_pid = {m["packet_id"]: m for m in manifest}
    filled, problems = validate(rows, strict)

    n_total, n_filled = len(rows), len(filled)
    print(
        f"packet: {n_total} chains, {n_filled} judged, {n_total - n_filled} outstanding"
    )
    for p in problems:
        print(f"  PROBLEM  {p}")

    wA, wB, split = population_weights()

    cells_A: dict[str, list] = {
        s: [] for s in STRATUM_TO_CODE if not s.startswith("gate_rejected")
    }
    cells_B: dict[str, list] = {
        s: [] for s in STRATUM_TO_CODE if s.startswith("gate_rejected")
    }
    for r in filled:
        m = by_pid[r["packet_id"]]
        (cells_B if m["arm"] == "gate_rejected" else cells_A)[m["stratum_code"]].append(
            r
        )

    report: dict = {
        "study": "human adjudication of 30 released chains -- issue #176",
        "stage": "2 of 2 -- one human annotator. The arm that licenses stage 1.",
        "estimand": (
            "post-stratified human chain precision on the 2,772-chain reporting unit, "
            "weights from #175's host-proportional 200-chain sample"
        ),
        "annotators": 1,
        "inter_annotator_agreement": None,
        "progress": {"n_chains": n_total, "n_judged": n_filled},
        "sheet_problems": problems,
    }

    if n_filled == 0:
        report["status"] = (
            "DRY RUN -- no verdicts yet. Weights and design are fixed below."
        )
        report["weights_arm_A_reporting_unit"] = {k: round(v, 4) for k, v in wA.items()}
        report["weights_arm_B_gate_rejected"] = {k: round(v, 4) for k, v in wB.items()}
        report["within_code_assert_split_from_179"] = {
            k: {str(kk): round(vv, 4) for kk, vv in v.items()} for k, v in split.items()
        }
        report["planned_cells"] = {
            s: {
                "n": sum(1 for m in manifest if m["stratum_code"] == s),
                "reason_code": STRATUM_TO_CODE[s][0],
                "asserted_split": STRATUM_TO_CODE[s][1],
            }
            for s in sorted({m["stratum_code"] for m in manifest})
        }
        print("\nDRY RUN. Design and weights fixed; no verdicts to analyse yet.")
        print("  arm A weights:", {k: round(v, 3) for k, v in wA.items()})
        print(
            "  #179 split   :",
            {k: {str(a): round(b, 3) for a, b in v.items()} for k, v in split.items()},
        )
        for s, d in sorted(report["planned_cells"].items()):
            print(f"    {d['n']:>2}  {s}")
    else:
        report["status"] = "PARTIAL" if n_filled < n_total else "COMPLETE"
        report["arm_A_reporting_unit"] = post_stratify(cells_A, wA, split)
        report["arm_B_gate_rejected"] = post_stratify(cells_B, wB, split)
        report["arm_B_CAVEAT"] = (
            "Six observations. This is a direction, not a confirmation of the 17.8 pp "
            "gate discrimination #175 measured. Do not print a human-anchored pp figure."
        )

        # How often the machine was right, on the ONE field worded identically to what it
        # was asked. This is the comparison the binary field exists for.
        conf = Counter()
        for r in filled:
            m = by_pid[r["packet_id"]]
            code = m["stratum_code"]
            machine = STRATUM_TO_CODE[code][1]
            if machine is None:
                continue
            conf[(machine, r["risk_link_asserted"] == "yes")] += 1
        report["machine_vs_human_on_link_asserted"] = {
            "n": sum(conf.values()),
            "agree": conf[(True, True)] + conf[(False, False)],
            "machine_yes_human_no": conf[(True, False)],
            "machine_no_human_yes": conf[(False, True)],
            "SCOPE": (
                "Only the strata where #179 gave a machine answer to this exact question. "
                "It is not a corpus accuracy rate for the confidence label."
            ),
        }

        raw = Counter(r["verdict"] for r in filled)
        k = raw["unsupported"]
        lo, hi = wilson(k, n_filled)
        report["unweighted_over_the_30"] = {
            **{v: raw[v] for v in VERDICTS},
            "unsupported_wilson_95": [round(lo, 4), round(hi, 4)],
            "WARNING": (
                "The 30 are a stratified, deliberately non-random sample. This block is a "
                "sanity check on the cells, NOT a rate for anything. Never quote it."
            ),
        }

        mins = [
            float(r["minutes_spent"])
            for r in filled
            if (r.get("minutes_spent") or "").strip()
        ]
        if mins:
            report["cost"] = {
                "n_timed": len(mins),
                "total_hours": round(sum(mins) / 60, 2),
                "median_minutes": sorted(mins)[len(mins) // 2],
                "projected_hours_for_30": round(sum(mins) / len(mins) * 30 / 60, 2),
            }

        print(
            f"\narm A (reporting unit), post-stratified: {report['arm_A_reporting_unit']['weighted']}"
        )
        print(
            f"  population coverage: {report['arm_A_reporting_unit']['population_coverage']}"
        )
        if report["arm_A_reporting_unit"]["uncovered_strata"]:
            print(f"  UNCOVERED: {report['arm_A_reporting_unit']['uncovered_strata']}")
        print(
            f"arm B (gate-rejected, n=6): {report['arm_B_gate_rejected']['weighted']}  [direction only]"
        )

    report["LIMITS"] = (
        "One annotator, so no inter-annotator agreement and reviewer R11 stays open. "
        "Precision only: this packet shows chains that were emitted, so it says nothing "
        "about omission and does not touch the 0.6% vs 28.8% discrepancy. Maturity labels "
        "are not rated. The 70% containment collapse is not evaluated. Non-chain-yielding "
        "documents are out of frame. Cells are 1-7 chains and the intervals are wide; "
        "report them."
    )

    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
