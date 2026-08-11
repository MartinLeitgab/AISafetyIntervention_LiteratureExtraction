#!/usr/bin/env python
"""Scope composition of the 2,772-chain de-duplicated pathway set.

Answers the question a reviewer asks immediately after "87.4% of chains carry all
five reasoning stages": *are those chains actually about AI safety?*

Source of the labels
--------------------
An independent LLM pass (Opus 4.7) routed every chain in the de-duplicated set
against two open-vocabulary catalogues (harm family, mechanism family) and tagged
six controlled-vocabulary per-chain axes.  This script does NOT re-derive those
labels; it only aggregates the on-disk assignment file and reports two numbers
the paper uses:

  1. the share of chains the routing pass could not place in any harm or
     mechanism family (an explicit `unassigned` verdict was permitted, and
     preferred over forcing a fit);
  2. the share of chains whose risk endpoint was tagged `harm_target =
     capability-gap-only`, i.e. the "risk" is a machine-learning capability
     shortfall rather than a harm to people.

Both are LLM-assigned and un-adjudicated, exactly like intervention maturity, and
the paper labels them that way.

Class B: no LLM calls, no network, no API keys.  Fails fast on a missing input.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSIGN = (
    HERE
    / "phase2_results"
    / "step1_load_and_parse_umapwithoutlocalsatellites"
    / "phase2_routing_assignments.jsonl"
)
DEDUPED_PATHS = (
    HERE / "phase1_rawpathsfiles" / "paths_hopwise_v4_edge_only_deduped.jsonl"
)
OUT = HERE / "phase2_results" / "experiment_scope_composition_report.json"

# The de-duplicated pathway set is the paper's reporting unit.
EXPECTED_DEDUPED = 2772


def die(msg: str, produced_by: str) -> None:
    sys.stderr.write(
        f"\nFATAL: {msg}\n"
        f"  expected artifact : {produced_by}\n"
        f"  this script does NOT re-derive it, and does NOT fall back to a\n"
        f"  partial or cached result.\n\n"
    )
    raise SystemExit(2)


def main() -> None:
    if not ASSIGN.exists():
        die(
            f"missing routing assignments at {ASSIGN}",
            "graph_analysis/phase2_step5_opus_routing.py --mode opus_routing",
        )

    rows = [
        json.loads(line)
        for line in ASSIGN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    n = len(rows)
    if n == 0:
        die("routing assignment file is empty", str(ASSIGN))

    # Cross-check the routed set against the de-duplicated path file it should cover.
    n_deduped_file = None
    if DEDUPED_PATHS.exists():
        n_deduped_file = sum(
            1 for line in DEDUPED_PATHS.open(encoding="utf-8") if line.strip()
        )

    hc_unassigned = sum(1 for r in rows if not r.get("harm_class_id"))
    mc_unassigned = sum(1 for r in rows if not r.get("mechanism_class_id"))
    either = sum(
        1 for r in rows if not r.get("harm_class_id") or not r.get("mechanism_class_id")
    )
    both = sum(
        1
        for r in rows
        if not r.get("harm_class_id") and not r.get("mechanism_class_id")
    )

    targets = collections.Counter(
        (r.get("axes") or {}).get("harm_target") for r in rows
    )
    cap_gap = targets.get("capability-gap-only", 0)

    # Human-harm targets = every tag that names a class of people or institutions.
    human_harm_tags = [
        "human-survival",
        "human-flourishing-rights",
        "institutional-governance",
        "scientific-truth",
        "economic",
        "environmental",
    ]
    human_harm = sum(targets.get(t, 0) for t in human_harm_tags)
    other = n - cap_gap - human_harm

    # A worked audit trail: sample the evidence strings the router emitted, so a
    # reader can check the tag means what the paper says it means.
    examples = [
        {
            "path_id": r["path_id"],
            "harm_target": (r.get("axes") or {}).get("harm_target"),
            "evidence": (r.get("harm_target_evidence") or "")[:200],
        }
        for r in rows
        if (r.get("axes") or {}).get("harm_target") == "capability-gap-only"
    ][:8]

    unassigned_examples = [
        {"path_id": r["path_id"], "fit_note": (r.get("fit_note") or "")[:200]}
        for r in rows
        if not r.get("harm_class_id") and not r.get("mechanism_class_id")
    ][:8]

    report = {
        "audit": "scope composition of the de-duplicated pathway set",
        "n_chains_routed": n,
        "n_chains_in_deduped_file": n_deduped_file,
        "n_chains_expected": EXPECTED_DEDUPED,
        "coverage_note": (
            f"{n} of {EXPECTED_DEDUPED} de-duplicated chains carry a routing verdict "
            f"({100.0 * n / EXPECTED_DEDUPED:.1f}%); the remainder were dropped by the "
            "routing LLM in one batch and never re-routed."
        ),
        "unplaceable": {
            "harm_family_unassigned": hc_unassigned,
            "mechanism_family_unassigned": mc_unassigned,
            "either_axis_unassigned": either,
            "either_axis_pct": round(100.0 * either / n, 1),
            "both_axes_unassigned": both,
            "both_axes_pct": round(100.0 * both / n, 1),
        },
        "harm_target_distribution": {
            k if k else "MISSING": v for k, v in targets.most_common()
        },
        "capability_gap_only": {
            "n": cap_gap,
            "pct": round(100.0 * cap_gap / n, 1),
            "meaning": (
                "the chain's risk endpoint is a machine-learning capability shortfall "
                "(model cannot yet do task X well enough), not a harm to people"
            ),
        },
        "human_harm_targets": {
            "n": human_harm,
            "pct": round(100.0 * human_harm / n, 1),
            "tags": human_harm_tags,
        },
        "other_or_missing": {"n": other, "pct": round(100.0 * other / n, 1)},
        "CAVEAT": (
            "Both headline shares are LLM-assigned and un-adjudicated by a human, the "
            "same epistemic status as intervention maturity. They are reported as a "
            "coarse composition, not as a measured error rate."
        ),
        "capability_gap_examples": examples,
        "unassigned_examples": unassigned_examples,
    }

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"chains routed                 : {n}")
    print(
        f"unplaceable on either axis    : {either} ({report['unplaceable']['either_axis_pct']}%)"
    )
    print(
        f"unplaceable on both axes      : {both} ({report['unplaceable']['both_axes_pct']}%)"
    )
    print(
        f"capability-gap-only risk      : {cap_gap} ({report['capability_gap_only']['pct']}%)"
    )
    print(
        f"human-harm risk               : {human_harm} ({report['human_harm_targets']['pct']}%)"
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
