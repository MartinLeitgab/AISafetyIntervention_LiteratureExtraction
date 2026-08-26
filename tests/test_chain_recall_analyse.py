#!/usr/bin/env python3
"""Tests for the half of the chain-recall study that runs AFTER the money is spent.

This project has already paid once for a study whose post-hoc half was untested. The
analysis here is the part that turns a batch of JSON into the number the paper would quote,
and three things about it must hold before any batch is submitted:

  1. The ablation arm has to actually detect a pair we deliberately deleted, and has to
     report LOW sensitivity when the judge misses them. Sensitivity governs whether any
     other number in the report may be published at all, so a bug here is the expensive
     kind: it would licence a miss rate the instrument cannot support.
  2. Cohorts must be counted separately. A document sits in more than one cohort (the
     packet-30 and human-validation-10 overlap by construction), so a naive counter would
     let one document's arguments land in one bucket only.
  3. Unparseable rows must be counted as errors, never silently dropped into a denominator.

No LLM, no network. Runs in well under a second.

    uv run python tests/test_chain_recall_analyse.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "cr", ROOT / "graph_analysis" / "experiment_review_chain_recall.py"
)
cr = importlib.util.module_from_spec(spec)
sys.modules["cr"] = cr
spec.loader.exec_module(cr)


def arg(status, risk="reward hacking", iv="human preference learning", steps=3):
    return {
        "risk": risk,
        "intervention": iv,
        "risk_quote": "q",
        "intervention_quote": "q",
        "supporting_steps": steps,
        "status": status,
        "matched_pair_index": 1 if status == "carried" else None,
        "why": "w",
    }


def test_ablated_intervention_is_detected_and_sets_sensitivity() -> None:
    """Detection keys on the deleted INTERVENTION, not on the (risk, intervention) cell.

    The first version of this study deleted one cell of what is really a cross-product of
    reachable risks by reachable interventions, so both endpoints usually stayed visible in
    other cells and there was nothing to detect. Measured: 8 of 20 deletions left BOTH
    endpoints on display and only 1 removed an endpoint outright. Sensitivity came back
    0/20 and meant nothing. Keying detection on the cell is what made it meaningless, so
    this test pins the endpoint contract.
    """
    sample = [
        {
            "custom_id": "rec-0000",
            "cohorts": ["audited_100"],
            "ablated_intervention": "learning from human preferences",
        },
        {
            "custom_id": "rec-0001",
            "cohorts": ["audited_100"],
            "ablated_intervention": "mechanistic interpretability audits",
        },
    ]
    results = [
        # surfaces the deleted intervention, worded differently, under a DIFFERENT risk than
        # any it was originally paired with -> still counts, because the endpoint is gone
        {
            "custom_id": "rec-0000",
            "verdict": {
                "arguments": [
                    arg(
                        "uncaptured_material",
                        "some entirely other risk",
                        "preferences learning from human",
                    )
                ]
            },
        },
        # misses it entirely -> not detected
        {"custom_id": "rec-0001", "verdict": {"arguments": [arg("carried")]}},
    ]
    got = cr.analyse(results, sample)
    a = got["ablation_arm"]
    assert a["n"] == 2, a
    assert a["detected"] == 1, a["detail"]
    assert a["sensitivity"] == 0.5, a
    assert got["headline_audited_100"]["arguments_enumerated"] == 2, got


def test_a_carried_verdict_on_the_deleted_intervention_is_not_a_detection() -> None:
    """The judge must FLAG it, not merely mention it. Naming the deleted intervention while
    calling it `carried` means the judge thinks a remaining pair covers it -- which is the
    failure mode the endpoint deletion exists to expose, not evidence against it."""
    sample = [
        {
            "custom_id": "rec-0000",
            "cohorts": ["audited_100"],
            "ablated_intervention": "adversarial training",
        }
    ]
    results = [
        {
            "custom_id": "rec-0000",
            "verdict": {
                "arguments": [arg("carried", "some risk", "adversarial training")]
            },
        }
    ]
    got = cr.analyse(results, sample)
    assert got["ablation_arm"]["detected"] == 0, got["ablation_arm"]["detail"]
    assert got["ablation_arm"]["sensitivity"] == 0.0


def test_low_sensitivity_still_reports_and_the_correction_tracks_it() -> None:
    """A miss rate divided by a small sensitivity must blow UP, not quietly stay small."""
    sample = [
        {
            "custom_id": f"rec-{i:04d}",
            "cohorts": ["audited_100"],
            "ablated_intervention": "intervention %d" % i,
        }
        for i in range(10)
    ]
    # 1 of 10 deletions detected; 2 of 10 documents report a material miss
    results = []
    for i in range(10):
        args = [arg("carried")]
        if i == 0:
            args.append(arg("uncaptured_material", "risk 0", "intervention 0"))
        elif i == 1:
            args.append(arg("uncaptured_material", "something else", "another thing"))
        results.append({"custom_id": f"rec-{i:04d}", "verdict": {"arguments": args}})
    got = cr.analyse(results, sample)
    assert got["ablation_arm"]["sensitivity"] == 0.1, got["ablation_arm"]
    raw = got["headline_audited_100"]["material_miss_rate_pct"]
    corrected = got["sensitivity_corrected_material_miss_rate_pct"]
    assert corrected > raw * 5, (raw, corrected)
    assert "may be published" in got["ablation_arm"]["HOW_TO_READ"]


def test_a_document_counts_in_every_cohort_it_belongs_to() -> None:
    sample = [
        {
            "custom_id": "rec-0000",
            "cohorts": ["human_validation_10", "packet_30"],
            "ablated_intervention": None,
        }
    ]
    results = [
        {
            "custom_id": "rec-0000",
            "verdict": {
                "arguments": [
                    arg("carried"),
                    arg("uncaptured_material"),
                    arg("uncaptured_thin", steps=1),
                ]
            },
        }
    ]
    got = cr.analyse(results, sample)
    for c in ("human_validation_10", "packet_30"):
        b = got["by_cohort"][c]
        assert b["documents"] == 1 and b["arguments_enumerated"] == 3, (c, b)
        assert b["uncaptured_material"] == 1 and b["uncaptured_thin"] == 1, (c, b)
        assert b["documents_with_a_material_miss"] == 1, (c, b)
    assert got["ablation_arm"]["n"] == 0
    assert got["sensitivity_corrected_material_miss_rate_pct"] is None


def test_unparseable_rows_are_errors_not_silent_denominator_padding() -> None:
    sample = [
        {
            "custom_id": "rec-0000",
            "cohorts": ["audited_100"],
            "ablated_intervention": None,
        },
        {
            "custom_id": "rec-0001",
            "cohorts": ["audited_100"],
            "ablated_intervention": None,
        },
    ]
    results = [
        {
            "custom_id": "rec-0000",
            "verdict": {"arguments": [arg("uncaptured_material")]},
        },
        {"custom_id": "rec-0001", "error": "unparseable: boom"},
    ]
    got = cr.analyse(results, sample)
    assert got["n_parsed"] == 1 and got["n_errors"] == 1, got
    assert got["headline_audited_100"]["documents"] == 1, got["headline_audited_100"]
    assert got["headline_audited_100"]["material_miss_rate_pct"] == 100.0, got


def test_an_unknown_status_is_surfaced_rather_than_folded_into_carried() -> None:
    sample = [
        {"custom_id": "rec-0000", "cohorts": ["audited_100"], "ablated_pair": None}
    ]
    results = [
        {
            "custom_id": "rec-0000",
            "verdict": {"arguments": [arg("probably_carried"), arg("carried")]},
        }
    ]
    got = cr.analyse(results, sample)
    b = got["headline_audited_100"]
    assert b["unparsed_status"] == 1, b
    assert b["carried"] == 1, b


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
            n += 1
    print(f"\n{n}/{n} passed")
