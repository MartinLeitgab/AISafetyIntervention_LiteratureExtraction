#!/usr/bin/env python3
"""Tests for the #176 post-stratified estimator.

The estimator is the whole reason a 30-chain sample can speak for 2,772 chains, and it is
easy to get wrong in a way that still prints a plausible number. These check the three
failure modes that would matter:

  1. Reweighting must actually move the answer away from the pooled rate. The sample
     over-represents the flagged classes on purpose, so pooling the 30 inflates the
     unsupported share. If weighted == pooled, the weights are not being applied.
  2. An uncovered stratum must reduce reported coverage, never be silently imputed. This
     is the specific hazard the 2026-08-26 rebuild existed to remove, and a regression
     here would re-introduce it invisibly.
  3. A verdict sheet whose ids no longer match the manifest must CRASH, not be salvaged by
     row order. A rebuilt packet re-shuffles ids, so order-matching would attach real
     verdicts to the wrong chains.

No LLM, no network, runs in well under a second.

    uv run python tests/test_human_adjudication_poststratify.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "graph_analysis" / "experiment_review_human_adjudication.py"

spec = importlib.util.spec_from_file_location("human_adj", MOD)
ha = importlib.util.module_from_spec(spec)
sys.modules["human_adj"] = ha
spec.loader.exec_module(ha)


def row(pid: str, verdict: str, asserted: str = "yes") -> dict:
    return {"packet_id": pid, "verdict": verdict, "risk_link_asserted": asserted}


# Population weights as measured by #175's host-proportional 200-chain sample.
WEIGHTS = {
    "faithful": 0.48,
    "intervention_not_proposed": 0.24,
    "risk_framing_invented": 0.165,
    "intermediate_unsupported": 0.075,
    "chain_belongs_to_a_different_document": 0.04,
}
SPLIT = {
    "faithful": {True: 0.788, False: 0.212},
    "risk_framing_invented": {True: 0.242, False: 0.758},
}


def test_weighting_moves_the_answer_away_from_the_pooled_rate() -> None:
    """The big faithful cell must dominate, even though it holds few chains."""
    cells = {
        # 4 chains carrying 37.8% of the population, all clean
        "faithful_both_agree": [row(f"F{i}", "faithful") for i in range(4)],
        # 7 chains carrying 24.0%, all bad -- over-represented in the sample by design
        "intervention_not_proposed": [row(f"I{i}", "unsupported") for i in range(7)],
        "model_disagreement_faithful_but_link_not_asserted": [
            row(f"D{i}", "unsupported") for i in range(3)
        ],
        "risk_framing_invented_both_agree": [
            row(f"V{i}", "unsupported") for i in range(3)
        ],
        "model_disagreement_invented_but_link_asserted": [
            row(f"M{i}", "faithful") for i in range(3)
        ],
        "intermediate_unsupported": [row(f"T{i}", "unsupported") for i in range(3)],
        "known_judge_false_positive": [row("K0", "faithful")],
    }
    got = ha.post_stratify(cells, WEIGHTS, SPLIT)

    pooled = sum(
        1 for rs in cells.values() for r in rs if r["verdict"] == "unsupported"
    ) / sum(len(rs) for rs in cells.values())
    weighted = got["weighted"]["unsupported"]

    # Pooled: 16 of 24 = 66.7%. Weighted: 0.24 + 0.48*0.212 + 0.165*0.758 + 0.075 = 54.2%.
    assert abs(pooled - 16 / 24) < 1e-9, pooled
    assert abs(weighted - 0.5418) < 1e-3, weighted
    assert weighted < pooled - 0.10, (
        f"weighted {weighted} should sit far below pooled {pooled}; if they match, the "
        "weights are not being applied and the reported rate is a rate for 30 hand-picked "
        "chains rather than for the reporting unit"
    )
    assert abs(got["population_coverage"] - 1.0) < 1e-4, got["population_coverage"]
    assert got["uncovered_strata"] == [], got["uncovered_strata"]


def test_an_empty_stratum_reduces_coverage_and_is_never_imputed() -> None:
    """The 2026-08-26 hazard: a blind stratum must show up as a hole, not vanish."""
    cells = {
        "faithful_both_agree": [row("F0", "faithful")],
        "intervention_not_proposed": [row("I0", "unsupported")],
        "model_disagreement_faithful_but_link_not_asserted": [],
        "risk_framing_invented_both_agree": [],
        "model_disagreement_invented_but_link_asserted": [],
        "intermediate_unsupported": [],  # 7.5% -- the cell the rebuild added
        "known_judge_false_positive": [],
    }
    got = ha.post_stratify(cells, WEIGHTS, SPLIT)

    covered = 0.48 * 0.788 + 0.24
    assert abs(got["population_coverage"] - covered) < 1e-4, got["population_coverage"]
    assert got["population_coverage"] < 0.62, "coverage must fall when strata are blind"

    holes = {u["stratum"] for u in got["uncovered_strata"]}
    assert "intermediate_unsupported" in holes, holes
    assert len(holes) == 5, holes

    total = sum(got["weighted"].values())
    assert abs(total - got["population_coverage"]) < 1e-3, (
        "the weighted shares must sum to coverage, not to 1.0 -- renormalising would "
        "silently assume the blind strata behave like the covered ones"
    )


def test_a_stale_verdict_sheet_crashes_rather_than_matching_on_order() -> None:
    import csv
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "manifest.json").write_text(
            json.dumps(
                [
                    {
                        "packet_id": "C01",
                        "arm": "real",
                        "stratum_code": "faithful_both_agree",
                    },
                    {
                        "packet_id": "C02",
                        "arm": "real",
                        "stratum_code": "faithful_both_agree",
                    },
                ]
            ),
            encoding="utf-8",
        )
        sheet = d / "verdict_sheet.csv"
        with sheet.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["packet_id", "verdict"])
            w.writeheader()
            w.writerow({"packet_id": "C01", "verdict": "faithful"})  # one row short

        ha.MANIFEST, ha.SHEET = d / "manifest.json", sheet
        ha.OUT = d / "out.json"
        try:
            ha.main()
        except SystemExit as e:
            assert "CANNOT be salvaged by matching on order" in str(e), str(e)
        else:
            raise AssertionError(
                "a sheet that disagrees with the manifest must crash; silently matching "
                "on row order attaches verdicts to the wrong chains"
            )


def test_an_unexplained_level_verdict_contradiction_is_surfaced() -> None:
    rows = [
        {
            "packet_id": "C01",
            "verdict": "faithful",
            "risk_link_asserted": "yes",
            "risk_inference_level": "3",
            "intervention_inference_level": "0",
            "body_inference_level": "0",
            "notes": "",
        }
    ]
    _, problems = ha.validate(rows, strict=False)
    assert any("levels imply 'unsupported'" in p for p in problems), problems

    rows[0]["notes"] = "level 3 on the risk but the intervention half is verbatim"
    _, problems = ha.validate(rows, strict=False)
    assert problems == [], problems


if __name__ == "__main__":
    saved = (ha.MANIFEST, ha.SHEET, ha.OUT)
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
            n += 1
    ha.MANIFEST, ha.SHEET, ha.OUT = saved
    print(f"\n{n}/{n} passed")
