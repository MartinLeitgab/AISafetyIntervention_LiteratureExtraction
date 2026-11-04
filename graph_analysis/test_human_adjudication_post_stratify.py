"""Test: post_stratify must not report SE = 0 when a covered cell holds one chain.

Before the 2026-11-04 fix, a single-chain cell gave p(1-p)/n = 0 and the stratified SE
collapsed to 0.0. Class B, no I/O beyond importing the module.

    cd graph_analysis
    python -u test_human_adjudication_post_stratify.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import experiment_review_human_adjudication as e  # noqa: E402

WEIGHTS = {"faithful": 0.5, "intervention_not_proposed": 0.5}


def row(v):
    return {"verdict": v}


def main():
    # Broken case: every covered cell has one chain -> SE must be undefined.
    one = e.post_stratify(
        {
            "faithful_both_agree": [row("unsupported")],
            "intervention_not_proposed": [row("faithful")],
        },
        WEIGHTS,
        {"faithful": {True: 1.0, False: 0.0}},
    )
    assert one["unsupported_se"] is None, one
    assert one["cells_with_one_chain"] == [
        "faithful_both_agree",
        "intervention_not_proposed",
    ], one
    assert one["weighted"]["unsupported"] == 0.5, one

    # Fixed path: cells of 2 give a real, positive SE.
    two = e.post_stratify(
        {
            "faithful_both_agree": [row("unsupported"), row("faithful")],
            "intervention_not_proposed": [row("faithful"), row("faithful")],
        },
        WEIGHTS,
        {"faithful": {True: 1.0, False: 0.0}},
    )
    assert two["unsupported_se"] is not None and two["unsupported_se"] > 0, two
    assert two["cells_with_one_chain"] == [], two
    print(
        "PASS: singleton cells give SE None; cells of 2 give SE", two["unsupported_se"]
    )


if __name__ == "__main__":
    main()
