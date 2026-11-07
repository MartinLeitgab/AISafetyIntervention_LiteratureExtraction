"""Zero-call tests for experiment_review_retrieval_oracle.py (review C2 follow-up).

Run from graph_analysis/:  python -m pytest tests/test_retrieval_oracle.py -q
Real-response fixtures are captured from the smoke run; tests needing them skip until then.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiment_review_retrieval_oracle as RO  # noqa: E402

FIX = Path(__file__).parent / "fixtures"


def _arm(eq="yes"):
    return {
        "material_equivalence": eq,
        "answers_question": "yes",
        "source_support": "partial",
        "support_spans": [],
        "mechanism_completeness": {
            "risk": "yes",
            "intermediate_steps": "partial",
            "intervention": "no",
        },
        "citation_correctness": "yes",
    }


def _judge(ranking=None, **over):
    d = {lab: _arm() for lab in RO.LABELS}
    d.update(over)
    d["ranking"] = ranking if ranking is not None else [["W", "X"], ["Y"], ["Z"]]
    return d


def test_parse_writer():
    assert RO.parse_writer('{"chain_id": 12, "answer": "A (https://x)"}') == {
        "status": "ok",
        "chain_id": 12,
        "answer": "A (https://x)",
        "declined": False,
    }
    # declined, as gemini wrote it 4 times in the full run
    dec = RO.parse_writer('```json\n{"chain_id": null, "answer": ""}\n```')
    assert dec["status"] == "ok" and dec["declined"] and dec["answer"] == RO.DECLINED
    assert RO.parse_writer('{"chain_id": 3, "answer": ""}')["status"] == "malformed"
    assert (
        RO.parse_writer('```json\n{"chain_id": null, "answer": "none"}\n```')[
            "chain_id"
        ]
        is None
    )
    assert RO.parse_writer('{"chain_id": "7", "answer": "a"}')["chain_id"] == 7
    assert (
        RO.parse_writer('{"chain_id": "seven", "answer": "a"}')["status"] == "malformed"
    )
    assert RO.parse_writer("no json")["status"] == "malformed"


def test_parse_judge4_ok_and_malformed():
    p = RO.parse_judge4(json.dumps(_judge()))
    assert p["status"] == "ok" and p["W"]["mc_intervention"] == "no"
    assert (
        RO.parse_judge4(json.dumps(_judge(ranking=[["W", "X", "Y"]])))["status"]
        == "malformed"
    )
    assert (
        RO.parse_judge4(json.dumps(_judge(ranking=[["W"], ["W", "X", "Y", "Z"]])))[
            "status"
        ]
        == "malformed"
    )
    assert (
        RO.parse_judge4(json.dumps(_judge(W=_arm(eq="mostly"))))["status"]
        == "malformed"
    )
    bad = _judge()
    del bad["Z"]
    assert RO.parse_judge4(json.dumps(bad))["status"] == "malformed"
    assert RO.parse_judge4("")["status"] == "malformed"


def test_pairwise_from_ranking():
    rk = [["W", "X"], ["Y"], ["Z"]]
    assert RO.pairwise(rk, "W", "X") == "tie"
    assert RO.pairwise(rk, "W", "Y") == "win"
    assert RO.pairwise(rk, "Z", "Y") == "loss"


def test_label_order_seeded_and_complete():
    a, b = RO.label_order("abc123"), RO.label_order("abc123")
    assert a == b and sorted(a) == sorted(RO.ARMS) and sorted(a.values()) == RO.LABELS
    # not the same assignment for every question
    assert len({tuple(RO.label_order(f"q{i}").values()) for i in range(20)}) > 1


def test_oracle_candidates_are_exactly_the_source_documents_chains():
    class Idx:
        chains = [
            {"chain_idx": 0, "url": "u1"},
            {"chain_idx": 1, "url": "u2"},
            {"chain_idx": 2, "url": "u1"},
        ]

    assert [c["chain_idx"] for c in RO.oracle_candidates(Idx, "u1")] == [0, 2]
    assert RO.oracle_candidates(Idx, "u3") == []


def _fixture(name):
    p = FIX / f"retrieval_oracle_{name}.json"
    if not p.exists():
        pytest.skip(
            f"real-response fixture {p} not captured yet (smoke run captures it)"
        )
    return json.loads(p.read_text(encoding="utf-8"))["text"]


@pytest.mark.parametrize("arm", RO.ARMS)
def test_real_writer_fixtures_parse(arm):
    assert RO.parse_writer(_fixture(arm))["status"] == "ok"


def test_real_judge_fixture_parses():
    assert RO.parse_judge4(_fixture("judge"))["status"] == "ok"


def test_real_declined_fixture():
    p = RO.parse_writer(_fixture("declined"))
    assert p["status"] == "ok" and p["declined"]
