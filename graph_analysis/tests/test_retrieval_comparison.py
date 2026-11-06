"""Zero-call tests for experiment_review_retrieval_comparison.py (review C2).

Run from graph_analysis/:  python -m pytest tests/test_retrieval_comparison.py -q
Real-response fixtures (tests/fixtures/retrieval_comparison_*.json) are captured from the
smoke run; the tests that need them skip with a reason until they exist.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiment_review_retrieval_comparison as RC  # noqa: E402

FIX = Path(__file__).parent / "fixtures"

JUDGE_OK = {
    "X": {
        "answers_question": "yes",
        "source_support": "partial",
        "support_spans": [
            {"claim": "c", "cited_url": "https://a", "span": "the quick brown fox"}
        ],
        "mechanism_completeness": {
            "risk": "yes",
            "intermediate_steps": "partial",
            "intervention": "yes",
        },
        "citation_correctness": "yes",
    },
    "Y": {
        "answers_question": "no",
        "source_support": "no",
        "support_spans": [],
        "mechanism_completeness": {
            "risk": "no",
            "intermediate_steps": "no",
            "intervention": "no",
        },
        "citation_correctness": "no",
    },
    "preference": "X",
}


def test_parse_qgen_ok_none_malformed():
    ok = RC.parse_qgen(
        '```json\n{"question": "Q?", "reference_answer": "A", "spans": ["s1"]}\n```'
    )
    assert ok["status"] == "ok" and ok["question"] == "Q?"
    assert RC.parse_qgen('{"none": "no intervention"}')["status"] == "none"
    assert RC.parse_qgen("I cannot do that")["status"] == "malformed"
    assert (
        RC.parse_qgen('{"question": "Q?", "reference_answer": "A", "spans": []}')[
            "status"
        ]
        == "malformed"
    )
    assert (
        RC.parse_qgen('{"question": "", "reference_answer": "A", "spans": ["s"]}')[
            "status"
        ]
        == "malformed"
    )


def test_parse_judge_ok_and_malformed():
    p = RC.parse_judge("Here is my verdict:\n" + json.dumps(JUDGE_OK))
    assert p["status"] == "ok" and p["preference"] == "X"
    assert p["X"]["mc_steps"] == "partial" and p["Y"]["mc_risk"] == "no"
    bad = json.loads(json.dumps(JUDGE_OK))
    bad["X"]["source_support"] = "mostly"
    assert RC.parse_judge(json.dumps(bad))["status"] == "malformed"
    bad = json.loads(json.dumps(JUDGE_OK))
    bad["preference"] = "A"
    assert RC.parse_judge(json.dumps(bad))["status"] == "malformed"
    assert (
        RC.parse_judge('{"X": {}, "Y": {}, "preference": "tie"}')["status"]
        == "malformed"
    )
    part = json.loads(json.dumps(JUDGE_OK))
    part["X"]["mechanism_completeness"]["risk"] = "partial"  # seen in the full run
    assert RC.parse_judge(json.dumps(part))["status"] == "ok"
    part["X"]["mechanism_completeness"]["risk"] = "maybe"
    assert RC.parse_judge(json.dumps(part))["status"] == "malformed"
    assert RC.parse_judge("")["status"] == "malformed"


def test_span_found_normalizes_whitespace_quotes_case():
    text = "The  Quick\nbrown ‘fox’ jumps"
    assert RC.span_found("the quick brown 'fox'", text)
    assert not RC.span_found("the slow brown fox", text)
    assert not RC.span_found("   ", text)


def test_cited_urls_dedup_and_trailing_punctuation():
    a = "Claim one (https://arxiv.org/abs/1). Claim two (https://arxiv.org/abs/1), and https://x.org/p."
    assert RC.cited_urls(a) == ["https://arxiv.org/abs/1", "https://x.org/p"]


def test_split_strata_20_5_with_replacement():
    chain = [f"c{i}" for i in range(30)]
    noch = [f"n{i}" for i in range(10)]
    st = RC.split_strata(chain, noch, {}, 20, 5)
    assert st["chain"]["pending"] == chain[:20] and st["nochain"]["pending"] == noch[:5]
    acc = {u: True for u in chain[:20]} | {u: True for u in noch[:5]}
    acc["c3"] = False  # generator returned NONE -> replaced by c20
    acc["n1"] = False
    st = RC.split_strata(chain, noch, acc, 20, 5)
    assert st["chain"]["pending"] == ["c20"] and st["nochain"]["pending"] == ["n5"]
    acc["c20"] = True
    acc["n5"] = True
    st = RC.split_strata(chain, noch, acc, 20, 5)
    assert len(st["chain"]["chosen"]) == 20 and "c3" not in st["chain"]["chosen"]
    assert st["chain"]["skipped"] == ["c3"] and st["nochain"]["skipped"] == ["n1"]
    assert len(st["nochain"]["chosen"]) == 5 and not st["chain"]["pending"]
    # strata never mix
    assert not set(st["chain"]["chosen"]) & set(st["nochain"]["chosen"])


def test_bm25_weights_match_direct_formula():
    docs = [
        ["risk", "reward", "hacking"],
        ["reward", "model"],
        ["risk", "risk", "oversight", "model"],
    ]
    vocab = {t: i for i, t in enumerate(sorted({t for d in docs for t in d}))}
    rows, cols, vals = [], [], []
    for i, d in enumerate(docs):
        for t in set(d):
            rows.append(i)
            cols.append(vocab[t])
            vals.append(d.count(t))
    tf = sparse.csr_matrix(
        (vals, (rows, cols)), shape=(3, len(vocab)), dtype=np.float32
    )
    w = RC.bm25_weights(tf).toarray()
    avgdl = sum(len(d) for d in docs) / 3
    for i, d in enumerate(docs):
        for t, j in vocab.items():
            f = d.count(t)
            df = sum(t in dd for dd in docs)
            idf = math.log(1 + (3 - df + 0.5) / (df + 0.5))
            want = (
                idf
                * f
                * (RC.BM25_K1 + 1)
                / (f + RC.BM25_K1 * (1 - RC.BM25_B + RC.BM25_B * len(d) / avgdl))
                if f
                else 0
            )
            assert w[i, j] == pytest.approx(want, rel=1e-5)


def _fixture(name):
    p = FIX / f"retrieval_comparison_{name}.json"
    if not p.exists():
        pytest.skip(
            f"real-response fixture {p} not captured yet (smoke run captures it)"
        )
    return json.loads(p.read_text(encoding="utf-8"))["text"]


def test_real_qgen_fixture_parses():
    assert RC.parse_qgen(_fixture("qgen"))["status"] in {"ok", "none"}


def test_real_synthesis_fixture_cites_urls():
    assert RC.cited_urls(_fixture("arm_b"))


def test_real_judge_fixture_parses():
    assert RC.parse_judge(_fixture("judge"))["status"] == "ok"
