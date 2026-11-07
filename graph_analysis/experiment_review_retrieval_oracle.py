#!/usr/bin/env python
"""Four-arm follow-up to the retrieval comparison (review C2, v5.1 review round).

The v5.1 reviewers asked to separate "the matcher is weak" from "the chain is unhelpful".
Their requests, as stated in paper/reviews_2026-11-06_v51/, and the arm that answers each:

  Opus 5.5 workshop Q1 / conference Q4 / GPT-6.1 Sol both: the source document's own
      chains, i.e. perfect matching ................................. arm ORACLE
  Opus 5.5 conference Q4: "an LLM that matches the question to chains and renders the
      selected chain in prose" ...................................... arm MATCHED
  GPT-6.1 Sol conference Q4: "a chain-plus-passage system" ......... arm CHAIN+PASSAGE
  the baseline every request compares against ........................ arm PASSAGE
  Opus 5.5 conference Q4: questions by a model different from the passage-arm
      synthesizer: the cached gpt-6.1-sol questions are kept and every arm is now written
      by gemini-3.1-pro-preview.
  GPT-6.1 Sol conference Q4: no comparison may change several components at once:
      MATCHED vs ORACLE differ only in the candidate chains; ORACLE vs PASSAGE only in the
      evidence (chains vs passages); CHAIN+PASSAGE vs PASSAGE only in the added chains.
  Opus 5.5 workshop Q1: success = "preferred, or tie": a ranking is kept.
  The author's measure (2026-11-06): material equivalence to the source document's answer,
      ignoring wording, length and format, reported first.

DESIGN (fixed here before any call; copied verbatim into the receipt)
----------------------------------------------------------------------
Questions: the 20 chain-yielding questions of experiment_review_retrieval_comparison.py,
reused unchanged (question, reference answer, spans). The 5 no-chain questions are not
re-run: no ORACLE arm exists for them.

Writer for all four arms: gemini-3.1-pro-preview, one call per arm per question, through the
Gemini batch API. Every prompt says to answer from the given evidence only and to cite the
source URL of each claim.
  * MATCHED: candidates are the chains rooted at the 20 risk nodes nearest the question by
    cosine (text-embedding-3-small, the cached question vector), among the 2,256 risk nodes
    that root a reporting-unit chain. The writer picks one chain, or none, and answers from it.
  * ORACLE: candidates are all reporting-unit chains whose source is the question's own
    document; same prompt as MATCHED.
  * PASSAGE: the ten BM25 passages of the first run, unchanged.
  * CHAIN+PASSAGE: the MATCHED candidates plus the ten passages; the writer answers from both.
A chain is shown as its node names and node descriptions in logical-chain order, each with
its stage label, plus its source URL: what the released graph stores for that chain.

Judge: claude-opus-5-5, effort high, through the Message Batches API, one call per question.
It receives the question, the reference answer and spans, the four answers in seeded random
order labeled W, X, Y, Z, and the full text of every corpus document any answer cites. Per
answer: material equivalence to the reference (same risk, same intervention, same mechanism,
ignoring wording, length, format and fluency; yes / partial / no); answers the question;
source support with a verbatim span per supported claim; citation correctness; mechanism
completeness (risk, intermediate steps, intervention). Then a ranking of the four, ties
allowed, by which answer a researcher would rather receive.

Analysis, fixed before the run: counts per measure per arm over the 20; pairwise
win / tie / loss from the ranking for ORACLE vs PASSAGE, MATCHED vs PASSAGE, CHAIN+PASSAGE
vs PASSAGE and MATCHED vs ORACLE; how often MATCHED's chosen chain comes from the source
document and how often its candidates include one. No pooled score, no decimals.

Limits: n = 20; model judge; the judge shares a provider with the v5 audit judge; the
reference answer and the questions come from gpt-6.1-sol, a provider used by no arm.

CLASS A (metered: Gemini batch, Anthropic batch). Run from graph_analysis/:
    python -u experiment_review_retrieval_oracle.py                  # dry run, tokens + USD
    python -u experiment_review_retrieval_oracle.py --gen --smoke    # 1 question, 4 calls
    python -u experiment_review_retrieval_oracle.py --judge --smoke
    python -u experiment_review_retrieval_oracle.py --gen            # all 20
    python -u experiment_review_retrieval_oracle.py --judge
    python -u experiment_review_retrieval_oracle.py --score
Requires the --prep caches and the --run raw responses of
experiment_review_retrieval_comparison.py. Environment: REVIEW_ANTHROPIC_ENV.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

import experiment_review_retrieval_comparison as RC

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
FIRST = ROOT / "phase2_results/experiment_review_retrieval_comparison_report.json"
WORK = ROOT / "phase2_results/retrieval_oracle_raw"
OUT = ROOT / "phase2_results/experiment_review_retrieval_oracle_report.json"
GEMINI_ENV = ROOT.parent.parent / ".env"
WRITER = "gemini-3.1-pro-preview"
JUDGE = "claude-opus-5-5"
N_ROOTS_MATCHED = 20
ARMS = ["matched", "oracle", "passage", "chain_passage"]
LABELS = ["W", "X", "Y", "Z"]
MAX_OUT = 32_000
# USD per million tokens at batch rates (half of list): Gemini 3.1 Pro 2 / 12 list
# (aggregators, 2026-11-06), claude-opus-5-5 4 / 20 list.
PRICE = {WRITER: (1.0, 6.0), JUDGE: (2.0, 10.0)}
LEVEL3 = {"yes", "partial", "no"}
DECLINED = "[No answer: none of the evidence given answers the question.]"


def log(msg: str) -> None:
    RC.log(msg)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def load_questions() -> list:
    RC.require(FIRST, "experiment_review_retrieval_comparison.py --run")
    rep = json.loads(FIRST.read_text(encoding="utf-8"))
    out = []
    for row in rep["per_query"]:
        if row["stratum"] != "chain":
            continue
        d = RC.RAW / row["qid"]
        qg = RC.parse_qgen(
            json.loads((d / "qgen.json").read_text(encoding="utf-8"))["text"]
        )
        if qg["status"] != "ok":
            RC.fail(f"cached question for {row['qid']} does not parse", d / "qgen.json")
        out.append(
            {
                "qid": row["qid"],
                "url": row["source_url"],
                "question": qg["question"],
                "reference_answer": qg["reference_answer"],
                "spans": qg["spans"],
                "qvec": np.array(
                    json.loads(
                        (d / "question_embedding.json").read_text(encoding="utf-8")
                    )["vector"],
                    dtype=np.float32,
                ),
                "passage_rows": [
                    p["chunk_row"]
                    for p in json.loads(
                        (d / "arm_b_passages.json").read_text(encoding="utf-8")
                    )
                ],
            }
        )
    if len(out) != 20:
        RC.fail(f"expected 20 chain-yielding questions, found {len(out)}", FIRST)
    return out


def chain_block(idx: RC.Index, meta: dict, c: dict) -> str:
    lines = [f"[Chain {c['chain_idx']}] source URL: {c['url']}"]
    for n in c["path"]:
        desc = (meta["description"].get(str(n)) or "").strip()
        lines.append(
            f"  {idx.stage[n].capitalize()}: {idx.name[n]}"
            + (f" -- {desc}" if desc else "")
        )
    return "\n".join(lines)


def matched_candidates(idx: RC.Index, qvec: np.ndarray) -> list:
    q = qvec / np.linalg.norm(qvec)
    rv = idx.vecs[[idx.row[r] for r in idx.roots]] @ q
    top = np.argsort(-rv)[:N_ROOTS_MATCHED]
    cands = []
    for i in top:
        cands += idx.by_root[idx.roots[i]]
    return sorted(cands, key=lambda c: c["chain_idx"])


def oracle_candidates(idx: RC.Index, url: str) -> list:
    return sorted(
        [c for c in idx.chains if c["url"] == url], key=lambda c: c["chain_idx"]
    )


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

CHAIN_PROMPT = """\
Below are mechanism chains from a graph of the AI safety literature. Each chain runs from a \
risk through intermediate stages to an intervention, and cites the source document it was \
extracted from.

Pick the ONE chain that best answers the question, or none if no chain answers it. Then \
answer the question in prose from that chain only. Do not add knowledge that is not in the \
chain. Cite the chain's source URL in parentheses after each claim.

Return only JSON: {{"chain_id": <number or null>, "answer": "..."}}

Question: {question}

{chains}"""

PASSAGE_PROMPT = """\
Answer the question below from these passages only. Do not add knowledge that is not in the \
passages. Cite the source URL of each claim in parentheses right after it. If the passages \
do not answer the question, say so.

Return only JSON: {{"chain_id": null, "answer": "..."}}

Question: {question}

{passages}"""

CHAIN_PASSAGE_PROMPT = """\
Answer the question below from the mechanism chains and the passages given, and from \
nothing else. Each chain runs from a risk through intermediate stages to an intervention and \
cites its source document; each passage gives its source URL. Do not add knowledge that is \
not in them. Cite the source URL of each claim in parentheses right after it. If they do not \
answer the question, say so.

Return only JSON: {{"chain_id": <number of a chain you used, or null>, "answer": "..."}}

Question: {question}

=== CHAINS ===
{chains}

=== PASSAGES ===
{passages}"""

JUDGE_PROMPT = """\
You are judging four answers to a question about the AI safety literature, labeled W, X, Y \
and Z. You have the answer that one source document gives (the reference), with verbatim \
spans from it, and the full text of every document any answer cites.

Score each answer separately:

1. material_equivalence: does it state the same risk, the same intervention and the same \
mechanism as the reference? Ignore wording, length, format, fluency and level of detail: an \
answer that says the same thing in fewer words is equivalent. "yes" = all three match; \
"partial" = one or two match; "no" = none matches.
2. answers_question: does it address the risk asked about and give a proposal and the \
mechanism by which it reduces that risk? The reference is one correct answer, not the only \
one. "yes" / "partial" / "no".
3. source_support: is every claim supported by the source it cites? "yes" / "partial" / \
"no". For every supported claim quote a VERBATIM span (5 to 40 words) from the cited \
document's text below.
4. mechanism_completeness: risk present; intermediate steps from the intervention to the \
risk reduction; intervention present. Each "yes" / "partial" / "no".
5. citation_correctness: is each cited document the one that makes the argument attributed \
to it? "yes" / "partial" / "no".

Then rank the four answers by which one a researcher would rather receive, best first, \
putting answers you consider equally good in the same group.

Return only JSON, no prose around it:
{{"W": {{"material_equivalence": "...", "material_equivalence_reason": "...",
        "answers_question": "...", "source_support": "...",
        "support_spans": [{{"claim": "...", "cited_url": "...", "span": "..."}}],
        "mechanism_completeness": {{"risk": "...", "intermediate_steps": "...", "intervention": "..."}},
        "citation_correctness": "..."}},
  "X": {{ same keys }}, "Y": {{ same keys }}, "Z": {{ same keys }},
  "ranking": [["best label(s)"], ["next"], ...], "ranking_reason": "..."}}

=== QUESTION ===
{question}

=== REFERENCE ANSWER (from the source document) ===
{reference_answer}

Reference spans:
{spans}

=== ANSWER W ===
{W}

=== ANSWER X ===
{X}

=== ANSWER Y ===
{Y}

=== ANSWER Z ===
{Z}

=== FULL TEXT OF EVERY CITED DOCUMENT ===
{cited_texts}"""


def passages_block(passages: list) -> str:
    return "\n\n".join(
        f"[Passage {k + 1}] source URL: {p['url']}\ntitle: {p['title']}\n{p['text']}"
        for k, p in enumerate(passages)
    )


def build_writer_prompts(q: dict, idx: RC.Index, meta: dict) -> dict:
    mc = matched_candidates(idx, q["qvec"])
    oc = oracle_candidates(idx, q["url"])
    if not oc:
        RC.fail(f"no chain for the source document of {q['qid']}", RC.PATHS)
    passages = [idx.chunk(r) for r in q["passage_rows"]]
    mblock = "\n\n".join(chain_block(idx, meta, c) for c in mc)
    return {
        "prompts": {
            "matched": CHAIN_PROMPT.format(question=q["question"], chains=mblock),
            "oracle": CHAIN_PROMPT.format(
                question=q["question"],
                chains="\n\n".join(chain_block(idx, meta, c) for c in oc),
            ),
            "passage": PASSAGE_PROMPT.format(
                question=q["question"], passages=passages_block(passages)
            ),
            "chain_passage": CHAIN_PASSAGE_PROMPT.format(
                question=q["question"], chains=mblock, passages=passages_block(passages)
            ),
        },
        "matched_candidate_chains": [c["chain_idx"] for c in mc],
        "matched_candidates_include_source": any(c["url"] == q["url"] for c in mc),
        "oracle_chains": [c["chain_idx"] for c in oc],
        "passage_urls": [p["url"] for p in passages],
    }


# --------------------------------------------------------------------------
# Parsers (tests/test_retrieval_oracle.py)
# --------------------------------------------------------------------------


def parse_writer(text: str) -> dict:
    d = RC.extract_json(text)
    if not isinstance(d, dict) or not isinstance(d.get("answer"), str):
        return {"status": "malformed"}
    cid = d.get("chain_id")
    if not d["answer"].strip():
        # The prompt allows "none"; the writer expresses it as chain_id null and an empty
        # answer (4 of 80 in the full run). Scored as a declined answer, not dropped.
        if cid is not None:
            return {"status": "malformed"}
        return {"status": "ok", "chain_id": None, "answer": DECLINED, "declined": True}
    if cid is not None:
        try:
            cid = int(cid)
        except (TypeError, ValueError):
            return {"status": "malformed"}
    return {
        "status": "ok",
        "chain_id": cid,
        "answer": d["answer"].strip(),
        "declined": False,
    }


def parse_judge4(text: str) -> dict:
    d = RC.extract_json(text)
    if not isinstance(d, dict):
        return {"status": "malformed"}
    out = {"status": "ok"}
    for lab in LABELS:
        a = d.get(lab)
        if not isinstance(a, dict):
            return {"status": "malformed"}
        mc = a.get("mechanism_completeness") or {}
        row = {
            "material_equivalence": a.get("material_equivalence"),
            "answers_question": a.get("answers_question"),
            "source_support": a.get("source_support"),
            "citation_correctness": a.get("citation_correctness"),
            "mc_risk": mc.get("risk"),
            "mc_steps": mc.get("intermediate_steps"),
            "mc_intervention": mc.get("intervention"),
            "support_spans": a.get("support_spans") or [],
        }
        for k, v in row.items():
            if k != "support_spans" and v not in LEVEL3:
                return {"status": "malformed"}
        out[lab] = row
    rk = d.get("ranking")
    if not isinstance(rk, list) or not all(isinstance(g, list) for g in rk):
        return {"status": "malformed"}
    flat = [x for g in rk for x in g]
    if sorted(flat) != LABELS:
        return {"status": "malformed"}
    out["ranking"] = rk
    return out


def label_order(qid: str) -> dict:
    """arm -> label, seeded per question."""
    labs = LABELS[:]
    random.Random(f"{RC.SEED}-oracle-{qid}").shuffle(labs)
    return dict(zip(ARMS, labs))


def pairwise(ranking: list, a: str, b: str) -> str:
    """'win' if label a ranks above b, 'tie' if same group, else 'loss'."""
    pos = {lab: i for i, g in enumerate(ranking) for lab in g}
    if pos[a] == pos[b]:
        return "tie"
    return "win" if pos[a] < pos[b] else "loss"


# --------------------------------------------------------------------------
# Batch transport
# --------------------------------------------------------------------------


def gemini_client():
    from google import genai

    return genai.Client(api_key=RC._env_key(GEMINI_ENV, "gemini_api_key"))


def wait(poll, describe: str, every: int = 60) -> None:
    """Heartbeat every 30 s; query the remote every `every` s until poll() returns True."""
    t0 = time.time()
    last_q = -every
    state = "?"
    while True:
        if time.time() - last_q >= every:
            last_q = time.time()
            done, state = poll()
            if done:
                log(f"{describe}: {state} after {time.time() - t0:.0f}s")
                return
        log(
            f"{describe}: waiting, last status {state}, {time.time() - t0:.0f}s elapsed"
        )
        time.sleep(30)


def run_gemini_batch(jobs: list, tag: str) -> None:
    """jobs: (key, prompt, out_path). Writes {"text","usage"} json per key."""
    todo = [j for j in jobs if not j[2].exists()]
    if not todo:
        log(f"{tag}: all {len(jobs)} writer responses cached")
        return
    client = gemini_client()
    idfile = WORK / f"gemini_batch_{tag}.json"
    if idfile.exists():
        meta = json.loads(idfile.read_text(encoding="utf-8"))
        if [k for k, _, _ in todo] != meta["keys"]:
            RC.fail("a submitted Gemini batch does not match the pending jobs", idfile)
    else:
        src = [
            {
                "contents": [{"role": "user", "parts": [{"text": p}]}],
                "config": {"max_output_tokens": MAX_OUT},
            }
            for _, p, _ in todo
        ]
        b = client.batches.create(
            model=WRITER, src=src, config={"display_name": f"c2-oracle-{tag}"}
        )
        meta = {"name": b.name, "keys": [k for k, _, _ in todo]}
        idfile.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        log(f"{tag}: submitted Gemini batch {b.name} with {len(todo)} requests")

    holder = {}

    def poll():
        b = client.batches.get(name=meta["name"])
        holder["b"] = b
        st = str(b.state)
        return any(s in st for s in ("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED")), st

    wait(poll, f"gemini {tag}")
    b = holder["b"]
    if "SUCCEEDED" not in str(b.state):
        RC.fail(f"Gemini batch ended {b.state}", idfile)
    resps = b.dest.inlined_responses
    if len(resps) != len(todo):
        RC.fail(f"Gemini batch returned {len(resps)} of {len(todo)}", idfile)
    for (key, _, out), r in zip(todo, resps):
        if r.error:
            RC.fail(f"Gemini request {key} failed: {r.error}", idfile)
        resp = r.response
        u = resp.usage_metadata
        text = resp.text or ""
        rec = {
            "model": WRITER,
            "text": text,
            "finish": str(resp.candidates[0].finish_reason)
            if resp.candidates
            else None,
            "usage": {
                "input_tokens": u.prompt_token_count,
                "output_tokens": (u.candidates_token_count or 0)
                + (u.thoughts_token_count or 0),
            },
        }
        out.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"{tag}: wrote {len(todo)} writer responses")


def run_anthropic_batch(jobs: list, tag: str) -> None:
    todo = [j for j in jobs if not j[2].exists()]
    if not todo:
        log(f"{tag}: all {len(jobs)} judge responses cached")
        return
    client = RC.anthropic_client()
    idfile = WORK / f"anthropic_batch_{tag}.json"
    if idfile.exists():
        meta = json.loads(idfile.read_text(encoding="utf-8"))
        if sorted(k for k, _, _ in todo) != sorted(meta["keys"]):
            RC.fail(
                "a submitted Anthropic batch does not match the pending jobs", idfile
            )
    else:
        reqs = [
            {
                "custom_id": k,
                "params": {
                    "model": JUDGE,
                    "max_tokens": MAX_OUT,
                    "output_config": {"effort": "high"},
                    "messages": [{"role": "user", "content": p}],
                },
            }
            for k, p, _ in todo
        ]
        b = client.messages.batches.create(requests=reqs)
        meta = {"id": b.id, "keys": [k for k, _, _ in todo]}
        idfile.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        log(f"{tag}: submitted Anthropic batch {b.id} with {len(todo)} requests")

    def poll():
        b = client.messages.batches.retrieve(meta["id"])
        return (
            b.processing_status == "ended",
            f"{b.processing_status} {b.request_counts}",
        )

    wait(poll, f"anthropic {tag}")
    outs = {k: o for k, _, o in todo}
    for r in client.messages.batches.results(meta["id"]):
        if r.result.type != "succeeded":
            RC.fail(f"judge request {r.custom_id} {r.result.type}", idfile)
        m = r.result.message
        if m.stop_reason not in ("end_turn", "stop_sequence"):
            RC.fail(f"judge request {r.custom_id} stop_reason {m.stop_reason}", idfile)
        rec = {
            "model": JUDGE,
            "text": "".join(
                b.text for b in m.content if getattr(b, "type", "") == "text"
            ),
            "usage": {
                "input_tokens": m.usage.input_tokens,
                "output_tokens": m.usage.output_tokens,
            },
        }
        outs[r.custom_id].write_text(
            json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    log(f"{tag}: wrote {len(todo)} judge responses")


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


def judge_prompt(q: dict, answers: dict, texts: dict) -> tuple:
    lab = label_order(q["qid"])
    cited = []
    for arm in ARMS:
        for u in RC.cited_urls(answers[arm]):
            if u not in cited:
                cited.append(u)
    blocks = [
        f"--- document {u} ---\n"
        + (
            texts[u]["text"]
            if u in texts
            else "[not a corpus document; no text available]"
        )
        for u in cited
    ]
    by_label = {lab[arm]: answers[arm] for arm in ARMS}
    prompt = JUDGE_PROMPT.format(
        question=q["question"],
        reference_answer=q["reference_answer"],
        spans="\n".join(f'- "{s}"' for s in q["spans"]),
        cited_texts="\n\n".join(blocks),
        **by_label,
    )
    return prompt, cited


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--gen", action="store_true")
    g.add_argument("--judge", action="store_true")
    g.add_argument("--score", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)

    na = RC.load_slim()
    chains = RC.load_chains(na)
    corpus = {a["url"] for a in na.values() if a.get("url")}
    del na
    qs = load_questions()
    if args.smoke:
        qs = qs[:1]
    idx = RC.Index(chains)
    meta = json.loads((RC.RAW / "chain_node_meta.json").read_text(encoding="utf-8"))

    built = {q["qid"]: build_writer_prompts(q, idx, meta) for q in qs}
    for q in qs:
        d = WORK / q["qid"]
        d.mkdir(exist_ok=True)
        info = {k: v for k, v in built[q["qid"]].items() if k != "prompts"}
        (d / "inputs.json").write_text(json.dumps(info, indent=1), encoding="utf-8")

    if not (args.gen or args.judge or args.score):
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")
        per_arm = Counter()
        for b in built.values():
            for arm, p in b["prompts"].items():
                per_arm[arm] += len(enc.encode(p, disallowed_special=()))
        gin = sum(per_arm.values())
        gout = 3000 * 4 * len(qs)
        print(
            json.dumps(
                {
                    "questions": len(qs),
                    "writer_input_tokens_by_arm": dict(per_arm),
                    "writer_usd_batch": round(
                        gin / 1e6 * PRICE[WRITER][0] + gout / 1e6 * PRICE[WRITER][1], 2
                    ),
                    "matched_candidates_include_source": sum(
                        b["matched_candidates_include_source"] for b in built.values()
                    ),
                    "matched_candidate_chains_mean": round(
                        float(
                            np.mean(
                                [
                                    len(b["matched_candidate_chains"])
                                    for b in built.values()
                                ]
                            )
                        ),
                        1,
                    ),
                    "oracle_chains_per_question": [
                        len(b["oracle_chains"]) for b in built.values()
                    ],
                    "judge_note": "judge input depends on the cited documents; estimated after --gen",
                },
                indent=1,
            )
        )
        return 0

    tag = "smoke" if args.smoke else "full"
    if args.gen:
        jobs = [
            (
                f"{q['qid']}:{arm}",
                built[q["qid"]]["prompts"][arm],
                WORK / q["qid"] / f"{arm}.json",
            )
            for q in qs
            for arm in ARMS
        ]
        run_gemini_batch(jobs, tag)
        bad = []
        for k, _, out in jobs:
            p = parse_writer(json.loads(out.read_text(encoding="utf-8"))["text"])
            if p["status"] != "ok":
                bad.append(str(out))
        if bad:
            RC.fail(f"{len(bad)} writer responses do not parse: {bad}")
        log("writer responses parse")
        return 0

    texts = RC.ard_texts(corpus)
    answers = {}
    for q in qs:
        answers[q["qid"]] = {
            arm: parse_writer(
                json.loads(
                    (WORK / q["qid"] / f"{arm}.json").read_text(encoding="utf-8")
                )["text"]
            )
            for arm in ARMS
        }
    if args.judge:
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")
        jobs, tok = [], 0
        for q in qs:
            prompt, _ = judge_prompt(
                q, {a: answers[q["qid"]][a]["answer"] for a in ARMS}, texts
            )
            tok += len(enc.encode(prompt, disallowed_special=()))
            jobs.append((q["qid"], prompt, WORK / q["qid"] / "judge.json"))
        log(
            f"judge input {tok:,} tokens, about USD {tok / 1e6 * PRICE[JUDGE][0] + len(qs) * 8000 / 1e6 * PRICE[JUDGE][1]:.2f} at batch rates"
        )
        run_anthropic_batch(jobs, tag)
        return 0

    # --score
    rows = []
    for q in qs:
        rec = json.loads((WORK / q["qid"] / "judge.json").read_text(encoding="utf-8"))
        pj = parse_judge4(rec["text"])
        if pj["status"] != "ok":
            RC.fail(
                f"judge response malformed for {q['qid']}",
                WORK / q["qid"] / "judge.json",
            )
        lab = label_order(q["qid"])
        b = built[q["qid"]]
        _, cited = judge_prompt(
            q, {a: answers[q["qid"]][a]["answer"] for a in ARMS}, texts
        )
        row = {
            "qid": q["qid"],
            "source_url": q["url"],
            "question": q["question"],
            "labels": lab,
        }
        for arm in ARMS:
            j = dict(pj[lab[arm]])
            spans = j.pop("support_spans")
            j["spans_total"] = len(spans)
            j["spans_found_verbatim"] = sum(
                RC.span_found(
                    (s or {}).get("span", ""),
                    texts.get((s or {}).get("cited_url"), {}).get("text", ""),
                )
                for s in spans
            )
            j["chain_id"] = answers[q["qid"]][arm]["chain_id"]
            j["declined"] = answers[q["qid"]][arm]["declined"]
            j["cited_urls"] = RC.cited_urls(answers[q["qid"]][arm]["answer"])
            row[arm] = j
        cid = row["matched"]["chain_id"]
        row["matched_chain_from_source"] = (
            cid is not None and chains[cid]["url"] == q["url"]
        )
        row["matched_candidates_include_source"] = b[
            "matched_candidates_include_source"
        ]
        row["ranking"] = pj["ranking"]
        row["pairwise"] = {
            f"{a}_vs_{c}": pairwise(pj["ranking"], lab[a], lab[c])
            for a, c in (
                ("oracle", "passage"),
                ("matched", "passage"),
                ("chain_passage", "passage"),
                ("matched", "oracle"),
            )
        }
        writer_usage = {
            arm: json.loads(
                (WORK / q["qid"] / f"{arm}.json").read_text(encoding="utf-8")
            )["usage"]
            for arm in ARMS
        }
        row["usage"] = {"writer": writer_usage, "judge": rec["usage"]}
        rows.append(row)

    tally = {}
    for arm in ARMS:
        t = {}
        for k in (
            "material_equivalence",
            "answers_question",
            "source_support",
            "citation_correctness",
            "mc_risk",
            "mc_steps",
            "mc_intervention",
        ):
            t[k] = {
                lv: sum(r[arm][k] == lv for r in rows)
                for lv in ("yes", "partial", "no")
            }
        t["spans_found_verbatim"] = (
            f"{sum(r[arm]['spans_found_verbatim'] for r in rows)}/{sum(r[arm]['spans_total'] for r in rows)}"
        )
        t["chose_no_chain"] = (
            sum(r[arm]["chain_id"] is None for r in rows) if arm != "passage" else None
        )
        t["declined_to_answer"] = sum(r[arm]["declined"] for r in rows)
        tally[arm] = t
    pw = {k: dict(Counter(r["pairwise"][k] for r in rows)) for k in rows[0]["pairwise"]}
    win_in = win_out = j_in = j_out = 0
    for r in rows:
        for u in r["usage"]["writer"].values():
            win_in += u["input_tokens"] or 0
            win_out += u["output_tokens"] or 0
        j_in += r["usage"]["judge"]["input_tokens"]
        j_out += r["usage"]["judge"]["output_tokens"]
    rep = {
        "study": "four-arm retrieval follow-up (review C2, v5.1 review round)",
        "smoke": args.smoke,
        "design": __doc__.split(
            "DESIGN (fixed here before any call; copied verbatim into the receipt)"
        )[1]
        .split("CLASS A")[0]
        .strip(),
        "models": {
            "writer_all_arms": WRITER,
            "judge": JUDGE,
            "questions_and_reference": RC.GEN_MODEL,
        },
        "n_questions": len(rows),
        "by_arm": tally,
        "pairwise_from_ranking": pw,
        "matched_chain_from_source": sum(r["matched_chain_from_source"] for r in rows),
        "matched_candidates_include_source": sum(
            r["matched_candidates_include_source"] for r in rows
        ),
        "usage_tokens": {
            "writer_in": win_in,
            "writer_out": win_out,
            "judge_in": j_in,
            "judge_out": j_out,
        },
        "usd_at_batch_rates": round(
            win_in / 1e6 * PRICE[WRITER][0]
            + win_out / 1e6 * PRICE[WRITER][1]
            + j_in / 1e6 * PRICE[JUDGE][0]
            + j_out / 1e6 * PRICE[JUDGE][1],
            2,
        ),
        "per_query": rows,
    }
    out = OUT.with_name(OUT.stem + "_smoke.json") if args.smoke else OUT
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        json.dumps(
            {
                k: rep[k]
                for k in (
                    "by_arm",
                    "pairwise_from_ranking",
                    "matched_chain_from_source",
                    "matched_candidates_include_source",
                    "usage_tokens",
                    "usd_at_batch_rates",
                )
            },
            indent=1,
        )
    )
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
