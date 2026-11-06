#!/usr/bin/env python
"""Chain lookup vs passage retrieval plus synthesis on mechanism questions (review C2).

All four reviews of the 2026-11-05 round ask whether looking up a stored chain answers a
mechanism question better than retrieving passages and having a model synthesize an answer.
No further human annotation is available, so the judge is a model; the paper says so.

DESIGN (fixed here before any call; copied verbatim into the receipt)
----------------------------------------------------------------------
Queries, n = 25. Questions are written from the SOURCE TEXT by a model that never sees the
extraction, so they do not favor the chain lookup.
  * 20 documents drawn at random (seed 20261106) from the 1,868 chain-yielding documents and
    5 from the 9,911 corpus documents with no chain. The 5 show what the chain lookup cannot
    answer because the gates removed it; they are reported separately, never pooled.
  * One call per document (gpt-6.1-sol, reasoning effort high): write one question of the
    form "what has been proposed to reduce risk R, and by what mechanism?", naming the risk in
    the generator's own words and not naming the intervention, plus the document's answer with
    verbatim quoted spans. The question goes to both arms; answer and spans only to the judge.
  * A document for which the generator finds no such proposal returns NONE and is replaced by
    the next document of the same stratum in the seeded order; skipped documents are listed.

Arm A, chain lookup (no LLM). Embed the question with text-embedding-3-small, the model of the
stored node embeddings. Take the 5 risk nodes nearest by cosine among the 2,256 risk nodes
that root a reporting-unit chain. Among the chains rooted at them, the top chain is the one
whose nodes have the highest mean cosine to the question. The answer is that chain's node
names in logical-chain order, each with its stage label, plus its source URL.

Arm B, passage retrieval plus synthesis. BM25 (k1 = 1.5, b = 0.75, Lucene idf
log(1 + (N - df + 0.5) / (df + 0.5)), the Elasticsearch default; a sparse-matrix
implementation in place of rank_bm25, whose per-chunk dictionaries do not fit in memory here)
over non-overlapping 500-word chunks of the full ARD text of all 11,779 corpus documents,
tokens = lowercase words of 2+ characters, English stop words removed. The top 10 chunks go
to gpt-6.1-sol (reasoning effort high) with "answer the question from these passages only,
citing the source URL of each claim".

Judge, blind: claude-opus-5-5, effort high. It receives the question, the reference answer and
spans, both answers in seeded random order labeled X and Y, and the full text of every corpus
document either answer cites. Per answer it scores:
  * answers the question: the answer addresses the risk asked about and gives a proposal and
    mechanism for it (yes / partial / no). Added to the handover's list: without it an answer
    about a different risk can score yes on every other measure;
  * source support: every claim supported by its cited source (yes / partial / no);
  * mechanism completeness: risk, intermediate steps, intervention present;
  * citation correctness: the cited document is the one that makes the argument;
  * and which answer a researcher would rather receive, or a tie.
A verbatim quoted span is required for every "yes"; the script checks each span against the
cited text and reports how many are not found verbatim.

Analysis, fixed before the run: counts per measure and per arm, separately for the
chain-yielding and the no-chain queries; no pooled rate across the two groups; no decimals.
Retrieval hit: how often arm A's top chain comes from the query's own source document (the
matching step's accuracy), and, for symmetry, how often arm B's 10 chunks include it.

Known limits, stated in the receipt: model judge, not human relevance judgments; the judge
shares a provider with the v5 audit judge but not with the arm-B synthesizer or the
extractor; arm A's answer is a list of node names and arm B's is prose, so the judge can tell
the arms apart and blinding hides only the label.

CLASS A (metered: OpenAI + Anthropic). Run from graph_analysis/:
    python -u experiment_review_retrieval_comparison.py --prep     # caches, no API calls
    python -u experiment_review_retrieval_comparison.py            # dry run: sample + tokens
    python -u experiment_review_retrieval_comparison.py --smoke    # 1 + 1 queries
    python -u experiment_review_retrieval_comparison.py --run      # all 25
    python -u experiment_review_retrieval_comparison.py --score    # receipt
Environment: REVIEW_ANTHROPIC_ENV = absolute path of a .env holding the Anthropic key.
Every response is persisted to phase2_results/retrieval_comparison_raw/ as it streams; a
re-run reuses any completed response and never re-buys it.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pickle
import random
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
REPO = ROOT.parent
NODES_SLIM = ROOT / "phase2_results/node_attrs_slim.pkl"
NODES_FULL = (
    ROOT
    / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/graph_node_attributes.pkl"
)
PATHS = ROOT / "phase1_rawpathsfiles/paths_hopwise_v4_edge_only_deduped.jsonl"
ARD = REPO / "intervention_graph_creation/data/raw/ard_json_full"
RAW = ROOT / "phase2_results/retrieval_comparison_raw"
OUT = ROOT / "phase2_results/experiment_review_retrieval_comparison_report.json"
OPENAI_ENV = ROOT / ".env"
OPENAI_KEY_VAR = "newopenai_api_key"
ANTHROPIC_ENV_VAR = "REVIEW_ANTHROPIC_ENV"

SEED = 20261106
N_CHAIN, N_NOCHAIN = 20, 5
N_ROOTS, N_CHUNKS, CHUNK_WORDS = 5, 10, 500
BM25_K1, BM25_B = 1.5, 0.75
GEN_MODEL = "gpt-6.1-sol"
JUDGE_MODEL = "claude-opus-5-5"
EMBED_MODEL = "text-embedding-3-small"
MAX_OUT = 32_000
WORKERS = 8
# USD per million tokens (handover, 2026-11-05 aggregator prices).
PRICE = {GEN_MODEL: (2.0, 10.0), JUDGE_MODEL: (4.0, 20.0)}

# Canonical logical-chain order. Never reorder.
STAGES = [
    "risk",
    "problem analysis",
    "theoretical insight",
    "design rationale",
    "implementation mechanism",
    "validation evidence",
    "intervention",
]

URL_RE = re.compile(r"https?://[^\s\)\]\}>\"'`,;]+")


def fail(msg: str, artifact=None, produced_by: str | None = None) -> None:
    lines = [f"FATAL: {msg}"]
    if artifact is not None:
        lines.append(f"  expected artifact: {artifact}")
    if produced_by:
        lines.append(f"  produced by: {produced_by}")
    lines.append(
        "  this script does NOT substitute another model, another index or a cached answer "
        "from a different design."
    )
    raise SystemExit("\n".join(lines))


# --------------------------------------------------------------------------
# Logging and heartbeat
# --------------------------------------------------------------------------

LOCK = threading.Lock()
T0 = time.time()
STATUS: Counter = Counter()


def log(msg: str) -> None:
    with LOCK:
        print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def heartbeat(stop: threading.Event, total: int) -> None:
    while not stop.wait(30):
        with LOCK:
            s = dict(STATUS)
        log(f"heartbeat: {s} of {total} queries")


# --------------------------------------------------------------------------
# Corpus, chains, sample
# --------------------------------------------------------------------------


def require(p: Path, how: str) -> None:
    if not p.exists():
        fail(f"missing input {p.name}", p, how)


def load_slim() -> dict:
    require(NODES_SLIM, "experiment_review_prep_slim_nodes.py")
    return pickle.load(open(NODES_SLIM, "rb"))


def load_chains(na: dict) -> list:
    require(PATHS, "phase1_dedup_paths.py")
    chains = []
    for i, line in enumerate(open(PATHS, encoding="utf-8")):
        p = json.loads(line)["path"]
        chains.append({"chain_idx": i, "path": p, "url": na[p[0]]["url"]})
    return chains


def stage_of(a: dict) -> str:
    if (a.get("type") or "").lower() == "intervention":
        return "intervention"
    return a.get("concept_category") or "unknown"


def ard_texts(wanted: set) -> dict:
    require(ARD, "intervention_graph_creation/data/raw/ard_json_full/download_ard.py")
    out = {}
    for fp in sorted(ARD.glob("*.jsonl")):
        for line in open(fp, encoding="utf-8"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            u = d.get("url")
            if u in wanted and u not in out and (d.get("text") or "").strip():
                out[u] = {"title": d.get("title") or "", "text": d["text"]}
    return out


def draw_sample(na: dict, chains: list) -> dict:
    """Seeded order of candidates per stratum; the first accepted N of each are used."""
    corpus = sorted({a["url"] for a in na.values() if a.get("url")})
    cy = sorted({c["url"] for c in chains})
    cy_set = set(cy)
    nochain = [u for u in corpus if u not in cy_set]
    rng = random.Random(SEED)
    cy_order = cy[:]
    rng.shuffle(cy_order)
    nc_order = nochain[:]
    rng.shuffle(nc_order)
    return {
        "n_corpus": len(corpus),
        "n_chain_yielding": len(cy),
        "n_no_chain": len(nochain),
        "chain": cy_order,
        "nochain": nc_order,
    }


# --------------------------------------------------------------------------
# Prep: chain index (embeddings) and BM25 index. No API calls.
# --------------------------------------------------------------------------


def parse_embedding(raw):
    if raw is None:
        return None
    if isinstance(raw, (list, tuple, np.ndarray)):
        return np.asarray(raw, dtype=np.float32)
    s = str(raw).strip()
    if s.startswith("<") and s.endswith(">"):
        s = s[1:-1]
    v = np.array([float(x) for x in s.split(",")], dtype=np.float32)
    return v


def prep_chain_index(na: dict, chains: list) -> None:
    require(NODES_FULL, "phase2_step1_loadandparse.py")
    needed = sorted({n for c in chains for n in c["path"]})
    log(f"loading {NODES_FULL.name} (3.3 GB) for {len(needed):,} chain nodes ...")
    full = pickle.load(open(NODES_FULL, "rb"))
    vecs = np.zeros((len(needed), 1536), dtype=np.float32)
    for i, n in enumerate(needed):
        v = parse_embedding(full[n].get("embedding"))
        if v is None or v.shape[0] != 1536 or not np.isfinite(v).all():
            fail(f"node {n} has no usable 1536-d embedding", NODES_FULL)
        vecs[i] = v / np.linalg.norm(v)
    names = {n: full[n].get("name") for n in needed}
    descs = {n: full[n].get("description") for n in needed}
    del full
    np.save(RAW / "chain_node_vecs.npy", vecs)
    (RAW / "chain_node_meta.json").write_text(
        json.dumps(
            {
                "node_ids": needed,
                "name": {str(n): names[n] for n in needed},
                "description": {str(n): descs[n] for n in needed},
                "stage": {str(n): stage_of(na[n]) for n in needed},
                "url": {str(n): na[n]["url"] for n in needed},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    log(f"chain index: {len(needed):,} nodes")


def chunk_words(text: str) -> list:
    w = text.split()
    return [" ".join(w[i : i + CHUNK_WORDS]) for i in range(0, len(w), CHUNK_WORDS)]


def bm25_weights(tf):
    """Per-(chunk, term) BM25 weight from a CSR term-count matrix (Lucene idf)."""
    tf = tf.tocsr().astype(np.float32)
    n_docs = tf.shape[0]
    dl = np.asarray(tf.sum(axis=1)).ravel()
    avgdl = dl.mean()
    df = np.bincount(tf.indices, minlength=tf.shape[1])
    idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
    rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
    f = tf.data
    denom = f + BM25_K1 * (1 - BM25_B + BM25_B * dl[rows] / avgdl)
    tf.data = (idf[tf.indices] * f * (BM25_K1 + 1) / denom).astype(np.float32)
    return tf


def prep_bm25(na: dict) -> None:
    from scipy import sparse
    from sklearn.feature_extraction.text import CountVectorizer

    corpus = {a["url"] for a in na.values() if a.get("url")}
    texts = ard_texts(corpus)
    missing = corpus - set(texts)
    if missing:
        fail(
            f"{len(missing)} corpus documents have no ARD text, e.g. {next(iter(missing))}",
            ARD,
        )
    meta = []
    with open(RAW / "bm25_chunks.jsonl", "w", encoding="utf-8") as fh:
        for u in sorted(texts):
            for j, ch in enumerate(chunk_words(texts[u]["text"])):
                meta.append(u)
                fh.write(
                    json.dumps(
                        {"url": u, "title": texts[u]["title"], "chunk": j, "text": ch},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    log(f"chunks: {len(meta):,} over {len(texts):,} documents; vectorizing ...")

    def chunk_iter():
        for line in open(RAW / "bm25_chunks.jsonl", encoding="utf-8"):
            yield json.loads(line)["text"]

    cv = CountVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w\w+\b",
        stop_words="english",
        dtype=np.float32,
    )
    tf = cv.fit_transform(chunk_iter()).tocsr()
    tf = bm25_weights(tf)
    n_docs = tf.shape[0]
    sparse.save_npz(RAW / "bm25_weights.npz", tf.tocsc())
    (RAW / "bm25_vocab.json").write_text(
        json.dumps({k: int(v) for k, v in cv.vocabulary_.items()}), encoding="utf-8"
    )
    log(f"BM25 index: {n_docs:,} chunks x {tf.shape[1]:,} terms, nnz {tf.nnz:,}")


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

QGEN_PROMPT = """\
Below is one document from the AI safety literature.

From this document, write ONE question of the form "What has been proposed to reduce \
<risk>, and by what mechanism?", where <risk> is a risk the document says an intervention \
reduces. Name the risk in your own words: do not copy the document's own term for it, do not \
name the intervention, and do not mention the document, its title or its authors. Then give \
the answer this document gives: the intervention and the mechanism by which it is proposed to \
reduce the risk, with the intermediate steps of the argument. Support the answer with 2 to 6 \
spans quoted VERBATIM from the document.

If the document proposes no intervention against any risk, return {"none": "<one-sentence reason>"}.

Return only JSON, no prose around it:
{"question": "...", "reference_answer": "...", "spans": ["...", "..."]}

=== DOCUMENT (url: {url}) ===
{text}
=== END DOCUMENT ==="""

SYNTH_PROMPT = """\
Answer the question below from these passages only. Do not use any knowledge that is not in \
the passages. Cite the source URL of each claim in parentheses right after it. If the \
passages do not answer the question, say so.

Question: {question}

{passages}"""

JUDGE_PROMPT = """\
You are judging two answers to a question about the AI safety literature. A researcher asked \
the question; each answer cites one or more documents. You have the reference answer that \
one source document gives, with verbatim spans from it, and the full text of every document \
either answer cites. The reference is one correct answer, not the only one: an answer that \
gives a different proposal for the same risk, correctly supported by its own source, can \
still answer the question.

Score each answer, X and Y, separately:

1. answers_question: does it address the risk asked about and give a proposal and the \
mechanism by which it reduces that risk? "yes" / "partial" / "no".
2. source_support: is every claim supported by the source it cites? "yes" = all claims \
supported, "partial" = some, "no" = none or no citation. For every supported claim quote a \
VERBATIM span from the cited document's text below (copy exactly, 5 to 40 words).
3. mechanism_completeness: risk present ("yes"/"no"); intermediate steps from the \
intervention to the risk reduction ("yes"/"partial"/"no"); intervention present ("yes"/"no").
4. citation_correctness: is each cited document the one that makes the argument attributed \
to it? "yes" / "partial" / "no".

Then preference: which answer would a researcher rather receive, "X", "Y" or "tie".

Return only JSON, no prose around it:
{{"X": {{"answers_question": "...", "answers_question_reason": "...",
        "source_support": "...", "support_spans": [{{"claim": "...", "cited_url": "...", "span": "..."}}],
        "unsupported_claims": ["..."],
        "mechanism_completeness": {{"risk": "...", "intermediate_steps": "...", "intervention": "..."}},
        "citation_correctness": "...", "citation_reason": "..."}},
  "Y": {{ same keys }},
  "preference": "...", "preference_reason": "..."}}

=== QUESTION ===
{question}

=== REFERENCE ANSWER (from the source document) ===
{reference_answer}

Reference spans:
{spans}

=== ANSWER X ===
{answer_x}

=== ANSWER Y ===
{answer_y}

=== FULL TEXT OF EVERY CITED DOCUMENT ===
{cited_texts}"""


# --------------------------------------------------------------------------
# Parsers (unit-tested in tests/test_retrieval_comparison.py)
# --------------------------------------------------------------------------

LEVEL3 = {"yes", "partial", "no"}


def extract_json(text: str):
    """The last top-level JSON object in a response; None if there is none."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    end = t.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(t[start : end + 1])
    except json.JSONDecodeError:
        return None


def parse_qgen(text: str) -> dict:
    d = extract_json(text)
    if not isinstance(d, dict):
        return {"status": "malformed"}
    if "none" in d and "question" not in d:
        return {"status": "none", "reason": str(d["none"])}
    q, a, s = d.get("question"), d.get("reference_answer"), d.get("spans")
    if not (isinstance(q, str) and q.strip() and isinstance(a, str) and a.strip()):
        return {"status": "malformed"}
    if not isinstance(s, list) or not s or not all(isinstance(x, str) for x in s):
        return {"status": "malformed"}
    return {
        "status": "ok",
        "question": q.strip(),
        "reference_answer": a.strip(),
        "spans": s,
    }


def _norm(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def span_found(span: str, text: str) -> bool:
    return bool(span.strip()) and _norm(span) in _norm(text)


def parse_judge(text: str) -> dict:
    d = extract_json(text)
    if not isinstance(d, dict):
        return {"status": "malformed"}
    out = {"status": "ok", "preference": d.get("preference")}
    if out["preference"] not in {"X", "Y", "tie"}:
        return {"status": "malformed"}
    for lab in ("X", "Y"):
        a = d.get(lab)
        if not isinstance(a, dict):
            return {"status": "malformed"}
        mc = a.get("mechanism_completeness") or {}
        row = {
            "answers_question": a.get("answers_question"),
            "source_support": a.get("source_support"),
            "citation_correctness": a.get("citation_correctness"),
            "mc_risk": mc.get("risk"),
            "mc_steps": mc.get("intermediate_steps"),
            "mc_intervention": mc.get("intervention"),
            "support_spans": a.get("support_spans") or [],
        }
        # mc_risk and mc_intervention were asked as yes/no; the judge answered "partial"
        # for one query of the full run (e4d3c0203329), so all fields accept three levels
        # and the receipt reports the counts as given (deviation recorded 2026-11-06).
        for k in (
            "answers_question",
            "source_support",
            "citation_correctness",
            "mc_steps",
            "mc_risk",
            "mc_intervention",
        ):
            if row[k] not in LEVEL3:
                return {"status": "malformed"}
        out[lab] = row
    out["raw"] = d
    return out


def cited_urls(answer: str) -> list:
    seen = []
    for u in URL_RE.findall(answer or ""):
        u = u.rstrip(".")
        if u not in seen:
            seen.append(u)
    return seen


def split_strata(
    order_chain: list, order_nochain: list, accepted: dict, n_chain: int, n_nochain: int
) -> dict:
    """First n accepted per stratum in seeded order; `accepted` maps url -> bool, absent =
    not yet generated. Returns the chosen urls, the next urls to generate, the skipped."""
    out = {}
    for name, order, n in (
        ("chain", order_chain, n_chain),
        ("nochain", order_nochain, n_nochain),
    ):
        chosen, pending, skipped = [], [], []
        for u in order:
            if len(chosen) + len(pending) >= n:
                break
            st = accepted.get(u)
            if st is True:
                chosen.append(u)
            elif st is False:
                skipped.append(u)
            else:
                pending.append(u)
        out[name] = {"chosen": chosen, "pending": pending, "skipped": skipped}
    return out


# --------------------------------------------------------------------------
# API clients, streamed and persisted
# --------------------------------------------------------------------------


def _env_key(path: Path, var: str) -> str:
    from dotenv import dotenv_values

    if not path.is_file():
        fail(f"key file not found (variable {var})", path)
    vals = {k.lower(): v for k, v in (dotenv_values(path) or {}).items()}
    key = (vals.get(var.lower()) or "").strip()
    if not key:
        fail(f"variable {var} empty or missing", path)
    return key


_CLIENTS: dict = {}


def openai_client():
    if "openai" not in _CLIENTS:
        from openai import OpenAI

        _CLIENTS["openai"] = OpenAI(
            api_key=_env_key(OPENAI_ENV, OPENAI_KEY_VAR), timeout=1800.0, max_retries=3
        )
    return _CLIENTS["openai"]


def anthropic_client():
    if "anthropic" not in _CLIENTS:
        import anthropic

        p = os.environ.get(ANTHROPIC_ENV_VAR)
        if not p:
            fail(
                f"{ANTHROPIC_ENV_VAR} not set",
                None,
                "export it to the .env holding the key",
            )
        _CLIENTS["anthropic"] = anthropic.Anthropic(
            api_key=_env_key(Path(p), "ANTHROPIC_API_KEY"),
            timeout=1800.0,
            max_retries=3,
        )
    return _CLIENTS["anthropic"]


def call_openai(prompt: str, out: Path) -> dict:
    """Stream gpt-6.1-sol into out.partial; on completion write out (json with text+usage)."""
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    part = out.with_suffix(".partial.txt")
    stream = openai_client().responses.create(
        model=GEN_MODEL,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        reasoning={"effort": "high"},
        max_output_tokens=MAX_OUT,
        stream=True,
    )
    chunks, usage, status = [], {}, None
    with open(part, "w", encoding="utf-8") as fh:
        for ev in stream:
            et = getattr(ev, "type", "")
            if et == "response.output_text.delta":
                chunks.append(ev.delta)
                fh.write(ev.delta)
                fh.flush()
            elif et in ("response.completed", "response.incomplete", "response.failed"):
                r = ev.response
                status = getattr(r, "status", et)
                u = getattr(r, "usage", None)
                if u is not None:
                    usage = {
                        "input_tokens": u.input_tokens,
                        "output_tokens": u.output_tokens,
                    }
            elif et == "error":
                raise RuntimeError(f"openai stream error: {ev}")
    if status != "completed":
        raise RuntimeError(f"openai response status {status} for {out.name}")
    rec = {
        "model": GEN_MODEL,
        "text": "".join(chunks),
        "usage": usage,
        "status": status,
    }
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    part.unlink(missing_ok=True)
    return rec


def call_anthropic(prompt: str, out: Path) -> dict:
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    part = out.with_suffix(".partial.txt")
    with open(part, "w", encoding="utf-8") as fh:
        with anthropic_client().messages.stream(
            model=JUDGE_MODEL,
            max_tokens=MAX_OUT,
            extra_body={"output_config": {"effort": "high"}},
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for t in stream.text_stream:
                fh.write(t)
                fh.flush()
            final = stream.get_final_message()
    if final.stop_reason not in ("end_turn", "stop_sequence"):
        raise RuntimeError(f"anthropic stop_reason {final.stop_reason} for {out.name}")
    text = "".join(b.text for b in final.content if getattr(b, "type", "") == "text")
    rec = {
        "model": JUDGE_MODEL,
        "text": text,
        "usage": {
            "input_tokens": final.usage.input_tokens,
            "output_tokens": final.usage.output_tokens,
        },
        "status": final.stop_reason,
    }
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    part.unlink(missing_ok=True)
    return rec


def embed(text: str, out: Path) -> np.ndarray:
    if out.exists():
        return np.array(
            json.loads(out.read_text(encoding="utf-8"))["vector"], dtype=np.float32
        )
    r = openai_client().embeddings.create(model=EMBED_MODEL, input=text)
    v = r.data[0].embedding
    out.write_text(
        json.dumps({"model": EMBED_MODEL, "vector": v, "usage": r.usage.total_tokens}),
        encoding="utf-8",
    )
    return np.array(v, dtype=np.float32)


# --------------------------------------------------------------------------
# The two arms
# --------------------------------------------------------------------------


class Index:
    def __init__(self, chains: list):
        from scipy import sparse

        for p, how in (
            (RAW / "chain_node_vecs.npy", "--prep"),
            (RAW / "chain_node_meta.json", "--prep"),
            (RAW / "bm25_weights.npz", "--prep"),
            (RAW / "bm25_vocab.json", "--prep"),
            (RAW / "bm25_chunks.jsonl", "--prep"),
        ):
            require(p, f"python -u experiment_review_retrieval_comparison.py {how}")
        self.vecs = np.load(RAW / "chain_node_vecs.npy")
        m = json.loads((RAW / "chain_node_meta.json").read_text(encoding="utf-8"))
        self.node_ids = m["node_ids"]
        self.row = {n: i for i, n in enumerate(self.node_ids)}
        self.name = {int(k): v for k, v in m["name"].items()}
        self.stage = {int(k): v for k, v in m["stage"].items()}
        self.chains = chains
        self.roots = sorted({c["path"][0] for c in chains})
        self.by_root: dict = {}
        for c in chains:
            self.by_root.setdefault(c["path"][0], []).append(c)
        self.bm25 = sparse.load_npz(RAW / "bm25_weights.npz").tocsc()
        self.vocab = json.loads((RAW / "bm25_vocab.json").read_text(encoding="utf-8"))
        self.chunk_offsets = []
        with open(RAW / "bm25_chunks.jsonl", "rb") as fh:
            off = 0
            for line in fh:
                self.chunk_offsets.append(off)
                off += len(line)
        if len(self.chunk_offsets) != self.bm25.shape[0]:
            fail("chunk file and BM25 matrix disagree on chunk count", RAW, "--prep")

    def chunk(self, i: int) -> dict:
        with open(RAW / "bm25_chunks.jsonl", "rb") as fh:
            fh.seek(self.chunk_offsets[i])
            return json.loads(fh.readline().decode("utf-8"))

    def arm_a(self, qvec: np.ndarray) -> dict:
        q = qvec / np.linalg.norm(qvec)
        rv = self.vecs[[self.row[r] for r in self.roots]] @ q
        top = np.argsort(-rv)[:N_ROOTS]
        roots = [(self.roots[i], float(rv[i])) for i in top]
        cands = []
        for r, _ in roots:
            for c in self.by_root[r]:
                s = float((self.vecs[[self.row[n] for n in c["path"]]] @ q).mean())
                cands.append((s, c["chain_idx"], c))
        cands.sort(key=lambda t: (-t[0], t[1]))
        best = cands[0][2]
        lines = [f"{self.stage[n].capitalize()}: {self.name[n]}" for n in best["path"]]
        answer = "\n".join(lines) + f"\nSource: {best['url']}"
        return {
            "answer": answer,
            "top_chain_idx": best["chain_idx"],
            "top_chain_url": best["url"],
            "top_chain_score": cands[0][0],
            "roots": [{"node": r, "cos": s} for r, s in roots],
            "candidate_urls": sorted({c[2]["url"] for c in cands}),
            "n_candidates": len(cands),
        }

    def bm25_top(self, question: str) -> list:
        from sklearn.feature_extraction.text import CountVectorizer

        an = CountVectorizer(
            lowercase=True, token_pattern=r"(?u)\b\w\w+\b", stop_words="english"
        ).build_analyzer()
        cols = [self.vocab[t] for t in an(question) if t in self.vocab]
        if not cols:
            fail(f"question has no indexed terms: {question!r}")
        scores = np.asarray(self.bm25[:, cols].sum(axis=1)).ravel()
        top = np.argsort(-scores)[:N_CHUNKS]
        return [
            dict(self.chunk(int(i)), score=float(scores[i]), chunk_row=int(i))
            for i in top
        ]


def synth_prompt(question: str, passages: list) -> str:
    blocks = [
        f"[Passage {k + 1}] source URL: {p['url']}\ntitle: {p['title']}\n{p['text']}"
        for k, p in enumerate(passages)
    ]
    return SYNTH_PROMPT.format(question=question, passages="\n\n".join(blocks))


# --------------------------------------------------------------------------
# Pipeline per query
# --------------------------------------------------------------------------


def qid(url: str) -> str:
    import hashlib

    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def run_qgen(url: str, texts: dict) -> dict:
    d = RAW / qid(url)
    d.mkdir(parents=True, exist_ok=True)
    prompt = QGEN_PROMPT.replace("{url}", url).replace("{text}", texts[url]["text"])
    rec = call_openai(prompt, d / "qgen.json")
    p = parse_qgen(rec["text"])
    if p["status"] == "ok":
        p["spans_found"] = sum(span_found(s, texts[url]["text"]) for s in p["spans"])
    return p


def run_query(url: str, stratum: str, qg: dict, idx: Index, texts: dict) -> dict:
    d = RAW / qid(url)
    STATUS["arms"] += 1
    qvec = embed(qg["question"], d / "question_embedding.json")
    a = idx.arm_a(qvec)
    (d / "arm_a.json").write_text(
        json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    passages = idx.bm25_top(qg["question"])
    (d / "arm_b_passages.json").write_text(
        json.dumps(
            [
                {k: p[k] for k in ("url", "chunk", "score", "chunk_row")}
                for p in passages
            ]
        ),
        encoding="utf-8",
    )
    b = call_openai(synth_prompt(qg["question"], passages), d / "arm_b.json")
    STATUS["arms"] -= 1

    STATUS["judge"] += 1
    rng = random.Random(f"{SEED}-{url}")
    a_is_x = rng.random() < 0.5
    ans = {"A": a["answer"], "B": b["text"]}
    x, y = ("A", "B") if a_is_x else ("B", "A")
    cites = {"A": cited_urls(ans["A"]), "B": cited_urls(ans["B"])}
    all_cited = []
    for u in cites["A"] + cites["B"]:
        if u not in all_cited:
            all_cited.append(u)
    blocks = []
    for u in all_cited:
        if u in texts:
            blocks.append(f"--- document {u} ---\n{texts[u]['text']}")
        else:
            blocks.append(
                f"--- document {u} ---\n[not a corpus document; no text available]"
            )
    prompt = JUDGE_PROMPT.format(
        question=qg["question"],
        reference_answer=qg["reference_answer"],
        spans="\n".join(f'- "{s}"' for s in qg["spans"]),
        answer_x=ans[x],
        answer_y=ans[y],
        cited_texts="\n\n".join(blocks),
    )
    j = call_anthropic(prompt, d / "judge.json")
    STATUS["judge"] -= 1
    pj = parse_judge(j["text"])
    if pj["status"] != "ok":
        raise RuntimeError(
            f"judge response malformed for {url}: see {d / 'judge.json'}"
        )
    by_arm = {x: pj["X"], y: pj["Y"]}
    for arm in ("A", "B"):
        sp = by_arm[arm]["support_spans"]
        found = 0
        for s in sp:
            t = texts.get((s or {}).get("cited_url"), {}).get("text", "")
            found += span_found((s or {}).get("span", ""), t)
        by_arm[arm]["spans_total"] = len(sp)
        by_arm[arm]["spans_found_verbatim"] = found
        del by_arm[arm]["support_spans"]
    pref = {"X": x, "Y": y, "tie": "tie"}[pj["preference"]]
    row = {
        "qid": qid(url),
        "stratum": stratum,
        "source_url": url,
        "question": qg["question"],
        "reference_spans_found_verbatim": f"{qg['spans_found']}/{len(qg['spans'])}",
        "arm_a_label": "X" if a_is_x else "Y",
        "arm_a_top_chain_url": a["top_chain_url"],
        "arm_a_hit_top": a["top_chain_url"] == url,
        "arm_a_hit_any_candidate": url in a["candidate_urls"],
        "arm_b_hit_passages": any(p["url"] == url for p in passages),
        "arm_b_cites_source": url in cites["B"],
        "arm_b_cited_urls": cites["B"],
        "arm_b_cites_outside_passages": [
            u for u in cites["B"] if u not in {p["url"] for p in passages}
        ],
        "A": by_arm["A"],
        "B": by_arm["B"],
        "preference": pref,
        "usage": {
            "qgen": json.loads((d / "qgen.json").read_text(encoding="utf-8"))["usage"],
            "arm_b": b["usage"],
            "judge": j["usage"],
        },
    }
    (d / "row.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    STATUS["done"] += 1
    log(f"done {stratum:7s} {qid(url)}  pref={pref}  A_hit={row['arm_a_hit_top']}")
    return row


def select_queries(sample: dict, texts: dict, n_chain: int, n_nochain: int) -> dict:
    """Generate questions in seeded order until n accepted per stratum (concurrently)."""
    accepted: dict = {}
    qgens: dict = {}
    with concurrent.futures.ThreadPoolExecutor(WORKERS) as ex:
        while True:
            st = split_strata(
                sample["chain"], sample["nochain"], accepted, n_chain, n_nochain
            )
            pend = st["chain"]["pending"] + st["nochain"]["pending"]
            no_text = [u for u in pend if u not in texts]
            for u in no_text:
                accepted[u] = False
                qgens[u] = {"status": "none", "reason": "no ARD text"}
            pend = [u for u in pend if u in texts]
            if not pend:
                if no_text:
                    continue
                break
            STATUS["qgen"] = len(pend)
            futs = {ex.submit(run_qgen, u, texts): u for u in pend}
            for f in concurrent.futures.as_completed(futs):
                u = futs[f]
                p = f.result()
                if p["status"] == "malformed":
                    fail(
                        f"question generator returned malformed JSON for {u}",
                        RAW / qid(u) / "qgen.json",
                    )
                accepted[u] = p["status"] == "ok"
                qgens[u] = p
                log(f"qgen {qid(u)} {p['status']}")
    STATUS["qgen"] = 0
    st = split_strata(sample["chain"], sample["nochain"], accepted, n_chain, n_nochain)
    return {
        "chain": st["chain"]["chosen"],
        "nochain": st["nochain"]["chosen"],
        "skipped": {
            k: [{"url": u, "reason": qgens[u].get("reason")} for u in st[k]["skipped"]]
            for k in ("chain", "nochain")
        },
        "qgens": qgens,
    }


# --------------------------------------------------------------------------
# Score
# --------------------------------------------------------------------------


def score(rows: list, sel: dict, sample: dict, smoke: bool) -> dict:
    def tally(rs: list) -> dict:
        out = {"n": len(rs)}
        for arm in ("A", "B"):
            t = {}
            for k in (
                "answers_question",
                "source_support",
                "citation_correctness",
                "mc_steps",
            ):
                t[k] = dict(Counter(r[arm][k] for r in rs))
            for k in ("mc_risk", "mc_intervention"):
                t[k] = dict(Counter(r[arm][k] for r in rs))
            t["spans_found_verbatim"] = (
                f"{sum(r[arm]['spans_found_verbatim'] for r in rs)}/{sum(r[arm]['spans_total'] for r in rs)}"
            )
            out[f"arm_{arm}"] = t
        out["preference"] = dict(Counter(r["preference"] for r in rs))
        out["arm_A_retrieval_hit_top_chain"] = sum(r["arm_a_hit_top"] for r in rs)
        out["arm_A_source_among_candidate_chains"] = sum(
            r["arm_a_hit_any_candidate"] for r in rs
        )
        out["arm_B_source_among_10_passages"] = sum(r["arm_b_hit_passages"] for r in rs)
        out["arm_B_cites_source"] = sum(r["arm_b_cites_source"] for r in rs)
        out["arm_B_answers_citing_outside_passages"] = sum(
            bool(r["arm_b_cites_outside_passages"]) for r in rs
        )
        return out

    usage = Counter()
    for r in rows:
        for k, u in r["usage"].items():
            model = JUDGE_MODEL if k == "judge" else GEN_MODEL
            usage[(model, "in")] += u.get("input_tokens") or 0
            usage[(model, "out")] += u.get("output_tokens") or 0
    for u, p in sel["qgens"].items():
        if p["status"] != "ok":
            rec = json.loads((RAW / qid(u) / "qgen.json").read_text(encoding="utf-8"))
            usage[(GEN_MODEL, "in")] += rec["usage"].get("input_tokens") or 0
            usage[(GEN_MODEL, "out")] += rec["usage"].get("output_tokens") or 0
    usd = sum(
        usage[(m, "in")] / 1e6 * PRICE[m][0] + usage[(m, "out")] / 1e6 * PRICE[m][1]
        for m in PRICE
    )
    return {
        "study": "chain lookup vs passage retrieval plus synthesis (review C2)",
        "smoke": smoke,
        "design": __doc__.split(
            "DESIGN (fixed here before any call; copied verbatim into the receipt)"
        )[1]
        .split("CLASS A")[0]
        .strip(),
        "models": {
            "question_generation": GEN_MODEL,
            "arm_B_synthesis": GEN_MODEL,
            "judge": JUDGE_MODEL,
            "embedding": EMBED_MODEL,
        },
        "population": {
            k: sample[k] for k in ("n_corpus", "n_chain_yielding", "n_no_chain")
        },
        "seed": SEED,
        "skipped_by_question_generator": sel["skipped"],
        "chain_yielding": tally([r for r in rows if r["stratum"] == "chain"]),
        "no_chain": tally([r for r in rows if r["stratum"] == "nochain"]),
        "DEVIATIONS": [
            "judge prompt asked risk/intervention presence as yes/no; one judge response used "
            "'partial', so the parser accepts three levels for those two fields",
            "answers_question added to the handover's measure list (see design)",
            "BM25 as a sparse-matrix implementation with Lucene idf instead of rank_bm25 (memory)",
        ],
        "NOT_POOLED": "the two strata are never combined into one rate; the no-chain gap is selection by the gates",
        "usage_tokens": {f"{m}_{d}": v for (m, d), v in usage.items()},
        "usd_at_list_prices": round(usd, 2),
        "per_query": rows,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--prep", action="store_true")
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--score", action="store_true")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)

    na = load_slim()
    chains = load_chains(na)
    if args.prep:
        prep_chain_index(na, chains)
        del na
        prep_bm25(load_slim())
        return 0

    sample = draw_sample(na, chains)
    log(
        f"population: corpus {sample['n_corpus']:,}, chain-yielding {sample['n_chain_yielding']:,}, "
        f"no-chain {sample['n_no_chain']:,}"
    )
    if (sample["n_chain_yielding"], sample["n_corpus"]) != (1868, 11779):
        fail(f"population differs from the paper's 1,868 / 11,779: {sample}", PATHS)
    # Candidate texts: generous prefix of each stratum's seeded order covers replacements.
    cand = set(sample["chain"][:200]) | set(sample["nochain"][:200])
    texts = ard_texts(cand)

    if not (args.smoke or args.run or args.score):
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")
        rows = []
        for name, n in (("chain", N_CHAIN), ("nochain", N_NOCHAIN)):
            for u in [u for u in sample[name] if u in texts][:n]:
                rows.append(
                    (name, u, len(enc.encode(texts[u]["text"], disallowed_special=())))
                )
        toks = [t for _, _, t in rows]
        print(
            json.dumps(
                {
                    "first_candidates": [
                        {"stratum": s, "url": u, "tokens": t} for s, u, t in rows
                    ],
                    "qgen_input_tokens_total": sum(toks),
                    "max": max(toks),
                },
                indent=1,
            )
        )
        return 0

    n_chain, n_nochain = (1, 1) if args.smoke else (N_CHAIN, N_NOCHAIN)
    stop = threading.Event()
    threading.Thread(
        target=heartbeat, args=(stop, n_chain + n_nochain), daemon=True
    ).start()
    try:
        sel = select_queries(sample, texts, n_chain, n_nochain)
        log(
            f"selected {len(sel['chain'])} chain-yielding + {len(sel['nochain'])} no-chain; skipped {sel['skipped']}"
        )
        if len(sel["chain"]) < n_chain or len(sel["nochain"]) < n_nochain:
            fail(
                "not enough accepted questions in the first 200 candidates of a stratum"
            )
        # Cited documents can be any corpus document: load every corpus text once.
        texts_all = ard_texts({a["url"] for a in na.values() if a.get("url")})
        del na
        log("loading indexes ...")
        idx = Index(chains)
        jobs = [(u, "chain") for u in sel["chain"]] + [
            (u, "nochain") for u in sel["nochain"]
        ]
        rows = []
        if args.score:
            for u, s in jobs:
                p = RAW / qid(u) / "row.json"
                require(p, "--run")
                rows.append(json.loads(p.read_text(encoding="utf-8")))
        else:
            with concurrent.futures.ThreadPoolExecutor(WORKERS) as ex:
                futs = [
                    ex.submit(run_query, u, s, sel["qgens"][u], idx, texts_all)
                    for u, s in jobs
                ]
                for f in concurrent.futures.as_completed(futs):
                    rows.append(f.result())
        order = {u: i for i, (u, _) in enumerate(jobs)}
        rows.sort(key=lambda r: order[r["source_url"]])
    finally:
        stop.set()
    rep = score(rows, sel, sample, smoke=args.smoke)
    rep["wall_clock_s"] = round(time.time() - T0, 1)
    out = OUT.with_name(OUT.stem + "_smoke.json") if args.smoke else OUT
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        json.dumps(
            {
                k: rep[k]
                for k in (
                    "chain_yielding",
                    "no_chain",
                    "usage_tokens",
                    "usd_at_list_prices",
                    "wall_clock_s",
                )
            },
            indent=1,
        )
    )
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
