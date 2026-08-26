#!/usr/bin/env python
"""Three models, one prompt, twenty documents: how much of an extraction is the model?

GitHub issue #168. OPEN_ITEMS.md S11, runbook R3.

sec:limitations says every extraction comes from one model under one prompt, and until
2026-08-16 it carried a rendered open block asking a co-author to recover an n=20
o3 / GPT-5 / Claude-4 check run by hand earlier in the project. Those numbers are gone.
This replaces them with a run that is reproducible.

Arms, over the SAME 20 documents -- the first 20 of the seed-42 chain-yielding sample the
schema ablation (#165) draws, so the two studies share a baseline:

    released   o3 as shipped, batch API, 2025      already paid for; read from the graph
    A          o3, synchronous /v1/responses       metered
    B          gpt-5, synchronous /v1/responses    metered
    C          claude-opus-5, Anthropic messages   metered

Arm C ran through the Claude Code CLI until 2026-08-16 and was described as free. It is
not: the CLI bills the interactive session's usage allowance, which is shared with every
other thing this project does and was being consumed at about one percent per minute at 43k
output tokens per document. It is an API call now, like the other two arms.

Identical prompt (the released PROMPT_EXTRACT), identical request shape, reasoning effort
medium where the model takes it. gpt-5.6-sol, the id in OPEN_ITEMS.md, is not in this
account's model list; gpt-5 is, and it matches the o3 / GPT-5 / Claude-4 design of the run
this replaces.

What is measured
----------------
1. counts      nodes, edges, risks, interventions per document per model
2. structure   whether each model's graph yields a chain under the released enumerator's
               constraints, and how its concept nodes distribute over the five stages
3. endpoints   whether the same risk-to-intervention pairs appear, by token-set Jaccard
               over normalized names at 0.6, with 0.5 and 0.7 reported beside it so the
               threshold is visible rather than load-bearing
4. noise floor arm A re-runs the corpus extractor on documents it already extracted, so
               arm A against the released graph is a repeat-extraction measurement at
               n=20 -- part of what S7 would buy, at no extra cost

Scoring code is imported from experiment_review_schema_ablation rather than copied: both
studies must apply the released enumerator's constraints the same way, and two copies
would drift.

CLASS A for arms A and B (metered OpenAI), subscription for arm C. Run from graph_analysis/:

    python -u experiment_review_multimodel_consistency.py                # dry run
    python -u experiment_review_multimodel_consistency.py --run A,B,C
    python -u experiment_review_multimodel_consistency.py --score

Output:
    phase2_results/multimodel_raw/<arm>/<doc_id>.json
    phase2_results/experiment_review_multimodel_consistency_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import tiktoken

import experiment_review_schema_ablation as ABL

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "phase2_results/multimodel_raw"
OUT = ROOT / "phase2_results/experiment_review_multimodel_consistency_report.json"

N_DOCS = 20
MAX_WORKERS = 4
ARMS = {
    "A_o3": {"provider": "openai", "model": "o3", "effort": "medium"},
    "B_gpt5": {"provider": "openai", "model": "gpt-5", "effort": "medium"},
    "C_opus5": {"provider": "anthropic", "model": "claude-opus-5", "effort": None},
}
# Opus emits ~43k output tokens on this prompt, so the ceiling has to clear that with room.
ANTHROPIC_MAX_TOKENS = 64000
# Same key file as the OpenAI arms; read at run time, never copied into this repo.
ENV_PATH = ABL.ENV_PATH

# USD per million tokens, SYNCHRONOUS. Inputs to this script, printed with the result.
RATES = {"o3": (2.00, 8.00), "gpt-5": (1.25, 10.00), "claude-opus-5": (0.0, 0.0)}
JACCARD_THRESHOLDS = [0.5, 0.6, 0.7]
MATCH_AT = 0.6

# A token-set criterion cannot separate "same concept, other words" from "different
# concept", and the o3 re-run scores only 19% against its own released extraction under it,
# so the lexical measure needs a semantic one beside it or every arm reads as disagreeing.
# The embedding model is the one the pipeline itself uses on node text.
EMBED_MODEL = "text-embedding-3-small"
EMBED_CACHE = ROOT / "phase2_results/multimodel_raw/name_embeddings.json"
COSINE_THRESHOLDS = [0.80, 0.85, 0.90]

_print_lock = threading.Lock()
_STOP = re.compile(r"\b(the|a|an|of|in|to|for|and|or|by|with|on|from|as|at|is|are)\b")


def norm_tokens(name: str) -> frozenset:
    s = _STOP.sub(" ", (name or "").lower())
    return frozenset(t for t in re.split(r"[^a-z0-9]+", s) if len(t) > 2)


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def best_match_rate(left: list[frozenset], right: list[frozenset], thr: float) -> float:
    """Share of LEFT items with some RIGHT item at Jaccard >= thr. Asymmetric on purpose:
    reported both ways, because a model that extracts twice as much matches trivially in
    one direction."""
    if not left:
        return 0.0
    return round(
        100
        * sum(1 for a in left if any(jaccard(a, b) >= thr for b in right))
        / len(left),
        1,
    )


def embed_names(client, names: list[str]) -> dict:
    """One embedding per distinct name, cached on disk so re-scoring costs nothing."""
    cache = (
        json.loads(EMBED_CACHE.read_text(encoding="utf-8"))
        if EMBED_CACHE.exists()
        else {}
    )
    todo = sorted({n for n in names if n and n not in cache})
    for i in range(0, len(todo), 256):
        chunk = todo[i : i + 256]
        r = client.embeddings.create(model=EMBED_MODEL, input=chunk)
        for name, item in zip(chunk, r.data):
            cache[name] = item.embedding
        print(f"  embedded {min(i + 256, len(todo))}/{len(todo)}", flush=True)
    if todo:
        EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
        EMBED_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return cache


def cosine(a: list, b: list) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return num / (na * nb) if na and nb else 0.0


def best_cosine_rate(left: list, right: list, emb: dict, thr: float) -> float:
    if not left:
        return 0.0
    hit = 0
    for a in left:
        va = emb.get(a)
        if va and any(cosine(va, emb[b]) >= thr for b in right if b in emb):
            hit += 1
    return round(100 * hit / len(left), 1)


def call_anthropic(prompt: str, document: str, model: str) -> dict:
    """Metered Anthropic API, in the shape paper/review_multi_model.py uses.

    This arm used to shell out to the Claude Code CLI on subscription auth, described in an
    earlier version of this file as costing USD 0. That was wrong in the way that matters:
    the CLI bills the interactive session's usage allowance, a shared and exhaustible
    resource, and it consumed it at about one percent per minute here. Dollars are the
    cheaper currency, and an API call is also the same kind of object as the other two arms
    rather than a subprocess with its own harness prompt.
    """
    import anthropic
    from dotenv import dotenv_values

    if not ENV_PATH.exists():
        fail_env()
    key = (dotenv_values(ENV_PATH) or {}).get("ANTHROPIC_API_KEY") or ""
    if not key.strip():
        fail_env()
    client = anthropic.Anthropic(api_key=key.strip(), timeout=1800.0, max_retries=3)
    t0 = time.time()
    r = client.messages.create(
        model=model,
        max_tokens=ANTHROPIC_MAX_TOKENS,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\nHere is the paper for analysis:\n\n{document}"
                ),
            }
        ],
    )
    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    return {
        "text": text,
        "wall_clock_s": round(time.time() - t0, 1),
        "stop_reason": r.stop_reason,
        "usage": {
            "input_tokens": r.usage.input_tokens,
            "output_tokens": r.usage.output_tokens,
            "reasoning_tokens": None,
        },
    }


def fail_env() -> None:
    raise SystemExit(
        "FATAL: ANTHROPIC_API_KEY not found\n"
        f"  expected artifact: {ENV_PATH}\n"
        "  produced by: runbook R0 in paper/OPEN_ITEMS.md\n"
        "  this script does NOT fall back to the Claude Code CLI: that path bills the "
        "interactive session's usage allowance rather than dollars."
    )


def call_openai(client, prompt: str, document: str, model: str, effort: str) -> dict:
    """One request in extractor.py's shape. `reasoning` is omitted entirely when the model
    has no reasoning parameter -- gpt-4.1 rejects it, and that arm exists precisely to be a
    non-reasoning comparison."""
    t0 = time.time()
    kwargs = {"reasoning": {"effort": effort}} if effort else {}
    r = client.responses.create(
        model=model,
        **kwargs,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_text",
                        "text": f"\n\nHere is the paper for analysis:\n\n{document}",
                    },
                ],
            }
        ],
    )
    usage = getattr(r, "usage", None)
    return {
        "text": r.output_text,
        "wall_clock_s": round(time.time() - t0, 1),
        "usage": {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "reasoning_tokens": getattr(
                getattr(usage, "output_tokens_details", None), "reasoning_tokens", None
            ),
        },
    }


def endpoint_names(nodes: dict) -> tuple[list, list]:
    risks = [
        v["name"]
        for v in nodes.values()
        if v["type"] == "concept"
        and (v.get("category") or "").strip().lower() == "risk"
    ]
    intvs = [v["name"] for v in nodes.values() if v["type"] == "intervention"]
    return risks, intvs


def endpoints(nodes: dict) -> tuple[list, list, list]:
    r, i = endpoint_names(nodes)
    risks = [norm_tokens(x) for x in r]
    intvs = [norm_tokens(x) for x in i]
    return risks, intvs, [a | b for a in risks for b in intvs][:400]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    prompt = ABL.load_released_prompt()
    sample = ABL.load_sample()
    urls = sample["picked"][:N_DOCS]
    texts = {u: sample["texts"][u]["text"] for u in urls}

    receipt = {
        "study": "multi-model extraction consistency (issue #168, S11, R3)",
        "arms": ARMS,
        "prompt": "the released PROMPT_EXTRACT, unmodified",
        "sample": {
            "n_documents": len(urls),
            "drawn_from": "first 20 of the seed-42 chain-yielding sample of issue #165",
            "urls": urls,
        },
        "rates_usd_per_million_ASSUMED": RATES,
    }

    if not args.run and not args.score:
        enc = tiktoken.get_encoding(ABL.ENCODING)
        ins = [
            len(enc.encode(prompt, disallowed_special=()))
            + len(enc.encode(texts[u], disallowed_special=()))
            for u in urls
        ]
        est = {
            "n_calls_per_arm": len(urls),
            "input_tokens_per_metered_arm": sum(ins),
            "input_tokens_both_metered_arms": 2 * sum(ins),
            "visible_output_assumed_per_call": 5361,
        }
        band = {}
        for r in [0.0, 1.0, 2.0, 4.0]:
            usd = 0.0
            for m in ("o3", "gpt-5"):
                ri, ro = RATES[m]
                usd += sum(ins) / 1e6 * ri + 5361 * len(urls) * (1 + r) / 1e6 * ro
            band[f"reasoning_{r:g}x"] = round(usd, 2)
        est["usd_band_metered_arms"] = band
        receipt["dry_run"] = est
        print(json.dumps(est, indent=2))
        return

    if args.run:
        client = None
        jobs = []
        for arm in [a.strip() for a in args.run.split(",") if a.strip()]:
            key = next(k for k in ARMS if k.startswith(arm) or k == arm)
            (RAW_DIR / key).mkdir(parents=True, exist_ok=True)
            for u in urls:
                dest = RAW_DIR / key / f"{ABL.doc_id(u)}.json"
                if not dest.exists():
                    jobs.append((key, u, dest))
        if args.limit:
            jobs = jobs[: args.limit]
        if any(ARMS[k]["provider"] == "openai" for k, _, _ in jobs):
            client = ABL.openai_client()
        print(f"{len(jobs)} calls to make", flush=True)

        def work(job):
            key, u, dest = job
            spec = ARMS[key]
            if spec["provider"] == "openai":
                res = call_openai(
                    client, prompt, texts[u], spec["model"], spec["effort"]
                )
            else:
                res = call_anthropic(prompt, texts[u], spec["model"])
            dest.write_text(
                json.dumps(
                    {"arm": key, "url": u, "model": spec["model"], **res},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return key, u, res

        # Anthropic jobs run after the OpenAI ones rather than interleaved, so a rate limit
        # on one vendor cannot be mistaken for a failure on the other.
        # Both providers are plain API calls now, so both go through the pool. The old
        # split existed because arm C was a CLI subprocess and had to stay serial.
        api_jobs = list(jobs)
        cli_jobs = []
        done = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(work, j): j for j in api_jobs}
            for f in as_completed(futs):
                key, u, _ = futs[f]
                done += 1
                try:
                    _, _, res = f.result()
                    msg = f"[{done}/{len(jobs)}] {key} {ABL.doc_id(u)[:36]} {res['wall_clock_s']}s out={res['usage']['output_tokens']}"
                except Exception as exc:  # noqa: BLE001 - reported, never swallowed
                    msg = f"[{done}/{len(jobs)}] {key} {ABL.doc_id(u)[:36]} FAILED {type(exc).__name__}: {exc}"
                with _print_lock:
                    print(msg, flush=True)
        for j in cli_jobs:
            done += 1
            try:
                key, u, res = work(j)
                print(
                    f"[{done}/{len(jobs)}] {key} {ABL.doc_id(u)[:36]} {res['wall_clock_s']}s",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[{done}/{len(jobs)}] {j[0]} {ABL.doc_id(j[1])[:36]} FAILED {type(exc).__name__}: {exc}",
                    flush=True,
                )

    if args.score:
        base = ABL.released_graphs(urls, sample["node_attrs"])
        per_model = {"released": {}}
        for u in urls:
            g = base[u]
            per_model["released"][u] = {
                "score": ABL.score_graph(g["nodes"], g["edges"]),
                "endpoints": endpoints(g["nodes"]),
                "names": endpoint_names(g["nodes"]),
            }
        for key in ARMS:
            d = RAW_DIR / key
            if not d.exists():
                continue
            per_model[key] = {}
            for fp in sorted(d.glob("*.json")):
                rec = json.loads(fp.read_text(encoding="utf-8"))
                ext = ABL.parse_extraction(rec.get("text") or "")
                if not ext:
                    per_model[key][rec["url"]] = {"parse_failure": True}
                    continue
                nodes, edges = ABL.graph_from_extraction(ext)
                per_model[key][rec["url"]] = {
                    "score": ABL.score_graph(nodes, edges),
                    "endpoints": endpoints(nodes),
                    "names": endpoint_names(nodes),
                    "usage": rec.get("usage"),
                }

        def agg(rows):
            ok = [r for r in rows.values() if not r.get("parse_failure")]
            if not ok:
                return {"n": 0}
            s = [r["score"] for r in ok]
            return {
                "n": len(ok),
                "parse_failures": sum(
                    1 for r in rows.values() if r.get("parse_failure")
                ),
                "mean_nodes": round(statistics.mean([x["n_nodes"] for x in s]), 1),
                "mean_edges": round(statistics.mean([x["n_edges"] for x in s]), 1),
                "mean_risk_nodes": round(
                    statistics.mean([x["n_risk_nodes"] for x in s]), 1
                ),
                "mean_interventions": round(
                    statistics.mean([x["n_interventions"] for x in s]), 1
                ),
                "pct_with_chain": round(
                    100 * sum(x["has_chain"] for x in s) / len(s), 1
                ),
                "mean_chains": round(statistics.mean([x["n_chains"] for x in s]), 2),
                # The mean is dominated by single documents -- one Opus graph emits 57,007
                # simple paths -- so the median and the maximum ship beside it.
                "median_chains": statistics.median([x["n_chains"] for x in s]),
                "max_chains": max(x["n_chains"] for x in s),
                "stage_mix_pct": stage_mix(s),
            }

        all_names = [
            n
            for rows in per_model.values()
            for r in rows.values()
            if not r.get("parse_failure")
            for half in r["names"]
            for n in half
        ]
        print(f"embedding {len(set(all_names))} distinct endpoint names", flush=True)
        emb = embed_names(ABL.openai_client(), all_names)

        pairs = {}
        for a, b in combinations(list(per_model), 2):
            shared = [
                u
                for u in urls
                if u in per_model[a]
                and u in per_model[b]
                and not per_model[a][u].get("parse_failure")
                and not per_model[b][u].get("parse_failure")
            ]
            if not shared:
                continue
            row = {"n_documents": len(shared)}
            for thr in JACCARD_THRESHOLDS:
                r_ab, r_ba, i_ab, i_ba = [], [], [], []
                for u in shared:
                    ra, ia, _ = per_model[a][u]["endpoints"]
                    rb, ib, _ = per_model[b][u]["endpoints"]
                    r_ab.append(best_match_rate(ra, rb, thr))
                    r_ba.append(best_match_rate(rb, ra, thr))
                    i_ab.append(best_match_rate(ia, ib, thr))
                    i_ba.append(best_match_rate(ib, ia, thr))
                row[f"jaccard_{thr}"] = {
                    "pct_risks_of_A_matched_in_B": round(statistics.mean(r_ab), 1),
                    "pct_risks_of_B_matched_in_A": round(statistics.mean(r_ba), 1),
                    "pct_interventions_of_A_matched_in_B": round(
                        statistics.mean(i_ab), 1
                    ),
                    "pct_interventions_of_B_matched_in_A": round(
                        statistics.mean(i_ba), 1
                    ),
                }
            for thr in COSINE_THRESHOLDS:
                r_ab, r_ba, i_ab, i_ba = [], [], [], []
                for u in shared:
                    ra, ia = per_model[a][u]["names"]
                    rb, ib = per_model[b][u]["names"]
                    r_ab.append(best_cosine_rate(ra, rb, emb, thr))
                    r_ba.append(best_cosine_rate(rb, ra, emb, thr))
                    i_ab.append(best_cosine_rate(ia, ib, emb, thr))
                    i_ba.append(best_cosine_rate(ib, ia, emb, thr))
                row[f"cosine_{thr}"] = {
                    "pct_risks_of_A_matched_in_B": round(statistics.mean(r_ab), 1),
                    "pct_risks_of_B_matched_in_A": round(statistics.mean(r_ba), 1),
                    "pct_interventions_of_A_matched_in_B": round(
                        statistics.mean(i_ab), 1
                    ),
                    "pct_interventions_of_B_matched_in_A": round(
                        statistics.mean(i_ba), 1
                    ),
                }
            pairs[f"{a} vs {b}"] = row

        # Which gate moves between runs, stated as counts rather than left in
        # per_document. The sample is conditioned on the shipped run yielding a chain, so
        # the released column is 20 of 20 by construction and these are upper bounds on
        # what a re-run loses, never symmetric stability rates.
        gate = {}
        for key in [k for k in per_model if k != "released"]:
            shared = [
                u
                for u in urls
                if u in per_model[key] and not per_model[key][u].get("parse_failure")
            ]
            gate[key] = {
                "documents_compared": len(shared),
                "released_with_a_mature_intervention": sum(
                    1
                    for u in shared
                    if per_model["released"][u]["score"]["n_interventions_mature"]
                ),
                "arm_with_a_mature_intervention": sum(
                    1
                    for u in shared
                    if per_model[key][u]["score"]["n_interventions_mature"]
                ),
                "released_with_a_chain": sum(
                    1 for u in shared if per_model["released"][u]["score"]["has_chain"]
                ),
                "arm_with_a_chain": sum(
                    1 for u in shared if per_model[key][u]["score"]["has_chain"]
                ),
                "released_mean_edges_conf_ge3": round(
                    statistics.mean(
                        [
                            per_model["released"][u]["score"]["n_edges_conf_ge3"]
                            for u in shared
                        ]
                    ),
                    2,
                ),
                "arm_mean_edges_conf_ge3": round(
                    statistics.mean(
                        [per_model[key][u]["score"]["n_edges_conf_ge3"] for u in shared]
                    ),
                    2,
                ),
            }
        receipt["gate_stability_against_the_shipped_extraction"] = gate
        receipt["headline"] = {k: agg(v) for k, v in per_model.items()}
        receipt["pairwise_endpoint_agreement"] = pairs
        receipt["match_threshold_used_in_prose"] = MATCH_AT
        receipt["per_document"] = {
            k: {
                u: {kk: vv for kk, vv in r.items() if kk not in ("endpoints", "names")}
                for u, r in v.items()
            }
            for k, v in per_model.items()
        }
        receipt["wall_clock_s"] = round(time.time() - t0, 1)
        OUT.write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(receipt["headline"], indent=2))
        print(json.dumps(pairs, indent=2))
        print(f"\nwrote {OUT}")


def stage_mix(scores: list) -> dict:
    tot = defaultdict(int)
    for s in scores:
        for k, v in s["category_counts"].items():
            tot[k] += v
    n = sum(tot.values()) or 1
    return {
        k: round(100 * v / n, 1)
        for k, v in sorted(tot.items(), key=lambda kv: -kv[1])[:12]
    }


if __name__ == "__main__":
    sys.exit(main())
