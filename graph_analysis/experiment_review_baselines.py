#!/usr/bin/env python
r"""Does the design earn its cost, and what does a re-run cost on an unconditioned sample?

GitHub issue #171. OPEN_ITEMS.md S6 remainder (arms B/C/D) and S7 (arm U).

#165 asked whether the structure is real, with a schema-blind prompt and degraded sources.
This asks the other half, which all six external reviewers raised (R26, R27): no design
choice in the paper is shown to be load-bearing. It also runs the control #168 could not,
whose sample is conditioned on the shipped run yielding a chain.

    B  abstract only, released prompt        does full text earn its cost?    25 arXiv docs
    C  gpt-4.1, non-reasoning, full text     does reasoning earn its cost?    30 docs
    D  flat triples, no stage schema         does the schema earn its cost?   30 docs
    U  o3 on UNCONDITIONED documents         symmetric repeat-extraction      20 docs

Arms C and D share #165's 30-document chain-yielding sample, so every arm is paired against
the same baseline and the two studies read together.

Arm B and the abstract problem
------------------------------
0 of the 1,869 chain-yielding documents carry an `abstract` field of 200+ characters in the
local ARD snapshot, and the arXiv records' `text` usually starts at "1 Introduction". An
abstract-only arm is therefore not runnable on the corpus as distributed. Taking the first N
characters of the body instead would measure truncation, not abstracts, so arm B draws 25
arXiv chain-yielding documents and fetches each abstract from the public arXiv API, with the
fetch date in the receipt. Non-arXiv sources are out of arm B by construction.

Scoring
-------
Structural, as in #165, by the released enumerator's constraints. Arms B and C use the
released prompt and score identically to arm A.

Arm D emits no category, no maturity and no confidence, so the enumerator cannot run on it.
Its measure is ENDPOINT RECOVERY: do its nodes contain the released chain's risk and
intervention, by embedding cosine at 0.80? The ceiling for that metric is not 100% but
46.5%, which is what an o3 re-run of the released prompt scores against its own shipped
extraction (#168). Arm D is read against that number.

CLASS A (metered OpenAI). Run from graph_analysis/:

    python -u experiment_review_baselines.py                  # dry run, no API call
    python -u experiment_review_baselines.py --run B,C,D,U
    python -u experiment_review_baselines.py --score

Output:
    phase2_results/baselines_raw/<arm>/<doc_id>.json
    phase2_results/experiment_review_baselines_report.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import threading
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tiktoken

import experiment_review_multimodel_consistency as MM
import experiment_review_schema_ablation as ABL

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "phase2_results/baselines_raw"
ABSTRACTS = ROOT / "phase2_results/baselines_raw/arxiv_abstracts.json"
OUT = ROOT / "phase2_results/experiment_review_baselines_report.json"

N_C_D = 30
N_B = 25
N_U = 20
SEED_B, SEED_U = 43, 44
MAX_WORKERS = 4

ARMS = {
    "B_abstract": {"model": "o3", "effort": "medium"},
    "C_gpt41": {"model": "gpt-4.1", "effort": None},
    "D_triples": {"model": "o3", "effort": "medium"},
    "U_unconditioned": {"model": "o3", "effort": "medium"},
}
# USD per million tokens, SYNCHRONOUS. Inputs to this script, printed with the result.
RATES = {"o3": (2.00, 8.00), "gpt-4.1": (2.00, 8.00)}
REASONING_RATIOS = [0.0, 1.0, 2.0, 4.0]

# The ceiling for arm D's endpoint-recovery metric: an o3 re-run of the released prompt
# recovers its own shipped risk names at this rate (#168). Reading arm D against 100%
# would charge it for run-to-run variation that has nothing to do with the schema.
ENDPOINT_CEILING_PCT = 46.5
COSINE_AT = 0.80

_print_lock = threading.Lock()

FLAT_TRIPLE_PROMPT = """# Knowledge Extraction for AI Safety Analysis

Extract the factual content of the data source as a flat set of subject-relation-object
triples. Do not organise them into pathways, do not classify the nodes into categories, and
do not rank or score them.

Guidance:
- About 15 triples per 5000 words of data source; prioritise accuracy over count.
- Name each subject and object so that the same concept from another data source would be
  named the same way: "[specific phenomenon or technique] in [context]", not a bare term.
- Use whatever relation verb the source supports.
- Cover the whole data source, not only its introduction or conclusion.

Return JSON only, in exactly this format:

```json
{
  "nodes": [
    {"name": "...", "type": "concept", "description": "1-2 sentences"}
  ],
  "edges": [
    {"type": "relation verb", "source_node": "exact node name",
     "target_node": "exact node name", "description": "the claim this triple makes"}
  ]
}
```

Now process the provided data source."""


def fail(msg: str, artifact, produced_by: str) -> None:
    raise SystemExit(
        f"FATAL: {msg}\n"
        f"  expected artifact: {artifact}\n"
        f"  produced by: {produced_by}\n"
        "  this script does NOT substitute a truncated body for an abstract, a different "
        "model for the one named, or a smaller sample for the one specified."
    )


def arxiv_id(url: str) -> str | None:
    m = re.search(
        r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}|[a-z-]+/[0-9]{7})",
        url or "",
        re.I,
    )
    return m.group(1) if m else None


def fetch_abstracts(urls: list[str]) -> dict:
    """Abstracts from the public arXiv API, cached. No key, no rate-limit games: one
    request per document with a courteous pause, and every failure reported."""
    cache = (
        json.loads(ABSTRACTS.read_text(encoding="utf-8")) if ABSTRACTS.exists() else {}
    )
    for u in urls:
        if u in cache:
            continue
        aid = arxiv_id(u)
        if not aid:
            cache[u] = None
            continue
        req = urllib.request.Request(
            f"http://export.arxiv.org/api/query?id_list={aid}",
            headers={"User-Agent": "AISafetyIntervention-research/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                xml = r.read().decode("utf-8", "replace")
            m = re.search(r"<summary>(.*?)</summary>", xml, re.S)
            cache[u] = re.sub(r"\s+", " ", m.group(1)).strip() if m else None
        except Exception as exc:  # noqa: BLE001 - recorded, never silently skipped
            print(f"  arXiv fetch failed for {aid}: {type(exc).__name__}", flush=True)
            cache[u] = None
        time.sleep(3.0)
        ABSTRACTS.parent.mkdir(parents=True, exist_ok=True)
        ABSTRACTS.write_text(
            json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    return cache


def build(sample: dict) -> dict:
    """Payloads per arm. Arm B and arm U carry their own populations."""
    prompt = ABL.load_released_prompt()
    picked = sample["picked"]
    payloads = defaultdict(dict)

    for u in picked[:N_C_D]:
        text = sample["texts"][u]["text"]
        payloads["C_gpt41"][u] = (prompt, text)
        payloads["D_triples"][u] = (FLAT_TRIPLE_PROMPT, text)

    # Arm B: arXiv chain-yielding documents, abstracts from the API.
    arxiv_chain = sorted(u for u in sample["all_chain_urls"] if arxiv_id(u))
    picked_b = random.Random(SEED_B).sample(arxiv_chain, min(N_B, len(arxiv_chain)))
    abstracts = fetch_abstracts(picked_b)
    got_b = [u for u in picked_b if (abstracts.get(u) or "").strip()]
    for u in got_b:
        payloads["B_abstract"][u] = (prompt, abstracts[u])

    # Arm U: unconditioned, from the whole corpus.
    picked_u = random.Random(SEED_U).sample(sorted(sample["corpus_with_text"]), N_U)
    for u in picked_u:
        payloads["U_unconditioned"][u] = (prompt, sample["corpus_texts"][u])

    return {
        "payloads": payloads,
        "arm_B": {
            "drawn": picked_b,
            "with_abstract": got_b,
            "n_failed_fetch": len(picked_b) - len(got_b),
        },
        "arm_U": {"drawn": picked_u},
    }


def load_inputs() -> dict:
    """The ablation's sample, plus the corpus-wide populations arms B and U need."""
    import pickle

    sample = ABL.load_sample()
    na = sample["node_attrs"]

    chain_urls = set()
    with open(ABL.PATHS, encoding="utf-8") as fh:
        for line in fh:
            u = na.get(json.loads(line)["path"][0], {}).get("url")
            if u:
                chain_urls.add(u)
    corpus_urls = {a.get("url") for a in na.values() if a.get("url")}
    unconditioned_pool = sorted(corpus_urls)

    rng = random.Random(SEED_U)
    candidates = rng.sample(unconditioned_pool, min(400, len(unconditioned_pool)))
    wanted = set(candidates)
    texts = {}
    for fp in sorted(ABL.ARD.glob("*.jsonl")):
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                u = d.get("url")
                if u in wanted and u not in texts and (d.get("text") or "").strip():
                    texts[u] = d["text"]
    if len(texts) < N_U:
        fail(
            f"only {len(texts)} unconditioned documents resolved to ARD source text",
            ABL.ARD,
            "download_ard.py",
        )
    del pickle
    sample["all_chain_urls"] = chain_urls
    sample["corpus_with_text"] = set(texts)
    sample["corpus_texts"] = texts
    return sample


def dry_run(built: dict) -> dict:
    enc = tiktoken.get_encoding(ABL.ENCODING)
    per_arm, total_in = {}, 0
    for arm, docs in built["payloads"].items():
        ins = [
            len(enc.encode(p, disallowed_special=()))
            + len(enc.encode(d, disallowed_special=()))
            for p, d in docs.values()
        ]
        per_arm[arm] = {
            "n_calls": len(ins),
            "input_tokens_total": sum(ins),
            "input_tokens_mean": round(statistics.mean(ins)) if ins else 0,
        }
        total_in += sum(ins)
    n_calls = sum(a["n_calls"] for a in per_arm.values())
    est_out = 5361 * n_calls
    band = {}
    for r in REASONING_RATIOS:
        band[f"reasoning_{r:g}x"] = round(
            total_in / 1e6 * 2.00 + est_out * (1 + r) / 1e6 * 8.00, 2
        )
    return {
        "per_arm": per_arm,
        "n_calls": n_calls,
        "input_tokens_total": total_in,
        "visible_output_assumed": est_out,
        "usd_band_by_reasoning_ratio": band,
        "wall_clock_estimate_min": round(n_calls * 70 / max(1, MAX_WORKERS) / 60, 1),
        "arm_B_fetch": {
            "drawn": len(built["arm_B"]["drawn"]),
            "with_abstract": len(built["arm_B"]["with_abstract"]),
            "failed_fetch": built["arm_B"]["n_failed_fetch"],
        },
    }


def run(arms: list[str], built: dict, limit: int = 0) -> None:
    client = ABL.openai_client()
    jobs = []
    for arm in arms:
        key = next(k for k in ARMS if k.startswith(arm) or k == arm)
        (RAW_DIR / key).mkdir(parents=True, exist_ok=True)
        for u, (prompt, doc) in built["payloads"][key].items():
            dest = RAW_DIR / key / f"{ABL.doc_id(u)}.json"
            if not dest.exists():
                jobs.append((key, u, prompt, doc, dest))
    if limit:
        jobs = jobs[:limit]
    print(f"{len(jobs)} calls to make ({MAX_WORKERS} at a time)", flush=True)

    def work(job):
        key, u, prompt, doc, dest = job
        spec = ARMS[key]
        res = MM.call_openai(client, prompt, doc, spec["model"], spec["effort"])
        dest.write_text(
            json.dumps(
                {"arm": key, "url": u, "model": spec["model"], **res},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return key, u, res

    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(work, j): j for j in jobs}
        for f in as_completed(futs):
            key, u = futs[f][0], futs[f][1]
            done += 1
            try:
                _, _, res = f.result()
                msg = (
                    f"[{done}/{len(jobs)}] {key} {ABL.doc_id(u)[:34]} "
                    f"{res['wall_clock_s']}s out={res['usage']['output_tokens']}"
                )
            except Exception as exc:  # noqa: BLE001 - reported, never swallowed
                msg = f"[{done}/{len(jobs)}] {key} {ABL.doc_id(u)[:34]} FAILED {type(exc).__name__}: {exc}"
            with _print_lock:
                print(msg, flush=True)


def score(sample: dict, built: dict) -> dict:
    urls_cd = sample["picked"][:N_C_D]
    urls_b = built["arm_B"]["with_abstract"]
    urls_u = built["arm_U"]["drawn"]
    all_urls = list(dict.fromkeys(urls_cd + urls_b + urls_u))
    base = ABL.released_graphs(all_urls, sample["node_attrs"])

    def base_row(u):
        g = base[u]
        return {
            "score": ABL.score_graph(g["nodes"], g["edges"]),
            "names": MM.endpoint_names(g["nodes"]),
        }

    released = {u: base_row(u) for u in all_urls}
    arms = {}
    for key in ARMS:
        d = RAW_DIR / key
        if not d.exists():
            continue
        rows = {}
        for fp in sorted(d.glob("*.json")):
            rec = json.loads(fp.read_text(encoding="utf-8"))
            text = rec.get("text") or ""
            ext = ABL.parse_extraction(text)
            if not ext:
                rows[rec["url"]] = {
                    "parse_failure": True,
                    "failure_kind": "declined_no_json_block"
                    if "```json" not in text
                    else "malformed_json",
                }
                continue
            nodes, edges = ABL.graph_from_extraction(ext)
            rows[rec["url"]] = {
                "score": ABL.score_graph(nodes, edges),
                "names": MM.endpoint_names(nodes),
                "all_node_names": [v["name"] for v in nodes.values()],
                "usage": rec.get("usage"),
            }
        arms[key] = rows

    # Endpoint recovery needs embeddings for arm D, whose nodes carry no category at all.
    names = [n for r in released.values() for half in r["names"] for n in half]
    for rows in arms.values():
        for r in rows.values():
            if not r.get("parse_failure"):
                names += r.get("all_node_names", [])
    emb = MM.embed_names(ABL.openai_client(), names) if names else {}

    def agg(rows, pop):
        ok = [r for r in rows.values() if not r.get("parse_failure")]
        if not ok:
            return {"n": 0}
        s = [r["score"] for r in ok]
        rec_r, rec_i = [], []
        for u, r in rows.items():
            if r.get("parse_failure") or u not in released:
                continue
            rr, ri = released[u]["names"]
            cand = r.get("all_node_names", [])
            rec_r.append(MM.best_cosine_rate(rr, cand, emb, COSINE_AT))
            rec_i.append(MM.best_cosine_rate(ri, cand, emb, COSINE_AT))
        return {
            "n_documents_attempted": len(rows),
            "n": len(ok),
            "no_graph_returned": dict(
                (k, sum(1 for r in rows.values() if r.get("failure_kind") == k))
                for k in {
                    r.get("failure_kind")
                    for r in rows.values()
                    if r.get("parse_failure")
                }
            ),
            "mean_nodes": round(statistics.mean([x["n_nodes"] for x in s]), 1),
            "mean_edges": round(statistics.mean([x["n_edges"] for x in s]), 1),
            "pct_of_attempted_yielding_a_chain": round(
                100
                * sum(r.get("score", {}).get("has_chain", False) for r in rows.values())
                / len(rows),
                1,
            ),
            "pct_chains_all_five": round(
                100
                * sum(x["n_chains_all_five_stages"] for x in s)
                / max(1, sum(x["n_chains"] for x in s)),
                1,
            ),
            "endpoint_recovery_pct": {
                "released_risks_found": round(statistics.mean(rec_r), 1)
                if rec_r
                else None,
                "released_interventions_found": round(statistics.mean(rec_i), 1)
                if rec_i
                else None,
                "ceiling_from_issue_168": ENDPOINT_CEILING_PCT,
            },
        }

    pops = {
        "B_abstract": urls_b,
        "C_gpt41": urls_cd,
        "D_triples": urls_cd,
        "U_unconditioned": urls_u,
    }
    head = {}
    for key, rows in arms.items():
        head[key] = agg(rows, pops[key])
        head[f"released_on_the_{key}_sample"] = {
            "n": len(pops[key]),
            "pct_of_attempted_yielding_a_chain": round(
                100
                * sum(released[u]["score"]["has_chain"] for u in pops[key])
                / max(1, len(pops[key])),
                1,
            ),
            "mean_nodes": round(
                statistics.mean([released[u]["score"]["n_nodes"] for u in pops[key]]), 1
            ),
        }
    return {
        "headline": head,
        "per_document": {
            k: {
                u: {
                    kk: vv
                    for kk, vv in r.items()
                    if kk not in ("names", "all_node_names")
                }
                for u, r in v.items()
            }
            for k, v in arms.items()
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    sample = load_inputs()
    built = build(sample)

    receipt = {
        "study": "baselines B/C/D and an unconditioned repeat arm (issue #171)",
        "arms": ARMS,
        "populations": {
            "C_and_D": {
                "n": N_C_D,
                "source": "the first 30 of issue #165's chain-yielding sample",
            },
            "B": {
                "n_drawn": len(built["arm_B"]["drawn"]),
                "n_with_abstract": len(built["arm_B"]["with_abstract"]),
                "n_failed_fetch": built["arm_B"]["n_failed_fetch"],
                "source": "arXiv chain-yielding documents; abstracts from the arXiv API",
                "why_not_from_ARD": (
                    "0 of the 1,869 chain-yielding documents carry an abstract field of "
                    "200+ characters in the local ARD snapshot"
                ),
            },
            "U": {
                "n": N_U,
                "seed": SEED_U,
                "source": "drawn from all 11,779 corpus documents",
            },
        },
        "rates_usd_per_million_ASSUMED": RATES,
    }

    if args.run:
        run([a.strip() for a in args.run.split(",") if a.strip()], built, args.limit)
    if args.score:
        receipt["results"] = score(sample, built)
    if not args.run and not args.score:
        receipt["dry_run"] = dry_run(built)
        print(json.dumps(receipt["dry_run"], indent=2))

    receipt["wall_clock_s"] = round(time.time() - t0, 1)
    if args.score:
        OUT.write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(receipt["results"]["headline"], indent=2))
        print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
