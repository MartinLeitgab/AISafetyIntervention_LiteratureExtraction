#!/usr/bin/env python
"""Does the audit hold on the unit the paper analyses? (S2, GitHub issue #172)

The first judge run sampled the population the released graph is built from. That is the
right population for the artifact we ship, and it leaves only 12 of 100 judged papers
inside the gate-selected chain set. This run judges 100 documents drawn from the 1,868
CHAIN-YIELDING ones instead, stratified by source type to match that population, so the
two runs differ in sampling frame and in nothing else.

Fidelity to the first run
-------------------------
  * model      claude-sonnet-4-5-20250929, the version the first run used
  * prompt     read at run time from extraction_validator/schema.py on branch
               judge_handoff_workshop_items_2_3, so it cannot drift from what ran
  * system     the KG-Judge system prompt from that branch's judge.py
  * max_tokens 32,000, as there
  * transport  Message Batches API -- half price, and server-side, so the batch survives a
               killed local process

One asymmetry, stated rather than hidden
----------------------------------------
The first run judged the extractor's own JSON, carrying node_rationale, edge_rationale and
the two intervention-rationale fields. The released graph does not store them, so each
extraction here is reconstructed without them. The SCHEMA-CHECK counts are therefore not
comparable across runs -- 84% of the first run's blocker flags were this exact mismatch --
while the COVERAGE and OMISSION measures, which are what this study exists for, are.

CLASS A (metered Anthropic batch API). Run from graph_analysis/:

    python -u experiment_review_judge_chainyielding.py                 # dry run
    python -u experiment_review_judge_chainyielding.py --submit        # create the batch
    python -u experiment_review_judge_chainyielding.py --poll          # fetch when ready
    python -u experiment_review_judge_chainyielding.py --score

Output:
    phase2_results/judge_chainyielding/batch_id.json, results.jsonl
    phase2_results/experiment_review_judge_chainyielding_report.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import tiktoken

import experiment_review_schema_ablation as ABL

ROOT = Path(__file__).parent
REPO = ROOT.parent
NODES = (
    ROOT
    / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/graph_node_attributes.pkl"
)
EDGES = (
    ROOT
    / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/graph_edge_data.pkl"
)
WORK = ROOT / "phase2_results/judge_chainyielding"
OUT = ROOT / "phase2_results/experiment_review_judge_chainyielding_report.json"

JUDGE_BRANCH = "origin/judge_handoff_workshop_items_2_3"
JUDGE_FILE = "extraction_validator/schema.py"
MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 32_000
SYSTEM_PROMPT = (
    "You are KG-Judge, a precise and rigorous auditor for knowledge graphs. "
    "Always return valid JSON in the exact format requested."
)
N_DOCS = 100
SEED = 45
# Batch rates, USD per million tokens: half the synchronous 3 / 15.
RATE_IN, RATE_OUT = 1.50, 7.50

# The first run's figures, for the comparison this study exists to make.
FIRST_RUN = {
    "n_papers": 100,
    "papers_with_an_added_node": 7,
    "added_nodes": 9,
    "nodes_in_those_extractions": 1617,
    "node_omission_pct": 0.6,
    "coverage_rows": 777,
    "coverage_missing": 302,
    "edges_in_those_extractions": 1667,
    "edge_omission_pct": 18.1,
}


def fail(msg: str, artifact, produced_by: str) -> None:
    raise SystemExit(
        f"FATAL: {msg}\n"
        f"  expected artifact: {artifact}\n"
        f"  produced by: {produced_by}\n"
        "  this script does NOT substitute a different model version, a rebuilt prompt or "
        "a synchronous call for the batch API."
    )


def judge_prompt_template() -> str:
    """The prompt as it ran, read from the branch rather than copied into this file."""
    try:
        src = subprocess.run(
            ["git", "show", f"{JUDGE_BRANCH}:{JUDGE_FILE}"],
            cwd=REPO,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout
    except subprocess.CalledProcessError:
        fail("judge prompt not readable", f"{JUDGE_BRANCH}:{JUDGE_FILE}", "git fetch")
    m = re.search(r'def create_validation_prompt\(.*?return f"""(.*?)"""', src, re.S)
    if not m:
        fail("create_validation_prompt template not found", JUDGE_FILE, JUDGE_BRANCH)
    return m.group(1)


def render_prompt(
    template: str, text: str, extraction: dict, prompt_extract: str
) -> str:
    out = template.replace("{original_text}", text)
    out = out.replace("{PROMPT_EXTRACT}", prompt_extract)
    out = out.replace(
        "{kg_output.model_dump_json(indent=2)}",
        json.dumps(extraction, indent=2, ensure_ascii=False),
    )
    # The template is an f-string: its literal JSON braces are doubled.
    out = out.replace("{{", "{").replace("}}", "}")
    if "{" + "original_text" + "}" in out:
        fail("prompt substitution failed", JUDGE_FILE, JUDGE_BRANCH)
    return out


def build_extractions() -> dict:
    """Reconstruct each document's extraction JSON from the released graph."""
    for p, what, how in [
        (NODES, "full node attributes", "phase2_step1_loadandparse.py"),
        (EDGES, "edge checkpoint", "phase2_step1_loadandparse.py"),
    ]:
        if not p.exists():
            fail(f"{what} not found", p, how)
    na = pickle.load(open(NODES, "rb"))
    by_url = defaultdict(list)
    for nid, a in na.items():
        if a.get("url"):
            by_url[a["url"]].append(nid)
    edges = pickle.load(open(EDGES, "rb"))
    edges_by_url = defaultdict(list)
    url_of = {nid: a.get("url") for nid, a in na.items()}
    for e in edges:
        if e.get("type") != "EDGE":
            continue
        u = url_of.get(e["source"])
        if u and url_of.get(e["target"]) == u:
            edges_by_url[u].append(e)
    del edges

    out = {}
    for u, nids in by_url.items():
        name = {n: na[n].get("name") for n in nids}
        nodes = []
        for n in nids:
            a = na[n]
            row = {
                "name": a.get("name"),
                "aliases": list(a.get("aliases") or [])[:3],
                "type": a.get("type"),
                "description": a.get("description"),
            }
            if (a.get("type") or "") == "intervention":
                row["intervention_lifecycle"] = a.get("intervention_lifecycle")
                row["intervention_maturity"] = a.get("intervention_maturity")
            else:
                row["concept_category"] = a.get("concept_category")
            nodes.append(row)
        eds = [
            {
                "type": e.get("subtype"),
                "source_node": name.get(e["source"]),
                "target_node": name.get(e["target"]),
                "description": e.get("description"),
                "edge_confidence": e.get("confidence"),
            }
            for e in edges_by_url.get(u, [])
        ]
        out[u] = {"nodes": nodes, "edges": eds}
    return out


def chain_yielding_urls(na_slim: dict) -> list:
    urls = set()
    with open(ABL.PATHS, encoding="utf-8") as fh:
        for line in fh:
            u = na_slim.get(json.loads(line)["path"][0], {}).get("url")
            if u:
                urls.add(u)
    return sorted(urls)


def stratified_sample(urls: list) -> list:
    """Proportional to the chain-yielding population's own source mix, largest remainder."""
    buckets = defaultdict(list)
    for u in urls:
        buckets[ABL.host_bucket(u)].append(u)
    total = len(urls)
    exact = {b: N_DOCS * len(v) / total for b, v in buckets.items()}
    alloc = {b: int(v) for b, v in exact.items()}
    rem = sorted(buckets, key=lambda b: exact[b] - alloc[b], reverse=True)
    i = 0
    while sum(alloc.values()) < N_DOCS:
        alloc[rem[i % len(rem)]] += 1
        i += 1
    rng = random.Random(SEED)
    picked = []
    for b, n in alloc.items():
        picked += rng.sample(buckets[b], min(n, len(buckets[b])))
    return sorted(picked)


def anthropic_client():
    from dotenv import dotenv_values

    import anthropic

    key = (dotenv_values(ABL.ENV_PATH) or {}).get("ANTHROPIC_API_KEY") or ""
    if not key.strip():
        fail("ANTHROPIC_API_KEY missing", ABL.ENV_PATH, "runbook R0")
    return anthropic.Anthropic(api_key=key.strip(), timeout=1800.0, max_retries=3)


def load_jobs() -> tuple[list, dict]:
    sample = ABL.load_sample()  # gives node_attrs and the ARD text index helpers
    na_slim = sample["node_attrs"]
    urls = stratified_sample(chain_yielding_urls(na_slim))
    wanted = set(urls)
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
    missing = [u for u in urls if u not in texts]
    if missing:
        print(f"  {len(missing)} sampled documents have no ARD text and are dropped")
    urls = [u for u in urls if u in texts]
    extractions = build_extractions()
    template = judge_prompt_template()
    prompt_extract = ABL.load_released_prompt()
    jobs = []
    for u in urls:
        if u not in extractions:
            continue
        jobs.append(
            {
                "url": u,
                "custom_id": ABL.doc_id(u)[:60],
                "prompt": render_prompt(
                    template, texts[u], extractions[u], prompt_extract
                ),
                "n_nodes": len(extractions[u]["nodes"]),
                "n_edges": len(extractions[u]["edges"]),
            }
        )
    return jobs, {
        "n_sampled": len(urls),
        "source_mix": dict(Counter(ABL.host_bucket(u) for u in urls)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--poll", action="store_true")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.poll or args.score:
        jobs, meta = (None, None)
    else:
        jobs, meta = load_jobs()

    if not any([args.submit, args.poll, args.score]):
        enc = tiktoken.get_encoding(ABL.ENCODING)
        ins = [len(enc.encode(j["prompt"], disallowed_special=())) for j in jobs]
        est_out = 4000 * len(jobs)
        print(
            json.dumps(
                {
                    "n_calls": len(jobs),
                    "input_tokens_total": sum(ins),
                    "input_tokens_mean": round(statistics.mean(ins)),
                    "assumed_output_tokens_total": est_out,
                    "usd_at_batch_rates": round(
                        sum(ins) / 1e6 * RATE_IN + est_out / 1e6 * RATE_OUT, 2
                    ),
                    "model": MODEL,
                    "sample": meta,
                },
                indent=2,
            )
        )
        return

    if args.submit:
        client = anthropic_client()
        requests = [
            {
                "custom_id": j["custom_id"],
                "params": {
                    "model": MODEL,
                    "max_tokens": MAX_TOKENS,
                    "system": SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": j["prompt"]}],
                },
            }
            for j in jobs
        ]
        batch = client.messages.batches.create(requests=requests)
        (WORK / "batch_id.json").write_text(
            json.dumps(
                {
                    "id": batch.id,
                    "model": MODEL,
                    "n_requests": len(requests),
                    "url_by_custom_id": {j["custom_id"]: j["url"] for j in jobs},
                    "extraction_size_by_custom_id": {
                        j["custom_id"]: {"nodes": j["n_nodes"], "edges": j["n_edges"]}
                        for j in jobs
                    },
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"submitted batch {batch.id} with {len(requests)} requests")
        return

    if args.poll:
        meta = json.loads((WORK / "batch_id.json").read_text(encoding="utf-8"))
        client = anthropic_client()
        b = client.messages.batches.retrieve(meta["id"])
        print(f"status={b.processing_status} counts={b.request_counts}")
        if b.processing_status != "ended":
            return
        with open(WORK / "results.jsonl", "w", encoding="utf-8") as fh:
            for r in client.messages.batches.results(meta["id"]):
                row = {"custom_id": r.custom_id, "type": r.result.type}
                if r.result.type == "succeeded":
                    m = r.result.message
                    row["text"] = "".join(
                        b.text for b in m.content if getattr(b, "type", "") == "text"
                    )
                    row["usage"] = {
                        "input_tokens": m.usage.input_tokens,
                        "output_tokens": m.usage.output_tokens,
                    }
                    row["stop_reason"] = m.stop_reason
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {WORK / 'results.jsonl'}")
        return

    if args.score:
        meta = json.loads((WORK / "batch_id.json").read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in (WORK / "results.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        cov = Counter()
        add_nodes = 0
        papers_with_add = 0
        parsed = 0
        unparsed = []
        n_nodes = n_edges = 0
        usage_in = usage_out = 0
        for r in rows:
            if r.get("type") != "succeeded":
                unparsed.append((r["custom_id"], r.get("type")))
                continue
            usage_in += r["usage"]["input_tokens"]
            usage_out += r["usage"]["output_tokens"]
            d = ABL.parse_extraction(r.get("text") or "")
            if d is None:
                m = re.search(r"\{.*\}", r.get("text") or "", re.S)
                try:
                    d = json.loads(m.group(0)) if m else None
                except json.JSONDecodeError:
                    d = None
            if not isinstance(d, dict) or "validation_report" not in d:
                unparsed.append((r["custom_id"], "unparseable"))
                continue
            parsed += 1
            size = meta["extraction_size_by_custom_id"].get(r["custom_id"], {})
            n_nodes += size.get("nodes", 0)
            n_edges += size.get("edges", 0)
            rep = d["validation_report"]
            for row in (rep.get("coverage") or {}).get(
                "expected_edges_from_source"
            ) or []:
                cov[(row.get("status") or "unknown").strip().lower()] += 1
            fixes = (d.get("proposed_fixes") or {}).get("add_nodes") or []
            add_nodes += len(fixes)
            papers_with_add += 1 if fixes else 0

        missing = cov.get("missing", 0)
        report = {
            "study": "judge run stratified on chain-yielding papers (issue #172, S2)",
            "model": MODEL,
            "transport": "Anthropic Message Batches API",
            "sampling_frame": "the 1,868 chain-yielding documents, stratified by source type",
            "n_requests": meta["n_requests"],
            "n_parsed": parsed,
            "unparsed": unparsed,
            "extraction_size_over_parsed": {"nodes": n_nodes, "edges": n_edges},
            "node_level": {
                "papers_with_an_added_node": papers_with_add,
                "added_nodes": add_nodes,
                "pct_of_nodes": round(100 * add_nodes / max(1, n_nodes), 1),
            },
            "edge_level": {
                "coverage_rows": sum(cov.values()),
                "by_status": dict(cov),
                "missing": missing,
                "missing_per_paper": round(missing / max(1, parsed), 2),
                "pct_of_edges": round(100 * missing / max(1, n_edges), 1),
            },
            "first_run_for_comparison": FIRST_RUN,
            "not_comparable_across_runs": (
                "schema_check counts: this run reconstructs extractions from the released "
                "graph, which does not store the four rationale fields, and 84% of the "
                "first run's blocker flags were that mismatch"
            ),
            "usage": {"input_tokens": usage_in, "output_tokens": usage_out},
            "usd_at_batch_rates": round(
                usage_in / 1e6 * RATE_IN + usage_out / 1e6 * RATE_OUT, 2
            ),
            "wall_clock_s": round(time.time() - t0, 1),
        }
        OUT.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            json.dumps({k: v for k, v in report.items() if k != "unparsed"}, indent=2)
        )
        print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
