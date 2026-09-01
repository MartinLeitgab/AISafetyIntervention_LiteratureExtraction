#!/usr/bin/env python3
"""What share of released chains assert something their source does not?

The paper measures omission four ways and precision zero ways. Appendix P exhibits one
invented safety framing (Euclid's proof read as "prime scarcity affecting cryptographic key
space") and arm F puts 30% on chains recovered from sentence-shuffled text, but nothing
measures how often a chain in the RELEASED reporting unit asserts a framing, or an
intervention, that its source never states. Two reviewers in the 2026-08-17 round asked for
exactly that number and one called it more decision-relevant than any omission figure in the
paper.

This is stage 1 of a hybrid design. It is NOT a human anchor and does not claim to be:

  stage 1 (this script)  an independent judge over 200 sampled chains, cross-provider from
                         the extractor, forced to cite a verbatim span from the source for
                         every "supported" verdict, plus a MISMATCHED-PAIR NULL ARM.
  stage 2 (a person)     adjudicate ~30: every chain stage 1 flagged, plus a random sample
                         it cleared. Stage 2 is what licenses any extrapolation from stage 1.

Read stage 1 one-sidedly. A judge that shares the extractor's priors should UNDER-detect
invention, so a high rate is a floor and is informative, while a low rate cannot distinguish
faithful extraction from a shared blind spot. The null arm is what makes the instrument
checkable at all: 10 items pair a chain with a DIFFERENT document's source text, drawn from
the same source type so register alone does not give it away. A judge that does not flag
those has told us its verdicts are worthless, for the price of ten calls. The meta-grader
stage this project already ran lacked such a control and could conclude nothing; do not
repeat that.

Class A: metered Anthropic batch API. Roughly USD 5 for the full run at 210 requests.
Run --dry-run first: it prints the measured input-token projection for the whole sample
before anything is submitted.

Usage
-----
    cd graph_analysis
    python -u experiment_review_chain_precision.py --dry-run
    python -u experiment_review_chain_precision.py --submit
    python -u experiment_review_chain_precision.py --collect <batch_id>
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEDUPED = HERE / "phase1_rawpathsfiles" / "paths_hopwise_v4_edge_only_deduped.jsonl"
NODE_ATTRS = HERE / "phase2_results" / "node_attrs_slim.pkl"
ARD_DIR = ROOT / "intervention_graph_creation" / "data" / "raw" / "ard_json_full"
OUT_DIR = HERE / "phase2_results"
RAW_OUT = OUT_DIR / "chain_precision_raw"
RECEIPT = OUT_DIR / "experiment_review_chain_precision_report.json"
BATCH_ID_FILE = RAW_OUT / "batch_id.txt"
SAMPLE_FILE = RAW_OUT / "sample.json"
# Arm B keeps its own manifest and batch id so the two arms can be collected independently,
# but they share the prompt, the model and the transport -- that is what makes them
# comparable, so do not "improve" the instrument between them.
CONTRAST_BATCH_ID_FILE = RAW_OUT / "batch_id_contrast.txt"
CONTRAST_SAMPLE_FILE = RAW_OUT / "sample_contrast.json"

# The key lives outside this repo and is never copied into it, logged, or written to a
# receipt. Same path the ablation and multi-model studies use.
KEY_ENV = Path.home() / "0_project_work" / "ExistentialRiskBenchmark" / ".env"
KEY_VAR = "ANTHROPIC_API_KEY"

# Pinned to the version the two judge runs used, so this is comparable to them and remains
# cross-provider against the o3 extractor.
MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 1200

N_CHAINS = 200
N_NULL = 10
N_CONTRAST = 100
SEED = 42

STAGE_ORDER = [
    "risk",
    "problem analysis",
    "theoretical insight",
    "design rationale",
    "implementation mechanism",
    "validation evidence",
]

SYSTEM = (
    "You audit whether a structured claim about a document is supported by that document. "
    "You are strict about attribution: a claim is supported only if you can quote the span "
    "that supports it. You never treat a plausible-sounding claim as supported."
)

PROMPT = """\
Below is the full text of a document, and a chain of claims that an automated pipeline
extracted from it. The chain is supposed to record an argument THE DOCUMENT ITSELF makes,
running from a risk it names to an intervention it proposes.

Your job is to decide whether the document actually makes that argument. You are not judging
whether the argument is correct, whether the intervention would work, or whether the document
is good research. A faithful record of a weak argument is faithful.

=== BEGIN DOCUMENT ===
TITLE: {title}

{text}
=== END DOCUMENT ===

=== BEGIN EXTRACTED CHAIN ===
{chain}
=== END EXTRACTED CHAIN ===

Answer in this exact JSON format and nothing else:

{{
  "risk_framing": {{
    "verdict": "supported" | "partial" | "unsupported",
    "quote": "a verbatim span from the document that names this risk, or \\"\\" if none exists"
  }},
  "intervention": {{
    "verdict": "supported" | "partial" | "unsupported",
    "quote": "a verbatim span in which the document proposes this intervention, or \\"\\""
  }},
  "intermediate_stages": {{
    "verdict": "supported" | "partial" | "unsupported",
    "note": "one sentence on any intermediate step the document does not support"
  }},
  "chain_is_a_fair_summary_of_an_argument_the_document_makes": true | false,
  "reason_code": "faithful" | "risk_framing_invented" | "intervention_not_proposed"
                 | "intermediate_unsupported" | "chain_belongs_to_a_different_document",
  "confidence": 1 | 2 | 3 | 4 | 5
}}

Rules. A "supported" verdict REQUIRES a quote copied verbatim from the document above; if you
cannot find one, the verdict is "unsupported" and the quote is the empty string. If the chain
appears to describe a different document altogether, say so with the
chain_belongs_to_a_different_document reason code. Do not repair, improve or reinterpret the
chain to make it fit.
"""


def die(msg: str) -> None:
    raise SystemExit(f"FATAL: {msg}")


def read_key() -> str:
    if not KEY_ENV.is_file():
        die(
            f"Anthropic key file not found.\n  expected: {KEY_ENV}\n"
            f"  expected variable: {KEY_VAR}\n"
            f"  This script does NOT fall back to an environment variable or another "
            f"provider, and never copies the key into this repo."
        )
    for line in KEY_ENV.read_text(encoding="utf-8", errors="replace").splitlines():
        name, _, value = line.strip().partition("=")
        if name.strip() == KEY_VAR and value.strip():
            return value.strip().strip('"').strip("'")
    die(f"{KEY_VAR} not present or empty in {KEY_ENV}")
    return ""


def load_sources() -> dict[str, dict]:
    files = sorted(glob.glob(str(ARD_DIR / "*.jsonl")))
    if not files:
        die(
            f"ARD source text not found.\n  expected: {ARD_DIR}/*.jsonl\n"
            f"  produced by: the ARD download (12 .jsonl, ~440 MB)\n"
            f"  This script does NOT reconstruct source text from the graph -- the whole "
            f"point is to compare a chain against the document it came from."
        )
    by_url: dict[str, dict] = {}
    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                url = (r.get("url") or "").strip()
                if url and (r.get("text") or "").strip():
                    by_url.setdefault(url, r)
    return by_url


def host_of(url: str) -> str:
    m = re.match(r"https?://([^/]+)", url or "")
    return (m.group(1) if m else "unknown").lower().replace("www.", "")


def build_sample() -> list[dict]:
    attrs = pickle.load(NODE_ATTRS.open("rb"))
    sources = load_sources()
    print(f"  {len(sources):,} ARD documents with text, keyed by URL", flush=True)

    chains = []
    with DEDUPED.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            nodes = json.loads(line)["path"]
            url = attrs.get(nodes[0], {}).get("url") or ""
            if url in sources:
                chains.append({"chain_index": i, "nodes": nodes, "url": url})
    print(
        f"  {len(chains):,} of the 2,772 chains have their source document on disk",
        flush=True,
    )
    if len(chains) < N_CHAINS + N_NULL:
        die(f"only {len(chains)} usable chains, need {N_CHAINS + N_NULL}")

    # Stratify by URL host, proportional to the reporting unit, so the sample carries the
    # venue mix the chain set has rather than the corpus mix.
    by_host: dict[str, list] = defaultdict(list)
    for c in chains:
        by_host[host_of(c["url"])].append(c)
    rng = random.Random(SEED)
    for v in by_host.values():
        rng.shuffle(v)
    total = len(chains)
    picked: list[dict] = []
    for host, items in sorted(by_host.items(), key=lambda kv: -len(kv[1])):
        take = round(N_CHAINS * len(items) / total)
        picked.extend(items[:take])
    rng.shuffle(picked)
    picked = picked[:N_CHAINS]
    if len(picked) < N_CHAINS:  # rounding shortfall, fill at random from the remainder
        chosen = {id(x) for x in picked}
        pool = [c for c in chains if id(c) not in chosen]
        rng.shuffle(pool)
        picked.extend(pool[: N_CHAINS - len(picked)])

    items = []
    for k, c in enumerate(picked):
        items.append(
            {
                "custom_id": f"real-{k:04d}",
                "arm": "real",
                "chain_index": c["chain_index"],
                "nodes": c["nodes"],
                "chain_url": c["url"],
                "source_url": c["url"],
                "host": host_of(c["url"]),
            }
        )

    # NULL ARM. Pair a chain with a different document of the same host, so the mismatch is
    # not detectable from register or topic vocabulary alone.
    chosen_idx = {c["chain_index"] for c in picked}
    spare = [c for c in chains if c["chain_index"] not in chosen_idx]
    rng.shuffle(spare)
    used: set[str] = set()
    for c in spare:
        if len(used) >= N_NULL:
            break
        h = host_of(c["url"])
        others = [d for d in spare if host_of(d["url"]) == h and d["url"] != c["url"]]
        if not others or c["url"] in used:
            continue
        other = rng.choice(others)
        used.add(c["url"])
        items.append(
            {
                "custom_id": f"null-{len(used) - 1:04d}",
                "arm": "null_mismatched_pair",
                "chain_index": c["chain_index"],
                "nodes": c["nodes"],
                "chain_url": c["url"],
                "source_url": other["url"],
                "host": h,
            }
        )
    if len(used) < N_NULL:
        die(f"could only build {len(used)} mismatched pairs, need {N_NULL}")

    # Render each item's prompt now, so the dry run measures exactly what is submitted.
    for it in items:
        src = sources[it["source_url"]]
        it["title"] = (src.get("title") or "(no title)").strip()
        it["text"] = src["text"]
        it["chain_str"] = render_chain(it["nodes"], attrs)
        it["prompt"] = PROMPT.format(
            title=it["title"], text=it["text"], chain=it["chain_str"]
        )
    return items


def render_chain(nodes: list[int], attrs: dict) -> str:
    lines = []
    for n in nodes:
        a = attrs.get(n, {})
        if a.get("type") == "intervention":
            label = f"INTERVENTION (model-assigned maturity {a.get('intervention_maturity')})"
        else:
            label = (a.get("concept_category") or "?").upper()
        lines.append(f"  {label}: {a.get('name')}")
    return "\n".join(lines)


def build_contrast_sample() -> list[dict]:
    """ARM B: chains the deployed gates REJECT, judged with the identical instrument.

    The point of the two arms is that the composite verdict of arm A cannot be read on its
    own. The extraction prompt licenses moderate inference where a stage is absent and caps
    it at edge confidence 2, and the reporting unit gates that out at confidence >= 3 and
    maturity >= 3. So a chain in arm A that a judge calls unsupported is a failure OF THE
    GATES, while the same verdict on a gate-rejected chain is the design working as intended.
    Only the difference between the arms measures whether the gates discriminate, which is
    reviewer R21 and is the one thing a single-arm precision number cannot answer.

    Gate-rejected is computed per chain rather than by set subtraction against the released
    file: a chain qualifies if its weakest hop carries confidence < 3 or its intervention
    endpoint carries maturity < 3. Set subtraction would also catch chains that pass the
    gates but lost the collapse to a different representative, which are not gate-rejected.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gate_sens", HERE / "experiment_review_gate_sensitivity.py"
    )
    gs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gs)
    sys.setrecursionlimit(50000)

    slim = pickle.load(gs.SLIM.open("rb"))
    edge_rows = pickle.load(gs.EDGES.open("rb"))
    print(f"  {len(slim):,} nodes, {len(edge_rows):,} edge rows", flush=True)

    risk_nodes = {
        n
        for n, a in slim.items()
        if (a.get("concept_category") or "").lower() == "risk"
    }
    body_nodes = {
        n
        for n, a in slim.items()
        if (a.get("concept_category") or "").lower() in set(gs.BODY)
    }
    interventions = {
        n for n, a in slim.items() if (a.get("type") or "").lower() == "intervention"
    }
    mat_ge = {
        m: {
            n for n in interventions if (slim[n].get("intervention_maturity") or 0) >= m
        }
        for m in (1, 3)
    }
    url_of = {n: a.get("url") for n, a in slim.items()}

    # Best confidence available on each unordered hop. The enumerator walks a hop if ANY
    # structural edge between the pair clears the gate, so max over parallel edges is what
    # the gate saw, and min over hops is the chain's weakest link.
    best_conf: dict[frozenset, int] = {}
    for e in edge_rows:
        if e.get("type") != "EDGE":
            continue
        c = e.get("confidence")
        if c is None or e["source"] == e["target"]:
            continue
        k = frozenset((e["source"], e["target"]))
        if c > best_conf.get(k, -1):
            best_conf[k] = c

    # Guard: reproduce the released enumeration at the deployed setting before trusting the
    # open one. Same assertion experiment_review_gate_sensitivity.py makes.
    released_raw = [
        json.loads(line)["path"]
        for line in DEDUPED.with_name("paths_hopwise_v4_edge_only.jsonl").open(
            encoding="utf-8"
        )
    ]
    adj3, _ = gs.build_adjacency(edge_rows, 3)
    base, _ = gs.enumerate_paths(adj3, risk_nodes, body_nodes, mat_ge[3])
    if Counter(map(tuple, base)) != Counter(map(tuple, released_raw)):
        die(
            "the in-memory enumerator does not reproduce the released 8,954-path file at "
            "the deployed gate setting, so nothing it produces at another setting can be "
            "trusted. Do NOT proceed."
        )
    print(
        f"  guard OK: reproduced the released {len(base):,}-path enumeration",
        flush=True,
    )

    adj1, _ = gs.build_adjacency(edge_rows, 1)
    open_paths, _ = gs.enumerate_paths(adj1, risk_nodes, body_nodes, mat_ge[1])
    urls_open = []
    for p in open_paths:
        u = {url_of.get(x) for x in p}
        u.discard(None)
        urls_open.append(sorted(u)[0] if u else None)
    keep, _ = gs.dedupe(open_paths, urls_open, 0.70)
    open_chains = [open_paths[i] for i in keep]
    print(
        f"  fully-open setting: {len(open_chains):,} chains after the 0.70 collapse "
        f"(tab:gates row 9 reports 31,740)",
        flush=True,
    )

    sources = load_sources()
    rejected = []
    for p in open_chains:
        endpoint_mat = slim.get(p[-1], {}).get("intervention_maturity") or 0
        hops = [best_conf.get(frozenset((u, v)), 0) for u, v in zip(p, p[1:])]
        min_conf = min(hops) if hops else 0
        if min_conf >= 3 and endpoint_mat >= 3:
            continue  # would pass the deployed gates; not a contrast item
        url = slim.get(p[0], {}).get("url") or ""
        if url in sources:
            rejected.append(
                {
                    "nodes": p,
                    "url": url,
                    "min_conf": min_conf,
                    "endpoint_mat": endpoint_mat,
                }
            )
    print(
        f"  gate-rejected chains with source text on disk: {len(rejected):,}",
        flush=True,
    )
    if len(rejected) < N_CONTRAST:
        die(f"only {len(rejected)} gate-rejected chains available, need {N_CONTRAST}")

    by_host: dict[str, list] = defaultdict(list)
    for c in rejected:
        by_host[host_of(c["url"])].append(c)
    rng = random.Random(SEED)
    for v in by_host.values():
        rng.shuffle(v)
    total = len(rejected)
    picked: list[dict] = []
    for host, its in sorted(by_host.items(), key=lambda kv: -len(kv[1])):
        picked.extend(its[: round(N_CONTRAST * len(its) / total)])
    rng.shuffle(picked)
    picked = picked[:N_CONTRAST]
    if len(picked) < N_CONTRAST:
        chosen = {tuple(x["nodes"]) for x in picked}
        pool = [c for c in rejected if tuple(c["nodes"]) not in chosen]
        rng.shuffle(pool)
        picked.extend(pool[: N_CONTRAST - len(picked)])

    items = []
    for k, c in enumerate(picked):
        src = sources[c["url"]]
        it = {
            "custom_id": f"rejected-{k:04d}",
            "arm": "gate_rejected",
            "nodes": c["nodes"],
            "chain_url": c["url"],
            "source_url": c["url"],
            "host": host_of(c["url"]),
            "min_edge_confidence": c["min_conf"],
            "endpoint_maturity": c["endpoint_mat"],
            "title": (src.get("title") or "(no title)").strip(),
            "text": src["text"],
        }
        it["chain_str"] = render_chain(it["nodes"], slim)
        it["prompt"] = PROMPT.format(
            title=it["title"], text=it["text"], chain=it["chain_str"]
        )
        items.append(it)
    print(
        f"  arm B: {len(items)} chains | min-confidence mix "
        f"{dict(Counter(i['min_edge_confidence'] for i in items))} | maturity mix "
        f"{dict(Counter(i['endpoint_maturity'] for i in items))}",
        flush=True,
    )
    return items


def approx_tokens(s: str) -> int:
    # Deliberately crude and deliberately conservative: characters / 3.6 runs above the
    # tokenizer for English prose, so the projection over-estimates rather than surprises.
    return int(len(s) / 3.6)


def project_cost(items: list[dict]) -> dict:
    inp = sum(approx_tokens(it["prompt"]) + approx_tokens(SYSTEM) for it in items)
    out = MAX_TOKENS // 3 * len(items)
    # Anthropic batch pricing is half the synchronous rate. Sonnet synchronous is USD 3 per
    # million input and 15 per million output at the time of writing; CHECK before running.
    dollars = inp / 1e6 * 1.50 + out / 1e6 * 7.50
    return {
        "requests": len(items),
        "projected_input_tokens": inp,
        "projected_output_tokens_upper": out,
        "projected_usd_at_sonnet_batch_rates": round(dollars, 2),
        "rate_basis_ASSUMED_not_measured": "USD 1.50/M in, 7.50/M out (batch = half of 3/15)",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--dry-run",
        action="store_true",
        help="build arm A, project the cost, and send exactly 3 synchronous requests",
    )
    g.add_argument(
        "--submit", action="store_true", help="submit arm A (200 real + 10 null)"
    )
    g.add_argument(
        "--build-contrast",
        action="store_true",
        help="build arm B (100 "
        "gate-rejected chains) and project its cost, without calling anything",
    )
    g.add_argument("--submit-contrast", action="store_true", help="submit arm B")
    g.add_argument(
        "--collect",
        metavar="BATCH_ID",
        nargs="?",
        const="",
        help="fetch a finished arm-A batch",
    )
    g.add_argument(
        "--collect-contrast",
        metavar="BATCH_ID",
        nargs="?",
        const="",
        help="fetch a finished arm-B batch",
    )
    args = ap.parse_args()

    RAW_OUT.mkdir(parents=True, exist_ok=True)

    if args.collect is not None:
        return collect(
            args.collect or BATCH_ID_FILE.read_text().strip(),
            SAMPLE_FILE,
            "results.jsonl",
        )
    if args.collect_contrast is not None:
        return collect(
            args.collect_contrast or CONTRAST_BATCH_ID_FILE.read_text().strip(),
            CONTRAST_SAMPLE_FILE,
            "results_contrast.jsonl",
        )

    if args.build_contrast or args.submit_contrast:
        print("building arm B (gate-rejected contrast) ...", flush=True)
        items = build_contrast_sample()
        proj = project_cost(items)
        print(f"\n  {len(items)} requests")
        print(f"  projected input tokens : {proj['projected_input_tokens']:,}")
        print(
            f"  projected cost         : USD "
            f"{proj['projected_usd_at_sonnet_batch_rates']}"
        )
        CONTRAST_SAMPLE_FILE.write_text(
            json.dumps(
                [
                    {k: v for k, v in it.items() if k not in ("text", "prompt")}
                    for it in items
                ],
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"  manifest -> {CONTRAST_SAMPLE_FILE}")
        if args.build_contrast:
            print("\nbuild only. Re-run with --submit-contrast to book the batch.")
            return 0
        import anthropic

        client = anthropic.Anthropic(api_key=read_key())
        batch = client.messages.batches.create(
            requests=[
                {
                    "custom_id": it["custom_id"],
                    "params": {
                        "model": MODEL,
                        "max_tokens": MAX_TOKENS,
                        "system": SYSTEM,
                        "messages": [{"role": "user", "content": it["prompt"]}],
                    },
                }
                for it in items
            ]
        )
        CONTRAST_BATCH_ID_FILE.write_text(batch.id, encoding="utf-8")
        print(f"\nsubmitted {len(items)} arm-B requests as {batch.id}")
        return 0

    for p in (DEDUPED, NODE_ATTRS):
        if not p.is_file():
            die(f"missing input: {p}")

    print("building sample ...", flush=True)
    items = build_sample()
    proj = project_cost(items)
    hosts = Counter(it["host"] for it in items if it["arm"] == "real")
    print(f"\n  {len(items)} requests ({N_CHAINS} real + {N_NULL} null)")
    print(f"  venue mix of the real arm: {dict(hosts.most_common(8))}")
    print(f"  projected input tokens : {proj['projected_input_tokens']:,}")
    print(f"  projected output (upper): {proj['projected_output_tokens_upper']:,}")
    print(
        f"  projected cost         : USD {proj['projected_usd_at_sonnet_batch_rates']}"
    )
    print(
        f"  longest prompt         : "
        f"{max(approx_tokens(it['prompt']) for it in items):,} tokens"
    )

    SAMPLE_FILE.write_text(
        json.dumps(
            [
                {k: v for k, v in it.items() if k not in ("text", "prompt")}
                for it in items
            ],
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"  sample manifest -> {SAMPLE_FILE}")

    import anthropic

    client = anthropic.Anthropic(api_key=read_key())

    if args.dry_run:
        print("\n--- dry run: 2 real + 1 null, synchronous ---", flush=True)
        probe = [it for it in items if it["arm"] == "real"][:2]
        probe += [it for it in items if it["arm"] == "null_mismatched_pair"][:1]
        for it in probe:
            t0 = time.monotonic()
            r = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM,
                messages=[{"role": "user", "content": it["prompt"]}],
            )
            body = r.content[0].text if r.content else ""
            (RAW_OUT / f"dryrun_{it['custom_id']}.json").write_text(
                json.dumps(
                    {
                        "item": {k: v for k, v in it.items() if k != "text"},
                        "response": body,
                    },
                    indent=1,
                ),
                encoding="utf-8",
            )
            print(
                f"\n[{it['custom_id']} / {it['arm']} / {it['host']}] "
                f"in={r.usage.input_tokens} out={r.usage.output_tokens} "
                f"{time.monotonic() - t0:.1f}s"
            )
            print(body[:900])
        print("\ndry run complete. Re-run with --submit to book the batch.")
        return 0

    requests = [
        {
            "custom_id": it["custom_id"],
            "params": {
                "model": MODEL,
                "max_tokens": MAX_TOKENS,
                "system": SYSTEM,
                "messages": [{"role": "user", "content": it["prompt"]}],
            },
        }
        for it in items
    ]
    batch = client.messages.batches.create(requests=requests)
    BATCH_ID_FILE.write_text(batch.id, encoding="utf-8")
    print(f"\nsubmitted {len(requests)} requests as {batch.id}")
    print(f"  id saved to {BATCH_ID_FILE}")
    print(f"  collect with: python -u {Path(__file__).name} --collect")
    return 0


def collect(batch_id: str, sample_file: Path, results_name: str) -> int:
    import anthropic

    client = anthropic.Anthropic(api_key=read_key())
    b = client.messages.batches.retrieve(batch_id)
    print(f"{batch_id}: {b.processing_status} counts={b.request_counts}")
    if b.processing_status != "ended":
        print("not finished; re-run later. Nothing written.")
        return 1

    manifest = {it["custom_id"]: it for it in json.loads(sample_file.read_text())}
    rows = []
    for res in client.messages.batches.results(batch_id):
        cid = res.custom_id
        rec = {"custom_id": cid, **{k: v for k, v in manifest.get(cid, {}).items()}}
        if res.result.type != "succeeded":
            rec["error"] = res.result.type
        else:
            body = (
                res.result.message.content[0].text if res.result.message.content else ""
            )
            rec["raw"] = body
            rec["usage"] = {
                "input_tokens": res.result.message.usage.input_tokens,
                "output_tokens": res.result.message.usage.output_tokens,
            }
            v = parse_verdict(body)
            if v is None:
                rec["parse_error"] = True
            else:
                rec["verdict"] = v
        rows.append(rec)

    (RAW_OUT / results_name).write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )

    # Rebuild the receipt from every arm collected so far, so the file always carries the
    # comparison rather than whichever arm was fetched last.
    rows = []
    for fn in ("results.jsonl", "results_contrast.jsonl"):
        fp = RAW_OUT / fn
        if fp.is_file():
            rows.extend(json.loads(x) for x in fp.read_text().splitlines() if x.strip())

    return write_receipt(rows, batch_id)


def parse_verdict(body: str) -> dict | None:
    """Pull the verdict object out of one response, or None if there is not one.

    Module-level, and deliberately separate from anything that makes a network call: this is
    the code that runs AFTER the money is spent, so it is the part that has to be provable
    without spending any. Exercised by tests/test_chain_precision_collect.py against the
    real dry-run responses on disk, including a deliberately malformed one.
    """
    m = re.search(r"\{.*\}", body, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def tally(rows: list[dict], arm: str) -> dict:
    sub = [r for r in rows if r.get("arm") == arm and "verdict" in r]
    n = len(sub)
    fair = sum(
        1
        for r in sub
        if r["verdict"].get("chain_is_a_fair_summary_of_an_argument_the_document_makes")
        is True
    )
    codes = Counter(r["verdict"].get("reason_code") for r in sub)
    risk_unsup = sum(
        1
        for r in sub
        if r["verdict"].get("risk_framing", {}).get("verdict") == "unsupported"
    )
    intv_unsup = sum(
        1
        for r in sub
        if r["verdict"].get("intervention", {}).get("verdict") == "unsupported"
    )
    quoted = sum(
        1
        for r in sub
        if r["verdict"].get("risk_framing", {}).get("verdict") == "supported"
        and (r["verdict"].get("risk_framing", {}).get("quote") or "").strip()
    )
    return {
        "n_parsed": n,
        "judged_fair_summary": fair,
        "judged_fair_summary_pct": round(100.0 * fair / n, 1) if n else None,
        "judged_not_fair": n - fair,
        "judged_not_fair_pct": round(100.0 * (n - fair) / n, 1) if n else None,
        "risk_framing_unsupported": risk_unsup,
        "intervention_unsupported": intv_unsup,
        "supported_risk_verdicts_carrying_a_quote": quoted,
        "reason_codes": dict(codes.most_common()),
    }


def write_receipt(rows: list[dict], batch_id: str) -> int:
    real, null = tally(rows, "real"), tally(rows, "null_mismatched_pair")
    rej = tally(rows, "gate_rejected")
    gate_delta = None
    if rej["n_parsed"] and real["n_parsed"]:
        gate_delta = {
            "question": (
                "Do the two quality gates select better-grounded chains? Arm A is the "
                "reporting unit (edge confidence >= 3, endpoint maturity >= 3); arm B is "
                "chains the same enumerator emits that those gates reject. Identical prompt, "
                "model and transport; the judge sees no gate label and cannot tell the arms "
                "apart."
            ),
            "arm_A_not_fair_pct": real["judged_not_fair_pct"],
            "arm_B_not_fair_pct": rej["judged_not_fair_pct"],
            "difference_pp": round(
                rej["judged_not_fair_pct"] - real["judged_not_fair_pct"], 1
            ),
            "reading": (
                "A large positive difference is evidence the gates discriminate on grounding "
                "and is the first direct answer to reviewer R21. A difference near zero says "
                "they do not, whatever arm A's absolute rate is. The absolute rate in arm A "
                "must not be read as a spuriousness rate for the corpus: the extraction "
                "prompt licenses moderate inference where a stage is absent and caps it at "
                "edge confidence 2, so what arm A measures is whether the gates removed it."
            ),
        }
    usage = {
        "input_tokens": sum(r.get("usage", {}).get("input_tokens", 0) for r in rows),
        "output_tokens": sum(r.get("usage", {}).get("output_tokens", 0) for r in rows),
    }
    receipt = {
        "study": "precision of the released chain set: does a chain assert what its source does not",
        "stage": "1 of 2 -- independent LLM judge. NOT a human anchor.",
        "model": MODEL,
        "transport": "Anthropic batch API",
        "batch_id": batch_id,
        "sample": {
            "unit": "chains from the 2,772-chain reporting unit",
            "n_real": N_CHAINS,
            "n_null_mismatched_pairs": N_NULL,
            "stratified_by": "URL host of the chain's risk node, proportional to the chain set",
            "seed": SEED,
        },
        "real_arm": real,
        "null_arm": null,
        "gate_rejected_arm": rej,
        "gate_discrimination": gate_delta,
        "instrument_check": {
            "question": "did the judge notice when the chain came from a different document?",
            "null_arm_flagged_pct": null["judged_not_fair_pct"],
            "reading": (
                "A null arm that is not flagged at a high rate invalidates the real arm. "
                "Read the real arm only if this is high."
            ),
        },
        "errors": sum(1 for r in rows if "error" in r or r.get("parse_error")),
        "usage": usage,
        "LIMITS": (
            "One model's opinion, cross-provider from the extractor but sharing its priors. "
            "A judge primed by the same schema should under-detect invention, so the real "
            "arm's not-fair share is a FLOOR, not an estimate. Stage 2 (human adjudication "
            "of every flagged chain plus a random sample of the cleared ones) is what "
            "licenses any extrapolation. No number here is a human-validated rate."
        ),
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    print(f"\n=== real arm (n={real['n_parsed']}) ===")
    print(f"  judged a fair summary : {real['judged_fair_summary_pct']}%")
    print(
        f"  judged NOT a fair summary : {real['judged_not_fair_pct']}% "
        f"({real['judged_not_fair']})"
    )
    print(f"  reason codes: {real['reason_codes']}")
    print(f"=== null arm (n={null['n_parsed']}) ===")
    print(
        f"  flagged: {null['judged_not_fair_pct']}% -- must be high or the real arm is void"
    )
    print(f"  reason codes: {null['reason_codes']}")
    if rej["n_parsed"]:
        print(f"=== gate-rejected arm (n={rej['n_parsed']}) ===")
        print(f"  judged NOT a fair summary : {rej['judged_not_fair_pct']}%")
        print(f"  reason codes: {rej['reason_codes']}")
    # gate_delta needs BOTH arms. Whichever batch ends first collects alone, so this must
    # survive a receipt that has arm B and not arm A -- which is exactly what happened on
    # 2026-08-17 and crashed the first collect after arm B landed five minutes early.
    if gate_delta:
        print(
            f"  GATE DISCRIMINATION: arm A {gate_delta['arm_A_not_fair_pct']}% vs "
            f"arm B {gate_delta['arm_B_not_fair_pct']}% "
            f"= {gate_delta['difference_pp']} pp"
        )
    elif rej["n_parsed"] or real["n_parsed"]:
        print("  GATE DISCRIMINATION: not computable yet, one arm is still outstanding")
    print(f"\nusage: {usage}; errors: {receipt['errors']}")
    print(f"wrote {RECEIPT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
