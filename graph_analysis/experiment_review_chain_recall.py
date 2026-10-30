#!/usr/bin/env python3
"""How many ARGUMENTS does the extraction miss? Recall, measured in the unit we report.

The project measures omission five ways -- 0.6%, 28.8%, 26.4% at node level and 18.1%,
21.7% at relationship level -- and they span two orders of magnitude. That spread is not a
finding about the corpus. It is an artifact of asking a SCHEMA-CONFORMANCE instrument a
RECALL question: the judge's `proposed_fixes.add_nodes` and `coverage.expected_edges_from_
source` were built to check that an extraction conforms to its schema, and both count
fragments. A concept added mid-argument does not change which risk reaches which
intervention, so neither number bounds what the artifact actually claims to hold.

experiment_review_omission_chain_impact.py already showed how far apart the units sit:
granting the judge every flagged relationship across 100 papers makes ONE further
risk-to-intervention pair reachable, out of 86 that were structurally available. That
bounds the flagged omissions. It does not measure recall, because the judge was never asked
to look for whole arguments in the first place.

This asks directly, and it is the instrument that should have been run instead of counting
added nodes.

    Per document: enumerate every risk-to-intervention argument the source makes. For each,
    decide whether the extraction already carries it -- against the extraction's own list of
    risk-to-intervention pairs, matched on MEANING and never on wording. What is left over
    is a missed argument, and only some of those matter.

MATERIALITY, because not every uncaptured argument is a defect
    `carried`              A listed pair makes this argument. Different wording is still
                           carried: the load-bearing content of a chain is a KIND of risk
                           linked to a KIND of intervention, and that survives rephrasing.
    `uncaptured_thin`      Not carried, but the document supports it with fewer than two
                           reasoning steps beyond the two endpoints. A mention, not an
                           argument. Counted, reported, and NOT part of the headline.
    `uncaptured_material`  Not carried, introduces a risk or an intervention no listed pair
                           has, AND the document gives at least two supporting reasoning
                           steps the extraction does not hold. This is the number.

THE ABLATION ARM IS THE POINT, not a nicety
    A miss rate from an unvalidated judge is worth what the last five were worth. So for
    N_ABLATE documents the list is shown with one INTERVENTION deleted -- every pair ending
    at it -- and we measure how often the judge surfaces the endpoint we removed. That is a
    measured sensitivity: recover 8 of 10 and an observed miss rate is a floor correctable
    by 1/0.8; recover 2 of 10 and the rate means nothing and we say so instead of publishing
    it. The meta-grader stage had no such control and could conclude nothing.

    🔴 THE FIRST VERSION OF THIS ARM WAS VOID, and the failure is worth keeping in view.
    It deleted a single (risk, intervention) CELL. But the pair list is a reachability
    CROSS-PRODUCT -- 2 risks and 6 interventions give 12 pairs -- so removing one cell
    almost never removes an endpoint. Measured over the 20: 8 deletions left BOTH endpoints
    visible elsewhere, 11 left the risk visible, and exactly 1 removed an endpoint. It
    returned 0/20 and that number said nothing whatever about the judge, because there was
    nothing in the input to find. The control caught a bug in the control. Deleting an
    endpoint is the only deletion this list shape can express.

POPULATION, chosen so the human arm can check the machine
    The 100 audited documents, so this is directly comparable to the 0.6% and 18.1% it is
    meant to replace, PLUS the 30 documents of the #176 packet, of which 10 are being
    enumerated by hand under the same definition. Those 10 are the validation set: machine
    at n=128, human at n=10 checking it. They overlap the audited 100 by only 2, which is
    why both go in.

Class A: metered Anthropic batch API. --dry-run measures the real token bill from the real
sample and prints it BEFORE anything is submitted. Do not submit without reading it.

Usage
-----
    cd graph_analysis
    python -u experiment_review_chain_recall.py --dry-run
    python -u experiment_review_chain_recall.py --submit
    python -u experiment_review_chain_recall.py --collect <batch_id>
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import random
import re
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SLIM = HERE / "phase2_results" / "node_attrs_slim.pkl"
EDGES = (
    HERE
    / "phase2_results"
    / "step1_load_and_parse_umapwithoutlocalsatellites"
    / "graph_edge_data.pkl"
)
PACKET_MANIFEST = HERE / "phase2_results" / "human_review_packet" / "manifest.json"
ARD_DIR = ROOT / "intervention_graph_creation" / "data" / "raw" / "ard_json_full"
RAW = HERE / "phase2_results" / "chain_recall_raw"
RAW_ABL = HERE / "phase2_results" / "chain_recall_ablation_raw"
OUT = HERE / "phase2_results" / "experiment_review_chain_recall_report.json"

KEY_ENV = Path.home() / "0_project_work" / "ExistentialRiskBenchmark" / ".env"
KEY_VAR = "ANTHROPIC_API_KEY"
MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 4000
SEED = 42
N_ABLATE = 20

# Batch pricing, Sonnet 4.5, USD per million. Batch is half of standard. ASSUMED -- check
# the rate card before quoting a cost anywhere, and the dry run prints this so the number
# is never silently stale.
RATE_IN, RATE_OUT = 1.50, 7.50

PROMPT = """You are auditing a knowledge-graph extraction for RECALL: what arguments in \
this document did the extraction fail to capture?

You will be given a source document and the list of risk-to-intervention pairs an automated \
extraction produced from it.

TASK
1. Read the document and enumerate every distinct risk-to-intervention argument it makes. \
An argument is: the document identifies some risk or problem, and proposes or endorses some \
intervention, technique or approach as reducing it. Enumerate what the DOCUMENT argues. Do \
not start from the list.
2. Only then, for each argument you enumerated, decide whether the EXTRACTED PAIRS already \
carry it.

MATCHING IS ON MEANING, NOT WORDING. If a listed pair expresses the same kind of risk linked \
to the same kind of intervention, it is carried, however differently it is phrased. \
"reward hacking" and "specification gaming" are the same risk. "RLHF" and "learning from \
human preferences" are the same intervention. Do not mark something uncaptured because the \
extraction named it differently, and do not mark it uncaptured because the extraction's \
version is less precise.

For each enumerated argument return:
  risk                  - the risk, in your own words, one short phrase
  intervention          - the intervention, one short phrase
  risk_quote            - a VERBATIM span from the document stating the risk
  intervention_quote    - a VERBATIM span in which the intervention is proposed
  supporting_steps      - how many distinct reasoning steps the document gives BETWEEN the \
risk and the intervention (mechanism, rationale, evidence). An integer.
  status                - one of:
      carried             a listed pair makes this same argument
      uncaptured_thin     no listed pair makes it, AND supporting_steps < 2
      uncaptured_material no listed pair makes it, it introduces a risk or an intervention \
that no listed pair has, AND supporting_steps >= 2
  matched_pair_index    - if carried, the 1-based index of the pair it matches; else null
  why                   - one sentence, and for any uncaptured status say what is new about it

Be conservative about `uncaptured_material`. If you are unsure whether a listed pair already \
covers it, mark it `carried`. A false "material" is worse than a missed one here, because \
this number is meant to bound how much the extraction misses and an inflated one is useless.

SOURCE DOCUMENT
---
{text}
---

RISK-TO-INTERVENTION PAIRS THE EXTRACTION HOLDS ({n_pairs})
{pairs}

Return ONLY a JSON object:
{{"arguments": [{{"risk": "...", "intervention": "...", "risk_quote": "...", \
"intervention_quote": "...", "supporting_steps": 0, "status": "...", \
"matched_pair_index": null, "why": "..."}}]}}
"""


def die(msg: str) -> None:
    raise SystemExit(f"FATAL: {msg}")


def read_key() -> str:
    if not KEY_ENV.is_file():
        die(
            f"no API key file at {KEY_ENV}\n"
            f"  expected {KEY_VAR}=... in that file. It is NOT in this repo and must not be\n"
            f"  copied into it. This script does not fall back to an environment variable."
        )
    for line in KEY_ENV.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(f"{KEY_VAR}="):
            return line.split("=", 1)[1].strip().strip("\"'")
    die(f"{KEY_VAR} not found in {KEY_ENV}")
    return ""


def load_sources() -> dict:
    files = sorted(glob.glob(str(ARD_DIR / "*.jsonl")))
    if not files:
        die(f"ARD source text not found: {ARD_DIR}/*.jsonl")
    by_url = {}
    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                u = (r.get("url") or "").strip()
                if u and (r.get("text") or "").strip():
                    by_url.setdefault(u, r)
    return by_url


def reachable_pairs(nids, slim, adj_all):
    """The (risk, intervention) pairs one document's extraction supports.

    Gate-free and undirected, matching the released enumerator's traversal and the list the
    #176 packet shows its annotator, so the machine and the human are answering against the
    SAME definition of "what the extraction holds". If these two ever diverge the human
    validation stops validating anything.
    """
    inside = set(nids)
    risks, ivs = [], []
    for n in nids:
        a = slim.get(n, {})
        if a.get("type") == "intervention":
            ivs.append(n)
        elif (a.get("concept_category") or "").lower() == "risk":
            risks.append(n)
    out = []
    for r in risks:
        seen, q = {r}, deque([r])
        while q:
            x = q.popleft()
            for m in adj_all.get(x, ()):
                if m in inside and m not in seen:
                    seen.add(m)
                    q.append(m)
        for i in ivs:
            if i in seen:
                out.append((slim[r].get("name"), slim[i].get("name")))
    return sorted(set(out))


DEDUPED = HERE / "phase1_rawpathsfiles" / "paths_hopwise_v4_edge_only_deduped.jsonl"
RAW_GATED = HERE / "phase2_results" / "chain_recall_gated_raw"
N_GATED = 100


def build_sample_gated() -> list[dict]:
    """Pair lists carrying EVERY cut, taken from the released reporting unit itself.

    The gate-free run answers "did the pipeline capture this argument". This one answers
    the question the paper's claims actually rest on: does the 2,772-chain REPORTING UNIT
    carry it? So the pair list is read straight off
    `paths_hopwise_v4_edge_only_deduped.jsonl`, which has every cut baked in by
    construction -- structural edges only, edge confidence >= 3 on every hop, intervention
    maturity >= 3, exactly one risk rooting the path, simple paths, first hop on an
    intermediate subtype, stop at the first qualifying intervention, the four-node floor,
    the thirty-hop ceiling, and the 70% sub-path collapse. Rebuilding those nine cuts by
    hand would be nine chances to get one wrong; reading the shipped file is zero.

    🔴 THE POPULATION HAS TO CHANGE WITH THE LIST, and this is not optional. Only 1,868 of
    11,779 documents yield a gated chain, and just 12 of the 99 audited documents do. Run
    this on the audited population and 87 of 99 pair lists are EMPTY, so every argument
    reads as uncaptured and the result is trivially 100% -- a measurement of the sampling
    frame, not of the artifact. Sampling from the chain-yielding population instead is what
    S2 had to do for the same reason.

    Host-stratified proportional to the chain set, matching how #175 drew its 200, so the
    two are comparable and the reason-code weights remain usable.
    """
    if not DEDUPED.is_file():
        die(f"missing {DEDUPED}")
    slim = pickle.load(SLIM.open("rb"))
    pairs: dict[str, set] = defaultdict(set)
    for line in DEDUPED.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        p = json.loads(line)["path"]
        u = slim.get(p[0], {}).get("url")
        if u:
            pairs[u].add((slim[p[0]].get("name"), slim[p[-1]].get("name")))
    if len(pairs) != 1868:
        die(
            f"the deduped file yields {len(pairs)} chain-yielding documents, not the 1,868 "
            "the paper reports. The reporting unit on disk is not the one described; fix "
            "that before measuring anything against it."
        )

    sources = load_sources()
    rng = random.Random(SEED)
    hv = {
        m["source_url"]
        for m in json.loads(PACKET_MANIFEST.read_text(encoding="utf-8"))
        if m.get("recall_arm")
    }

    def host(u):
        m = re.match(r"https?://([^/]+)", u or "")
        return (m.group(1) if m else "unknown").lower().replace("www.", "")

    by_host = defaultdict(list)
    for u in sorted(pairs):
        if u in sources:
            by_host[host(u)].append(u)
    total = sum(len(v) for v in by_host.values())
    picked = []
    for h, us in sorted(by_host.items()):
        k = round(N_GATED * len(us) / total)
        rng.shuffle(us)
        picked += us[:k]
    picked = sorted(set(picked) | (hv & set(pairs) & set(sources)))

    items = []
    for u in picked:
        prs = sorted(pairs[u])
        items.append(
            {
                "url": u,
                "cohorts": ["gated_reporting_unit"]
                + (["human_validation"] if u in hv else []),
                "pairs": prs,
                "text_chars": len(sources[u]["text"]),
                "_text": sources[u]["text"],
            }
        )
    eligible = [
        i for i, it in enumerate(items) if len({b for _, b in it["pairs"]}) >= 2
    ]
    for i in rng.sample(eligible, min(N_ABLATE, len(eligible))):
        it = items[i]
        victim = rng.choice(sorted({b for _, b in it["pairs"]}))
        it["ablated_intervention"] = victim
        it["pairs_shown"] = [(a, b) for a, b in it["pairs"] if b != victim]
    for it in items:
        it.setdefault("ablated_intervention", None)
        it.setdefault("pairs_shown", it["pairs"])
    return items


def build_sample(judge_dir: Path | None) -> list[dict]:
    for p in (SLIM, EDGES, PACKET_MANIFEST):
        if not p.exists():
            die(f"missing input: {p}")
    slim = pickle.load(SLIM.open("rb"))
    nodes_by_url = defaultdict(list)
    for nid, rec in slim.items():
        if rec.get("url"):
            nodes_by_url[rec["url"]].append(nid)
    adj_all = defaultdict(set)
    for e in pickle.load(EDGES.open("rb")):
        if e.get("type") != "EDGE":
            continue
        adj_all[e["source"]].add(e["target"])
        adj_all[e["target"]].add(e["source"])

    urls: dict[str, set] = defaultdict(set)
    if judge_dir and judge_dir.is_dir():
        for p in sorted(judge_dir.glob("*.json")):
            if p.name in ("summary.json", "errors.json"):
                continue
            u = json.loads(p.read_text(encoding="utf-8")).get("url")
            if u:
                urls[u].add("audited_100")
    for m in json.loads(PACKET_MANIFEST.read_text(encoding="utf-8")):
        urls[m["source_url"]].add("packet_30")
        if m.get("recall_arm"):
            urls[m["source_url"]].add("human_validation_10")

    sources = load_sources()
    rng = random.Random(SEED)
    items = []
    for u in sorted(urls):
        src = sources.get(u)
        nids = nodes_by_url.get(u)
        if src is None or not nids:
            continue
        prs = reachable_pairs(nids, slim, adj_all)
        if not prs:
            continue
        items.append(
            {
                "url": u,
                "cohorts": sorted(urls[u]),
                "pairs": prs,
                "text_chars": len(src["text"]),
                "_text": src["text"],
            }
        )

    # Ablation arm: delete an entire INTERVENTION, meaning every pair that ends at it.
    #
    # 🔴 The first version of this deleted a single (risk, intervention) CELL and it was
    # very nearly a no-op. The pair list is a reachability CROSS-PRODUCT -- a document with
    # 2 risks and 6 interventions yields 12 pairs -- so removing one cell usually leaves
    # both of its endpoints on display in other cells. Measured on the 2026-08-26 run: of
    # 20 deletions, 8 left BOTH endpoints visible, 11 left the risk visible, and exactly 1
    # removed an endpoint outright. Sensitivity came back 0/20, which said nothing about
    # the judge because there was nothing there to find. Deleting an endpoint is the only
    # deletion this list shape can actually express.
    eligible = [
        i for i, it in enumerate(items) if len({b for _, b in it["pairs"]}) >= 2
    ]
    for i in rng.sample(eligible, min(N_ABLATE, len(eligible))):
        it = items[i]
        victim = rng.choice(sorted({b for _, b in it["pairs"]}))
        it["ablated_intervention"] = victim
        it["pairs_shown"] = [(a, b) for a, b in it["pairs"] if b != victim]
        it["ablated_pair"] = [
            sorted({a for a, b in it["pairs"] if b == victim})[0],
            victim,
        ]
    for it in items:
        it.setdefault("ablated_pair", None)
        it.setdefault("ablated_intervention", None)
        it.setdefault("pairs_shown", it["pairs"])
    return items


def render(it: dict) -> str:
    pairs = "\n".join(
        f"{k + 1}. RISK: {a}  ->  INTERVENTION: {b}"
        for k, (a, b) in enumerate(it["pairs_shown"])
    )
    return PROMPT.format(text=it["_text"], pairs=pairs, n_pairs=len(it["pairs_shown"]))


def out_dir(items: list[dict]) -> Path:  # noqa: D401
    if items and "gated_reporting_unit" in items[0]["cohorts"]:
        return RAW_GATED
    return _out_dir_ungated(items)


def _out_dir_ungated(items: list[dict]) -> Path:
    """Ablation re-runs write to their OWN directory.

    Learned 2026-08-26, immediately: an --ablation-only dry run clobbered the completed
    run's sample.json, which is the only thing that joins its custom_ids back to documents.
    The results were already collected so nothing was lost, but a re-collect would have been
    impossible. Never let a repair overwrite the artifact it is repairing.
    """
    return RAW_ABL if all(it.get("ablated_intervention") for it in items) else RAW


def dry_run(items: list[dict]) -> int:
    try:
        import anthropic
    except ImportError:
        die("pip install anthropic")
    client = anthropic.Anthropic(api_key=read_key())
    rng = random.Random(SEED)
    probe = rng.sample(items, min(8, len(items)))
    measured = []
    for it in probe:
        n = client.messages.count_tokens(
            model=MODEL, messages=[{"role": "user", "content": render(it)}]
        ).input_tokens
        measured.append((n, it["text_chars"]))
    tok_per_char = sum(n for n, _ in measured) / sum(c for _, c in measured)
    total_in = sum(it["text_chars"] * tok_per_char for it in items)
    total_out = len(items) * MAX_TOKENS * 0.45  # observed batch fill on the #175 run
    cost = total_in / 1e6 * RATE_IN + total_out / 1e6 * RATE_OUT

    coh = Counter(c for it in items for c in it["cohorts"])
    print(
        f"DRY RUN -- nothing submitted. model {MODEL}, batch pricing ASSUMED "
        f"{RATE_IN}/{RATE_OUT} USD per M in/out"
    )
    print(f"  documents            : {len(items)}")
    for k, v in sorted(coh.items()):
        print(f"    {k:<22}: {v}")
    print(
        f"  ablation arm         : {sum(1 for it in items if it.get('ablated_intervention'))} documents "
        f"with one pair deleted from the list shown"
    )
    print(
        f"  pairs per document   : median "
        f"{sorted(len(it['pairs']) for it in items)[len(items) // 2]}, "
        f"max {max(len(it['pairs']) for it in items)}"
    )
    print(
        f"  measured tokens/char : {tok_per_char:.4f} over {len(probe)} probe documents"
    )
    print(f"  projected input      : {total_in / 1e6:.2f}M tokens")
    print(
        f"  projected output     : {total_out / 1e6:.2f}M tokens (at 45% of the "
        f"{MAX_TOKENS} cap)"
    )
    print(f"  PROJECTED COST       : USD {cost:.2f}")
    print(
        "\n  wall clock: one batch, the #175 run of 210 requests returned inside an hour."
    )
    d = out_dir(items)
    d.mkdir(parents=True, exist_ok=True)
    (d / "sample.json").write_text(
        json.dumps(
            [{k: v for k, v in it.items() if k != "_text"} for it in items], indent=1
        ),
        encoding="utf-8",
    )
    print(f"\n  wrote {d / 'sample.json'} (sample without source text)")
    return 0


def submit(items: list[dict]) -> int:
    import anthropic

    client = anthropic.Anthropic(api_key=read_key())
    reqs = [
        {
            "custom_id": f"rec-{i:04d}",
            "params": {
                "model": MODEL,
                "max_tokens": MAX_TOKENS,
                "messages": [{"role": "user", "content": render(it)}],
            },
        }
        for i, it in enumerate(items)
    ]
    batch = client.messages.batches.create(requests=reqs)
    d = out_dir(items)
    d.mkdir(parents=True, exist_ok=True)
    (d / "batch_id.txt").write_text(batch.id, encoding="utf-8")
    (d / "sample.json").write_text(
        json.dumps(
            [
                {
                    "custom_id": f"rec-{i:04d}",
                    **{k: v for k, v in it.items() if k != "_text"},
                }
                for i, it in enumerate(items)
            ],
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"submitted {len(reqs)} requests, batch {batch.id}  ->  {d}")
    print(f"  collect with: python -u {Path(__file__).name} --collect {batch.id}")
    return 0


STATUSES = ("carried", "uncaptured_thin", "uncaptured_material")


def _norm(s):
    import re

    s = re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())
    return {t for t in s.split() if len(t) > 2}


def _same(a, b, thr=0.4):
    """Loose token-overlap match, used ONLY to decide whether the judge resurfaced the pair
    we deleted. Loose on purpose: a STRICT test would under-count detections and flatter the
    instrument by making its measured sensitivity look worse than it is, which would in turn
    inflate the corrected miss rate. Erring loose keeps the correction conservative."""
    ta, tb = _norm(a), _norm(b)
    return bool(ta and tb) and len(ta & tb) / len(ta | tb) >= thr


def analyse(results: list[dict], sample: list[dict]) -> dict:
    by_id = {s["custom_id"]: s for s in sample}
    per_cohort = defaultdict(lambda: Counter())
    docs_with_material = defaultdict(set)
    abl_total = abl_detected = 0
    abl_detail = []
    parsed = errors = 0

    for r in results:
        cid = r.get("custom_id")
        s = by_id.get(cid)
        if s is None or "arguments" not in (r.get("verdict") or {}):
            errors += 1
            continue
        parsed += 1
        args = r["verdict"]["arguments"]
        for c in s["cohorts"]:
            per_cohort[c]["documents"] += 1
            per_cohort[c]["arguments"] += len(args)
            for a in args:
                st = (a.get("status") or "").strip().lower()
                per_cohort[c][st if st in STATUSES else "unparsed_status"] += 1
                if st == "uncaptured_material":
                    docs_with_material[c].add(cid)

        want_i = s.get("ablated_intervention")
        if want_i:
            # What was deleted is an INTERVENTION -- every pair ending at it. Detection is
            # therefore on the intervention alone: any risk that reached it will do, since
            # the whole endpoint left the list. Matching on the (risk, intervention) cell
            # instead is what made the first run's sensitivity meaningless.
            abl_total += 1
            hit = any(
                _same(a.get("intervention"), want_i)
                and (a.get("status") or "").startswith("uncaptured")
                for a in args
            )
            abl_detected += hit
            abl_detail.append(
                {"custom_id": cid, "deleted_intervention": want_i, "detected": hit}
            )

    def block(c):
        k = per_cohort[c]
        n = k["arguments"]
        return {
            "documents": k["documents"],
            "arguments_enumerated": n,
            "carried": k["carried"],
            "uncaptured_thin": k["uncaptured_thin"],
            "uncaptured_material": k["uncaptured_material"],
            "material_miss_rate_pct": round(100 * k["uncaptured_material"] / n, 1)
            if n
            else None,
            "documents_with_a_material_miss": len(docs_with_material[c]),
            "unparsed_status": k["unparsed_status"],
        }

    sens = round(abl_detected / abl_total, 3) if abl_total else None
    overall = block("audited_100")
    corrected = None
    if sens and overall["material_miss_rate_pct"] is not None and sens > 0:
        corrected = round(overall["material_miss_rate_pct"] / sens, 1)

    return {
        "study": "chain-level recall: how many ARGUMENTS does the extraction miss",
        "why_this_replaces_the_node_counts": (
            "0.6%, 28.8%, 26.4% count nodes and 18.1%, 21.7% count relationships. All five "
            "come from a schema-conformance instrument answering a recall question, which "
            "is why they span two orders of magnitude. This counts arguments, the unit the "
            "paper reports on."
        ),
        "model": MODEL,
        "n_parsed": parsed,
        "n_errors": errors,
        "by_cohort": {c: block(c) for c in sorted(per_cohort)},
        "ablation_arm": {
            "question": (
                "One known pair was deleted from the list shown, for these documents. How "
                "often does the judge surface the pair we removed?"
            ),
            "n": abl_total,
            "detected": abl_detected,
            "sensitivity": sens,
            "detail": abl_detail,
            "HOW_TO_READ": (
                "This is the measured sensitivity of the instrument and it governs every "
                "other number here. High sensitivity means an observed miss rate is close "
                "to the truth. Low sensitivity means the judge cannot see missing arguments "
                "even when we put one there on purpose, and then NO miss rate from this run "
                "may be published -- report the sensitivity and stop. Do not quote a "
                "corrected rate without quoting the sensitivity beside it."
            ),
        },
        "headline_audited_100": overall,
        "sensitivity_corrected_material_miss_rate_pct": corrected,
        "human_validation": (
            "The human_validation_10 cohort is enumerated by hand under the same definition "
            "in #176 (recall_enumeration.csv). Compare the two before believing either. The "
            "machine runs at n=127 and the human at n=10; the human is the check, not the "
            "sample."
        ),
        "LIMITS": (
            "One model, cross-provider from the extractor but sharing its priors, so it "
            "should UNDER-detect missing arguments in the same way the extractor missed "
            "them -- the ablation arm bounds that but does not remove it. Pair lists are "
            "gate-free reachability over each document's extracted subgraph, matching what "
            "the #176 packet shows its human annotator. Nothing here is human-adjudicated "
            "except the 10-document validation cohort."
        ),
    }


def collect(batch_id: str) -> int:
    import anthropic

    # Resolve which run this batch belongs to by matching the id recorded at submit time,
    # rather than assuming RAW. The ablation repair writes to its own directory, and
    # collecting it against the main run's sample would join every result to the wrong
    # document while looking perfectly healthy.
    d = next(
        (
            c
            for c in (RAW, RAW_ABL, RAW_GATED)
            if (c / "batch_id.txt").is_file()
            and (c / "batch_id.txt").read_text(encoding="utf-8").strip() == batch_id
        ),
        None,
    )
    if d is None:
        die(
            f"no directory records batch {batch_id}.\n"
            f"  looked for a matching batch_id.txt in {RAW} and {RAW_ABL}.\n"
            "  Collecting against another run's sample would silently join every result to\n"
            "  the wrong document, so this refuses rather than guessing."
        )
    sample_fp = d / "sample.json"
    if not sample_fp.is_file():
        die(f"missing {sample_fp}; written by --submit and needed to join results.")
    sample = json.loads(sample_fp.read_text(encoding="utf-8"))
    print(f"collecting {batch_id} against {sample_fp}")

    client = anthropic.Anthropic(api_key=read_key())
    rows = []
    for res in client.messages.batches.results(batch_id):
        rec = {"custom_id": res.custom_id}
        if res.result.type != "succeeded":
            rec["error"] = res.result.type
        else:
            txt = res.result.message.content[0].text
            rec["raw"] = txt
            try:
                start, stop = txt.index("{"), txt.rindex("}") + 1
                rec["verdict"] = json.loads(txt[start:stop])
            except (ValueError, json.JSONDecodeError) as e:
                rec["error"] = f"unparseable: {e}"
        rows.append(rec)

    d.mkdir(parents=True, exist_ok=True)
    with (d / "results.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    report = analyse(rows, sample)
    suffix = {RAW: "", RAW_ABL: "_ablation", RAW_GATED: "_gated"}[d]
    out = OUT if not suffix else OUT.with_name(OUT.stem + suffix + ".json")
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"parsed {report['n_parsed']}, errors {report['n_errors']}")
    print(
        f"  ablation sensitivity: {report['ablation_arm']['detected']}"
        f"/{report['ablation_arm']['n']} = {report['ablation_arm']['sensitivity']}"
    )
    print(f"  audited_100: {report['headline_audited_100']}")
    print(
        f"  corrected material miss rate: "
        f"{report['sensitivity_corrected_material_miss_rate_pct']}%"
    )
    # Print the path actually written. This said {OUT} while writing to {out}, which made
    # an ablation collect look as though it had overwritten the completed run's report.
    print(f"\nwrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judge-reports", help="dir of the audited-100 judge reports")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--submit", action="store_true")
    g.add_argument("--collect", metavar="BATCH_ID")
    ap.add_argument("--gated", action="store_true")
    ap.add_argument(
        "--ablation-only",
        action="store_true",
        help=(
            "submit ONLY the ablation documents. Used to repair the control after the "
            "2026-08-26 run, whose cell-deletion ablation was a no-op; the real arm from "
            "that run is unaffected and is not re-paid for."
        ),
    )
    a = ap.parse_args()

    if a.collect:
        return collect(a.collect)
    items = (
        build_sample_gated()
        if a.gated
        else build_sample(Path(a.judge_reports) if a.judge_reports else None)
    )
    if a.ablation_only:
        items = [it for it in items if it.get("ablated_intervention")]
        print(f"ABLATION ONLY: {len(items)} documents")
    if not items:
        die("sample is empty; check --judge-reports and the packet manifest.")
    return dry_run(items) if a.dry_run else submit(items)


if __name__ == "__main__":
    sys.exit(main())
