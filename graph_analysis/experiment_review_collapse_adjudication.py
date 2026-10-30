#!/usr/bin/env python3
"""Is a path the 70% collapse dropped a DIFFERENT argument, or the same one restated?

The sub-path collapse defines the reporting unit: 8,954 enumerated chains become the 2,772
the paper reports on, and 6,182 traversals are dropped. Everything we know about that step
is structural. 78.3% of the drops touch a node their container lacks, 28.0% end at an
intervention the container never reaches, 18.0% of distinct risk-to-intervention pairs
vanish. `experiment_review_containment_semantics.py` says outright what none of that
settles, and the manuscript repeats it:

    "Whether the 78.3% are distinct arguments or restatements at a different grain is a
     question about content, and the structural evidence above does not settle it. An
     annotation pass over a sample would; sec:limitations records it as not performed."

Reviewers asked for exactly this twice (R22, R23; GC and GW, both marked NEW in the
register and both unaddressed): "no annotation study shows retained paths are distinct
arguments", and "validate the gates and the collapse rule on a small hand-checked sample".
This is that pass, run by a judge rather than a hand, and it is the last structural claim
in the paper resting on a node-overlap heuristic.

THE QUESTION, asked of one displaced pair at a time
    Given the source document, the chain the collapse KEPT, and the chain it DROPPED
    because the kept one contained 70% of its nodes: does the dropped chain assert an
    argument the kept chain does not?

    `same_argument`       The two describe one argument at different grain. The drop is a
                          repeat and the collapse did its job.
    `different_argument`  The dropped chain asserts something the kept one does not -- a
                          different intervention, a different risk, or a materially
                          different route between the same endpoints. The collapse cost the
                          reporting unit an argument.
    `unclear`             A real answer. Better than a coin flip on a long document.

WHY THE NULL ARM IS NOT OPTIONAL
    A judge that answers `same_argument` reflexively would produce a flattering result and
    no way to catch it. So N_NULL items pair a kept chain with a dropped chain from a
    DIFFERENT document. Those are unambiguously different arguments; a judge that does not
    say so has told us its verdicts are worthless, for the price of a few calls. The same
    control that made #175 readable, and its absence is why the meta-grader stage could
    conclude nothing.

Class A: metered Anthropic batch API. --dry-run measures the real bill first.

    cd graph_analysis
    python -u experiment_review_collapse_adjudication.py --dry-run
    python -u experiment_review_collapse_adjudication.py --submit
    python -u experiment_review_collapse_adjudication.py --collect <batch_id>
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SLIM = HERE / "phase2_results" / "node_attrs_slim.pkl"
RAW_PATHS = HERE / "phase1_rawpathsfiles" / "paths_hopwise_v4_edge_only.jsonl"
ARD_DIR = ROOT / "intervention_graph_creation" / "data" / "raw" / "ard_json_full"
RAW = HERE / "phase2_results" / "collapse_adjudication_raw"
OUT = HERE / "phase2_results" / "experiment_review_collapse_adjudication_report.json"

KEY_ENV = Path.home() / "0_project_work" / "ExistentialRiskBenchmark" / ".env"
KEY_VAR = "ANTHROPIC_API_KEY"
MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 1500
SEED = 42
N_REAL = 120
N_NULL = 15
CONTAINMENT_THRESHOLD = 0.70
EXPECTED_KEPT = 2772
RATE_IN, RATE_OUT = 1.50, 7.50

PROMPT = """Two chains were extracted from the same source document. A de-duplication step \
KEPT the first and DROPPED the second, because the kept chain's node set contains at least \
70% of the dropped chain's nodes.

Your question: does the DROPPED chain assert an argument that the KEPT chain does not?

Judge the ARGUMENT, not the wording. Two chains that name the same risk and the same \
intervention differently are the same argument. Two chains that walk between the same \
endpoints through different intermediate reasoning are the same argument UNLESS the route \
itself is the claim -- a different mechanism by which the intervention is said to work is a \
different argument. A dropped chain that ends at a different intervention, or starts from a \
different risk, is a different argument.

Answer one of:
  same_argument       one argument at two levels of detail; dropping it loses nothing
  different_argument  the dropped chain asserts something the kept chain does not
  unclear             you cannot tell from this document

SOURCE DOCUMENT
---
{text}
---

KEPT CHAIN
{kept}

DROPPED CHAIN
{dropped}

Return ONLY JSON:
{{"verdict": "same_argument|different_argument|unclear", "what_the_dropped_chain_adds": \
"one phrase, or empty if nothing", "evidence_quote": "a verbatim span from the document \
supporting your answer, or empty", "why": "one sentence"}}
"""


def die(msg: str) -> None:
    raise SystemExit(f"FATAL: {msg}")


def read_key() -> str:
    if not KEY_ENV.is_file():
        die(f"no API key file at {KEY_ENV}; expected {KEY_VAR}=...")
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


def render_chain(nodes, slim) -> str:
    out = []
    for i, n in enumerate(nodes):
        a = slim.get(n, {})
        lab = (
            "INTERVENTION"
            if a.get("type") == "intervention"
            else (a.get("concept_category") or "?").upper()
        )
        out.append(f"  {i + 1}. [{lab}] {a.get('name')}")
    return "\n".join(out)


def rerun_collapse(slim):
    """Re-derive which kept chain displaced each drop, and refuse to proceed unless the
    re-implementation reproduces the released 2,772. Copied deliberately from
    experiment_review_containment_semantics.py rather than imported, because that script
    exits on its own checks; the guard below is the same one and must not be relaxed."""
    if not RAW_PATHS.is_file():
        die(f"missing {RAW_PATHS}")
    raw = []
    for i, line in enumerate(RAW_PATHS.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if line:
            r = json.loads(line)
            r["path_id"] = i
            raw.append(r)

    by_url = defaultdict(list)
    for p in raw:
        urls = {slim[n]["url"] for n in p["path"] if n in slim and slim[n].get("url")}
        by_url[next(iter(urls)) if len(urls) == 1 else f"_x_{p['path_id']}"].append(p)

    keep, drops = set(), []
    for url, plist in by_url.items():
        ordered = sorted(plist, key=lambda p: -len(p["path"]))
        sets = [(p["path_id"], frozenset(p["path"])) for p in ordered]
        for i, (pid_i, ns_i) in enumerate(sets):
            if not ns_i:
                continue
            container = None
            for pid_j, ns_j in sets[:i]:
                if pid_j not in keep or not ns_j:
                    continue
                small, large = (ns_i, ns_j) if len(ns_i) <= len(ns_j) else (ns_j, ns_i)
                if len(small & large) / len(small) >= CONTAINMENT_THRESHOLD:
                    container = pid_j
                    break
            if container is None:
                keep.add(pid_i)
            elif not url.startswith("_x_"):
                drops.append((pid_i, container, url))

    if len(keep) != EXPECTED_KEPT:
        die(
            f"re-implementation kept {len(keep)}, released file has {EXPECTED_KEPT}. The "
            "pairing does not reproduce the released substrate; nothing here is reportable."
        )
    return {p["path_id"]: p for p in raw}, drops


def build_sample():
    if not SLIM.is_file():
        die(f"missing {SLIM}")
    slim = pickle.load(SLIM.open("rb"))
    paths, drops = rerun_collapse(slim)
    sources = load_sources()
    rng = random.Random(SEED)

    usable = [d for d in drops if d[2] in sources]
    rng.shuffle(usable)
    items = []
    for pid, cid, url in usable[:N_REAL]:
        items.append(
            {
                "arm": "real",
                "url": url,
                "dropped_path_id": pid,
                "kept_path_id": cid,
                "kept": render_chain(paths[cid]["path"], slim),
                "dropped": render_chain(paths[pid]["path"], slim),
                "_text": sources[url]["text"],
                "text_chars": len(sources[url]["text"]),
            }
        )
    # Null arm: the dropped chain comes from a DIFFERENT document than the kept one and the
    # source shown. These are unambiguously different arguments.
    pool = usable[N_REAL:]
    for k in range(min(N_NULL, len(pool) // 2)):
        a, b = pool[2 * k], pool[2 * k + 1]
        if a[2] == b[2]:
            continue
        items.append(
            {
                "arm": "null_mismatched",
                "url": a[2],
                "dropped_path_id": b[0],
                "kept_path_id": a[1],
                "kept": render_chain(paths[a[1]]["path"], slim),
                "dropped": render_chain(paths[b[0]]["path"], slim),
                "_text": sources[a[2]]["text"],
                "text_chars": len(sources[a[2]]["text"]),
            }
        )
    return items


def render(it) -> str:
    return PROMPT.format(text=it["_text"], kept=it["kept"], dropped=it["dropped"])


def analyse(results, sample):
    by_id = {s["custom_id"]: s for s in sample}
    per_arm = defaultdict(Counter)
    adds, parsed, errors = [], 0, 0
    for r in results:
        s = by_id.get(r.get("custom_id"))
        v = (r.get("verdict") or {}).get("verdict")
        if s is None or not v:
            errors += 1
            continue
        parsed += 1
        per_arm[s["arm"]][v.strip().lower()] += 1
        if s["arm"] == "real" and v.strip().lower() == "different_argument":
            adds.append(
                {
                    "custom_id": r["custom_id"],
                    "adds": (r["verdict"].get("what_the_dropped_chain_adds") or "")[
                        :160
                    ],
                }
            )

    real = per_arm["real"]
    nreal = sum(real.values())
    null = per_arm["null_mismatched"]
    nnull = sum(null.values())
    null_flag = round(100 * null["different_argument"] / nnull, 1) if nnull else None

    return {
        "study": "does the 70% collapse drop distinct arguments (R22/R23)",
        "answers": (
            "R22 (no annotation study shows retained paths are distinct arguments) and R23 "
            "(validate the collapse rule on a small hand-checked sample). Both were marked "
            "NEW in the reviewer register and both were unaddressed."
        ),
        "model": MODEL,
        "n_parsed": parsed,
        "n_errors": errors,
        "real_arm": {
            "n": nreal,
            **{k: real[k] for k in ("same_argument", "different_argument", "unclear")},
            "pct_different_argument": round(100 * real["different_argument"] / nreal, 1)
            if nreal
            else None,
        },
        "null_arm": {
            "n": nnull,
            "pct_flagged_different": null_flag,
            "reading": (
                "The null pairs a kept chain with a dropped chain from ANOTHER document. "
                "These are different arguments by construction. A judge that does not flag "
                "them at a high rate has answered same_argument reflexively and the real "
                "arm must not be read. Check this BEFORE reading anything above."
            ),
        },
        "what_the_dropped_chains_add": adds[:40],
        "HOW_TO_READ": (
            "This is the first content-level evidence about the step that DEFINES the "
            "reporting unit. It sits against the structural bounds already reported: 78.3% "
            "of drops touch a node the container lacks and 28.0% end at an intervention it "
            "never reaches, which together bound the answer from one side without settling "
            "it. A low different_argument share means the collapse is doing what it claims; "
            "a high one means the 2,772 is losing arguments and the 18.0% pair-loss figure "
            "is the number to lead with."
        ),
        "LIMITS": (
            "One model, no human adjudication. Sampled uniformly over the 6,182 drops, so "
            "it estimates the drop population and not the kept one. The pairing is "
            "re-derived and the script refuses to run unless it reproduces the released "
            "2,772 exactly."
        ),
    }


def collect(batch_id: str) -> int:
    import anthropic

    fp = RAW / "sample.json"
    if not fp.is_file():
        die(f"missing {fp}; written by --submit and needed to join results.")
    sample = json.loads(fp.read_text(encoding="utf-8"))
    client = anthropic.Anthropic(api_key=read_key())
    rows = []
    for res in client.messages.batches.results(batch_id):
        rec = {"custom_id": res.custom_id}
        if res.result.type != "succeeded":
            rec["error"] = res.result.type
        else:
            t = res.result.message.content[0].text
            rec["raw"] = t
            try:
                rec["verdict"] = json.loads(t[t.index("{") : t.rindex("}") + 1])
            except (ValueError, json.JSONDecodeError) as e:
                rec["error"] = f"unparseable: {e}"
        rows.append(rec)
    RAW.mkdir(parents=True, exist_ok=True)
    with (RAW / "results.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    rep = analyse(rows, sample)
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"parsed {rep['n_parsed']}, errors {rep['n_errors']}")
    print(
        f"  NULL ARM (read first): {rep['null_arm']['pct_flagged_different']}% flagged "
        f"different, n={rep['null_arm']['n']}"
    )
    print(f"  real arm: {rep['real_arm']}")
    print(f"\nwrote {OUT}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--submit", action="store_true")
    g.add_argument("--collect", metavar="BATCH_ID")
    a = ap.parse_args()
    if a.collect:
        return collect(a.collect)

    items = build_sample()
    if not items:
        die("empty sample")
    import anthropic

    client = anthropic.Anthropic(api_key=read_key())
    rng = random.Random(SEED)
    probe = rng.sample(items, min(6, len(items)))
    meas = [
        (
            client.messages.count_tokens(
                model=MODEL, messages=[{"role": "user", "content": render(it)}]
            ).input_tokens,
            it["text_chars"],
        )
        for it in probe
    ]
    tpc = sum(n for n, _ in meas) / sum(c for _, c in meas)
    tin = sum(it["text_chars"] * tpc for it in items)
    tout = len(items) * MAX_TOKENS * 0.35
    cost = tin / 1e6 * RATE_IN + tout / 1e6 * RATE_OUT
    arms = Counter(it["arm"] for it in items)

    if a.dry_run:
        print(
            f"DRY RUN -- nothing submitted. model {MODEL}, batch rates ASSUMED "
            f"{RATE_IN}/{RATE_OUT} per M"
        )
        print(f"  items          : {len(items)}  {dict(arms)}")
        print(f"  tokens/char    : {tpc:.4f} over {len(probe)} probes")
        print(f"  projected in   : {tin / 1e6:.2f}M")
        print(f"  projected out  : {tout / 1e6:.2f}M")
        print(f"  PROJECTED COST : USD {cost:.2f}")
        return 0

    reqs = [
        {
            "custom_id": f"col-{i:04d}",
            "params": {
                "model": MODEL,
                "max_tokens": MAX_TOKENS,
                "messages": [{"role": "user", "content": render(it)}],
            },
        }
        for i, it in enumerate(items)
    ]
    batch = client.messages.batches.create(requests=reqs)
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "batch_id.txt").write_text(batch.id, encoding="utf-8")
    (RAW / "sample.json").write_text(
        json.dumps(
            [
                {
                    "custom_id": f"col-{i:04d}",
                    **{k: v for k, v in it.items() if k != "_text"},
                }
                for i, it in enumerate(items)
            ],
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"submitted {len(reqs)} requests ({dict(arms)}), batch {batch.id}")
    print(f"  projected cost USD {cost:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
