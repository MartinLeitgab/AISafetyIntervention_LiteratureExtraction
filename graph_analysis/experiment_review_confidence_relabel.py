#!/usr/bin/env python3
"""Is the edge-confidence label reproducible, or does it just not discriminate?

#175 found 33 chains in the reporting unit whose risk framing a judge could not quote from
the source, and every one of them cleared the confidence >= 3 gate. The extraction prompt
requires confidence 1 or 2 wherever inference was applied, so those first hops were labelled
in violation of the prompt's own rule. Two explanations, with different consequences:

  run-to-run noise      the label is unstable, and a second pass would assign 1-2. Then a
                        cheap fix exists: re-label and re-gate.
  the rubric does not   a second annotator assigns 3 to the same link, because the rubric
  discriminate          cannot separate an evidenced link from an inferred one on this kind
                        of text. Then NO threshold on this label will ever help, and raising
                        the gate is theatre.

This arm decides between them. A model from a different provider than the extractor is given
the source, ONE link, and the confidence rubric verbatim from the extraction prompt, and asked
what value the source's support warrants. It never sees the stored value. Same design as the
stage-agreement arm (#161), applied to the other gate attribute.

The sample is balanced on the thing in question: the 33 chains #175 called
risk_framing_invented against 33 it called faithful. If the label is doing any work, the two
groups re-label differently.

NOT run, and a different question: an o3 self-consistency arm (does the extractor itself
re-label its own link lower on a second pass). That measures whether a second pass would fix
it; this measures whether the label means anything to begin with.

Class A: metered Anthropic batch API, roughly USD 2 at 66 requests.

    cd graph_analysis
    python -u experiment_review_confidence_relabel.py --dry-run
    python -u experiment_review_confidence_relabel.py --submit
    python -u experiment_review_confidence_relabel.py --collect
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREC_RAW = HERE / "phase2_results" / "chain_precision_raw"
NODE_ATTRS = HERE / "phase2_results" / "node_attrs_slim.pkl"
EDGES = (
    HERE
    / "phase2_results"
    / "step1_load_and_parse_umapwithoutlocalsatellites"
    / "graph_edge_data.pkl"
)
ARD_DIR = ROOT / "intervention_graph_creation" / "data" / "raw" / "ard_json_full"
RAW_OUT = HERE / "phase2_results" / "confidence_relabel_raw"
RECEIPT = HERE / "phase2_results" / "experiment_review_confidence_relabel_report.json"
BATCH_ID_FILE = RAW_OUT / "batch_id.txt"
SAMPLE_FILE = RAW_OUT / "sample.json"

KEY_ENV = Path.home() / "0_project_work" / "ExistentialRiskBenchmark" / ".env"
KEY_VAR = "ANTHROPIC_API_KEY"
MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 900
N_PER_GROUP = 33
SEED = 42

SYSTEM = (
    "You assign an evidence-strength label to one asserted link, using a fixed rubric and "
    "the source document. You are strict: the label describes what the SOURCE supports, "
    "never how plausible the link sounds to you."
)

# Verbatim from src/prompt/final_primary_prompt.py, the edge_confidence rubric the extractor
# was given. Reproduced unchanged so the two annotators answer the same question.
RUBRIC = """\
- 5: Proofs, rigorous studies, replications
- 4: Controlled experiments
- 3: Systematic qualitative evidence
- 2: Weak, single examples, preliminary results, limited case studies, or light inference
     (must be 2 if light inference applied)
- 1: Speculative, theoretical hypotheses, speculative connections, or moderate inference
     (must be 1 if moderate inference applied)"""

PROMPT = """\
Below is a document, and one link that an automated pipeline extracted from it. Assign the
link an evidence-confidence value using the rubric.

=== BEGIN DOCUMENT ===
TITLE: {title}

{text}
=== END DOCUMENT ===

=== BEGIN THE LINK ===
FROM (risk):        {src_name}
TO ({dst_cat}):     {dst_name}
RELATION:           {rel}
HOW THE PIPELINE DESCRIBED THE LINK:
{description}
=== END THE LINK ===

=== CONFIDENCE RUBRIC (use exactly this) ===
{rubric}
=== END RUBRIC ===

Note the two rules built into the rubric: a link the source supports only by light inference
must be 2, and one supported only by moderate inference must be 1. A link the document does
not assert at all, and which you reached by reasoning about what the document implies, is
inference by that definition.

Answer in this exact JSON format and nothing else:

{{
  "confidence": 1 | 2 | 3 | 4 | 5,
  "quote": "the verbatim span from the document that supports the link at the value you gave, or \\"\\" if the value is 1 or 2 because no such span exists",
  "is_this_link_asserted_by_the_document": true | false,
  "one_line_reason": "why that value and not one higher"
}}

A value of 3 or above REQUIRES a verbatim quote. If you cannot produce one, the value is 1
or 2 by the rubric's own wording.
"""


def die(msg: str) -> None:
    raise SystemExit(f"FATAL: {msg}")


def read_key() -> str:
    if not KEY_ENV.is_file():
        die(f"Anthropic key file not found: {KEY_ENV} (expects {KEY_VAR})")
    for line in KEY_ENV.read_text(encoding="utf-8", errors="replace").splitlines():
        name, _, value = line.strip().partition("=")
        if name.strip() == KEY_VAR and value.strip():
            return value.strip().strip('"').strip("'")
    die(f"{KEY_VAR} not present in {KEY_ENV}")
    return ""


def load_sources() -> dict[str, dict]:
    files = sorted(glob.glob(str(ARD_DIR / "*.jsonl")))
    if not files:
        die(f"ARD source text not found: {ARD_DIR}/*.jsonl")
    by_url: dict[str, dict] = {}
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


def build_sample() -> list[dict]:
    for p in (NODE_ATTRS, EDGES, PREC_RAW / "results.jsonl"):
        if not p.is_file():
            die(f"missing input: {p}")

    attrs = pickle.load(NODE_ATTRS.open("rb"))
    sources = load_sources()

    # Best edge on each unordered pair, with its relation type and description -- the
    # traversal is undirected, so the pair is the key the enumerator itself used.
    best: dict[frozenset, dict] = {}
    for e in pickle.load(EDGES.open("rb")):
        if e.get("type") != "EDGE":
            continue
        c = e.get("confidence")
        if c is None:
            continue
        k = frozenset((e["source"], e["target"]))
        if c > best.get(k, {}).get("confidence", -1):
            best[k] = {
                "confidence": c,
                "relation": e.get("subtype") or "unknown",
                "description": (e.get("description") or "").strip(),
            }

    rows = [
        json.loads(x)
        for x in (PREC_RAW / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        if r.get("arm") != "real" or "verdict" not in r:
            continue
        code = r["verdict"]["reason_code"]
        if code in ("risk_framing_invented", "faithful"):
            groups[code].append(r)

    rng = random.Random(SEED)
    items = []
    for code in ("risk_framing_invented", "faithful"):
        cands = sorted(groups[code], key=lambda r: r["custom_id"])
        if len(cands) < N_PER_GROUP:
            die(f"group {code} has {len(cands)}, need {N_PER_GROUP}")
        rng.shuffle(cands)
        for k, r in enumerate(cands[:N_PER_GROUP]):
            p = r["nodes"]
            edge = best.get(frozenset((p[0], p[1])))
            if edge is None:
                die(f"no structural edge for the first hop of {r['custom_id']}")
            src = sources[r["source_url"]]
            items.append(
                {
                    "custom_id": f"{'inv' if code == 'risk_framing_invented' else 'fai'}-{k:03d}",
                    "group": code,
                    "precision_custom_id": r["custom_id"],
                    "source_url": r["source_url"],
                    "stored_confidence": edge["confidence"],
                    "relation": edge["relation"],
                    "risk_name": attrs.get(p[0], {}).get("name"),
                    "dst_name": attrs.get(p[1], {}).get("name"),
                    "dst_cat": (attrs.get(p[1], {}).get("concept_category") or "?"),
                    "prompt": PROMPT.format(
                        title=(src.get("title") or "(no title)").strip(),
                        text=src["text"],
                        src_name=attrs.get(p[0], {}).get("name"),
                        dst_cat=(attrs.get(p[1], {}).get("concept_category") or "?"),
                        dst_name=attrs.get(p[1], {}).get("name"),
                        rel=edge["relation"],
                        description=edge["description"] or "(no description stored)",
                        rubric=RUBRIC,
                    ),
                }
            )
    rng.shuffle(items)
    return items


def approx_tokens(s: str) -> int:
    return int(len(s) / 3.6)


def summarise(rows: list[dict]) -> dict:
    out = {}
    for group in ("risk_framing_invented", "faithful"):
        sub = [r for r in rows if r.get("group") == group and "verdict" in r]
        if not sub:
            out[group] = {"n": 0}
            continue
        vals = [r["verdict"].get("confidence") for r in sub]
        vals = [v for v in vals if isinstance(v, int)]
        stored = [r["stored_confidence"] for r in sub]
        asserted = sum(
            1
            for r in sub
            if r["verdict"].get("is_this_link_asserted_by_the_document") is True
        )
        below = sum(
            1
            for r in sub
            if isinstance(r["verdict"].get("confidence"), int)
            and r["verdict"]["confidence"] < 3
        )
        out[group] = {
            "n": len(sub),
            "stored_mean": round(sum(stored) / len(stored), 2),
            "relabelled_mean": round(sum(vals) / len(vals), 2) if vals else None,
            "relabelled_hist": {str(k): v for k, v in sorted(Counter(vals).items())},
            "relabelled_below_the_gate": below,
            "relabelled_below_the_gate_pct": round(100.0 * below / len(sub), 1),
            "link_judged_asserted_by_the_document": asserted,
            "link_judged_asserted_pct": round(100.0 * asserted / len(sub), 1),
        }
    a, b = out.get("risk_framing_invented", {}), out.get("faithful", {})
    if a.get("n") and b.get("n"):
        out["discrimination"] = {
            "question": (
                "Does a second annotator, blind to the stored value, separate the links "
                "#175 called invented from the ones it called faithful?"
            ),
            "relabelled_mean_invented": a["relabelled_mean"],
            "relabelled_mean_faithful": b["relabelled_mean"],
            "gap": round(b["relabelled_mean"] - a["relabelled_mean"], 2),
            "below_gate_invented_pct": a["relabelled_below_the_gate_pct"],
            "below_gate_faithful_pct": b["relabelled_below_the_gate_pct"],
            "reading": (
                "A large gap, with the invented group falling below 3, means the label is "
                "unstable rather than meaningless: a second pass would catch these and "
                "re-gating is a real fix. A small gap means the rubric cannot separate an "
                "evidenced link from an inferred one on this text, and NO threshold on this "
                "attribute will help -- raising the gate would be theatre."
            ),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--submit", action="store_true")
    g.add_argument("--collect", nargs="?", const="", metavar="BATCH_ID")
    args = ap.parse_args()

    RAW_OUT.mkdir(parents=True, exist_ok=True)
    import anthropic

    client = anthropic.Anthropic(api_key=read_key())

    if args.collect is not None:
        bid = args.collect or BATCH_ID_FILE.read_text().strip()
        b = client.messages.batches.retrieve(bid)
        print(f"{bid}: {b.processing_status} counts={b.request_counts}", flush=True)
        if b.processing_status != "ended":
            print("not finished; nothing written.")
            return 1
        manifest = {i["custom_id"]: i for i in json.loads(SAMPLE_FILE.read_text())}
        rows = []
        for res in client.messages.batches.results(bid):
            rec = dict(manifest.get(res.custom_id, {}))
            rec["custom_id"] = res.custom_id
            if res.result.type != "succeeded":
                rec["error"] = res.result.type
            else:
                body = res.result.message.content[0].text
                rec["raw"] = body
                m = re.search(r"\{.*\}", body, re.S)
                if m:
                    try:
                        rec["verdict"] = json.loads(m.group(0))
                    except json.JSONDecodeError:
                        rec["parse_error"] = True
                else:
                    rec["parse_error"] = True
            rows.append(rec)
        (RAW_OUT / "results.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
        )
        summary = summarise(rows)
        RECEIPT.write_text(
            json.dumps(
                {
                    "study": "blind re-labelling of the first-hop edge confidence",
                    "model": MODEL,
                    "transport": "Anthropic batch API",
                    "batch_id": bid,
                    "sample": {
                        "unit": "the risk -> first-body-node hop of a reporting-unit chain",
                        "groups": "33 #175-invented against 33 #175-faithful",
                        "blind": "the annotator never sees the stored confidence",
                        "seed": SEED,
                    },
                    "groups": summary,
                    "errors": sum(
                        1 for r in rows if "error" in r or r.get("parse_error")
                    ),
                    "LIMITS": (
                        "One model, cross-provider from the extractor but sharing its "
                        "priors, applying a rubric written for a different task shape. It "
                        "measures whether the label is reproducible, never whether it is "
                        "correct. No human adjudicated any of it. An o3 self-consistency "
                        "arm -- does the extractor re-label its own link -- is a different "
                        "question and was not run."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        for grp in ("risk_framing_invented", "faithful"):
            s = summary.get(grp, {})
            if s.get("n"):
                print(
                    f"  {grp:24s} n={s['n']:2d} stored={s['stored_mean']} "
                    f"relabelled={s['relabelled_mean']} hist={s['relabelled_hist']} "
                    f"below-gate={s['relabelled_below_the_gate_pct']}% "
                    f"asserted={s['link_judged_asserted_pct']}%"
                )
        if "discrimination" in summary:
            d = summary["discrimination"]
            print(
                f"\n  DISCRIMINATION: faithful {d['relabelled_mean_faithful']} vs invented "
                f"{d['relabelled_mean_invented']} = {d['gap']} | below gate "
                f"{d['below_gate_invented_pct']}% vs {d['below_gate_faithful_pct']}%"
            )
        print(f"\nwrote {RECEIPT}")
        return 0

    print("building sample ...", flush=True)
    items = build_sample()
    inp = sum(approx_tokens(i["prompt"]) for i in items)
    print(f"  {len(items)} requests | projected input {inp:,} tokens")
    print(
        f"  projected cost at assumed Sonnet batch rates: USD {inp / 1e6 * 1.5 + 0.15:.2f}"
    )
    print(
        f"  stored confidence: {dict(Counter(i['stored_confidence'] for i in items))}"
    )
    SAMPLE_FILE.write_text(
        json.dumps(
            [{k: v for k, v in i.items() if k != "prompt"} for i in items], indent=1
        ),
        encoding="utf-8",
    )

    if args.dry_run:
        for it in items[:2]:
            r = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM,
                messages=[{"role": "user", "content": it["prompt"]}],
            )
            print(
                f"\n[{it['custom_id']} / {it['group']} / stored={it['stored_confidence']}] "
                f"in={r.usage.input_tokens} out={r.usage.output_tokens}"
            )
            print(r.content[0].text[:600])
        print("\ndry run complete.")
        return 0

    batch = client.messages.batches.create(
        requests=[
            {
                "custom_id": i["custom_id"],
                "params": {
                    "model": MODEL,
                    "max_tokens": MAX_TOKENS,
                    "system": SYSTEM,
                    "messages": [{"role": "user", "content": i["prompt"]}],
                },
            }
            for i in items
        ]
    )
    BATCH_ID_FILE.write_text(batch.id, encoding="utf-8")
    print(f"\nsubmitted {len(items)} requests as {batch.id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
