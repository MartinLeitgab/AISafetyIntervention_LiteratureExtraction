#!/usr/bin/env python
"""Would a second model, from a different provider, assign the same logical-chain stage?

The probe in sec:r-stages recovers a node's stage from its own embedding at 98.8% against a
20.0% chance rate. Four of six external reviewers make the same objection: one o3 call wrote
both the text and the label, so the probe measures the extractor's internal consistency and
cannot show that the stage vocabulary carves anything a second annotator would recognise.

This is the second-annotator arm. It strips the stage labels from the body nodes of a
50-document sample, gives a different-provider model (Claude, against the o3 extractor) the
node's name and description and the same five stage definitions the extraction prompt used,
and asks it to assign a stage. It reports Cohen's kappa against the extractor's assignment,
a confusion matrix, and the pre-registered prediction that disagreement concentrates on the
problem-analysis / theoretical-insight and design-rationale / implementation-mechanism
boundaries -- the two places the probe's own errors already fall.

DESIGN NOTES, all of which bear on how the number should be read:

  * Nodes are pooled across documents and shuffled globally before batching. Presenting one
    document's nodes together would let the model reconstruct the paper's chain and infer
    stages from position, which is not what the probe measures and would inflate agreement.
  * The sample is stratified 25/25 between chain-yielding and non-chain-yielding documents,
    so the result can be read on the analysed population as well as on the corpus. Every
    other verification number in this paper is drawn from the corpus population only, and
    that mismatch is the paper's most-cited weakness.
  * The second model sees name and description. The probe's embedding covers name, aliases
    and description, so this arm sees slightly less. It is the harder condition.
  * Kappa here is agreement between two model assignments. It is not a human anchor and
    does not become one; a high kappa says the vocabulary is reproducible across models, a
    low kappa says it is idiosyncratic to the extractor. Neither says it is correct.

Billing: Claude Code CLI on subscription auth, run with --safe-mode so no CLAUDE.md, skill,
hook or MCP config enters the context, and an explicit --system-prompt so the harness prompt
does not either. ANTHROPIC_API_KEY is stripped from the child environment: this must not
fall through to metered API billing.

Class A (makes LLM calls). Run from graph_analysis/:

    python -u experiment_review_stage_agreement.py --smoke     # one batch, then stop
    python -u experiment_review_stage_agreement.py             # full run, resumable
    python -u experiment_review_stage_agreement.py --score-only

Per-batch output is written to phase2_results/stage_agreement_batches/batch_NNN.json before
the next call is made, so an interrupted run loses at most one batch and a re-run skips what
already landed.

Output: graph_analysis/phase2_results/experiment_review_stage_agreement_report.json
"""

import argparse
import json
import os
import pickle
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
NODES = (
    ROOT
    / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/graph_node_attributes.pkl"
)
RAW = ROOT / "phase1_rawpathsfiles/paths_hopwise_v4_edge_only.jsonl"
BATCHDIR = ROOT / "phase2_results/stage_agreement_batches"
OUT = ROOT / "phase2_results/experiment_review_stage_agreement_report.json"

SEED = 42
N_DOCS_PER_STRATUM = 25
BATCH_SIZE = 40
MODEL = "claude-opus-5"

# The five intermediate stages, in logical-chain order. Never reorder these.
STAGES = [
    "problem analysis",
    "theoretical insight",
    "design rationale",
    "implementation mechanism",
    "validation evidence",
]
ABBR = {
    "problem analysis": "pa",
    "theoretical insight": "ti",
    "design rationale": "dr",
    "implementation mechanism": "im",
    "validation evidence": "va",
}

# Verbatim from the extraction prompt's step_2 (reproduced in app:prompt), so the second
# model is judged against the same definitions the extractor was given. Adding or sharpening
# a definition here would make the arm easier than the task the extractor performed.
RUBRIC = """You are labelling concept nodes extracted from AI safety papers. Each node
belongs to exactly one stage of a causal-interventional chain that runs
risk -> problem analysis -> theoretical insight -> design rationale ->
implementation mechanism -> validation evidence -> intervention.

The five stages you must choose between, with the naming template each was written to:

  problem analysis          [Mechanism causing risk] in [context]
  theoretical insight       [Assumption or hypothesized resolution] in [context]
  design rationale          [Solution approach] in [context]
  implementation mechanism  [Technique] in [context]
  validation evidence       [Measurement and result] in [context]

For each node below, output the one stage it belongs to. Nodes come from many different
papers in random order; there is no chain to reconstruct and no ordering to exploit.

Return ONLY a JSON array, one object per node, no prose before or after:
[{"id": <the integer id>, "stage": "<one of the five stage names, exactly as written>"}]
"""

AUTH_VARS = ("ANTHROPIC_API_KEY", "anthropic_api_key", "ANTHROPIC_AUTH_TOKEN")


def child_env():
    """Strip auth vars so the run bills to the subscription, not the metered API."""
    return {
        k: v
        for k, v in os.environ.items()
        if k not in AUTH_VARS
        and k not in ("CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX")
    }


def build_sample():
    with open(NODES, "rb") as f:
        na = pickle.load(f)

    chain_docs = set()
    with open(RAW, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for nid in json.loads(line)["path"]:
                u = na.get(int(nid), {}).get("url")
                if u:
                    chain_docs.add(u)

    by_doc = defaultdict(list)
    for nid, r in na.items():
        if r.get("type") != "concept":
            continue
        cat = (r.get("concept_category") or "").strip().lower()
        if cat in STAGES and r.get("url") and (r.get("name") or "").strip():
            by_doc[r["url"]].append(int(nid))

    yielding = sorted(d for d in by_doc if d in chain_docs)
    other = sorted(d for d in by_doc if d not in chain_docs)
    if min(len(yielding), len(other)) < N_DOCS_PER_STRATUM:
        raise SystemExit(
            f"FATAL: strata too small ({len(yielding)} / {len(other)}); expected at least "
            f"{N_DOCS_PER_STRATUM} documents in each"
        )

    rng = random.Random(SEED)
    docs = rng.sample(yielding, N_DOCS_PER_STRATUM) + rng.sample(
        other, N_DOCS_PER_STRATUM
    )

    items = []
    for d in docs:
        for nid in by_doc[d]:
            r = na[nid]
            items.append(
                {
                    "id": nid,
                    "url": d,
                    "stratum": "chain_yielding" if d in chain_docs else "other",
                    "name": (r.get("name") or "").strip(),
                    "description": (r.get("description") or "").strip(),
                    "true_stage": r["concept_category"].strip().lower(),
                }
            )
    rng.shuffle(items)  # pool across documents so no chain order is visible
    return items


def render(batch):
    out = []
    for it in batch:
        desc = it["description"]
        if len(desc) > 700:
            desc = desc[:700] + " [...]"
        out.append(f"id {it['id']}\nname: {it['name']}\ndescription: {desc}")
    return "\n\n".join(out)


def call_claude(prompt):
    # The prompt goes on stdin, not argv: a 40-node batch runs past the 8191-character
    # Windows command-line limit and cmd.exe fails with "The command line is too long"
    # before claude is ever reached.
    cmd = [
        "claude",
        "-p",
        "--safe-mode",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--system-prompt",
        "You are a careful annotator. Follow the output format exactly.",
        "--tools",
        "",
        "--model",
        MODEL,
        "--output-format",
        "json",
    ]
    if os.name == "nt":
        cmd = ["cmd.exe", "/c"] + cmd
    p = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=900,
        env=child_env(),
        encoding="utf-8",
    )
    if p.returncode != 0:
        raise RuntimeError(f"claude exited {p.returncode}: {(p.stderr or '')[:400]}")
    env = json.loads(p.stdout)
    if env.get("subtype") != "success":
        raise RuntimeError(f"claude returned subtype={env.get('subtype')}")
    return env["result"], env.get("usage", {})


def parse_labels(text):
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```")[1]
        s = s[s.index("\n") + 1 :] if "\n" in s else s
        s = s.replace("json", "", 1) if s.lstrip().startswith("json") else s
    i, j = s.find("["), s.rfind("]")
    if i < 0 or j < 0:
        raise ValueError(f"no JSON array in response: {s[:200]}")
    rows = json.loads(s[i : j + 1])
    return {int(r["id"]): str(r["stage"]).strip().lower() for r in rows}


def cohen_kappa(pairs):
    n = len(pairs)
    labels = STAGES
    obs = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    exp = sum((ca[x] / n) * (cb[x] / n) for x in labels)
    return (obs - exp) / (1 - exp), obs, exp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run one batch and stop")
    ap.add_argument(
        "--score-only", action="store_true", help="score whatever has landed"
    )
    a = ap.parse_args()

    for p, how in (
        (NODES, "run phase2_step1_loadandparse.py against the FalkorDB dump"),
        (RAW, "the released 8,954-chain path file ships with the repo"),
    ):
        if not p.exists():
            raise SystemExit(f"FATAL: {p} not found. To produce it: {how}.")

    items = build_sample()
    batches = [items[i : i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
    BATCHDIR.mkdir(parents=True, exist_ok=True)
    print(
        f"sample: {len(items)} body nodes from {N_DOCS_PER_STRATUM * 2} documents "
        f"-> {len(batches)} batches of up to {BATCH_SIZE}",
        flush=True,
    )

    if not a.score_only:
        for bi, batch in enumerate(batches):
            fp = BATCHDIR / f"batch_{bi:03d}.json"
            if fp.exists():
                print(f"  batch {bi:03d}: already on disk, skipping", flush=True)
                continue
            prompt = RUBRIC + "\n\nNODES:\n\n" + render(batch)
            text, usage = call_claude(prompt)
            labels = parse_labels(text)
            missing = [it["id"] for it in batch if it["id"] not in labels]
            fp.write_text(
                json.dumps(
                    {
                        "batch": bi,
                        "model": MODEL,
                        "n_sent": len(batch),
                        "n_returned": len(labels),
                        "missing_ids": missing,
                        "usage": usage,
                        "labels": {str(k): v for k, v in labels.items()},
                    },
                    indent=1,
                ),
                encoding="utf-8",
            )
            print(
                f"  batch {bi:03d}: {len(labels)}/{len(batch)} labelled"
                f"{' MISSING ' + str(len(missing)) if missing else ''}",
                flush=True,
            )
            if a.smoke:
                print(
                    "\nsmoke test done -- inspect the batch file, then re-run without --smoke"
                )
                return

    # ---- score --------------------------------------------------------------------------
    got = {}
    usage_total = Counter()
    for fp in sorted(BATCHDIR.glob("batch_*.json")):
        d = json.loads(fp.read_text(encoding="utf-8"))
        got.update({int(k): v for k, v in d["labels"].items()})
        for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens"):
            usage_total[k] += (d.get("usage") or {}).get(k, 0) or 0

    by_id = {it["id"]: it for it in items}
    pairs, unusable = [], Counter()
    per_stratum = defaultdict(list)
    for nid, pred in got.items():
        it = by_id.get(nid)
        if it is None:
            unusable["id_not_in_sample"] += 1
            continue
        if pred not in STAGES:
            unusable["label_outside_the_five_stages"] += 1
            continue
        pairs.append((it["true_stage"], pred))
        per_stratum[it["stratum"]].append((it["true_stage"], pred))

    if not pairs:
        raise SystemExit("FATAL: no usable label pairs; nothing to score")

    kappa, obs, exp = cohen_kappa(pairs)
    cm = defaultdict(Counter)
    for t, p_ in pairs:
        cm[t][p_] += 1

    per_class = {}
    for s in STAGES:
        tp = cm[s][s]
        fp_ = sum(cm[o][s] for o in STAGES if o != s)
        fn = sum(cm[s][o] for o in STAGES if o != s)
        prec = tp / (tp + fp_) if tp + fp_ else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_class[s] = {
            "n_true": sum(cm[s].values()),
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
        }

    def offdiag(a_, b_):
        return cm[a_][b_] + cm[b_][a_]

    n_disagree = sum(1 for t, p_ in pairs if t != p_)
    adjacent = {
        "pa_vs_ti": offdiag("problem analysis", "theoretical insight"),
        "ti_vs_dr": offdiag("theoretical insight", "design rationale"),
        "dr_vs_im": offdiag("design rationale", "implementation mechanism"),
        "im_vs_va": offdiag("implementation mechanism", "validation evidence"),
    }

    strat = {}
    for k, v in per_stratum.items():
        if v:
            kk, oo, _ = cohen_kappa(v)
            strat[k] = {
                "n": len(v),
                "cohen_kappa": round(kk, 3),
                "raw_agreement": round(oo, 3),
            }

    report = {
        "experiment": "stage-assignment agreement with a second model (S3, reviewer R25)",
        "question": (
            "The stage probe is circular: one o3 call wrote both the node text and its stage "
            "label. Would a different-provider model assign the same stage from the text?"
        ),
        "design": {
            "second_model": MODEL,
            "extractor": "o3",
            "different_provider": True,
            "n_documents": N_DOCS_PER_STRATUM * 2,
            "stratification": "25 chain-yielding documents, 25 not",
            "n_nodes_sent": len(items),
            "n_label_pairs_scored": len(pairs),
            "nodes_pooled_and_shuffled_across_documents": True,
            "model_sees": "node name and description only",
            "probe_sees": "embedding of name, aliases and description",
            "seed": SEED,
        },
        "headline": {
            "cohen_kappa": round(kappa, 3),
            "raw_agreement": round(obs, 3),
            "chance_agreement": round(exp, 3),
            "n_disagreements": n_disagree,
        },
        "by_stratum": strat,
        "per_class": per_class,
        "confusion_matrix_true_by_predicted": {
            t: {p_: cm[t][p_] for p_ in STAGES} for t in STAGES
        },
        "adjacent_stage_confusions": {
            **adjacent,
            "pa_ti_plus_dr_im_share_of_disagreements": (
                round(
                    100 * (adjacent["pa_vs_ti"] + adjacent["dr_vs_im"]) / n_disagree, 1
                )
                if n_disagree
                else None
            ),
            "PRE_REGISTERED_PREDICTION": (
                "Disagreement concentrates on pa/ti and dr/im, the two boundaries where the "
                "probe's own errors fall (37 and 17 of its 113 test errors). Stated in "
                "OPEN_ITEMS.md S3 before this ran."
            ),
        },
        "unusable_responses": dict(unusable),
        "usage_subscription_billed": dict(usage_total),
        "LIMITS": (
            "Two model assignments, not a human anchor. A high kappa shows the vocabulary is "
            "reproducible across providers; it cannot show the stages are the right ones, "
            "and neither model's label was adjudicated by a person."
        ),
    }

    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report["headline"], indent=1))
    print(json.dumps(report["by_stratum"], indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
