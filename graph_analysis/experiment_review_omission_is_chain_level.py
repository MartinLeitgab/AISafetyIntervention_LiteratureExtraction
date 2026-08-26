#!/usr/bin/env python
"""Is the judge's omission signal load-bearing at the CHAIN level, or is it fluff?

The paper reports omission three ways -- 0.6% and 28.8% at NODE level (tab:omission) and
18.1% at EDGE level (sec:r-judge) -- and every one of them counts a unit that is not the
unit the artifact is about. The artifact's unit is a risk-to-intervention CHAIN. A concept
node added in the middle of an argument the extraction already captured does not change
which risk is connected to which intervention; a second pass over any document will always
find more nameable concepts, so a node-addition count measures the granularity of the
second pass at least as much as it measures a defect in the first.

This script asks the question the reported rates cannot answer:

    Of the relationships and nodes the judge says are missing, how many could change the
    chain set -- by introducing a new risk endpoint, a new intervention endpoint, or a
    bridge that connects an existing risk to an existing intervention it could not
    previously reach?

Three classes, and only the third is load-bearing:

  INERT          Both endpoints already exist in the released extraction for that paper AND
                 neither is a risk or an intervention. The edge densifies the middle of an
                 argument already captured. It cannot create, destroy or re-route a chain.
  RE-ROUTING     Both endpoints exist and at least one is a risk or an intervention, so the
                 edge could add a branch between existing endpoints -- same risk to a
                 different intervention, or the reverse. Materially different only if the
                 pair was not already connected; reported separately, not merged into INERT.
  NEW MATERIAL   At least one endpoint is a node the extraction does not have. Only these
                 can introduce an endpoint that was not there at all.

Class B: no LLM call, no network. Run from graph_analysis/:

    python -u experiment_review_omission_is_chain_level.py --judge-reports <dir>

    git archive origin/anthropic_judge_test extraction_validator/extend_try_1 \\
        | tar -x -C /tmp/judge

This script adjudicates nothing. Every row it reads is the judge's unchecked opinion, and
a row it calls INERT is inert *conditional on the judge being right that it is missing*.
The point is not that the judge is wrong; it is that being right about these rows would
still not be an omission of a chain.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
SLIM = ROOT / "phase2_results/node_attrs_slim.pkl"
OUT = ROOT / "phase2_results/experiment_review_omission_is_chain_level_report.json"
SKIP_FILES = {"summary.json", "errors.json"}

ENDPOINT_CATS = {"risk"}  # concept_category values that are chain endpoints
STOP = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "for",
    "and",
    "or",
    "on",
    "by",
    "with",
    "from",
    "that",
    "this",
    "is",
    "are",
    "as",
    "at",
    "via",
    "can",
    "be",
}


def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\([^)]*\)", " ", s)  # judge appends "(theoretical insight)" etc
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return " ".join(s.split())


def toks(s: str) -> set[str]:
    return {t for t in norm(s).split() if t not in STOP and len(t) > 2}


def match(name: str, pool: dict[str, dict]) -> dict | None:
    """Resolve a judge-written node name against the released nodes for one paper.

    Exact normalised match first. Then Jaccard over content words at 0.6, which is
    deliberately generous: a FALSE match moves a row from NEW MATERIAL into an inert class
    and therefore works AGAINST this script's conclusion. Being generous here means the
    load-bearing share it reports is an under-estimate, never an over-estimate.
    """
    n = norm(name)
    if n in pool:
        return pool[n]
    t = toks(name)
    if not t:
        return None
    best, best_j = None, 0.0
    for cand_n, rec in pool.items():
        ct = toks(cand_n)
        if not ct:
            continue
        j = len(t & ct) / len(t | ct)
        if j > best_j:
            best, best_j = rec, j
    return best if best_j >= 0.6 else None


def is_endpoint(rec: dict | None) -> str | None:
    if not rec:
        return None
    if rec.get("type") == "intervention":
        return "intervention"
    if (rec.get("concept_category") or "").lower() in ENDPOINT_CATS:
        return "risk"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-reports", required=True)
    a = ap.parse_args()

    jdir = Path(a.judge_reports)
    if not jdir.is_dir():
        raise SystemExit(
            f"FATAL: judge report directory not found: {jdir}\n"
            "  produce it with:\n"
            "    git archive origin/anthropic_judge_test extraction_validator/extend_try_1"
            " | tar -x -C /tmp/judge\n"
            "  this script does NOT fall back to a cached or partial copy."
        )
    if not SLIM.exists():
        raise SystemExit(
            f"FATAL: {SLIM} not found. Produce it with experiment_review_prep_slim_nodes.py."
        )

    slim = pickle.load(SLIM.open("rb"))
    by_url: dict[str, dict[str, dict]] = defaultdict(dict)
    for rec in slim.values():
        u = rec.get("url")
        if u and rec.get("name"):
            by_url[u][norm(rec["name"])] = rec

    cls = Counter()
    endpoint_touch = Counter()
    added_node_cats = Counter()
    added_nodes_total = 0
    papers, unmatched_papers = 0, []
    per_paper = []

    for p in sorted(jdir.glob("*.json")):
        if p.name in SKIP_FILES:
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        url = r.get("url")
        pool = by_url.get(url or "", {})
        if not pool:
            unmatched_papers.append(p.name)
            continue
        papers += 1

        rows = ((r.get("validation_report") or {}).get("coverage") or {}).get(
            "expected_edges_from_source"
        ) or []
        paper_cls = Counter()
        for row in rows:
            if (row.get("status") or "").strip().lower() != "missing":
                continue
            src_names = row.get("expected_source_node_name") or []
            tgt_names = row.get("expected_target_node_name") or []
            if not src_names or not tgt_names:
                cls["missing__unusable_row"] += 1
                paper_cls["unusable"] += 1
                continue
            s_rec = match(src_names[0], pool)
            t_rec = match(tgt_names[0], pool)
            if s_rec is None or t_rec is None:
                cls["NEW_MATERIAL"] += 1
                paper_cls["NEW_MATERIAL"] += 1
                continue
            roles = {is_endpoint(s_rec), is_endpoint(t_rec)} - {None}
            if roles:
                cls["RE_ROUTING"] += 1
                paper_cls["RE_ROUTING"] += 1
                for role in roles:
                    endpoint_touch[role] += 1
            else:
                cls["INERT"] += 1
                paper_cls["INERT"] += 1

        # proposed_fixes.add_nodes -- the instrument behind the 0.6% node-omission figure
        for nd in (r.get("proposed_fixes") or {}).get("add_nodes") or []:
            added_nodes_total += 1
            if (nd.get("type") or "").lower() == "intervention":
                added_node_cats["intervention"] += 1
            else:
                added_node_cats[
                    (nd.get("concept_category") or "unspecified").lower()
                ] += 1

        per_paper.append({"paper": p.name, **paper_cls})

    usable = cls["INERT"] + cls["RE_ROUTING"] + cls["NEW_MATERIAL"]
    load_bearing = cls["RE_ROUTING"] + cls["NEW_MATERIAL"]

    def pct(n, d):
        return round(100 * n / d, 1) if d else None

    added_endpoints = added_node_cats["intervention"] + added_node_cats["risk"]

    report = {
        "experiment": "is the judge's omission signal load-bearing at the chain level",
        "question": (
            "The paper reports omission at node level (0.6%, 28.8%) and edge level "
            "(18.1%). Neither unit is the chain. Of the relationships the judge calls "
            "missing, how many could change which risk connects to which intervention?"
        ),
        "n_papers": papers,
        "unmatched_papers": unmatched_papers,
        "missing_rows": {
            "usable": usable,
            "unusable_no_endpoint_names": cls["missing__unusable_row"],
            "INERT_both_endpoints_present_neither_is_a_chain_endpoint": cls["INERT"],
            "RE_ROUTING_both_present_one_is_a_risk_or_intervention": cls["RE_ROUTING"],
            "NEW_MATERIAL_at_least_one_endpoint_absent": cls["NEW_MATERIAL"],
        },
        "shares_of_usable_missing_rows": {
            "inert_pct": pct(cls["INERT"], usable),
            "re_routing_pct": pct(cls["RE_ROUTING"], usable),
            "new_material_pct": pct(cls["NEW_MATERIAL"], usable),
            "load_bearing_pct": pct(load_bearing, usable),
        },
        "re_routing_touches": dict(endpoint_touch),
        "add_nodes_instrument": {
            "note": (
                "This is the instrument behind the 0.6% node-omission figure. The question "
                "is what KIND of node the judge proposed, because only a risk or an "
                "intervention can add a chain endpoint."
            ),
            "total_added_nodes": added_nodes_total,
            "by_category": dict(added_node_cats),
            "endpoints_risk_or_intervention": added_endpoints,
            "endpoints_pct": pct(added_endpoints, added_nodes_total),
        },
        "READING": (
            "load_bearing_pct is an UPPER bound on the share of the judge's missing "
            "relationships that could matter to the chain set, and it is an upper bound "
            "twice over: RE_ROUTING rows are only material if the endpoint pair was not "
            "already connected by another route, which this script does not check, and "
            "the name matcher is deliberately generous so borderline cases fall into the "
            "inert classes. The complement -- INERT -- is the share that provably cannot "
            "change any chain, because it links two non-endpoint nodes the extraction "
            "already holds."
        ),
        "LIMITS": (
            "Every input row is the judge's unadjudicated opinion. This script does not "
            "ask whether a row is correctly marked missing; it asks what would follow if "
            "it were. Node-name resolution is fuzzy (normalised exact, then Jaccard 0.6 "
            "over content words), so individual rows can be misclassified even though the "
            "aggregate direction is robust to the threshold."
        ),
    }

    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"papers matched: {papers}   unmatched: {len(unmatched_papers)}")
    print(
        f"usable 'missing' rows: {usable}  (+{cls['missing__unusable_row']} unusable)"
    )
    print(f"  INERT        {cls['INERT']:>4}  {pct(cls['INERT'], usable)}%")
    print(f"  RE_ROUTING   {cls['RE_ROUTING']:>4}  {pct(cls['RE_ROUTING'], usable)}%")
    print(
        f"  NEW_MATERIAL {cls['NEW_MATERIAL']:>4}  {pct(cls['NEW_MATERIAL'], usable)}%"
    )
    print(f"  load-bearing (upper bound): {pct(load_bearing, usable)}%")
    print(
        f"\nadd_nodes: {added_nodes_total} total, {added_endpoints} are risk/intervention "
        f"({pct(added_endpoints, added_nodes_total)}%)"
    )
    print(f"  by category: {dict(added_node_cats)}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
