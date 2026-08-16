"""experiment_query_demo.py

(A) Demonstrates the dataset's clustering-FREE retrieval value: "by what mechanism
    does intervention I reduce risk R" answered by returning an actual extracted
    end-to-end logical chain (risk -> pa -> ti -> dr -> im -> va -> intervention),
    with node names. No clustering, no centrality, no SIM. 3 worked examples across
    distinct themes (adversarial / reward-learning / governance).

(B) Coverage context for "why only 1,868 papers with a complete chain": counts
    distinct source papers (url) that have >=1 risk node, >=1 intervention node,
    and both — bounding how many papers *could* yield a chain, vs the 1,868 that do
    after the maturity>=3 + EDGE-conf>=3 + single-risk strict cuts.

Class B (no LLM). Run from graph_analysis/:
    python -u experiment_query_demo.py
"""

from __future__ import annotations
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
PATHS = ROOT / "phase1_rawpathsfiles/paths_hopwise_v4_edge_only.jsonl"
OUT = ROOT / "phase2_results/experiment_query_demo_report.json"

SUB_ABBR = {
    "risk": "RISK",
    "problem analysis": "pa",
    "theoretical insight": "ti",
    "design rationale": "dr",
    "implementation mechanism": "im",
    "validation evidence": "va",
}
THEMES = {
    "adversarial robustness": ["adversarial", "perturbation", "robust"],
    "reward / preference learning": ["reward", "preference", "rlhf", "human feedback"],
    "governance / compute": [
        "governance",
        "compute",
        "export",
        "oversight",
        "regulat",
        "treaty",
    ],
}


def main():
    na = pickle.load(open(STEP1 / "graph_node_attributes.pkl", "rb"))

    # ---------- (B) paper coverage bound ----------
    risk_papers, interv_papers = set(), set()
    for a in na.values():
        url = a.get("url")
        if not url:
            continue
        cc = (a.get("concept_category") or "").lower()
        ty = (a.get("type") or "").lower()
        if cc == "risk":
            risk_papers.add(url)
        if ty == "intervention":
            interv_papers.add(url)
    both = risk_papers & interv_papers
    print("(B) paper coverage bound:")
    print(f"  papers with >=1 risk node:         {len(risk_papers)}")
    print(f"  papers with >=1 intervention node: {len(interv_papers)}")
    print(f"  papers with BOTH:                  {len(both)}")
    print("  -> upper bound on papers that could yield a risk->intervention chain")
    print(
        "     (1,868 achieve a COMPLETE chain after maturity>=3 + edge-conf>=3 + single-risk cuts)"
    )

    def nm(nid):
        return na.get(nid, {}).get("name", f"<{nid}>")

    def subtype(nid):
        a = na.get(nid, {})
        if (a.get("type") or "").lower() == "intervention":
            return "INTERVENTION"
        return SUB_ABBR.get((a.get("concept_category") or "").lower(), "?")

    def maturity(nid):
        return na.get(nid, {}).get("intervention_maturity")

    # ---------- (A) pick 1 complete (all-5) length-7 chain per theme ----------
    chosen = {}
    with open(PATHS) as f:
        for line in f:
            if len(chosen) == len(THEMES):
                break
            d = json.loads(line)
            nodes = d["path"]
            if len(nodes) != 7:
                continue
            body = nodes[1:-1]
            subs = [(na.get(n, {}).get("concept_category") or "").lower() for n in body]
            if set(subs) != {
                "problem analysis",
                "theoretical insight",
                "design rationale",
                "implementation mechanism",
                "validation evidence",
            }:
                continue
            rname = nm(nodes[0]).lower()
            iname = nm(nodes[-1]).lower()
            for theme, kws in THEMES.items():
                if theme in chosen:
                    continue
                if any(k in rname or k in iname for k in kws):
                    chosen[theme] = nodes
                    break

    print(
        "\n(A) worked retrieval examples — 'by what mechanism does intervention reduce risk':\n"
    )
    demo = {}
    for theme, nodes in chosen.items():
        print(f"### Theme: {theme}")
        print(f"  RISK:          {nm(nodes[0])}")
        # order body in canonical chain order
        order = [
            "problem analysis",
            "theoretical insight",
            "design rationale",
            "implementation mechanism",
            "validation evidence",
        ]
        body = nodes[1:-1]
        bysub = {}
        for n in body:
            bysub.setdefault(
                (na.get(n, {}).get("concept_category") or "").lower(), []
            ).append(n)
        chain_render = [{"role": "RISK", "name": nm(nodes[0])}]
        for st in order:
            for n in bysub.get(st, []):
                print(f"   -> {SUB_ABBR[st]:3s}: {nm(n)}")
                chain_render.append({"role": SUB_ABBR[st], "name": nm(n)})
        iv = nodes[-1]
        print(f"   => INTERVENTION (maturity {maturity(iv)}): {nm(iv)}\n")
        chain_render.append(
            {"role": "INTERVENTION", "maturity": maturity(iv), "name": nm(iv)}
        )
        demo[theme] = {"path_node_ids": nodes, "chain": chain_render}

    out = {
        "coverage_bound": {
            "papers_with_risk": len(risk_papers),
            "papers_with_intervention": len(interv_papers),
            "papers_with_both": len(both),
            "papers_with_complete_strict_chain": 1868,
        },
        "query_demo": demo,
    }
    OUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"wrote {OUT}\nDONE.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
