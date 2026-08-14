"""experiment_xrisk_race_correlation.py

Direct test: does xrisk and race framing co-occur at the path level, independent
of merging? If YES, then race-framing's appearance near xrisk hubs after merging
is partly intrinsic (the same papers that discuss x-risk also discuss race
dynamics — a discourse-community pattern). If NO, the co-occurrence is entirely
manufactured by the merging-induced topology shift.

Method: walk every path in `paths_hopwise_v4_edge_only.jsonl`. For each path:
  - Does the risk node have an xrisk keyword in its name? (xrisk_path = True/False)
  - Does ANY body node with category=problem_analysis have a race keyword in its
    name? (race_pa = True/False)
Build 2x2 contingency table; compute odds ratio + chi-squared.

Also tabulate at the RISK-NODE level (not path level): for each unique risk
node, get its EDGE-type PA neighbors; check xrisk vs race-PA presence.

Class B, no LLM calls. Uses our existing PKL data and the 8,954-path file.
"""

from __future__ import annotations
import json
import pickle
from collections import defaultdict
from pathlib import Path
from math import log, sqrt

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
PATHS_FP = ROOT / "phase1_rawpathsfiles/paths_hopwise_v4_edge_only.jsonl"
OUT = ROOT / "phase2_results/experiment_xrisk_race_correlation_report.json"

XRISK_KEYWORDS = ["existential", "extinction", "catastrophic misalign"]
RACE_KEYWORDS = ["race", "competitive", "competition", "racing"]


def has_kw(text, keywords):
    t = (text or "").lower()
    return any(kw in t for kw in keywords)


def chi2_2x2(a, b, c, d):
    """2x2 contingency [[a,b],[c,d]] χ² with Yates correction; returns (chi2, p_approx)."""
    n = a + b + c + d
    if n == 0:
        return 0.0, 1.0
    expected = [
        (a + b) * (a + c) / n,
        (a + b) * (b + d) / n,
        (c + d) * (a + c) / n,
        (c + d) * (b + d) / n,
    ]
    obs = [a, b, c, d]
    chi2 = sum(((abs(o - e) - 0.5) ** 2) / e for o, e in zip(obs, expected) if e > 0)
    # Rough p approximation for 1 d.o.f. via standard cutoffs (don't need scipy)
    p_approx = (
        1.0
        if chi2 < 0.5
        else 0.5
        if chi2 < 2.7
        else 0.1
        if chi2 < 3.84
        else 0.05
        if chi2 < 6.63
        else 0.01
        if chi2 < 10.83
        else 0.001
    )
    return chi2, p_approx


def odds_ratio(a, b, c, d):
    """Haldane-Anscombe corrected odds ratio + 95% CI."""
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_ = (a * d) / (b * c)
    log_se = sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    log_or = log(or_)
    return or_, (
        round(2.71828 ** (log_or - 1.96 * log_se), 3),
        round(2.71828 ** (log_or + 1.96 * log_se), 3),
    )


def main():
    print("loading node_attrs ...", flush=True)
    with open(STEP1 / "graph_node_attributes.pkl", "rb") as f:
        na = pickle.load(f)
    print(f"  {len(na)} nodes", flush=True)

    print("loading paths ...", flush=True)
    paths = []
    with open(PATHS_FP, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                paths.append(json.loads(line))
    print(f"  {len(paths)} paths", flush=True)

    # ===== Path-level 2x2 =====
    # For each path: xrisk_path = risk_node name has xrisk kw; race_pa = any PA body node
    # has race kw
    a = b = c = d = 0  # [xrisk & race] [xrisk & !race] [!xrisk & race] [!xrisk & !race]
    for p in paths:
        nodes = p.get("path", [])
        cats = p.get("categories", [])
        if not nodes or not cats:
            continue
        # risk node is first
        risk_attrs = na.get(int(nodes[0])) or na.get(nodes[0]) or {}
        risk_name = risk_attrs.get("name", "")
        is_xrisk = has_kw(risk_name, XRISK_KEYWORDS)
        # any PA body node with race kw
        race_pa = False
        for nid, cat in zip(nodes, cats):
            if cat == "problem_analysis":
                attrs = na.get(int(nid)) or na.get(nid) or {}
                if has_kw(attrs.get("name", ""), RACE_KEYWORDS):
                    race_pa = True
                    break
        if is_xrisk and race_pa:
            a += 1
        elif is_xrisk and not race_pa:
            b += 1
        elif (not is_xrisk) and race_pa:
            c += 1
        else:
            d += 1

    total = a + b + c + d
    p_xrisk = round((a + b) / total * 100, 2)
    p_race_given_xrisk = round(a / (a + b) * 100, 2) if (a + b) > 0 else 0
    p_race_given_no_xrisk = round(c / (c + d) * 100, 2) if (c + d) > 0 else 0
    or_, ci = odds_ratio(a, b, c, d)
    chi2, p_ = chi2_2x2(a, b, c, d)

    print("\n" + "=" * 70)
    print("PATH-LEVEL contingency: xrisk path × race-framed PA")
    print("=" * 70)
    print(f"  total paths: {total}")
    print(f"  xrisk fraction: {p_xrisk}%")
    print("  2x2:")
    print(f"    xrisk + race_pa     = {a:>5}")
    print(f"    xrisk + no_race_pa  = {b:>5}")
    print(f"    !xrisk + race_pa    = {c:>5}")
    print(f"    !xrisk + no_race_pa = {d:>5}")
    print(f"  P(race_pa | xrisk):     {p_race_given_xrisk}%")
    print(f"  P(race_pa | !xrisk):    {p_race_given_no_xrisk}%")
    print(f"  Odds ratio (race_pa | xrisk vs !xrisk): {or_:.3f}  CI95: {ci}")
    print(f"  chi2: {chi2:.2f}  p_approx={p_}")

    # ===== Risk-node-level co-occurrence =====
    # Build EDGE-type adjacency
    print("\nloading edge_data for node-level adjacency ...", flush=True)
    with open(STEP1 / "graph_edge_data.pkl", "rb") as f:
        ed = pickle.load(f)
    adj = defaultdict(set)
    for e in ed:
        if (e.get("type") or "").upper() != "EDGE":
            continue
        s, t = e.get("source"), e.get("target")
        if s is None or t is None or s == t:
            continue
        adj[s].add(t)
        adj[t].add(s)

    a2 = b2 = c2 = d2 = 0
    for nid, attrs in na.items():
        if (attrs.get("concept_category") or "").lower() != "risk":
            continue
        is_xrisk = has_kw(attrs.get("name", ""), XRISK_KEYWORDS)
        has_race_pa = False
        for nbr in adj.get(nid, ()):
            nbr_attrs = na.get(nbr) or {}
            if (nbr_attrs.get("concept_category") or "").lower() == "problem analysis":
                if has_kw(nbr_attrs.get("name", ""), RACE_KEYWORDS):
                    has_race_pa = True
                    break
        if is_xrisk and has_race_pa:
            a2 += 1
        elif is_xrisk:
            b2 += 1
        elif has_race_pa:
            c2 += 1
        else:
            d2 += 1

    total2 = a2 + b2 + c2 + d2
    p_xr2 = round((a2 + b2) / total2 * 100, 2) if total2 else 0
    p_race_xr = round(a2 / (a2 + b2) * 100, 2) if (a2 + b2) else 0
    p_race_nxr = round(c2 / (c2 + d2) * 100, 2) if (c2 + d2) else 0
    or2, ci2 = odds_ratio(a2, b2, c2, d2)
    chi22, p2 = chi2_2x2(a2, b2, c2, d2)
    print("\n" + "=" * 70)
    print(
        "RISK-NODE-LEVEL contingency: xrisk node × race-framed PA neighbor (EDGE-type)"
    )
    print("=" * 70)
    print(f"  total risk nodes: {total2}")
    print(f"  xrisk fraction: {p_xr2}%")
    print("  2x2:")
    print(f"    xrisk + race_pa_neighbor     = {a2:>5}")
    print(f"    xrisk + no_race_pa_neighbor  = {b2:>5}")
    print(f"    !xrisk + race_pa_neighbor    = {c2:>5}")
    print(f"    !xrisk + no_race_pa_neighbor = {d2:>5}")
    print(f"  P(race_pa_neighbor | xrisk):     {p_race_xr}%")
    print(f"  P(race_pa_neighbor | !xrisk):    {p_race_nxr}%")
    print(f"  Odds ratio: {or2:.3f}  CI95: {ci2}")
    print(f"  chi2: {chi22:.2f}  p_approx={p2}")

    out_doc = {
        "experiment": "direct xrisk × race-framing co-occurrence (no merging)",
        "xrisk_keywords": XRISK_KEYWORDS,
        "race_keywords": RACE_KEYWORDS,
        "path_level": {
            "total_paths": total,
            "xrisk_path_pct": p_xrisk,
            "contingency_abcd": {
                "a_xrisk_race": a,
                "b_xrisk_norace": b,
                "c_noxrisk_race": c,
                "d_noxrisk_norace": d,
            },
            "p_race_given_xrisk_pct": p_race_given_xrisk,
            "p_race_given_no_xrisk_pct": p_race_given_no_xrisk,
            "odds_ratio": round(or_, 3),
            "or_ci95": ci,
            "chi2": round(chi2, 2),
            "p_approx": p_,
        },
        "risk_node_level": {
            "total_risk_nodes": total2,
            "xrisk_node_pct": p_xr2,
            "contingency_abcd": {
                "a_xrisk_race": a2,
                "b_xrisk_norace": b2,
                "c_noxrisk_race": c2,
                "d_noxrisk_norace": d2,
            },
            "p_race_given_xrisk_pct": p_race_xr,
            "p_race_given_no_xrisk_pct": p_race_nxr,
            "odds_ratio": round(or2, 3),
            "or_ci95": ci2,
            "chi2": round(chi22, 2),
            "p_approx": p2,
        },
    }
    OUT.write_text(json.dumps(out_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
