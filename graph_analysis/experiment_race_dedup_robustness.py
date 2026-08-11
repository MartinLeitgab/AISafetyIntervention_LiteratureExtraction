"""experiment_race_dedup_robustness.py

Tests whether OUR OR=3.75 xrisk x race-PA co-occurrence finding survives
de-duplication of the xrisk near-duplicate hub (pseudo-replication concern):
near-duplicate risk nodes from related papers are NOT independent observations,
so the node-level 2x2 over-counts and shrinks the CI artificially.

Three units of analysis:
  (A) node-level   — every risk node (== prior experiment_xrisk_race_correlation)
  (B) merged-node  — collapse risk near-dups (alias Phase0 + cos>=0.88 AND
                     name+alias Jaccard>=0.05, within risk block); canonical risk
                     is xrisk if ANY member is; has race-PA if ANY member does
  (C) paper-level  — unit = distinct source paper url; xrisk-paper if it has any
                     xrisk risk node; race-paper if any of its risk nodes has a
                     race-flagged structural PA neighbor

race-PA neighbor = structural (EDGE) problem-analysis neighbor whose name matches
race keywords. (Matches packet J.6 definition; topology-independent of EC/hubs.)

Class B (no LLM). Run from graph_analysis/:
    python -u experiment_race_dedup_robustness.py
"""

from __future__ import annotations
import json
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
OUT_JSON = ROOT / "phase2_results/experiment_race_dedup_robustness_report.json"

COS_THR = 0.88
JAC_THR = 0.05
RACE_KW = ["race", "racing", "competi"]
XRISK_KW = [
    "existential",
    "extinction",
    "catastroph",
    "superintellig",
    "takeover",
    "disempower",
    "annihilat",
    "omnicid",
    "extinct",
]
_TOK = re.compile(r"[a-z0-9]+")


def toks(s):
    return set(_TOK.findall((s or "").lower()))


def parse_aliases(raw):
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    s = str(raw).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    return [p.strip() for p in s.split(",") if p.strip()]


def has_kw(name, kws):
    n = (name or "").lower()
    return any(k in n for k in kws)


def is_risk(a):
    return (a.get("concept_category") or "").lower() == "risk"


def is_pa(a):
    return (a.get("concept_category") or "").lower() == "problem analysis"


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        r = x
        while self.p[r] != r:
            r = self.p[r]
        while self.p[x] != r:
            self.p[x], x = r, self.p[x]
        return r

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def two_by_two(n_xr_race, n_xr_norace, n_nx_race, n_nx_norace, label):
    table = np.array([[n_xr_race, n_xr_norace], [n_nx_race, n_nx_norace]])
    a, b, c, d = n_xr_race, n_xr_norace, n_nx_race, n_nx_norace
    # Haldane-Anscombe 0.5 correction for OR CI if any zero
    aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    orr = (aa * dd) / (bb * cc)
    se = np.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    lo, hi = np.exp(np.log(orr) - 1.96 * se), np.exp(np.log(orr) + 1.96 * se)
    chi2, p, _, _ = chi2_contingency(table, correction=False)
    p_xr = a / (a + b) if (a + b) else 0
    p_nx = c / (c + d) if (c + d) else 0
    res = {
        "label": label,
        "n_xrisk_total": a + b,
        "n_nonxrisk_total": c + d,
        "xrisk_race": a,
        "xrisk_norace": b,
        "nonxrisk_race": c,
        "nonxrisk_norace": d,
        "P_race_given_xrisk_pct": round(100 * p_xr, 2),
        "P_race_given_nonxrisk_pct": round(100 * p_nx, 2),
        "odds_ratio": round(orr, 3),
        "OR_CI95": [round(lo, 3), round(hi, 3)],
        "chi2": round(chi2, 2),
        "p_value": p,
    }
    print(f"\n[{label}]")
    print(f"  xrisk total={a + b}  non-xrisk total={c + d}")
    print(f"  P(race|xrisk)={100 * p_xr:.2f}%  P(race|non-xrisk)={100 * p_nx:.2f}%")
    print(f"  OR={orr:.3f}  CI95=[{lo:.3f},{hi:.3f}]  chi2={chi2:.2f}  p={p:.2e}")
    return res


def main():
    t0 = time.time()
    print("loading...", flush=True)
    na = pickle.load(open(STEP1 / "graph_node_attributes.pkl", "rb"))
    ed = pickle.load(open(STEP1 / "graph_edge_data.pkl", "rb"))
    print(f"  {len(na)} nodes, {len(ed)} edges", flush=True)

    # structural EDGE adjacency
    struct = defaultdict(set)
    for e in ed:
        if (e.get("type") or "").upper() != "EDGE":
            continue
        s, t = e.get("source"), e.get("target")
        if s is None or t is None or s == t:
            continue
        struct[s].add(t)
        struct[t].add(s)

    risk_nids = [n for n, a in na.items() if is_risk(a)]
    print(f"  risk nodes: {len(risk_nids)}", flush=True)

    def risk_has_race_pa(nid):
        for nb in struct.get(nid, ()):
            a = na.get(nb) or {}
            if is_pa(a) and has_kw(a.get("name"), RACE_KW):
                return True
        return False

    node_xr = {n: has_kw(na[n].get("name"), XRISK_KW) for n in risk_nids}
    node_race = {n: risk_has_race_pa(n) for n in risk_nids}

    # ---------- (A) node-level ----------
    a = sum(1 for n in risk_nids if node_xr[n] and node_race[n])
    b = sum(1 for n in risk_nids if node_xr[n] and not node_race[n])
    c = sum(1 for n in risk_nids if not node_xr[n] and node_race[n])
    d = sum(1 for n in risk_nids if not node_xr[n] and not node_race[n])
    resA = two_by_two(a, b, c, d, "A node-level (== prior experiment, our keywords)")

    # ---------- (B) merged-node-level ----------
    print(
        "\nmerging risk near-dups (alias Phase0 + cos>=0.88 AND name+alias Jaccard>=0.05)...",
        flush=True,
    )
    uf = UF()
    name_idx = defaultdict(list)
    for n in risk_nids:
        nm = (na[n].get("name") or "").lower().strip()
        if nm:
            name_idx[nm].append(n)
    for grp in name_idx.values():
        for i in range(1, len(grp)):
            uf.union(grp[0], grp[i])
    for n in risk_nids:
        for al in parse_aliases(na[n].get("aliases")):
            key = al.lower().strip()
            if key in name_idx:
                for m in name_idx[key]:
                    if m != n:
                        uf.union(n, m)
    # cos+jaccard within risk block
    E, idx_nid, tset = [], [], []
    for n in risk_nids:
        emb = na[n].get("embedding")
        if not (isinstance(emb, str) and emb.startswith("<")):
            continue
        try:
            v = np.fromstring(emb[1:-1], sep=",", dtype=np.float32)
        except Exception:
            continue
        if v.shape[0] != 1536:
            continue
        nrm = np.linalg.norm(v)
        if nrm == 0:
            continue
        E.append(v / nrm)
        idx_nid.append(n)
        ts = toks(na[n].get("name"))
        for al in parse_aliases(na[n].get("aliases")):
            ts |= toks(al)
        tset.append(ts)
    E = np.stack(E)
    B = 1000
    for s0 in range(0, len(idx_nid), B):
        s1 = min(s0 + B, len(idx_nid))
        S = E[s0:s1] @ E.T
        for ri, gi in enumerate(range(s0, s1)):
            cand = np.where(S[ri] >= COS_THR)[0]
            for gj in cand:
                if gj <= gi:
                    continue
                if (
                    len(tset[gi] & tset[gj]) / max(1, len(tset[gi] | tset[gj]))
                    >= JAC_THR
                ):
                    uf.union(idx_nid[gi], idx_nid[gj])
    groups = defaultdict(list)
    for n in risk_nids:
        groups[uf.find(n)].append(n)
    print(
        f"  canonical risk groups: {len(groups)} (from {len(risk_nids)} nodes)",
        flush=True,
    )
    a = b = c = d = 0
    for root, members in groups.items():
        xr = any(node_xr[m] for m in members)
        rc = any(node_race[m] for m in members)
        if xr and rc:
            a += 1
        elif xr:
            b += 1
        elif rc:
            c += 1
        else:
            d += 1
    resB = two_by_two(a, b, c, d, "B merged-node (risk near-dups collapsed)")

    # ---------- (C) paper-level ----------
    print("\npaper-level aggregation by url...", flush=True)
    paper_xr = defaultdict(bool)
    paper_race = defaultdict(bool)
    paper_has_risk = set()
    for n in risk_nids:
        url = na[n].get("url") or ""
        if not url:
            continue
        paper_has_risk.add(url)
        if node_xr[n]:
            paper_xr[url] = True
        if node_race[n]:
            paper_race[url] = True
    a = sum(1 for u in paper_has_risk if paper_xr[u] and paper_race[u])
    b = sum(1 for u in paper_has_risk if paper_xr[u] and not paper_race[u])
    c = sum(1 for u in paper_has_risk if not paper_xr[u] and paper_race[u])
    d = sum(1 for u in paper_has_risk if not paper_xr[u] and not paper_race[u])
    resC = two_by_two(a, b, c, d, "C paper-level (unit = source paper url)")

    out = {
        "experiment": "race x xrisk co-occurrence robustness to de-duplication",
        "keywords": {"xrisk": XRISK_KW, "race": RACE_KW},
        "note": "tests pseudo-replication: does OR survive collapsing near-dup xrisk nodes / aggregating to papers",
        "A_node_level": resA,
        "B_merged_node_level": resB,
        "C_paper_level": resC,
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}\nwall {time.time() - t0:.1f}s\nDONE.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
