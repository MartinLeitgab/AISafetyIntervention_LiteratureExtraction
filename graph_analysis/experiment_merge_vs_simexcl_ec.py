"""experiment_merge_vs_simexcl_ec.py

Tests the user's claim (review items 7-8): on a MERGED graph, the xrisk super-hub
inherits its members' EDGE edges, so it dominates eigenvector centrality (EC) by
EDGE topology alone -> excluding risk<->risk SIM is NOT sufficient
you must UNDO
the merge first.

Four conditions (top-10 risks by EC, count how many are xrisk-keyword):
  1. un-merged, full within-cat SIM            (== J3 P1
  expect ~10/10 xrisk)
  2. un-merged, risk<->risk SIM EXCLUDED       (== J3 P2)
  3. merged(risk), full within-cat SIM
  4. merged(risk), risk<->risk SIM EXCLUDED    <- THE KEY TEST

Risk-block merge = alias Phase0 + cos>=0.88 AND name+alias Jaccard>=0.05 (Gleb rule).
Only the risk block is merged (other blocks identity) — the question is risk-side EC
domination, and risk-merge is the relevant collapse.

Class B (no LLM). Run from graph_analysis/:
    python -u experiment_merge_vs_simexcl_ec.py
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
import scipy.sparse as sp

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
OUT = ROOT / "phase2_results/experiment_merge_vs_simexcl_ec_report.json"

COS_THR_MERGE = 0.88
JAC_THR = 0.05
SIM_THR = 0.80
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
    s = str(raw).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    return [p.strip() for p in s.split(",") if p.strip()]


def is_xr(name):
    n = (name or "").lower()
    return any(k in n for k in XRISK_KW)


def cos_from_score(s):
    return 1.0 - float(s) ** 2 / 2.0


def cat_key(a):
    if (a.get("type") or "").lower() == "intervention":
        return "intervention"
    return (a.get("concept_category") or "").lower().strip() or None


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


def merge_risk_block(na):
    risk = [
        n for n, a in na.items() if (a.get("concept_category") or "").lower() == "risk"
    ]
    uf = UF()
    name_idx = defaultdict(list)
    for n in risk:
        nm = (na[n].get("name") or "").lower().strip()
        if nm:
            name_idx[nm].append(n)
    for grp in name_idx.values():
        for i in range(1, len(grp)):
            uf.union(grp[0], grp[i])
    for n in risk:
        for al in parse_aliases(na[n].get("aliases")):
            k = al.lower().strip()
            if k in name_idx:
                for m in name_idx[k]:
                    if m != n:
                        uf.union(n, m)
    E, ids, ts = [], [], []
    for n in risk:
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
        ids.append(n)
        t = toks(na[n].get("name"))
        for al in parse_aliases(na[n].get("aliases")):
            t |= toks(al)
        ts.append(t)
    E = np.stack(E)
    for s0 in range(0, len(ids), 1000):
        s1 = min(s0 + 1000, len(ids))
        S = E[s0:s1] @ E.T
        for ri, gi in enumerate(range(s0, s1)):
            for gj in np.where(S[ri] >= COS_THR_MERGE)[0]:
                if gj <= gi:
                    continue
                if len(ts[gi] & ts[gj]) / max(1, len(ts[gi] | ts[gj])) >= JAC_THR:
                    uf.union(ids[gi], ids[gj])
    groups = defaultdict(list)
    for n in risk:
        groups[uf.find(n)].append(n)
    mapping = {}
    member_count = {}
    for root, mem in groups.items():
        canon = max(
            mem, key=lambda x: (len(na.get(x, {}).get("description") or ""), -x)
        )
        for m in mem:
            mapping[m] = canon
        member_count[canon] = len(mem)
    return mapping, member_count, len(risk), len(groups)


def build_A(na, ed, risk_map, exclude_risk_sim):
    """mapping applies to risk nodes; others identity. Returns (A_csr, nids, nid2idx)."""

    def cn(x):
        return risk_map.get(x, x)

    cat = {n: cat_key(a) for n, a in na.items()}
    # canonical node set
    nodes = set()
    for n in na:
        nodes.add(cn(n))
    nids = sorted(nodes)
    idx = {n: i for i, n in enumerate(nids)}
    seen = set()
    rows, cols = [], []
    edeg = defaultdict(int)  # EDGE-only degree on canonical graph
    for e in ed:
        et = (e.get("type") or "").upper()
        s, t = e.get("source"), e.get("target")
        if s is None or t is None:
            continue
        cs, ct = cn(s), cn(t)
        if cs == ct:
            continue
        if cs not in idx or ct not in idx:
            continue
        if et == "EDGE":
            key = (min(cs, ct), max(cs, ct), "E")
            if key in seen:
                continue
            seen.add(key)
            rows.append(idx[cs])
            cols.append(idx[ct])
            rows.append(idx[ct])
            cols.append(idx[cs])
            edeg[cs] += 1
            edeg[ct] += 1
        elif et == "SIMILARITY":
            sc = e.get("similarity_score")
            if sc is None or cos_from_score(sc) < SIM_THR:
                continue
            ccs, cct = cat.get(cs), cat.get(ct)
            if ccs != cct:
                continue
            if exclude_risk_sim and ccs == "risk" and cct == "risk":
                continue
            key = (min(cs, ct), max(cs, ct), "S")
            if key in seen:
                continue
            seen.add(key)
            rows.append(idx[cs])
            cols.append(idx[ct])
            rows.append(idx[ct])
            cols.append(idx[cs])
    A = sp.csr_matrix(
        (np.ones(len(rows), np.float32), (rows, cols)), shape=(len(nids), len(nids))
    )
    return A, nids, idx, edeg


def ec(A, max_iter=500, tol=1e-8):
    n = A.shape[0]
    x = np.ones(n) / np.sqrt(n)
    for it in range(max_iter):
        y = A @ x
        nrm = np.linalg.norm(y)
        if nrm == 0:
            break
        y /= nrm
        if np.linalg.norm(y - x) < tol:
            return x, it + 1
        x = y
    return x, max_iter


def top10(na, ecv, nids, idx, risk_map, member_count, label):
    inv_members = member_count
    risks = [
        (ecv[idx[n]], n)
        for n in nids
        if (na.get(n, {}).get("concept_category") or "").lower() == "risk"
    ]
    risks.sort(reverse=True)
    rows = []
    nxr = 0
    for val, n in risks[:10]:
        xr = is_xr(na.get(n, {}).get("name"))
        nxr += xr
        rows.append(
            {
                "nid": n,
                "ec": float(val),
                "xrisk": bool(xr),
                "merged_members": inv_members.get(n, 1),
                "name": na.get(n, {}).get("name", "")[:70],
            }
        )
    print(f"\n[{label}]  xrisk in top-10: {nxr}/10")
    for r in rows:
        tag = "XR" if r["xrisk"] else "  "
        print(f"  {tag} ec={r['ec']:.4f} members={r['merged_members']:>3}  {r['name']}")
    return {"label": label, "xrisk_in_top10": nxr, "top10": rows}


def main():
    t0 = time.time()
    na = pickle.load(open(STEP1 / "graph_node_attributes.pkl", "rb"))
    ed = pickle.load(open(STEP1 / "graph_edge_data.pkl", "rb"))
    print(f"{len(na)} nodes, {len(ed)} edges", flush=True)

    ident = {}  # un-merged
    print("\nmerging risk block...", flush=True)
    risk_map, member_count, n_risk, n_groups = merge_risk_block(na)
    biggest = max(member_count.items(), key=lambda kv: kv[1])
    print(
        f"  risk {n_risk} -> {n_groups} canonical; biggest merged node = {member_count[biggest[0]]} members "
        f"({na.get(biggest[0], {}).get('name', '')[:60]})",
        flush=True,
    )

    results = []
    for label, mp, excl in [
        ("1 un-merged, full SIM", ident, False),
        ("2 un-merged, risk<->risk SIM EXCLUDED", ident, True),
        ("3 merged(risk), full SIM", risk_map, False),
        ("4 merged(risk), risk<->risk SIM EXCLUDED", risk_map, True),
    ]:
        print(f"\n=== building {label} ===", flush=True)
        A, nids, idx, edeg = build_A(na, ed, mp, excl)
        x, iters = ec(A)
        print(f"  EC iters={iters}", flush=True)
        r = top10(na, x, nids, idx, mp, member_count, label)
        # report EDGE-only degree of biggest merged risk node (conditions 3/4)
        if mp:
            r["biggest_merged_node_edge_degree"] = int(edeg.get(biggest[0], 0))
            r["biggest_merged_node_members"] = int(member_count[biggest[0]])
            print(
                f"  biggest merged risk node EDGE-degree = {edeg.get(biggest[0], 0)} "
                f"(from {member_count[biggest[0]]} merged members)",
                flush=True,
            )
        results.append(r)

    OUT.write_text(
        json.dumps(
            {
                "conditions": results,
                "biggest_merged_members": member_count[biggest[0]],
                "wall_sec": round(time.time() - t0, 1),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT}\nwall {time.time() - t0:.1f}s\nDONE.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
