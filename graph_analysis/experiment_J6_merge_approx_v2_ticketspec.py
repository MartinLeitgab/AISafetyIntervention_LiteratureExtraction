"""experiment_J6_merge_approx_v2_ticketspec.py

Re-run of the Gleb merge approximation incorporating the FULLER spec that GitHub
Issue #139 (Stage-1 comment, 2025-12-01) documents but the paper text omits:

  (T1) Phase 0 exact-match is ALIAS-BASED: union nodes where one node's name
       equals another node's alias (`n2.name in n1.aliases`), not just identical
       normalized names. (prev script used normalized-name-only Phase 0)
  (T2) The cos+Jaccard candidate Jaccard is computed over NAME + ALIASES tokens,
       not name tokens only. (prev script used name tokens only)
  (T3) Candidate search is FAISS IndexIVFFlat with 50 nearest neighbors per query
       -> approximate, capped at ~50 candidates/node. We approximate the *cap*
       (not the IVF recall loss) by keeping each node's top-K cosine neighbors.

Blocking (already matched prev script): concepts by (type, concept_category),
interventions by (type,) only. category_key() encodes this.

Variants reported:
  V0  name-only Phase0 + name-only Jaccard + exhaustive cos>=0.88   (== prev report)
  V1  alias Phase0   + name+alias Jaccard  + exhaustive cos>=0.88   (T1+T2)
  V2  alias Phase0   + name+alias Jaccard  + top-50 cap per node    (T1+T2+T3)

Gleb reference: 4,411 candidate pairs -> 2,385 nodes removed -> 200,061 canonical;
169,083 within-cat SIM edges
J.6 isolation 51/100.

Class B (no LLM). Run from graph_analysis/:
    python -u experiment_J6_merge_approx_v2_ticketspec.py
"""

from __future__ import annotations
import json
import pickle
import re
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
OUT_JSON = ROOT / "phase2_results/experiment_J6_merge_approx_v2_report.json"

COS_THR_MERGE = 0.88
JACCARD_THR_MERGE = 0.05
SIM_THR_GRAPH = 0.80
SIM_HOP_CAP = 2
TOP_N_RISKS = 100
RACE_KEYWORDS = ["race", "racing", "competi"]
BATCH = 1000
TOPK_NN = 50  # T3: FAISS "50 nearest neighbors per query"

_TOK = re.compile(r"[a-z0-9]+")


def tokset(s):
    return set(_TOK.findall((s or "").lower()))


def parse_aliases(raw):
    """Aliases stored as bracketed comma-joined string: '[a, b, c]'. Return list[str]."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    s = str(raw).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if not s.strip():
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def norm_name(s):
    return (s or "").lower().strip()


def jaccard(a, b):
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def cos_sim_from_score(s):
    return 1.0 - float(s) ** 2 / 2.0


def is_race_name(name):
    n = (name or "").lower()
    return any(kw in n for kw in RACE_KEYWORDS)


def category_key(attrs):
    if (attrs.get("type") or "").lower() == "intervention":
        return "intervention"
    cc = (attrs.get("concept_category") or "").lower().strip()
    return cc if cc else None


def load_pkls():
    print("loading node_attrs ...", flush=True)
    with open(STEP1 / "graph_node_attributes.pkl", "rb") as f:
        na = pickle.load(f)
    print(f"  {len(na)} nodes", flush=True)
    print("loading edge_data ...", flush=True)
    with open(STEP1 / "graph_edge_data.pkl", "rb") as f:
        ed = pickle.load(f)
    print(f"  {len(ed)} edges", flush=True)
    return na, ed


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        if x not in self.p:
            self.p[x] = x
            return x
        root = x
        while self.p[root] != root:
            root = self.p[root]
        while self.p[x] != root:
            self.p[x], x = root, self.p[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def provenance_diagnostic(na):
    print("\n" + "=" * 70)
    print("PROVENANCE DIAGNOSTIC (is our PKL un-merged? does it carry Gleb cols?)")
    print("=" * 70)
    n = len(na)
    # embedding dim
    dims = set()
    n_emb = 0
    for a in na.values():
        e = a.get("embedding")
        if isinstance(e, str) and e.startswith("<") and e.endswith(">"):
            try:
                v = np.fromstring(e[1:-1], sep=",", dtype=np.float32)
                dims.add(v.shape[0])
                n_emb += 1
            except Exception:
                pass
        if n_emb >= 2000:
            break
    n_semcl = sum(1 for a in na.values() if a.get("semantic_cluster") is not None)
    semvals = set()
    for a in na.values():
        sc = a.get("semantic_cluster")
        if sc is not None:
            semvals.add(sc)
    n_pr = sum(1 for a in na.values() if a.get("pagerank") is not None)
    n_umap = sum(1 for a in na.values() if a.get("embedding_umap_150d") is not None)
    # exact-name dup count within category (evidence of un-merged duplicates)
    name_groups = defaultdict(int)
    for a in na.values():
        ck = category_key(a)
        if ck is None:
            continue
        nm = norm_name(a.get("name"))
        if nm:
            name_groups[(ck, nm)] += 1
    n_exact_dup_nodes = sum(c - 1 for c in name_groups.values() if c > 1)
    n_exact_dup_groups = sum(1 for c in name_groups.values() if c > 1)
    diag = {
        "n_nodes": n,
        "embedding_dims_seen": sorted(dims),
        "n_with_semantic_cluster": n_semcl,
        "n_distinct_semantic_cluster_values": len(semvals),
        "semantic_cluster_value_range": [min(semvals), max(semvals)]
        if semvals
        else None,
        "n_with_pagerank": n_pr,
        "n_with_embedding_umap_150d": n_umap,
        "n_exact_name_dup_groups_within_category": n_exact_dup_groups,
        "n_exact_name_dup_nodes_within_category": n_exact_dup_nodes,
        "gleb_raw_nodes": 202446,
        "gleb_merged_nodes": 200061,
    }
    for k, v in diag.items():
        print(f"  {k}: {v}", flush=True)
    return diag


# ---------- Phase 0 ----------
def phase0_name_only(na):
    """V0 Phase 0: union by identical normalized name within category."""
    uf = UF()
    name_to_nids = defaultdict(list)
    for nid, a in na.items():
        ck = category_key(a)
        if ck is None:
            continue
        nm = norm_name(a.get("name"))
        if not nm:
            continue
        name_to_nids[(ck, nm)].append(nid)
    for nids in name_to_nids.values():
        for i in range(1, len(nids)):
            uf.union(nids[0], nids[i])
    return uf


def phase0_alias_based(na):
    """V1/V2 Phase 0: union by identical name OR name==another's alias, within category."""
    uf = UF()
    name_index = defaultdict(list)  # (cat, norm_name) -> [nids]
    for nid, a in na.items():
        ck = category_key(a)
        if ck is None:
            continue
        nm = norm_name(a.get("name"))
        if nm:
            name_index[(ck, nm)].append(nid)
    # exact-name unions
    for nids in name_index.values():
        for i in range(1, len(nids)):
            uf.union(nids[0], nids[i])
    # alias unions: n's alias == m's name -> union(n, m)
    n_alias_unions = 0
    for nid, a in na.items():
        ck = category_key(a)
        if ck is None:
            continue
        for al in parse_aliases(a.get("aliases")):
            key = (ck, norm_name(al))
            if key in name_index:
                for m in name_index[key]:
                    if m != nid:
                        uf.union(nid, m)
                        n_alias_unions += 1
    print(f"  phase0 alias-based: {n_alias_unions} alias->name union ops", flush=True)
    return uf


def canonicals_from_uf(uf, na):
    groups = defaultdict(list)
    for nid in list(uf.p.keys()):
        groups[uf.find(nid)].append(nid)
    mapping = {}
    n_collapsed = 0
    for root, members in groups.items():
        if len(members) == 1:
            mapping[members[0]] = members[0]
            continue
        best = max(
            members, key=lambda x: (len(na.get(x, {}).get("description") or ""), -x)
        )
        for m in members:
            mapping[m] = best
        n_collapsed += len(members) - 1
    for nid in na:
        if nid not in mapping:
            mapping[nid] = nid
    return mapping, n_collapsed


# ---------- embeddings / blocks ----------
def build_blocks(na, only_nids, use_alias_jaccard):
    blocks = defaultdict(list)
    skipped = 0
    for nid in only_nids:
        a = na.get(nid)
        if a is None:
            skipped += 1
            continue
        emb = a.get("embedding")
        if isinstance(emb, str):
            s = emb.strip()
            if not (s.startswith("<") and s.endswith(">")):
                skipped += 1
                continue
            try:
                v = np.fromstring(s[1:-1], sep=",", dtype=np.float32)
            except Exception:
                skipped += 1
                continue
        else:
            try:
                v = np.asarray(emb, dtype=np.float32)
            except Exception:
                skipped += 1
                continue
        if v.ndim != 1 or v.shape[0] != 1536:
            skipped += 1
            continue
        norm = np.linalg.norm(v)
        if norm == 0 or not np.isfinite(norm):
            skipped += 1
            continue
        v = v / norm
        ck = category_key(a)
        if ck is None:
            skipped += 1
            continue
        name = a.get("name") or ""
        if use_alias_jaccard:
            ts = tokset(name)
            for al in parse_aliases(a.get("aliases")):
                ts |= tokset(al)
        else:
            ts = tokset(name)
        blocks[ck].append((nid, v, ts))
    print(f"  blocks built; skipped(no emb)={skipped}", flush=True)
    return blocks


def find_pairs(block_items, topk=None):
    """Return (pairs, n_cos_pass, n_jac_pass). topk=None -> exhaustive; else cap per node."""
    n = len(block_items)
    if n < 2:
        return [], 0, 0
    E = np.stack([it[1] for it in block_items])
    nids = [it[0] for it in block_items]
    toks = [it[2] for it in block_items]
    pairset = set()
    n_cos = 0
    n_jac = 0
    for i_start in range(0, n, BATCH):
        i_end = min(i_start + BATCH, n)
        S = E[i_start:i_end] @ E.T  # (rows, n)
        for ri, gi in enumerate(range(i_start, i_end)):
            row = S[ri]
            cand = np.where(row >= COS_THR_MERGE)[0]
            cand = cand[cand != gi]
            if topk is not None and cand.size > topk:
                # keep top-k by cosine
                order = np.argsort(row[cand])[::-1][:topk]
                cand = cand[order]
            for gj in cand:
                a, b = (gi, int(gj)) if gi < gj else (int(gj), gi)
                if a == b:
                    continue
                key = (a, b)
                if key in pairset:
                    continue
                n_cos += 1
                if jaccard(toks[a], toks[b]) >= JACCARD_THR_MERGE:
                    n_jac += 1
                    pairset.add(key)
    pairs = [(nids[a], nids[b]) for (a, b) in pairset]
    return pairs, n_cos, n_jac


def run_variant(label, na, p0_mapping, use_alias_jaccard, topk):
    print("\n" + "=" * 70)
    print(f"VARIANT {label}")
    print("=" * 70)
    p0_canon = set(p0_mapping.values())
    n_p0 = sum(1 for nid, c in p0_mapping.items() if nid != c)
    print(
        f"  Phase0 collapsed: {n_p0}; canonicals after P0: {len(p0_canon)}", flush=True
    )
    blocks = build_blocks(na, p0_canon, use_alias_jaccard)
    uf = UF()
    tot_cos = 0
    tot_jac = 0
    for ck, items in blocks.items():
        t1 = time.time()
        pairs, nc, nj = find_pairs(items, topk=topk)
        tot_cos += nc
        tot_jac += nj
        for a, b in pairs:
            uf.union(a, b)
        print(
            f"    block {ck!r} n={len(items)}: cos_pass={nc}, both_pass={nj} ({time.time() - t1:.1f}s)",
            flush=True,
        )
    p2_mapping, n_p2 = canonicals_from_uf(uf, na)
    # compose
    mapping = {}
    for nid in na:
        mapping[nid] = p2_mapping.get(
            p0_mapping.get(nid, nid), p0_mapping.get(nid, nid)
        )
    n_total = sum(1 for nid, c in mapping.items() if nid != c)
    n_canon = len(set(mapping.values()))
    print(f"  Phase2 candidate pairs (both pass): {tot_jac}  (Gleb: 4,411)", flush=True)
    print(f"  Phase2 nodes collapsed: {n_p2}", flush=True)
    print(f"  TOTAL nodes collapsed (P0+P2): {n_total}  (Gleb: 2,385)", flush=True)
    print(f"  canonical nodes remaining: {n_canon}  (Gleb: 200,061)", flush=True)
    return {
        "label": label,
        "phase0_collapsed": n_p0,
        "phase2_cos_pass": tot_cos,
        "phase2_both_pass": tot_jac,
        "phase2_nodes_collapsed": n_p2,
        "total_nodes_collapsed": n_total,
        "canonical_nodes": n_canon,
    }, mapping


# ---------- merged-graph + J.6 (only for chosen variant) ----------
def build_merged_graph(na, ed, mapping):
    cat_cache = {n: category_key(a) for n, a in na.items()}
    struct_adj = defaultdict(set)
    sim_adj = defaultdict(set)
    n_struct = n_sim = 0
    for e in ed:
        et = (e.get("type") or "").upper()
        s, t = e.get("source"), e.get("target")
        if s is None or t is None:
            continue
        cs, ct = mapping.get(s, s), mapping.get(t, t)
        if cs == ct:
            continue
        if et == "EDGE":
            if ct not in struct_adj[cs]:
                struct_adj[cs].add(ct)
                struct_adj[ct].add(cs)
                n_struct += 1
        elif et == "SIMILARITY":
            sc = e.get("similarity_score")
            if sc is None or cos_sim_from_score(sc) < SIM_THR_GRAPH:
                continue
            if cat_cache.get(cs) != cat_cache.get(ct):
                continue
            if ct not in sim_adj[cs]:
                sim_adj[cs].add(ct)
                sim_adj[ct].add(cs)
                n_sim += 1
    merged = set(mapping.values())
    interv = {
        n for n in merged if (na.get(n, {}).get("type") or "").lower() == "intervention"
    }
    all_pa = {
        n
        for n in merged
        if (na.get(n, {}).get("concept_category") or "").lower() == "problem analysis"
    }
    race_pa = {n for n in all_pa if is_race_name(na.get(n, {}).get("name", ""))}
    return struct_adj, sim_adj, interv, all_pa, race_pa, merged, n_struct, n_sim


def ec(struct_adj, sim_adj, nids, max_iter=300, tol=1e-7):
    import scipy.sparse as sp

    idx = {n: i for i, n in enumerate(nids)}
    rows, cols = [], []
    for adj in (struct_adj, sim_adj):
        for s, nbrs in adj.items():
            if s not in idx:
                continue
            for t in nbrs:
                if t in idx:
                    rows.append(idx[s])
                    cols.append(idx[t])
    A = sp.csr_matrix(
        (np.ones(len(rows), np.float32), (rows, cols)), shape=(len(nids), len(nids))
    )
    x = np.ones(len(nids)) / np.sqrt(len(nids))
    for it in range(max_iter):
        y = A @ x
        nrm = np.linalg.norm(y)
        if nrm == 0:
            break
        y /= nrm
        if np.linalg.norm(y - x) < tol:
            print(f"  EC converged {it + 1} iter", flush=True)
            return x, idx
        x = y
    print("  EC max-iter", flush=True)
    return x, idx


def bfs_reach(start, struct_adj, sim_adj, interv, removed, cap):
    if start in removed:
        return False
    seen = {(start, 0)}
    q = deque()
    for nb in struct_adj.get(start, ()):
        if nb in removed:
            continue
        if nb in interv:
            return True
        st = (nb, 0)
        if st not in seen:
            seen.add(st)
            q.append(st)
    while q:
        node, sh = q.popleft()
        for nb in struct_adj.get(node, ()):
            if nb in removed:
                continue
            if nb in interv:
                return True
            st = (nb, sh)
            if st not in seen:
                seen.add(st)
                q.append(st)
        if sh < cap:
            for nb in sim_adj.get(node, ()):
                if nb in removed:
                    continue
                if nb in interv:
                    return True
                st = (nb, sh + 1)
                if st not in seen:
                    seen.add(st)
                    q.append(st)
    return False


def run_j6(na, ed, mapping, label):
    print("\n" + "=" * 70)
    print(f"J.6 isolation on merged graph ({label})")
    print("=" * 70)
    struct_adj, sim_adj, interv, all_pa, race_pa, merged, n_struct, n_sim = (
        build_merged_graph(na, ed, mapping)
    )
    print(
        f"  merged nodes={len(merged)} struct={n_struct} within-cat-sim={n_sim} (Gleb sim 169,083)",
        flush=True,
    )
    print(f"  PA={len(all_pa)} race-PA={len(race_pa)} interv={len(interv)}", flush=True)
    nids = sorted(merged)
    ecv, idx = ec(struct_adj, sim_adj, nids)
    risks = [
        (ecv[idx[n]], n)
        for n in nids
        if (na.get(n, {}).get("concept_category") or "").lower() == "risk"
    ]
    risks.sort(reverse=True)
    top100 = [n for _, n in risks[:TOP_N_RISKS]]
    base = iso = 0
    examples = []
    for n in top100:
        if bfs_reach(n, struct_adj, sim_adj, interv, frozenset(), SIM_HOP_CAP):
            base += 1
            if not bfs_reach(n, struct_adj, sim_adj, interv, race_pa, SIM_HOP_CAP):
                iso += 1
                if len(examples) < 10:
                    examples.append({"nid": n, "name": na.get(n, {}).get("name", "")})
    print(
        f"  baseline reachable {base}/100; isolated by race-PA removal {iso} (Gleb 51)",
        flush=True,
    )
    return {
        "merged_nodes": len(merged),
        "merged_struct_edges": n_struct,
        "merged_within_cat_sim_edges": n_sim,
        "n_pa": len(all_pa),
        "n_race_pa": len(race_pa),
        "baseline_reachable": base,
        "isolated_after_race_removal": iso,
        "isolated_examples": examples,
    }


def main():
    t0 = time.time()
    na, ed = load_pkls()
    diag = provenance_diagnostic(na)

    print("\nbuilding Phase-0 mappings ...", flush=True)
    p0_name = phase0_name_only(na)
    p0_name_map, _ = canonicals_from_uf(p0_name, na)
    p0_alias = phase0_alias_based(na)
    p0_alias_map, n_p0_alias = canonicals_from_uf(p0_alias, na)
    print(
        f"  Phase0 name-only collapsed={sum(1 for k, v in p0_name_map.items() if k != v)}",
        flush=True,
    )
    print(f"  Phase0 alias-based collapsed={n_p0_alias}", flush=True)

    v0, _ = run_variant(
        "V0 name-only P0 + name Jaccard + exhaustive", na, p0_name_map, False, None
    )
    v1, map_v1 = run_variant(
        "V1 alias P0 + name+alias Jaccard + exhaustive (T1+T2)",
        na,
        p0_alias_map,
        True,
        None,
    )
    v2, map_v2 = run_variant(
        "V2 alias P0 + name+alias Jaccard + top-50 cap (T1+T2+T3)",
        na,
        p0_alias_map,
        True,
        TOPK_NN,
    )

    # J.6 on V2 (closest-to-Gleb merge expected)
    j6_v2 = run_j6(na, ed, map_v2, "V2")

    out = {
        "experiment": "J.6 merge-approx v2 — ticket-spec (alias Phase0 + name+alias Jaccard + 50-NN cap)",
        "gleb_reference": {
            "candidate_pairs": 4411,
            "nodes_removed": 2385,
            "canonical_nodes": 200061,
            "within_cat_sim_edges": 169083,
            "j6_isolated": 51,
        },
        "provenance_diagnostic": diag,
        "variants": {"V0_prev": v0, "V1_T1_T2": v1, "V2_T1_T2_T3": v2},
        "j6_on_V2": j6_v2,
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}", flush=True)
    print(f"total wall {time.time() - t0:.1f}s\nDONE.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
