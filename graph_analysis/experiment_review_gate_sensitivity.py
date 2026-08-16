#!/usr/bin/env python
"""How much of the reporting unit is a function of the two LLM-assigned quality gates?

Reviewer item W-3 / Q4 (both simulated reviews, 2026-08-14): the 2,772-chain reporting
unit is selected by `intervention_maturity >= 3` and `edge_confidence >= 3`, both
LLM-assigned attributes that the judge protocol does not score. The reviews ask for the
headline descriptives recomputed at relaxed gates, and for the sensitivity of the chain
count to the 70% containment threshold used to reduce path enumeration to the reporting
unit (reviewer item Q-W9 / W-11), plus evidence that the containment step drops only
sub-paths that carry nothing new.

The enumeration is re-implemented in memory from the Step-1 checkpoints, with the same
semantics as phase2_step4_F2v4_hopwise_falkordb.py (the FalkorDB builder that produced the
released path set):

  - undirected adjacency over EDGE relations with edge_confidence >= C
  - one DFS per risk node; first hop must land on a body-subtype node
  - risk nodes are never revisited inside a path
  - the walk terminates on the first intervention with intervention_maturity >= M
  - emitted paths have >= 3 hops and <= 30 hops

The baseline (C=3, M=3) is asserted to reproduce the released 8,954-path file EXACTLY,
path for path, before any relaxed configuration is reported. Without that assertion a
sensitivity table would be measuring this script rather than the pipeline.

Class B (no LLM, no network). Run from graph_analysis/:
    python -u experiment_review_gate_sensitivity.py

Output: graph_analysis/phase2_results/experiment_review_gate_sensitivity_report.json
"""

import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
SLIM = ROOT / "phase2_results/node_attrs_slim.pkl"
EDGES = STEP1 / "graph_edge_data.pkl"
RAW = ROOT / "phase1_rawpathsfiles/paths_hopwise_v4_edge_only.jsonl"
DEDUP = ROOT / "phase1_rawpathsfiles/paths_hopwise_v4_edge_only_deduped.jsonl"
OUT = ROOT / "phase2_results/experiment_review_gate_sensitivity_report.json"

BODY = [
    "problem analysis",
    "theoretical insight",
    "design rationale",
    "implementation mechanism",
    "validation evidence",
]
MIN_HOPS, MAX_HOPS = 3, 30
PATH_CAP = (
    5_000_000  # safeguard; a config that hits it is reported, never truncated silently
)

HOST_TO_ARD = {
    "arxiv.org": "arxiv",
    "www.arxiv.org": "arxiv",
    "lesswrong.com": "lesswrong",
    "www.lesswrong.com": "lesswrong",
    "alignmentforum.org": "alignmentforum",
    "www.alignmentforum.org": "alignmentforum",
    "forum.effectivealtruism.org": "eaforum",
    "arbital.com": "arbital",
    "aisafety.info": "aisafety.info",
    "www.youtube.com": "youtube",
    "youtube.com": "youtube",
    "intelligence.org": "miri",
    "docs.google.com": "special_docs",
}


def ard_type(url):
    h = (urlparse(url).netloc or "").lower()
    return HOST_TO_ARD.get(h, "other_web")


def pct(n, d, nd=1):
    return round(100 * n / d, nd) if d else None


# ---------------------------------------------------------------------------------
# enumeration
# ---------------------------------------------------------------------------------
def build_adjacency(edge_rows, conf_min):
    adj = defaultdict(set)
    n = 0
    for e in edge_rows:
        if e.get("type") != "EDGE":
            continue
        c = e.get("confidence")
        if c is None or c < conf_min:
            continue
        s, t = e["source"], e["target"]
        if s == t:
            continue
        adj[s].add(t)
        adj[t].add(s)
        n += 1
    return adj, n


def enumerate_paths(adj, risk_nodes, body_nodes, terminal_nodes):
    """DFS with the semantics of phase2_step4_F2v4_hopwise_falkordb.py, EDGE-only mode."""
    out = []
    capped = False
    for R in sorted(risk_nodes):
        if capped:
            break
        visited = {R}
        parent = {R: None}

        stack_paths = []
        first_hops = [nb for nb in adj.get(R, ()) if nb in body_nodes]

        def dfs(curr, depth):
            nonlocal capped
            if capped:
                return
            if curr in terminal_nodes:
                path, cur = [], curr
                while cur is not None:
                    path.append(cur)
                    cur = parent.get(cur)
                path.reverse()
                if len(path) - 1 >= MIN_HOPS:
                    stack_paths.append(path)
                    if len(out) + len(stack_paths) >= PATH_CAP:
                        capped = True
                return
            if depth >= MAX_HOPS:
                return
            for nb in adj.get(curr, ()):
                if nb in visited or nb in risk_nodes:
                    continue
                visited.add(nb)
                parent[nb] = curr
                dfs(nb, depth + 1)
                visited.discard(nb)
                parent.pop(nb, None)
                if capped:
                    return

        for nb in first_hops:
            if capped:
                break
            visited.add(nb)
            parent[nb] = R
            dfs(nb, 1)
            visited.discard(nb)
            parent.pop(nb, None)
        out.extend(stack_paths)
    return out, capped


# ---------------------------------------------------------------------------------
# containment de-duplication (phase1_dedup_paths.py semantics)
# ---------------------------------------------------------------------------------
def dedupe(paths, url_of_path, threshold):
    by_url = defaultdict(list)
    for i, p in enumerate(paths):
        by_url[url_of_path[i]].append(i)
    keep, dropped_pairs = [], []
    for _url, idxs in by_url.items():
        order = sorted(idxs, key=lambda i: -len(paths[i]))
        kept_here = []
        for i in order:
            ns_i = frozenset(paths[i])
            if not ns_i:
                continue
            dup_of = None
            for j in kept_here:
                ns_j = frozenset(paths[j])
                small, large = (ns_i, ns_j) if len(ns_i) <= len(ns_j) else (ns_j, ns_i)
                if len(small & large) / len(small) >= threshold:
                    dup_of = j
                    break
            if dup_of is None:
                kept_here.append(i)
            else:
                dropped_pairs.append((i, dup_of))
        keep.extend(kept_here)
    return sorted(keep), dropped_pairs


# ---------------------------------------------------------------------------------
# descriptives
# ---------------------------------------------------------------------------------
def describe(paths, slim, label):
    n = len(paths)
    if n == 0:
        return {"label": label, "n_chains": 0}
    all5 = 0
    hist = Counter()
    per_url = Counter()
    ri = set()
    subtype_present = Counter()
    for p in paths:
        cats = {
            (slim.get(x, {}).get("concept_category") or "").lower() for x in p[1:-1]
        }
        present = cats & set(BODY)
        for st in present:
            subtype_present[st] += 1
        if len(present) == 5:
            all5 += 1
        hist[len(p)] += 1
        urls = {slim.get(x, {}).get("url") for x in p}
        urls.discard(None)
        if len(urls) == 1:
            per_url[next(iter(urls))] += 1
        ri.add((p[0], p[-1]))
    return {
        "label": label,
        "n_chains": n,
        "n_source_papers": len(per_url),
        "corpus_yield_pct": pct(len(per_url), 11779),
        "all5_pct": pct(all5, n),
        "pct_len_eq7": pct(hist[7], n),
        "pct_len_gt7": pct(sum(v for k, v in hist.items() if k > 7), n),
        "pct_len_lt7": pct(sum(v for k, v in hist.items() if k < 7), n),
        "len_min": min(hist),
        "len_max": max(hist),
        "max_chains_one_paper": per_url.most_common(1)[0][1] if per_url else 0,
        "distinct_ri_node_pairs": len(ri),
        "subtype_presence_pct": {st: pct(subtype_present[st], n) for st in BODY},
    }


def source_mix(paths, slim):
    c = Counter()
    for p in paths:
        urls = {slim.get(x, {}).get("url") for x in p}
        urls.discard(None)
        if len(urls) == 1:
            c[ard_type(next(iter(urls)))] += 1
    tot = sum(c.values())
    return {t: pct(v, tot) for t, v in sorted(c.items(), key=lambda kv: -kv[1])}


def main():
    t0 = time.time()
    sys.setrecursionlimit(50000)
    for p in (SLIM, EDGES, RAW, DEDUP):
        if not p.exists():
            raise SystemExit(
                f"FATAL: required input missing: {p}\n"
                "  node_attrs_slim.pkl comes from experiment_review_prep_slim_nodes.py;\n"
                "  the graph checkpoints and path files come from the Step-1 and\n"
                "  F2v4 hop-wise stages of the pipeline. No fallback path exists."
            )

    slim = pickle.load(open(SLIM, "rb"))
    edge_rows = pickle.load(open(EDGES, "rb"))
    print(f"loaded {len(slim):,} nodes, {len(edge_rows):,} edge rows", flush=True)

    risk_nodes = {
        n
        for n, a in slim.items()
        if (a.get("concept_category") or "").lower() == "risk"
    }
    body_nodes = {
        n
        for n, a in slim.items()
        if (a.get("concept_category") or "").lower() in set(BODY)
    }
    interventions = {
        n for n, a in slim.items() if (a.get("type") or "").lower() == "intervention"
    }
    by_mat = {
        m: {
            n for n in interventions if (slim[n].get("intervention_maturity") or 0) >= m
        }
        for m in (1, 2, 3)
    }
    url_of = {n: a.get("url") for n, a in slim.items()}

    # ---- baseline reproduction check --------------------------------------------
    released_raw = [json.loads(line)["path"] for line in open(RAW, encoding="utf-8")]
    released_dedup = [
        json.loads(line)["path"] for line in open(DEDUP, encoding="utf-8")
    ]

    adj3, n_edges3 = build_adjacency(edge_rows, 3)
    t1 = time.time()
    base_paths, base_capped = enumerate_paths(adj3, risk_nodes, body_nodes, by_mat[3])
    print(
        f"baseline enumeration: {len(base_paths):,} paths in {time.time() - t1:.0f}s",
        flush=True,
    )
    same_multiset = Counter(map(tuple, base_paths)) == Counter(map(tuple, released_raw))

    def url_list(paths):
        out = []
        for p in paths:
            urls = {url_of.get(x) for x in p}
            urls.discard(None)
            out.append(next(iter(urls)) if len(urls) == 1 else f"_mixed_{len(out)}")
        return out

    base_urls = url_list(base_paths)
    keep70, dropped70 = dedupe(base_paths, base_urls, 0.70)
    base_dedup = [base_paths[i] for i in keep70]
    dedup_matches = Counter(map(tuple, base_dedup)) == Counter(
        map(tuple, released_dedup)
    )

    reproduction = {
        "released_raw_paths": len(released_raw),
        "reenumerated_raw_paths": len(base_paths),
        "raw_path_multisets_identical": same_multiset,
        "released_deduped_paths": len(released_dedup),
        "reenumerated_deduped_paths": len(base_dedup),
        "deduped_path_multisets_identical": dedup_matches,
        "edge_edges_after_conf3_filter": n_edges3,
        "hit_path_cap": base_capped,
    }
    print(json.dumps(reproduction, indent=1), flush=True)
    if not same_multiset:
        raise SystemExit(
            "FATAL: the in-memory re-enumeration does not reproduce the released "
            "8,954-path file exactly. Every sensitivity number below would then be a "
            "property of this script rather than of the pipeline. Fix the semantics "
            "before reporting anything."
        )

    # ---- gate grid ---------------------------------------------------------------
    grid = {}
    for conf in (3, 2, 1):
        adj, n_e = build_adjacency(edge_rows, conf) if conf != 3 else (adj3, n_edges3)
        for mat in (3, 2, 1):
            key = f"conf>={conf}, maturity>={mat}"
            t2 = time.time()
            paths, capped = enumerate_paths(adj, risk_nodes, body_nodes, by_mat[mat])
            urls = url_list(paths)
            keep, _ = dedupe(paths, urls, 0.70)
            ded = [paths[i] for i in keep]
            d = describe(ded, slim, key)
            d["source_type_mix_pct"] = source_mix(ded, slim)
            grid[key] = {
                **d,
                "raw_paths_before_containment_dedup": len(paths),
                "edge_edges_in_adjacency": n_e,
                "n_terminal_interventions": len(by_mat[mat]),
                "hit_path_cap": capped,
                "seconds": round(time.time() - t2, 1),
            }
            print(
                f"  {key:26s} raw={len(paths):7,} deduped={len(ded):6,} "
                f"papers={d['n_source_papers']:5,} all5={d['all5_pct']}% "
                f"({time.time() - t2:.0f}s)",
                flush=True,
            )

    # ---- containment threshold sensitivity ---------------------------------------
    containment = {}
    for th in (0.60, 0.70, 0.80, 0.90, 1.00):
        keep, dropped = dedupe(base_paths, base_urls, th)
        ded = [base_paths[i] for i in keep]
        d = describe(ded, slim, f"containment {th:.2f}")
        containment[f"{th:.2f}"] = {
            "n_chains": d["n_chains"],
            "n_source_papers": d["n_source_papers"],
            "all5_pct": d["all5_pct"],
            "pct_len_eq7": d["pct_len_eq7"],
            "max_chains_one_paper": d["max_chains_one_paper"],
            "distinct_ri_node_pairs": d["distinct_ri_node_pairs"],
        }
        print(f"  containment {th:.2f}: {d['n_chains']} chains", flush=True)

    # ---- is the containment step lossless? ---------------------------------------
    # For every dropped path, count the nodes it carries that its container does not.
    novel_hist = Counter()
    examples = []
    for i, j in dropped70:
        ns_i, ns_j = set(base_paths[i]), set(base_paths[j])
        novel = ns_i - ns_j
        novel_hist[len(novel)] += 1
        if novel and len(examples) < 15:
            examples.append(
                {
                    "dropped_path_len": len(base_paths[i]),
                    "container_path_len": len(base_paths[j]),
                    "n_nodes_not_in_container": len(novel),
                    "novel_node_names": [
                        slim.get(x, {}).get("name") for x in sorted(novel)
                    ][:4],
                    "novel_node_subtypes": [
                        slim.get(x, {}).get("concept_category") for x in sorted(novel)
                    ][:4],
                }
            )
    n_dropped = len(dropped70)
    n_with_novel = sum(v for k, v in novel_hist.items() if k > 0)
    # how many of the dropped paths' novel nodes survive somewhere in the kept set
    kept_nodes = set()
    for i in keep70:
        kept_nodes.update(base_paths[i])
    lost_nodes = set()
    for i, _ in dropped70:
        lost_nodes.update(set(base_paths[i]) - kept_nodes)

    all_raw_nodes = set()
    for p in base_paths:
        all_raw_nodes.update(p)
    lost_subtypes = Counter(
        (slim.get(x, {}).get("concept_category") or slim.get(x, {}).get("type") or "?")
        for x in lost_nodes
    )
    lost_papers = {url_of.get(x) for x in lost_nodes}

    containment_losslessness = {
        "n_paths_dropped_at_0.70": n_dropped,
        "n_distinct_nodes_in_the_raw_8954_path_set": len(all_raw_nodes),
        "pct_of_raw_chain_set_nodes_lost": pct(len(lost_nodes), len(all_raw_nodes), 2),
        "lost_node_subtypes": dict(lost_subtypes.most_common()),
        "n_papers_with_at_least_one_lost_node": len(lost_papers),
        "n_dropped_paths_carrying_a_node_absent_from_their_container": n_with_novel,
        "pct_dropped_paths_carrying_a_novel_node": pct(n_with_novel, n_dropped),
        "novel_node_count_histogram": dict(sorted(novel_hist.items())),
        "mean_novel_nodes_per_dropped_path": round(
            sum(k * v for k, v in novel_hist.items()) / n_dropped, 2
        )
        if n_dropped
        else None,
        "n_nodes_present_in_a_dropped_path_and_in_no_kept_path": len(lost_nodes),
        "examples": examples,
        "INTERPRETATION": (
            "A dropped path is 'lossless' only if the argument it carries survives in the "
            "kept set. Two measurements: how often a dropped path carries a node its own "
            "container lacks, and how many nodes disappear from the chain set entirely. "
            "The second is the one that bounds information loss, because a node absent "
            "from one container may still be covered by another kept chain of the same "
            "paper."
        ),
    }

    out = {
        "experiment": "quality-gate and containment-threshold sensitivity (W-3/Q4, Q-W9/W-11)",
        "method": (
            "In-memory re-implementation of the F2v4 hop-wise EDGE-only enumeration, "
            "asserted to reproduce the released 8,954-path set exactly before any "
            "relaxed gate is reported."
        ),
        "deployed_configuration": "edge_confidence >= 3, intervention_maturity >= 3, containment 0.70",
        "baseline_reproduction_check": reproduction,
        "gate_grid": grid,
        "containment_threshold_sensitivity": containment,
        "containment_losslessness": containment_losslessness,
        "intervention_counts_by_maturity_floor": {
            f">={m}": len(v) for m, v in by_mat.items()
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
