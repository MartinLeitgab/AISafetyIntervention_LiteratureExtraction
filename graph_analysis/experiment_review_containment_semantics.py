#!/usr/bin/env python
"""What exactly does the 70% sub-path collapse drop, in terms a reader can check?

Reviewers of the 2026-08-15 round raised the same objection twice: the containment rule
compares NODE SETS, so it is blind to edge identity, to order, and to whether the dropped
traversal is an argument the kept one does not carry. The paper answers with two numbers
(78.3% of dropped paths carry a novel node; 6.1% of chain-set nodes survive in no kept
chain) and concedes the step is not lossless. Neither number says whether a dropped path
was already traced by its container.

This script re-implements the collapse exactly as phase1_dedup_paths.py ran it -- greedy,
longest-first, threshold 0.70, within source paper -- asserts it reproduces the released
2,772/8,954 split before reporting anything, then records for every dropped path WHICH kept
path displaced it and how the two relate:

  * node-set relation      -- is the dropped path's node set a subset of its container's?
  * order                  -- is the dropped node sequence a contiguous sub-path of the
                              container, a non-contiguous subsequence, or neither?
  * edge identity          -- is every consecutive pair the dropped path traverses also
                              consecutive in the container? If not, the dropped path walks
                              a relation the container never walks, which is precisely what
                              a node-set rule cannot see.
  * relation semantics     -- the edge subtypes on those unmatched hops, and whether the
                              released graph carries parallel edges (same node pair, two
                              relation types) at all.
  * stage coverage         -- do the dropped path's novel nodes introduce a logical-chain
                              stage the container lacks?

Class B (no LLM call, no network). Run from graph_analysis/:

    python -u experiment_review_containment_semantics.py

Output: graph_analysis/phase2_results/experiment_review_containment_semantics_report.json

What this cannot do: decide whether two traversals are the same ARGUMENT. That is an
annotation question (OPEN_ITEMS.md S4/R23). It bounds the question from below -- a dropped
path that is a contiguous sub-path of its container, over the same edges, is a restatement
by construction, and the rest are where an annotator would have to look.
"""

import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
SLIM = ROOT / "phase2_results/node_attrs_slim.pkl"
EDGES = (
    ROOT
    / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites/graph_edge_data.pkl"
)
RAW = ROOT / "phase1_rawpathsfiles/paths_hopwise_v4_edge_only.jsonl"
DEDUPED = ROOT / "phase1_rawpathsfiles/paths_hopwise_v4_edge_only_deduped.jsonl"
OUT = ROOT / "phase2_results/experiment_review_containment_semantics_report.json"

CONTAINMENT_THRESHOLD = (
    0.70  # phase1_dedup_paths.py: the code uses 0.70, docstring says 80%
)
EXPECTED_RAW, EXPECTED_KEPT = 8954, 2772

STAGES = (
    "risk",
    "problem analysis",
    "theoretical insight",
    "design rationale",
    "implementation mechanism",
    "validation evidence",
)


def pct(n, d, nd=1):
    return round(100 * n / d, nd) if d else None


def load_paths(fp):
    out = []
    with open(fp, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d["path_id"] = f"path_{i:05d}"
            out.append(d)
    return out


def is_contiguous_subpath(small, large):
    n = len(small)
    return any(large[i : i + n] == small for i in range(len(large) - n + 1))


def is_subsequence(small, large):
    it = iter(large)
    return all(any(x == y for y in it) for x in small)


def main():
    for p, how in (
        (SLIM, "run experiment_review_prep_slim_nodes.py"),
        (EDGES, "run phase2_step1_loadandparse.py against the FalkorDB dump"),
        (RAW, "the released 8,954-chain path file ships with the repo"),
        (DEDUPED, "the released 2,772-chain path file ships with the repo"),
    ):
        if not p.exists():
            raise SystemExit(f"FATAL: {p} not found. To produce it: {how}.")

    slim = pickle.load(open(SLIM, "rb"))
    url_of = {int(n): r.get("url") for n, r in slim.items()}
    cat_of = {
        int(n): (
            r.get("concept_category") if r.get("type") == "concept" else "intervention"
        )
        for n, r in slim.items()
    }

    edges = pickle.load(open(EDGES, "rb"))
    # Hop-wise enumeration walks structural edges without regard to direction: 100% of the
    # hops in both released path files match an edge as an unordered (source, target) pair,
    # so pairs are keyed unordered here. Keying them directed would report every reversed hop
    # as an edge the graph does not contain.
    # CORRECTED 2026-08-17. This comment used to say 94.1% of hops match a DIRECTED
    # source -> target edge. That figure is not reproducible and must not be cited -- it also
    # reached the body of GitHub issue #157. experiment_review_chain_order_semantics.py
    # measures the quantity properly and the manuscript now quotes it at instance level:
    # 92.1% of the raw set's 62,923 hops and 95.5% of the reporting unit's 17,829 are stored
    # in the direction the chain walks them. The nearest figure to 94.1% is 94.6%, which is
    # the same measure over DISTINCT hop pairs of the raw file rather than hop instances;
    # if that is what 94.1% was, it was measuring pairs and reporting them as hops.
    subtypes_of_pair = defaultdict(set)
    for e in edges:
        if e.get("type") != "EDGE":
            continue
        subtypes_of_pair[frozenset((e["source"], e["target"]))].add(e.get("subtype"))
    parallel_pairs = sum(1 for v in subtypes_of_pair.values() if len(v) > 1)

    raw = load_paths(RAW)
    deduped = load_paths(DEDUPED)
    if len(raw) != EXPECTED_RAW:
        raise SystemExit(
            f"FATAL: raw path file has {len(raw)}, expected {EXPECTED_RAW}"
        )

    # ---- re-implement the collapse, recording the container of every drop ---------------
    by_url = defaultdict(list)
    for p in raw:
        urls = {url_of.get(int(n)) for n in p["path"]}
        urls.discard(None)
        if len(urls) == 1:
            by_url[next(iter(urls))].append(p)
        elif not urls:
            by_url[f"_orphan_{p['path_id']}"].append(p)
        else:
            by_url[f"_mixed_{p['path_id']}"].append(p)

    keep, drops = set(), []
    for _url, plist in by_url.items():
        ordered = sorted(plist, key=lambda p: -len(p["path"]))
        sets = [(p["path_id"], frozenset(p["path"])) for p in ordered]
        for i, (pid_i, ns_i) in enumerate(sets):
            if not ns_i:
                drops.append((pid_i, None))
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
            else:
                drops.append((pid_i, container))

    if len(keep) != EXPECTED_KEPT:
        raise SystemExit(
            f"FATAL: re-implementation kept {len(keep)}, released file has {EXPECTED_KEPT}. "
            "The procedure below does not reproduce the released substrate; nothing here "
            "is reportable until it does."
        )
    released_ids = {tuple(p["path"]) for p in deduped}
    kept_ids = {tuple(p["path"]) for p in raw if p["path_id"] in keep}
    if released_ids != kept_ids:
        raise SystemExit(
            "FATAL: re-implementation keeps a different SET of chains than the released "
            f"file ({len(released_ids ^ kept_ids)} symmetric-difference members)."
        )

    path_of = {p["path_id"]: p["path"] for p in raw}

    # ---- characterise every drop --------------------------------------------------------
    rel = Counter()
    order = Counter()
    edge_id = Counter()
    endpoint = Counter()
    unmatched_hop_subtypes = Counter()
    novel_stage = Counter()
    shared_prefix_len = Counter()
    n_novel_node = 0
    empty_drops = 0

    for pid, cid in drops:
        if cid is None:
            empty_drops += 1
            continue
        d, c = path_of[pid], path_of[cid]
        ds, cs = set(d), set(c)

        # Does the dropped chain propose an intervention the kept chain never reaches?
        # This is the reviewers' question in the form the data can answer: a chain ending
        # at a different remedy is a different risk-to-intervention claim, whatever the
        # node-set overlap.
        endpoint[
            "ends_at_an_intervention_absent_from_the_container"
            if d[-1] not in cs
            else "ends_at_an_intervention_the_container_also_contains"
        ] += 1

        k = 0
        for x, y in zip(d, c):
            if x != y:
                break
            k += 1
        shared_prefix_len[k] += 1

        subset = ds <= cs
        rel[
            "node_set_is_subset_of_container" if subset else "partial_overlap_only"
        ] += 1
        if not subset:
            n_novel_node += 1
            novel = ds - cs
            cont_stages = {cat_of.get(int(n)) for n in c}
            if any(cat_of.get(int(n)) not in cont_stages for n in novel):
                novel_stage["novel_node_introduces_a_stage_the_container_lacks"] += 1
            else:
                novel_stage[
                    "novel_nodes_are_all_in_stages_the_container_already_has"
                ] += 1

        if is_contiguous_subpath(d, c):
            order["contiguous_sub_path_of_container"] += 1
        elif is_subsequence(d, c):
            order["non_contiguous_subsequence"] += 1
        else:
            order["not_a_subsequence_reordered_or_branching"] += 1

        chops = {frozenset(h) for h in zip(c, c[1:])}
        dhops = [frozenset(h) for h in zip(d, d[1:])]
        unmatched = [h for h in dhops if h not in chops]
        if unmatched:
            edge_id["traverses_at_least_one_hop_the_container_never_traverses"] += 1
            # Not every unmatched hop is equal. A hop whose two endpoints both sit in the
            # container is a chord -- the dropped path takes a direct relation across a node
            # the container walks through, so the two trace the same material at different
            # granularity. A hop touching a node the container lacks is a genuine departure.
            if all(h <= cs for h in unmatched):
                edge_id["unmatched_hops_are_all_chords_across_container_nodes"] += 1
            else:
                edge_id[
                    "at_least_one_unmatched_hop_touches_a_node_the_container_lacks"
                ] += 1
            for h in unmatched:
                for s in subtypes_of_pair.get(h, {"<absent from released EDGE set>"}):
                    unmatched_hop_subtypes[s] += 1
        else:
            edge_id["every_hop_is_also_a_hop_of_the_container"] += 1

    n_drops = len(drops)
    strict = order["contiguous_sub_path_of_container"]

    ri_raw = {(p["path"][0], p["path"][-1]) for p in raw}
    ri_kept = {(p["path"][0], p["path"][-1]) for p in deduped}
    lost_pairs = ri_raw - ri_kept

    out = {
        "experiment": "semantics of the 70% sub-path collapse (R22/R23)",
        "question": (
            "The containment rule compares node sets. Reviewers asked what it does to edge "
            "identity, order, and to traversals that are arguments in their own right."
        ),
        "reproduction_check": {
            "raw_paths": len(raw),
            "kept": len(keep),
            "dropped": n_drops,
            "released_deduped_file": len(deduped),
            "keeps_the_same_set_as_the_released_file": True,
            "note": (
                "The script exits non-zero unless the re-implementation reproduces the "
                "released chain set exactly, so every number below describes the substrate "
                "the paper reports on."
            ),
        },
        "node_set_relation": {
            **dict(rel),
            "pct_subset": pct(rel["node_set_is_subset_of_container"], n_drops),
            "pct_partial_overlap_only": pct(rel["partial_overlap_only"], n_drops),
            "cross_check_novel_node_share_pct": pct(n_novel_node, n_drops),
            "cross_check_note": (
                "experiment_review_gate_sensitivity_report.json -> "
                "containment_losslessness reports 78.3% of dropped paths carrying at least "
                "one node their container lacks. That is the same quantity as "
                "pct_partial_overlap_only and must agree."
            ),
        },
        "order_relation": {
            **dict(order),
            "pct_contiguous_sub_path": pct(strict, n_drops),
            "reading": (
                "A contiguous sub-path over the same node order is a restatement by "
                "construction: the container traces it and continues. Those drops are "
                "unambiguously safe. Everything else is where an annotator would have to "
                "decide, and that share is what the paper should state."
            ),
        },
        "edge_identity": {
            **dict(edge_id),
            "pct_with_an_unmatched_hop": pct(
                edge_id["traverses_at_least_one_hop_the_container_never_traverses"],
                n_drops,
            ),
            "pct_chords_only": pct(
                edge_id["unmatched_hops_are_all_chords_across_container_nodes"], n_drops
            ),
            "pct_touching_a_node_the_container_lacks": pct(
                edge_id[
                    "at_least_one_unmatched_hop_touches_a_node_the_container_lacks"
                ],
                n_drops,
            ),
            "unmatched_hop_relation_types": dict(unmatched_hop_subtypes.most_common()),
            "reading": (
                "This is the reviewers' objection made numeric, and it splits two ways. "
                "Every dropped path traverses at least one relation its container does not, "
                "so no drop is a strict re-walk. But where the unmatched hops are chords "
                "across nodes the container also holds, the two chains trace the same "
                "material at different granularity, and calling the drop a repeat is "
                "defensible. The other class is a genuine departure."
            ),
        },
        "does_the_drop_remove_a_distinct_remedy": {
            **dict(endpoint),
            "pct_ending_at_an_intervention_the_container_lacks": pct(
                endpoint["ends_at_an_intervention_absent_from_the_container"], n_drops
            ),
            "distinct_risk_to_intervention_pairs_raw": len(ri_raw),
            "distinct_risk_to_intervention_pairs_kept": len(ri_kept),
            "distinct_pairs_lost_to_the_collapse": len(lost_pairs),
            "pct_of_raw_pairs_lost": pct(len(lost_pairs), len(ri_raw)),
            "reading": (
                "This is the sharpest structural form of the reviewers' question. A dropped "
                "chain that terminates at an intervention the kept chain never reaches "
                "proposes a remedy the reporting unit no longer carries for that risk. The "
                "pair counts must agree with experiment_paper_claim_audit.json -> "
                "distinct_ri_node_pairs (raw 3,222 / deduped 2,643)."
            ),
        },
        "shared_prefix_with_container": {
            "length_histogram_nodes": {
                str(k): v for k, v in sorted(shared_prefix_len.items())
            },
            "pct_sharing_the_risk_root": pct(
                sum(v for k, v in shared_prefix_len.items() if k >= 1), n_drops
            ),
            "reading": (
                "Enumeration is risk-rooted, so a shared prefix is expected; the length at "
                "which two chains diverge is where one paper's argument branches."
            ),
        },
        "novel_node_stage_coverage": dict(novel_stage),
        "parallel_edges_in_the_released_graph": {
            "node_pairs_carrying_more_than_one_relation_type": parallel_pairs,
            "reading": (
                "If this is zero, the collapse cannot conflate two different relations "
                "between the same pair of nodes, and the edge-identity question reduces to "
                "which node pairs are traversed."
            ),
        },
        "empty_paths_dropped": empty_drops,
        "HEADLINE": (
            f"Of {n_drops} dropped paths, {pct(strict, n_drops)}% are contiguous sub-paths of "
            f"the chain that displaced them; "
            f"{pct(edge_id['traverses_at_least_one_hop_the_container_never_traverses'], n_drops)}% "
            "traverse at least one relation the kept chain does not, and "
            f"{pct(endpoint['ends_at_an_intervention_absent_from_the_container'], n_drops)}% "
            f"end at an intervention the kept chain never reaches. {len(lost_pairs)} distinct "
            f"risk-to-intervention pairs ({pct(len(lost_pairs), len(ri_raw))}% of the raw set) "
            "are not represented in the reporting unit at all."
        ),
        "LIMITS": (
            "Structural only. Two traversals over disjoint hops may still be the same "
            "argument, and two over identical hops may be read differently. Settling that "
            "needs the annotation study of OPEN_ITEMS.md S4."
        ),
    }

    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
