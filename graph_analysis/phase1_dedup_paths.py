"""phase1_dedup_paths.py — collapse near-duplicate paths within each source paper.

Two paths from the same paper are near-duplicates when one's node set is >=80%
contained in the other's. The shorter path is dropped; longer one is kept.

Output: `phase1_rawpathsfiles/paths_hopwise_v4_edge_only_deduped.jsonl`
Class B (no LLM tokens). ~1 min wall-clock on 8954 paths.
"""

import json
import pickle
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
IN_FP = ROOT / "phase1_rawpathsfiles/paths_hopwise_v4_edge_only.jsonl"
OUT_FP = ROOT / "phase1_rawpathsfiles/paths_hopwise_v4_edge_only_deduped.jsonl"
CONTAINMENT_THRESHOLD = 0.70


def main():
    print("loading node_attrs (for source URLs) ...", flush=True)
    with open(STEP1 / "graph_node_attributes.pkl", "rb") as f:
        na = pickle.load(f)
    print(f"  {len(na)} nodes", flush=True)

    print("loading paths ...", flush=True)
    paths = []
    with open(IN_FP, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                d = json.loads(line)
                d["path_id"] = f"path_{i:05d}"
                paths.append(d)
    print(f"  {len(paths)} paths", flush=True)

    # Attach source URL per path (each path's nodes share one URL since EDGE-type
    # extraction is within-paper)
    paths_by_url = defaultdict(list)
    for p in paths:
        urls = set()
        for nid in p.get("path", []):
            attrs = na.get(int(nid)) or na.get(nid) or {}
            url = attrs.get("url")
            if url:
                urls.add(url)
        if len(urls) == 1:
            paths_by_url[next(iter(urls))].append(p)
        elif len(urls) == 0:
            # No URL — treat as own group
            paths_by_url[f"_orphan_{p['path_id']}"].append(p)
        else:
            # Cross-paper path (shouldn't happen for EDGE-only hopwise but defensive)
            paths_by_url[f"_mixed_{p['path_id']}"].append(p)

    print(f"  unique source URLs: {len(paths_by_url)}", flush=True)

    keep = set()
    n_dropped = 0
    for url, plist in paths_by_url.items():
        # Sort longest-first so shorter dups attach to the longer canonical
        plist_sorted = sorted(plist, key=lambda p: -len(p.get("path", [])))
        node_sets = [(p["path_id"], frozenset(p.get("path", []))) for p in plist_sorted]
        for i, (pid_i, ns_i) in enumerate(node_sets):
            if not ns_i:
                # Empty path — drop
                n_dropped += 1
                continue
            # Check if this path is mostly contained in any LONGER (earlier in sort) kept path
            is_dup = False
            for pid_j, ns_j in node_sets[:i]:
                if pid_j not in keep:
                    continue
                if not ns_j:
                    continue
                # Containment of smaller in larger
                small, large = (ns_i, ns_j) if len(ns_i) <= len(ns_j) else (ns_j, ns_i)
                if len(small & large) / len(small) >= CONTAINMENT_THRESHOLD:
                    is_dup = True
                    break
            if not is_dup:
                keep.add(pid_i)
            else:
                n_dropped += 1

    print(f"\n  kept: {len(keep)}", flush=True)
    print(
        f"  dropped (>={int(CONTAINMENT_THRESHOLD * 100)}% contained): {n_dropped}",
        flush=True,
    )
    print(f"  reduction: {100 * n_dropped / len(paths):.1f}%", flush=True)

    # Write deduped jsonl preserving original order
    n_written = 0
    with open(OUT_FP, "w", encoding="utf-8") as fout:
        for p in paths:
            if p["path_id"] in keep:
                # Remove path_id we added (recomputed downstream)
                p2 = {k: v for k, v in p.items() if k != "path_id"}
                fout.write(json.dumps(p2, ensure_ascii=False) + "\n")
                n_written += 1
    print(f"\nwrote {OUT_FP}  ({n_written} paths)", flush=True)
    print(
        "  use --use-deduped-paths flag in main script to read this instead of "
        "the original",
        flush=True,
    )


if __name__ == "__main__":
    main()
