"""recover_consolidation_001.py — Recover the consolidation_001 output that
hit an unterminated-string JSON parse error on Opus's `summary` field.

Class B (no LLM tokens). Mirrors the apply logic inside run_consolidation.
"""

from __future__ import annotations
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import phase2_step4_phase2_doublet_llm_grouping as M
import phase2_step5_opus_routing as R

PARTIAL_FP = R.STEP1 / "phase2_routing_partial.txt"
DURATION_SEC = 170.0


def main():
    if not PARTIAL_FP.exists():
        sys.exit(f"ERROR: partial missing at {PARTIAL_FP}")
    raw = PARTIAL_FP.read_text(encoding="utf-8")
    m = re.search(r"END_SENTINEL_[a-f0-9]+", raw)
    if not m:
        sys.exit("ERROR: no END_SENTINEL in partial")
    payload = raw[: m.start()]
    parsed, method = R._robust_json_parse(payload, '{"merges_applied":')
    print(f"recovered via {method}", flush=True)
    print(f"  keys: {list(parsed.keys())}", flush=True)

    catalog = R._load_active_catalog()
    state = R._load_state()
    consolidation_idx = state.get("total_consolidations_run", 0) + 1

    # Save raw output to consolidation dir
    R.CONSOLIDATION_DIR.mkdir(parents=True, exist_ok=True)
    out = R.CONSOLIDATION_DIR / f"consolidation_{consolidation_idx:03d}.json"
    M.atomic_write_json(
        out,
        {
            "consolidation_idx": consolidation_idx,
            "duration_sec": DURATION_SEC,
            "n_batches_consolidated": 5,
            "batches_range": [0, state.get("total_batches_run", 5)],
            "raw_output": parsed,
            "recovered_from_partial": True,
            "recovery_method": method,
        },
    )
    print(f"wrote {out.name}")

    # ===== APPLY LOGIC (mirrors run_consolidation lines 1188+) =====
    n_merges = len(parsed.get("merges_applied", []))
    n_renames = len(parsed.get("renames_applied", []))
    n_axis_ext = len(parsed.get("axis_extensions_applied", []))

    hc_by_id = {h["class_id"]: h for h in catalog["harm_classes"]}
    mc_by_id = {m["class_id"]: m for m in catalog["mechanism_classes"]}

    # Apply renames
    for ren in parsed.get("renames_applied", []):
        gid = ren.get("class_id", "")
        target = hc_by_id.get(gid) or mc_by_id.get(gid)
        if target:
            if ren.get("new_name"):
                target["class_name"] = ren["new_name"]
            if ren.get("new_description"):
                target["class_description"] = ren["new_description"]

    # Apply merges
    remap_hc = {}
    remap_mc = {}
    for mrg in parsed.get("merges_applied", []):
        src = mrg.get("from")
        dst = mrg.get("to")
        if src in hc_by_id and dst in hc_by_id and src != dst:
            remap_hc[src] = dst
        elif src in mc_by_id and dst in mc_by_id and src != dst:
            remap_mc[src] = dst
    catalog["harm_classes"] = [
        h for h in catalog["harm_classes"] if h["class_id"] not in remap_hc
    ]
    catalog["mechanism_classes"] = [
        m for m in catalog["mechanism_classes"] if m["class_id"] not in remap_mc
    ]

    # Apply generalizations (broader name + description)
    n_generalizations_applied = 0
    for gen in parsed.get("generalizations_applied", []):
        gid = gen.get("class_id", "")
        target = hc_by_id.get(gid) or mc_by_id.get(gid)
        if target:
            if gen.get("broader_name"):
                target["class_name"] = gen["broader_name"]
            if gen.get("broader_description"):
                target["class_description"] = gen["broader_description"]
            n_generalizations_applied += 1

    # Apply axis extensions
    axes_by_name = {a["axis_name"]: a for a in catalog["axes"]}
    for ext in parsed.get("axis_extensions_applied", []):
        ax = axes_by_name.get(ext.get("axis", ""))
        new_val = ext.get("new_value", "")
        if ax and new_val and new_val not in ax.get("values", []):
            ax["values"].append(new_val)

    # Apply splits: allocate new ids, add to catalog, append per-path reassignments
    splits_applied_count = 0
    splits_dropped = []
    for split in parsed.get("splits_applied", []):
        from_id = split.get("from_class_id", "")
        new_name = split.get("new_class_name", "").strip()
        new_desc = split.get("new_class_description", "").strip()
        members = split.get("member_path_ids_to_move", []) or []
        rationale = split.get("rationale", "")
        if not new_name or not new_desc:
            splits_dropped.append((from_id, "missing name/description"))
            continue
        if len(members) < R.MIN_GROUP_SIZE:
            splits_dropped.append((from_id, f"only {len(members)} members"))
            continue
        kind = from_id[:2]
        if kind not in ("HC", "MC"):
            splits_dropped.append((from_id, "bad from_class_id"))
            continue
        new_id = R._next_class_id(catalog, kind)
        new_entry = {
            "class_id": new_id,
            "class_name": new_name,
            "class_description": new_desc,
            "source": f"consolidation_{consolidation_idx:03d}_split_from_{from_id}",
        }
        if kind == "HC":
            new_entry["is_capability_gap"] = bool(split.get("is_capability_gap", False))
            catalog["harm_classes"].append(new_entry)
        else:
            catalog["mechanism_classes"].append(new_entry)
        for pid in members:
            kw = {"new_hc": new_id} if kind == "HC" else {"new_mc": new_id}
            R._append_path_reassignment(
                pid,
                source=f"consolidation_{consolidation_idx:03d}_split",
                rationale=f"split-out from {from_id}: {rationale[:200]}",
                **kw,
            )
        splits_applied_count += 1
        print(
            f"  applied split: {from_id} -> {new_id} ({new_name[:60]}), "
            f"{len(members)} members moved",
            flush=True,
        )
    for d in splits_dropped:
        print(f"  DROPPED split: {d[0]} ({d[1]})", flush=True)

    # Apply reassignments_applied
    reassigns_applied = 0
    for ra in parsed.get("reassignments_applied", []):
        pid = ra.get("path_id", "")
        to_id = ra.get("to_class_id", "")
        side = (ra.get("side") or "").lower()
        if not pid or not to_id:
            continue
        if side not in ("harm", "mechanism"):
            side = (
                "harm"
                if to_id.startswith("HC")
                else ("mechanism" if to_id.startswith("MC") else "")
            )
        if side not in ("harm", "mechanism"):
            continue
        kw = {"new_hc": to_id} if side == "harm" else {"new_mc": to_id}
        R._append_path_reassignment(
            pid,
            source=f"consolidation_{consolidation_idx:03d}_reassign",
            rationale=ra.get("rationale", ""),
            **kw,
        )
        reassigns_applied += 1

    R._save_active_catalog(
        catalog, kind_suffix=f"consolidation_{consolidation_idx:03d}_recovered"
    )

    # Persist merge remap
    remap_fp = R.STEP1 / "phase2_routing_class_remap.json"
    if remap_fp.exists():
        cur = json.loads(remap_fp.read_text(encoding="utf-8"))
    else:
        cur = {"hc": {}, "mc": {}}
    cur.setdefault("hc", {}).update(remap_hc)
    cur.setdefault("mc", {}).update(remap_mc)
    M.atomic_write_json(remap_fp, cur)

    # State
    state["last_consolidation_at_batch_idx"] = state.get("total_batches_run", 5)
    state["total_consolidations_run"] = consolidation_idx
    R._save_state(state)

    # Rebuild jsonl + xlsx
    n_rows = R._rebuild_routing_assignments_jsonl()
    print(f"rebuilt merged jsonl: {n_rows} rows", flush=True)
    try:
        from phase2_routing_to_xlsx import build_routing_xlsx

        res = build_routing_xlsx()
        print(
            f"xlsx refreshed: {res['n_paths']} paths, {res['n_harm_classes']} HC + "
            f"{res['n_mech_classes']} MC",
            flush=True,
        )
    except Exception as e:
        print(f"xlsx rebuild warning: {e}", flush=True)

    # Auto-append to watch_items
    extra = [
        f"**Applied:** {n_merges} merges, {n_renames} renames, "
        f"{n_generalizations_applied} generalizations, {n_axis_ext} axis-ext, "
        f"{splits_applied_count} splits, {reassigns_applied} reassignments"
    ]
    if splits_dropped:
        extra.append(
            f"**Splits DROPPED:** {[(d[0], d[1]) for d in splits_dropped[:6]]}"
        )
    if parsed.get("split_dives_scheduled"):
        extra.append("**Split deep-dives scheduled (final_misfit_sweep):**")
        for s in parsed.get("split_dives_scheduled", []):
            extra.append(f"- {s.get('class_id')}: {s.get('rationale', '')[:140]}")
    if parsed.get("defended_homogeneous_classes"):
        extra.append("**Defended as homogeneous:**")
        for d in parsed.get("defended_homogeneous_classes", []):
            extra.append(f"- {d.get('class_id')}: {d.get('rationale', '')[:140]}")
    R._append_to_watch_items(
        "consolidation", consolidation_idx, parsed.get("summary", ""), extra_lines=extra
    )

    print(f"\n=== Consolidation {consolidation_idx:03d} RECOVERED + APPLIED ===")
    print(f"  {extra[0]}")
    print(f"  scheduled deep-dives: {len(parsed.get('split_dives_scheduled', []))}")


if __name__ == "__main__":
    main()
