"""audit_pilot_v2_misfits.py — Class B (no LLM tokens) heuristic misfit audit
for pilot v2's HC and MC assignments.

Approach: for each path-member of each HC/MC, compute a simple keyword-overlap
between (path risk/intervention/body) and (class name + description). Surfaces:
  - Members with the lowest text-overlap to their assigned class
  - Members with low fit_score already (Opus self-flagged)
  - Cross-axis sanity (severity vs harm_target consistency at member level)

This is a TRIAGE pass — flags candidates for human review, not a final
adjudication. Output: phase2_pilot_v2_misfit_audit.md
"""

from __future__ import annotations
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import phase2_step4_phase2_doublet_llm_grouping as M

PILOT_V2_FP = M.STEP1 / "phase2_pilot_v2_100paths_discovery.json"
OUT_MD = M.STEP1 / "phase2_pilot_v2_misfit_audit.md"
OUT_JSON_ATTN = M.STEP1 / "phase2_routing_heuristic_misfits.json"

STOPWORDS = set(
    """a an the of and or to in on for from with by is are was were be
been being have has had do does did this that these those it its their there
than then so as at into out over via using used use we our you your they them
i me my also more most less few many some any all each every both either
neither not no nor through against between within without before after during
about under above below up down off only just very still already yet whether
which who whom whose when where why how what may might can could should would
will shall must ai ml model models system systems training train trained learn
learning data agent agents large advanced""".split()
)


def _toks(s):
    return [
        t
        for t in re.findall(r"[a-z][a-z\-]+", (s or "").lower())
        if t not in STOPWORDS and len(t) > 2
    ]


def _overlap(a, b):
    a_set, b_set = set(a), set(b)
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / min(len(a_set), len(b_set))


def main():
    v2 = json.loads(PILOT_V2_FP.read_text(encoding="utf-8"))
    raw = v2["raw_output"]
    hcs = {h["class_id"]: h for h in raw["harm_classes"]}
    mcs = {m["class_id"]: m for m in raw["mechanism_classes"]}

    paths = {}
    with open(
        M.ROOT / "phase1_rawpathsfiles/paths_hopwise_v4_edge_only_deduped.jsonl",
        encoding="utf-8",
    ) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                paths[f"path_{i:05d}_dedup"] = json.loads(line)
    with open(M.STEP1 / "graph_node_attributes.pkl", "rb") as f:
        na = pickle.load(f)

    def path_text_sides(p):
        """Return (risk_side_text, mech_side_text) from path nodes + first/last bodies."""
        nodes = p.get("path", [])
        cats = p.get("categories", [])
        risk_idx = [i for i, c in enumerate(cats) if c == "risk"]
        interv_idx = [i for i, c in enumerate(cats) if c == "intervention"]
        body_idx = [i for i, c in enumerate(cats) if c not in ("risk", "intervention")]
        # risk side = risk node + first 1-2 body nodes
        risk_nodes = []
        if risk_idx:
            risk_nodes.append(nodes[risk_idx[0]])
        risk_nodes.extend([nodes[i] for i in body_idx[:2]])
        # mech side = last 1-2 body + intervention
        mech_nodes = []
        mech_nodes.extend([nodes[i] for i in body_idx[-2:]])
        if interv_idx:
            mech_nodes.append(nodes[interv_idx[-1]])

        def text_of(nids):
            out = []
            for nid in nids:
                a = na.get(int(nid)) or {}
                out.append(a.get("name", "") or "")
                out.append(a.get("description", "") or "")
            return " ".join(out)

        return text_of(risk_nodes), text_of(mech_nodes)

    # Build member groups
    hc_members = defaultdict(list)
    mc_members = defaultdict(list)
    for a in raw["assignments"]:
        hc = a.get("harm_class_id")
        mc = a.get("mechanism_class_id")
        if isinstance(hc, str):
            hc_members[hc].append(a)
        if isinstance(mc, str):
            mc_members[mc].append(a)

    # Audit each member
    hc_audit = []
    for hc_id, members in hc_members.items():
        hc = hcs[hc_id]
        hc_text = hc["class_name"] + " " + hc.get("class_description", "")
        hc_toks = _toks(hc_text)
        for a in members:
            p = paths.get(a["path_id"])
            if not p:
                continue
            risk_text, _ = path_text_sides(p)
            overlap = _overlap(_toks(risk_text), hc_toks)
            hc_audit.append(
                {
                    "class_id": hc_id,
                    "class_name": hc["class_name"],
                    "path_id": a["path_id"],
                    "fit_score": a.get("fit_score"),
                    "confidence": a.get("confidence"),
                    "risk_overlap": round(overlap, 3),
                    "fit_note": a.get("fit_note", ""),
                    "risk_text_head": (risk_text or "")[:160],
                }
            )

    mc_audit = []
    for mc_id, members in mc_members.items():
        mc = mcs[mc_id]
        mc_text = mc["class_name"] + " " + mc.get("class_description", "")
        mc_toks = _toks(mc_text)
        for a in members:
            p = paths.get(a["path_id"])
            if not p:
                continue
            _, mech_text = path_text_sides(p)
            overlap = _overlap(_toks(mech_text), mc_toks)
            mc_audit.append(
                {
                    "class_id": mc_id,
                    "class_name": mc["class_name"],
                    "path_id": a["path_id"],
                    "fit_score": a.get("fit_score"),
                    "confidence": a.get("confidence"),
                    "mech_overlap": round(overlap, 3),
                    "fit_note": a.get("fit_note", ""),
                    "mech_text_head": (mech_text or "")[:160],
                }
            )

    # Sort: low fit_score first, then low overlap
    def hc_sort_key(r):
        f = r.get("fit_score")
        if f is None:
            f = 99
        return (f, r["risk_overlap"])

    def mc_sort_key(r):
        f = r.get("fit_score")
        if f is None:
            f = 99
        return (f, r["mech_overlap"])

    hc_audit.sort(key=hc_sort_key)
    mc_audit.sort(key=mc_sort_key)

    # Render
    out = ["# Pilot v2 misfit audit (heuristic triage)\n"]
    out.append(
        "Auto-generated. Flags candidates for human review based on "
        "Opus self-reported fit_score + keyword overlap between path's "
        "risk-side / mechanism-side text and class definition.\n"
    )
    out.append(
        "Lower fit_score + lower overlap = more likely misfit. "
        "Manual review of top entries recommended before launching production routing.\n"
    )

    out.append(
        "## TOP CANDIDATES — Harm class misfits (sorted by fit_score asc, overlap asc)\n"
    )
    out.append("| fit | risk_overlap | path_id | HC | risk_text_head | fit_note |")
    out.append("|---|---|---|---|---|---|")
    for r in hc_audit[:30]:
        out.append(
            f"| {r.get('fit_score', '?')} | {r['risk_overlap']} | "
            f"{r['path_id']} | {r['class_id']} {r['class_name'][:35]} | "
            f"{r['risk_text_head'][:120]} | {r['fit_note'][:80]} |"
        )
    out.append("")

    out.append("## TOP CANDIDATES — Mechanism class misfits\n")
    out.append("| fit | mech_overlap | path_id | MC | mech_text_head | fit_note |")
    out.append("|---|---|---|---|---|---|")
    for r in mc_audit[:30]:
        out.append(
            f"| {r.get('fit_score', '?')} | {r['mech_overlap']} | "
            f"{r['path_id']} | {r['class_id']} {r['class_name'][:35]} | "
            f"{r['mech_text_head'][:120]} | {r['fit_note'][:80]} |"
        )

    # Stats
    out.append("\n## Stats")
    hc_by_avg = defaultdict(list)
    for r in hc_audit:
        hc_by_avg[r["class_id"]].append(r["risk_overlap"])
    out.append(
        "\n### Per-HC mean risk-text overlap (lower = more potentially-misfit)\n"
    )
    out.append("| HC | mean_overlap | n_members |")
    out.append("|---|---|---|")
    for hid, ovs in sorted(hc_by_avg.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        mean = round(sum(ovs) / len(ovs), 3)
        out.append(f"| {hid} {hcs[hid]['class_name'][:40]} | {mean} | {len(ovs)} |")

    mc_by_avg = defaultdict(list)
    for r in mc_audit:
        mc_by_avg[r["class_id"]].append(r["mech_overlap"])
    out.append("\n### Per-MC mean mech-text overlap\n")
    out.append("| MC | mean_overlap | n_members |")
    out.append("|---|---|---|")
    for mid, ovs in sorted(mc_by_avg.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        mean = round(sum(ovs) / len(ovs), 3)
        out.append(f"| {mid} {mcs[mid]['class_name'][:40]} | {mean} | {len(ovs)} |")

    OUT_MD.write_text("\n".join(out), encoding="utf-8")

    # Top-K misfit candidates for the routing-prompt attention queue.
    # Exclude rows Opus already self-flagged via fit_score <= 3 (those are
    # already surfaced via the low-fit attention-queue section). Only include
    # high-confidence Opus assignments (fit=4 or 5) that the heuristic still
    # flags as poor-overlap — those are the cases Opus missed.
    def is_heuristic_candidate(r, overlap_field):
        return (r.get("fit_score") in (4, 5)) and r[overlap_field] < 0.10

    top_hc = [r for r in hc_audit if is_heuristic_candidate(r, "risk_overlap")]
    top_mc = [r for r in mc_audit if is_heuristic_candidate(r, "mech_overlap")]
    import json as _json

    OUT_JSON_ATTN.write_text(
        _json.dumps(
            {
                "generated_from": PILOT_V2_FP.name,
                "n_hc_audited": len(hc_audit),
                "n_mc_audited": len(mc_audit),
                "top_hc_misfits": top_hc[:25],
                "top_mc_misfits": top_mc[:25],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"wrote {OUT_MD.name}", flush=True)
    print(
        f"wrote {OUT_JSON_ATTN.name}: {len(top_hc)} HC + {len(top_mc)} MC heuristic misfits",
        flush=True,
    )
    print(f"  HC member entries: {len(hc_audit)}", flush=True)
    print(f"  MC member entries: {len(mc_audit)}", flush=True)
    print(
        f"  Low fit (<=3) HC: {sum(1 for r in hc_audit if r.get('fit_score', 5) <= 3)}",
        flush=True,
    )
    print(
        f"  Low fit (<=3) MC: {sum(1 for r in mc_audit if r.get('fit_score', 5) <= 3)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
