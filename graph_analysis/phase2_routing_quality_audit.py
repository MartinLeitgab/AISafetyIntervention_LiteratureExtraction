"""phase2_routing_quality_audit.py — Class B (no LLM tokens) quality metrics.

Computes diff-able metrics from a routing-assignments jsonl so you can compare
runs (old pilot vs calibration vs main-routing checkpoints) and track whether
prompt/architecture changes are actually improving fit.

Metrics:
  - coverage:      n_assigned / n_input, n_unassigned (hc/mc), pending force-fit
  - fit quality:   mean fit_score, fit_score distribution, %low-fit (<=3)
  - confidence:    mean confidence, %low-confidence (<=3)
  - axis coverage: per-axis %assigned + per-axis value distribution
  - cross-axis:    severity x harm_target consistency, emergence_stage x
                   lifecycle_stage consistency
  - per-HC/MC:     n_members, singletons (count=1), small-groups (count<3)
  - misfit flags:  reassign_pending count, fit_note present count

Outputs a markdown report + JSON dump for diffing.

Usage:
  python phase2_routing_quality_audit.py --asg pilot.jsonl --label pilot
  python phase2_routing_quality_audit.py --asg calib.jsonl --label calib
  python phase2_routing_quality_audit.py --compare pilot calib   # diff
"""

from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
DEFAULT_ASG = STEP1 / "phase2_routing_assignments.jsonl"
AUDIT_DIR = STEP1 / "phase2_routing_quality_audits"

# Consistency rules — flag violations
CROSS_AXIS_RULES = [
    {
        "name": "catastrophic-existential => harm_target in {human-survival, institutional-governance, human-flourishing-rights}",
        "premise": ("severity", "catastrophic-existential"),
        "allowed_target": (
            "harm_target",
            {
                "human-survival",
                "institutional-governance",
                "human-flourishing-rights",
                "scientific-truth",
            },
        ),
    },
    {
        "name": "harm_target=capability-gap-only => severity != catastrophic-existential",
        "premise": ("harm_target", "capability-gap-only"),
        "disallowed_target": ("severity", {"catastrophic-existential"}),
    },
    # NOTE: removed "pre-train -> emergence in {training-time, scaling}" rule on
    # 2026-05-18. The intervention-stage axis (lifecycle_stage) and the
    # risk-emergence-stage axis (emergence_stage) are intentionally orthogonal:
    # pre-train interventions legitimately address deployment-time risks (OOD
    # robustness, generalization). v2 pilot data confirmed all 3 "violations"
    # were correct cross-stage targeting, not errors.
]


def load_assignments(fp):
    rows = []
    for line in Path(fp).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _axis_val(r, name):
    return (r.get("axes") or {}).get(name)


def compute_metrics(rows):
    n = len(rows)
    m = {"n_paths": n}

    # Coverage
    hc_assigned = sum(1 for r in rows if r.get("harm_class_id"))
    mc_assigned = sum(1 for r in rows if r.get("mechanism_class_id"))
    hc_unassigned = sum(1 for r in rows if r.get("harm_class_status") == "unassigned")
    mc_unassigned = sum(
        1 for r in rows if r.get("mechanism_class_status") == "unassigned"
    )
    hc_force_fit = sum(
        1 for r in rows if r.get("harm_class_status") == "force_fit_pending"
    )
    mc_force_fit = sum(
        1 for r in rows if r.get("mechanism_class_status") == "force_fit_pending"
    )
    reassign_pending = sum(1 for r in rows if r.get("reassign_pending"))
    fit_notes = sum(1 for r in rows if r.get("fit_note"))
    m["coverage"] = {
        "hc_assigned_pct": round(100 * hc_assigned / n, 1) if n else 0,
        "mc_assigned_pct": round(100 * mc_assigned / n, 1) if n else 0,
        "hc_unassigned": hc_unassigned,
        "mc_unassigned": mc_unassigned,
        "hc_force_fit_pending": hc_force_fit,
        "mc_force_fit_pending": mc_force_fit,
        "reassign_pending": reassign_pending,
        "fit_notes_present": fit_notes,
    }

    # Fit/confidence
    confs = [r.get("confidence") for r in rows if r.get("confidence") is not None]
    fits = [r.get("fit_score") for r in rows if r.get("fit_score") is not None]
    m["fit_confidence"] = {
        "n_with_confidence": len(confs),
        "n_with_fit_score": len(fits),
        "mean_confidence": round(sum(confs) / len(confs), 2) if confs else None,
        "mean_fit_score": round(sum(fits) / len(fits), 2) if fits else None,
        "pct_low_confidence_le3": round(
            100 * sum(1 for c in confs if c <= 3) / len(confs), 1
        )
        if confs
        else None,
        "pct_low_fit_le3": round(100 * sum(1 for f in fits if f <= 3) / len(fits), 1)
        if fits
        else None,
        "fit_score_distribution": dict(Counter(fits)),
        "confidence_distribution": dict(Counter(confs)),
    }

    # Per-axis coverage + value distribution
    axes_seen = set()
    for r in rows:
        for k in r.get("axes") or {}:
            axes_seen.add(k)
    axes_metrics = {}
    for ax in sorted(axes_seen):
        vals = [_axis_val(r, ax) for r in rows]
        vals = [v for v in vals if v]
        cnt = Counter(vals)
        axes_metrics[ax] = {
            "coverage_pct": round(100 * len(vals) / n, 1) if n else 0,
            "distinct_values": len(cnt),
            "top_5": cnt.most_common(5),
            "other_freetext_pct": round(
                100 * sum(1 for v in vals if v.startswith("OTHER:")) / len(vals), 1
            )
            if vals
            else 0,
        }
    m["axes"] = axes_metrics

    # Cross-axis consistency
    cross = []
    for rule in CROSS_AXIS_RULES:
        prem_ax, prem_val = rule["premise"]
        matched = [r for r in rows if _axis_val(r, prem_ax) == prem_val]
        viol = 0
        if "allowed_target" in rule:
            tax, allowed = rule["allowed_target"]
            for r in matched:
                v = _axis_val(r, tax)
                if v and v not in allowed:
                    viol += 1
        elif "disallowed_target" in rule:
            tax, disallowed = rule["disallowed_target"]
            for r in matched:
                v = _axis_val(r, tax)
                if v in disallowed:
                    viol += 1
        cross.append(
            {
                "rule": rule["name"],
                "n_matched_premise": len(matched),
                "n_violations": viol,
                "violation_pct": round(100 * viol / len(matched), 1) if matched else 0,
            }
        )
    m["cross_axis_consistency"] = cross

    # Per-HC/MC distribution
    hc_counts = Counter(r.get("harm_class_id") for r in rows if r.get("harm_class_id"))
    mc_counts = Counter(
        r.get("mechanism_class_id") for r in rows if r.get("mechanism_class_id")
    )
    m["class_distribution"] = {
        "hc_distinct": len(hc_counts),
        "mc_distinct": len(mc_counts),
        "hc_singletons": sum(1 for c in hc_counts.values() if c == 1),
        "mc_singletons": sum(1 for c in mc_counts.values() if c == 1),
        "hc_below_min": sum(1 for c in hc_counts.values() if c < 3),
        "mc_below_min": sum(1 for c in mc_counts.values() if c < 3),
        "hc_top_5": hc_counts.most_common(5),
        "mc_top_5": mc_counts.most_common(5),
    }
    return m


def render_md(label, m):
    out = [f"# Quality audit — {label}\n"]
    out.append(f"**Total paths:** {m['n_paths']}\n")

    c = m["coverage"]
    out.append("## Coverage\n")
    out.append(
        f"- HC assigned: **{c['hc_assigned_pct']}%**, MC assigned: **{c['mc_assigned_pct']}%**"
    )
    out.append(f"- Unassigned: HC={c['hc_unassigned']}, MC={c['mc_unassigned']}")
    out.append(
        f"- Force-fit pending: HC={c['hc_force_fit_pending']}, MC={c['mc_force_fit_pending']}"
    )
    out.append(f"- Reassign-pending tags: {c['reassign_pending']}")
    out.append(f"- Fit notes present: {c['fit_notes_present']}\n")

    f = m["fit_confidence"]
    out.append("## Fit / confidence\n")
    out.append(
        f"- mean confidence: **{f['mean_confidence']}** (n={f['n_with_confidence']}); low-conf (<=3): {f['pct_low_confidence_le3']}%"
    )
    out.append(
        f"- mean fit_score: **{f['mean_fit_score']}** (n={f['n_with_fit_score']}); low-fit (<=3): {f['pct_low_fit_le3']}%"
    )
    out.append(f"- fit_score dist: {f['fit_score_distribution']}")
    out.append(f"- confidence dist: {f['confidence_distribution']}\n")

    out.append("## Axes\n")
    out.append("| axis | coverage | distinct | top-5 | OTHER:* % |")
    out.append("|---|---|---|---|---|")
    for ax, a in m["axes"].items():
        top = ", ".join(f"{v}({n})" for v, n in a["top_5"])
        out.append(
            f"| {ax} | {a['coverage_pct']}% | {a['distinct_values']} | {top} | {a['other_freetext_pct']}% |"
        )
    out.append("")

    out.append("## Cross-axis consistency\n")
    out.append("| rule | matched | violations | viol% |")
    out.append("|---|---|---|---|")
    for r in m["cross_axis_consistency"]:
        out.append(
            f"| {r['rule']} | {r['n_matched_premise']} | {r['n_violations']} | {r['violation_pct']}% |"
        )
    out.append("")

    d = m["class_distribution"]
    out.append("## Class distribution\n")
    out.append(
        f"- HC distinct: {d['hc_distinct']}, singletons: {d['hc_singletons']}, below-min(<3): {d['hc_below_min']}"
    )
    out.append(
        f"- MC distinct: {d['mc_distinct']}, singletons: {d['mc_singletons']}, below-min(<3): {d['mc_below_min']}"
    )
    out.append(f"- HC top-5: {d['hc_top_5']}")
    out.append(f"- MC top-5: {d['mc_top_5']}\n")
    return "\n".join(out)


def render_diff(label_a, m_a, label_b, m_b):
    out = [f"# Quality audit DIFF — {label_a} vs {label_b}\n"]
    out.append(f"|metric|{label_a}|{label_b}|delta|")
    out.append("|---|---|---|---|")

    def row(name, va, vb):
        try:
            delta = round(vb - va, 2)
            arrow = "UP" if delta > 0 else ("DN" if delta < 0 else "=")
            out.append(f"|{name}|{va}|{vb}|{arrow} {delta}|")
        except (TypeError, ValueError):
            out.append(f"|{name}|{va}|{vb}|—|")

    row("n_paths", m_a["n_paths"], m_b["n_paths"])
    for k in [
        "hc_assigned_pct",
        "mc_assigned_pct",
        "hc_unassigned",
        "mc_unassigned",
        "hc_force_fit_pending",
        "mc_force_fit_pending",
        "reassign_pending",
        "fit_notes_present",
    ]:
        row(f"coverage.{k}", m_a["coverage"][k], m_b["coverage"][k])
    for k in [
        "mean_confidence",
        "mean_fit_score",
        "pct_low_confidence_le3",
        "pct_low_fit_le3",
    ]:
        row(f"fit.{k}", m_a["fit_confidence"][k], m_b["fit_confidence"][k])
    for k in [
        "hc_distinct",
        "mc_distinct",
        "hc_singletons",
        "mc_singletons",
        "hc_below_min",
        "mc_below_min",
    ]:
        row(f"class.{k}", m_a["class_distribution"][k], m_b["class_distribution"][k])

    out.append("\n## Per-axis coverage delta\n")
    out.append("| axis | a-cov | b-cov | a-OTHER% | b-OTHER% |")
    out.append("|---|---|---|---|---|")
    for ax in sorted(set(m_a["axes"]) | set(m_b["axes"])):
        aa = m_a["axes"].get(ax, {})
        bb = m_b["axes"].get(ax, {})
        out.append(
            f"| {ax} | {aa.get('coverage_pct')}% | {bb.get('coverage_pct')}% | "
            f"{aa.get('other_freetext_pct')}% | {bb.get('other_freetext_pct')}% |"
        )
    out.append("\n## Cross-axis violations delta\n")
    out.append("| rule | a-viol% | b-viol% |")
    out.append("|---|---|---|")
    rules_a = {r["rule"]: r for r in m_a["cross_axis_consistency"]}
    rules_b = {r["rule"]: r for r in m_b["cross_axis_consistency"]}
    for k in sorted(set(rules_a) | set(rules_b)):
        out.append(
            f"| {k} | {rules_a.get(k, {}).get('violation_pct')}% | {rules_b.get(k, {}).get('violation_pct')}% |"
        )
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--asg", type=str, default=str(DEFAULT_ASG), help="assignments jsonl to audit"
    )
    ap.add_argument(
        "--label",
        type=str,
        required=False,
        help="label for this audit (used in filename)",
    )
    ap.add_argument(
        "--compare",
        nargs=2,
        metavar=("LABEL_A", "LABEL_B"),
        help="instead of computing, diff two prior labels",
    )
    args = ap.parse_args()
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    if args.compare:
        la, lb = args.compare
        m_a = json.loads((AUDIT_DIR / f"audit_{la}.json").read_text(encoding="utf-8"))
        m_b = json.loads((AUDIT_DIR / f"audit_{lb}.json").read_text(encoding="utf-8"))
        diff_md = render_diff(la, m_a, lb, m_b)
        out_md = AUDIT_DIR / f"diff_{la}_vs_{lb}.md"
        out_md.write_text(diff_md, encoding="utf-8")
        print(diff_md)
        print(f"\nwrote {out_md}")
        return

    if not args.label:
        ap.error("--label required when not using --compare")
    rows = load_assignments(args.asg)
    m = compute_metrics(rows)
    out_json = AUDIT_DIR / f"audit_{args.label}.json"
    out_md = AUDIT_DIR / f"audit_{args.label}.md"
    out_json.write_text(
        json.dumps(m, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    out_md.write_text(render_md(args.label, m), encoding="utf-8")
    print(render_md(args.label, m))
    print(f"\nwrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
