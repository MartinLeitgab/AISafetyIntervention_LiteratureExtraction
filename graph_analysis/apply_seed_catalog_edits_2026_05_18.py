"""apply_seed_catalog_edits_2026_05_18.py — One-shot seed-catalog edits applied
before launching the 37-batch routing pipeline.

Edits applied (per discussion 2026-05-18):
  1. HC015 (RL sample-ineff cap-gap)  → MERGED into HC026 (general ML cap-gap)
  2. HC007 (Uncontrolled agentic LLM) → SPLIT into:
        HC007 — Manipulative / jailbroken LLM outputs in chatbot deployments
        HC034 — Autonomous agentic LLM goal-pursuit & containment failure
  3. modality axis: drop `NLP`, all NLP assignments → `LLM`
  4. HC015 + HC026 + HC026 cap-gap: clarify class_description that the tag is
     on the harm chain (no human-harm chain), NOT the intervention's safety value
  5. path_01508 harm_target=capability-gap-only → REASSIGN to human-survival
     (risk is spec gaming; intervention is a registry, but the underlying risk is canonical alignment)
  6. paths 00085, 00274, 02013, 01724 — tagged reassign_pending=true in
     phase2_routing_assignments.jsonl (no class change here; routing will adjudicate)

Class B (no LLM tokens). Idempotent: re-running on already-edited catalog is a no-op.

Backup written to phase2_routing_active_catalog.json.pre_2026_05_18.bak
"""

from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent
STEP1 = ROOT / "phase2_results/step1_load_and_parse_umapwithoutlocalsatellites"
CAT_FP = STEP1 / "phase2_routing_active_catalog.json"
ASG_FP = STEP1 / "phase2_routing_assignments.jsonl"
PILOT_FP = STEP1 / "phase2_pilot_100paths_axis_discovery.json"

BACKUP_CAT = STEP1 / "phase2_routing_active_catalog.json.pre_2026_05_18.bak"
BACKUP_ASG = STEP1 / "phase2_routing_assignments.jsonl.pre_2026_05_18.bak"


def main():
    if not CAT_FP.exists():
        sys.exit(f"ERROR: {CAT_FP} not found — run bootstrap first.")
    if not BACKUP_CAT.exists():
        shutil.copy2(CAT_FP, BACKUP_CAT)
        print(f"backed up catalog -> {BACKUP_CAT.name}", flush=True)
    if ASG_FP.exists() and not BACKUP_ASG.exists():
        shutil.copy2(ASG_FP, BACKUP_ASG)
        print(f"backed up assignments -> {BACKUP_ASG.name}", flush=True)

    cat = json.loads(CAT_FP.read_text(encoding="utf-8"))
    hc_by_id = {h["class_id"]: h for h in cat["harm_classes"]}

    # Track applied edits for idempotence
    applied = []

    # ----- Edit 1: HC015 -> HC026 merge -----
    if "HC015" in hc_by_id:
        cat["harm_classes"] = [
            h for h in cat["harm_classes"] if h["class_id"] != "HC015"
        ]
        applied.append("HC015 removed (merged -> HC026)")
        # Update HC026 description to absorb HC015's RL-specific scope
        for h in cat["harm_classes"]:
            if h["class_id"] == "HC026":
                h["class_description"] = (
                    "Capability-gap-only harm chain (not a human-harm chain). Covers "
                    "training instability, sample efficiency (including RL sparse-reward "
                    "and partial-observability exploration failure), compute waste, "
                    "semantic representation, NAS cost, and energy. The capability-gap "
                    "tag is on the harm side; interventions here can still be "
                    "safety-load-bearing via downstream chains (e.g., sample-efficient "
                    "exploration enables safer training-time evaluation)."
                )
                break

    # Persist group_remap so downstream readers translate old HC015 references
    cat.setdefault("group_remap", {})["HC015"] = "HC026"

    # ----- Edit 2: HC007 split (HC007 keeps id; HC034 is new) -----
    if "HC007" in hc_by_id:
        for h in cat["harm_classes"]:
            if h["class_id"] == "HC007":
                if (
                    "manipulative" not in (h["class_description"] or "").lower()
                    or "autonomous" in (h["class_description"] or "").lower()
                ):
                    h["class_name"] = (
                        "Manipulative / jailbroken LLM outputs in chatbot deployments"
                    )
                    h["class_description"] = (
                        "LLMs deployed as chat/text systems produce harmful, "
                        "manipulative, deceptive, or jailbroken outputs to users. Focus on "
                        "output-content harm to end-users; NOT on goal-directed agentic "
                        "behavior (see HC034 for that)."
                    )
                    applied.append("HC007 tightened (manipulative-output focus)")
                break
    if "HC034" not in hc_by_id:
        cat["harm_classes"].append(
            {
                "class_id": "HC034",
                "class_name": "Autonomous agentic LLM goal-pursuit & containment failure",
                "class_description": (
                    "LLM-based agents pursue persistent goals across tool-use and "
                    "interaction loops, with risks of resisting interruption, escaping "
                    "containment, or expanding influence beyond operator intent. Distinct "
                    "from HC007 which covers chatbot output harms (no goal-directedness)."
                ),
                "is_capability_gap": False,
            }
        )
        applied.append("HC034 added (autonomous-agent split from HC007)")

    # ----- Edit 3: modality axis prune NLP, fold to LLM -----
    for ax in cat.get("axes", []):
        if ax["axis_name"] == "modality":
            if "NLP" in (ax.get("values") or []):
                ax["values"] = [v for v in ax["values"] if v != "NLP"]
                applied.append("modality.NLP retired (fold to LLM)")
            break

    # ----- Edit 4: capability-gap clarification on HC026 already done above -----
    # Done in Edit 1 description update.

    # Write updated catalog
    CAT_FP.write_text(json.dumps(cat, indent=2, ensure_ascii=False), encoding="utf-8")

    # ----- Edits to assignments jsonl -----
    asg_rows = []
    if ASG_FP.exists():
        for line in ASG_FP.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                asg_rows.append(json.loads(line))

    misfit_path_ids = {
        "path_00085_dedup",
        "path_00274_dedup",
        "path_02013_dedup",
        "path_01724_dedup",
    }
    reassign_count = 0
    nlp_fold = 0
    hc015_remap = 0
    cap_gap_fix = 0
    for r in asg_rows:
        # HC015 -> HC026 remap on existing rows
        if r.get("harm_class_id") == "HC015":
            r["harm_class_id"] = "HC026"
            r.setdefault("history", []).append(
                {"edit": "HC015->HC026 merge", "date": "2026-05-18"}
            )
            hc015_remap += 1
        # modality NLP -> LLM
        axes = r.get("axes") or {}
        if axes.get("modality") == "NLP":
            axes["modality"] = "LLM"
            r["axes"] = axes
            r.setdefault("history", []).append(
                {"edit": "modality NLP->LLM fold", "date": "2026-05-18"}
            )
            nlp_fold += 1
        # path_01508 harm_target fix
        if r.get("path_id") == "path_01508_dedup":
            axes2 = r.get("axes") or {}
            if axes2.get("harm_target") == "capability-gap-only":
                axes2["harm_target"] = "human-survival"
                r["axes"] = axes2
                r.setdefault("history", []).append(
                    {
                        "edit": "harm_target cap-gap->human-survival (risk is spec gaming, a safety risk)",
                        "date": "2026-05-18",
                    }
                )
                cap_gap_fix += 1
        # Misfit reassign_pending tag
        if r.get("path_id") in misfit_path_ids and not r.get("reassign_pending"):
            r["reassign_pending"] = True
            r.setdefault("history", []).append(
                {
                    "edit": "tagged reassign_pending (see phase2_watch_items.md)",
                    "date": "2026-05-18",
                }
            )
            reassign_count += 1

    # Write back
    with open(ASG_FP, "w", encoding="utf-8") as f:
        for r in asg_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print()
    print("=" * 60)
    print("Seed-catalog edits applied")
    print("=" * 60)
    for line in applied:
        print(f"  - {line}")
    print("  - assignments edits:")
    print(f"      HC015 -> HC026 remap rows:            {hc015_remap}")
    print(f"      modality NLP -> LLM rows:             {nlp_fold}")
    print(f"      path_01508 harm_target cap-gap fix:   {cap_gap_fix}")
    print(f"      misfit reassign_pending tags:         {reassign_count}")
    print(f"  catalog HC count: {len(cat['harm_classes'])}")
    print(f"  catalog MC count: {len(cat['mechanism_classes'])}")
    print(f"  catalog axes:     {len(cat.get('axes', []))}")
    print(f"  catalog group_remap entries: {len(cat.get('group_remap', {}))}")


if __name__ == "__main__":
    main()
