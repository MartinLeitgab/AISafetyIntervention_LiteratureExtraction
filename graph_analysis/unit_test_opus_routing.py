"""unit_test_opus_routing.py — Class B (no LLM tokens) smoke test for the new
Opus routing pipeline in phase2_step5_opus_routing.py.

Exercises:
  - prompt makers (routing, consolidation, final_audit)
  - _resolve_and_enforce with MIN_GROUP_SIZE drop-and-force-fit
  - _append_flags + _rebuild_routing_assignments_jsonl
  - _init_active_catalog_from_pilot

Uses synthetic data in tempdir; monkey-patches paths.
"""

from __future__ import annotations
import json
import sys
import tempfile
import shutil
from pathlib import Path
from collections import Counter

import phase2_step5_opus_routing as R
import phase2_step4_phase2_doublet_llm_grouping as M


def _seed_pilot():
    return {
        "pilot_n": 10,
        "raw_output": {
            "axes": [
                {
                    "axis_name": "lifecycle_stage",
                    "axis_kind": "intervention",
                    "values": ["pre-train", "fine-tune", "deployment", "OTHER"],
                },
                {
                    "axis_name": "modality",
                    "axis_kind": "intervention",
                    "values": ["LLM", "RL-agent", "vision", "general", "OTHER"],
                },
                {
                    "axis_name": "severity",
                    "axis_kind": "risk",
                    "values": ["catastrophic", "moderate", "minor", "OTHER"],
                },
            ],
            "harm_classes": [
                {
                    "class_id": "HC001",
                    "class_name": "Reward hacking",
                    "class_description": "Specification gaming.",
                    "is_capability_gap": False,
                },
                {
                    "class_id": "HC002",
                    "class_name": "LLM hallucination",
                    "class_description": "Plausible falsehoods.",
                    "is_capability_gap": False,
                },
                {
                    "class_id": "HC003",
                    "class_name": "Capability gap RL",
                    "class_description": "Sample inefficiency.",
                    "is_capability_gap": True,
                },
            ],
            "mechanism_classes": [
                {
                    "class_id": "MC001",
                    "class_name": "RLHF",
                    "class_description": "Preference learning.",
                },
                {
                    "class_id": "MC002",
                    "class_name": "Constitutional AI",
                    "class_description": "Self-critique.",
                },
            ],
            "assignments": [
                {
                    "path_id": f"path_{i:05d}_dedup",
                    "harm_class_id": "HC001",
                    "mechanism_class_id": "MC001",
                    "axis_values": {
                        "lifecycle_stage": "fine-tune",
                        "modality": "LLM",
                        "severity": "moderate",
                    },
                    "confidence": 4,
                }
                for i in range(1, 11)
            ],
        },
    }


def setup_tempdir():
    tmp = Path(tempfile.mkdtemp(prefix="opus_routing_smoke_"))
    step1 = tmp / "phase2_results" / "step1_load_and_parse_umapwithoutlocalsatellites"
    step1.mkdir(parents=True, exist_ok=True)
    # Monkey-patch all path constants
    R.STEP1 = step1
    R.PILOT_FP = step1 / "phase2_pilot_100paths_axis_discovery.json"
    R.ACTIVE_CATALOG_FP = step1 / "phase2_routing_active_catalog.json"
    R.STATE_FP = step1 / "phase2_routing_state.json"
    R.BATCH_DIR = step1 / "phase2_routing_batches"
    R.FLAGS_LOG_FP = step1 / "phase2_routing_flags.jsonl"
    R.CONSOLIDATION_DIR = step1 / "phase2_routing_consolidations"
    R.FINAL_AUDIT_FP = step1 / "phase2_routing_final_audit.json"
    R.ASSIGNMENTS_FP = step1 / "phase2_routing_assignments.jsonl"
    R.PARTIAL_FP = step1 / "phase2_routing_partial.txt"
    # Write synthetic pilot
    R.PILOT_FP.write_text(json.dumps(_seed_pilot(), indent=2), encoding="utf-8")
    R.BATCH_DIR.mkdir(parents=True, exist_ok=True)
    R.CONSOLIDATION_DIR.mkdir(parents=True, exist_ok=True)
    return tmp


def test_init_catalog():
    print("[1] _init_active_catalog_from_pilot")
    R._init_active_catalog_from_pilot()
    cat = R._load_active_catalog()
    assert len(cat["harm_classes"]) == 3, (
        f"expected 3 HC, got {len(cat['harm_classes'])}"
    )
    assert len(cat["mechanism_classes"]) == 2, "expected 2 MC"
    assert len(cat["axes"]) == 3, "expected 3 axes"
    print(
        f"    catalog loaded: {len(cat['harm_classes'])} HC + {len(cat['mechanism_classes'])} MC + {len(cat['axes'])} axes  OK"
    )


def test_pilot_seeding():
    print("[2] pilot_assignments seeding into merged jsonl")
    n = R._rebuild_routing_assignments_jsonl()
    assert n == 10, f"expected 10 pilot rows, got {n}"
    print(f"    merged jsonl seeded with {n} pilot rows  OK")


def test_prompt_makers():
    print("[3] prompt-maker tests")
    cat = R._load_active_catalog()
    hc_counts = Counter({"HC001": 5, "HC002": 3, "HC003": 2})
    mc_counts = Counter({"MC001": 6, "MC002": 4})

    # Routing prompt
    sentinel = "abc123"
    fake_path = {
        "path_id": "path_99999_dedup",
        "path": [1, 2, 3],
        "categories": ["risk", "design_rationale", "intervention"],
    }
    fake_attrs = {
        1: {"name": "Risk X", "description": "..."},
        2: {
            "name": "Body X",
            "description": "...",
            "concept_category": "design_rationale",
        },
        3: {"name": "Interv X", "description": "...", "type": "intervention"},
    }
    p_routing = R.make_routing_prompt(
        [fake_path], cat, hc_counts, mc_counts, fake_attrs, sentinel
    )
    assert f"END_SENTINEL_{sentinel}" in p_routing
    assert "HC001" in p_routing and "MC001" in p_routing
    assert "axis_values" in p_routing and "catalog_flags" in p_routing
    print(f"    routing prompt OK ({len(p_routing)} chars)")

    # Consolidation
    p_consol = R.make_consolidation_prompt(
        cat,
        [
            {
                "batch_idx": 0,
                "flags": {
                    "merge_candidates": [
                        {"a": "HC001", "b": "HC002", "rationale": "test"}
                    ]
                },
            }
        ],
        hc_counts,
        mc_counts,
        sentinel,
    )
    assert f"END_SENTINEL_{sentinel}" in p_consol
    assert "merges_applied" in p_consol
    print(f"    consolidation prompt OK ({len(p_consol)} chars)")

    # Final audit
    p_audit = R.make_final_audit_prompt(cat, hc_counts, mc_counts, 100, sentinel)
    assert f"END_SENTINEL_{sentinel}" in p_audit
    assert "singleton_decisions" in p_audit
    print(f"    final audit prompt OK ({len(p_audit)} chars)")


def test_resolve_and_enforce():
    print("[4] _resolve_and_enforce with MIN_GROUP_SIZE drop")
    cat = R._load_active_catalog()
    # Synthetic Opus output: 5 paths
    #   - p1, p2: existing HC001 / new MC "Distillation" (3 members -> survives)
    #   - p3:    existing HC001 / new MC "Distillation"
    #   - p4:    new HC "Privacy" (singleton -> DROPPED)
    #            / new MC "Distillation"
    #   - p5:    new HC "Privacy" / existing MC001 — also dropped due to HC<min
    parsed = {
        "assignments": [
            {
                "path_id": "p1",
                "harm_class": {"existing": "HC001"},
                "mechanism_class": {
                    "new": {
                        "class_name": "Distillation",
                        "class_description": "Knowledge distill.",
                    }
                },
                "axis_values": {},
                "confidence": 4,
            },
            {
                "path_id": "p2",
                "harm_class": {"existing": "HC001"},
                "mechanism_class": {
                    "new": {
                        "class_name": "Distillation",
                        "class_description": "Knowledge distill.",
                    }
                },
                "axis_values": {},
                "confidence": 4,
            },
            {
                "path_id": "p3",
                "harm_class": {"existing": "HC001"},
                "mechanism_class": {
                    "new": {
                        "class_name": "Distillation",
                        "class_description": "Knowledge distill.",
                    }
                },
                "axis_values": {},
                "confidence": 4,
            },
            {
                "path_id": "p4",
                "harm_class": {
                    "new": {
                        "class_name": "Privacy violation",
                        "class_description": "Privacy.",
                        "is_capability_gap": False,
                    }
                },
                "mechanism_class": {
                    "new": {
                        "class_name": "Distillation",
                        "class_description": "Knowledge distill.",
                    }
                },
                "axis_values": {},
                "confidence": 3,
            },
            {
                "path_id": "p5",
                "harm_class": {
                    "new": {
                        "class_name": "Privacy violation",
                        "class_description": "Privacy.",
                        "is_capability_gap": False,
                    }
                },
                "mechanism_class": {"existing": "MC001"},
                "axis_values": {},
                "confidence": 3,
            },
        ],
        "catalog_flags": {},
    }
    cat2, resolved, forced, dropped_hc, dropped_mc = R._resolve_and_enforce(
        parsed, cat, 0
    )
    # Distillation has 4 members -> SURVIVES (>=3)
    mc_names = {m["class_name"] for m in cat2["mechanism_classes"]}
    assert "Distillation" in mc_names, "Distillation should survive (4 members)"
    # Privacy violation has 2 members -> DROPPED
    hc_names = {h["class_name"] for h in cat2["harm_classes"]}
    assert "Privacy violation" not in hc_names, (
        "Privacy violation should be dropped (only 2 members)"
    )
    assert "Privacy violation" in dropped_hc, (
        f"dropped_hc should contain Privacy violation, got {dropped_hc}"
    )
    # Forced fits: p4 and p5 both have hc_id = None
    p4 = next(r for r in resolved if r["path_id"] == "p4")
    p5 = next(r for r in resolved if r["path_id"] == "p5")
    assert p4["harm_class_id"] is None, "p4 hc should be None (dropped)"
    assert p5["harm_class_id"] is None, "p5 hc should be None (dropped)"
    assert "hc_new_dropped" in (p4["fit_note"] or ""), (
        "p4 should have fit_note about drop"
    )
    assert forced == 2, f"expected 2 forced fits, got {forced}"
    print(
        f"    Distillation (4 members) survived, Privacy violation (2) dropped; "
        f"forced={forced}, dropped_hc={dropped_hc}, dropped_mc={dropped_mc}  OK"
    )


def test_flags_and_jsonl_rebuild():
    print("[5] _append_flags + _rebuild_routing_assignments_jsonl")
    parsed = {
        "catalog_flags": {
            "merge_candidates": [
                {"a": "HC001", "b": "HC002", "rationale": "redundant"}
            ],
            "axis_value_extensions": [
                {"axis": "modality", "new_value": "biological", "rationale": "..."}
            ],
        }
    }
    n = R._append_flags(parsed, 0)
    assert n == 2, f"expected 2 flags appended, got {n}"
    assert R.FLAGS_LOG_FP.exists()
    print(f"    {n} flags appended to log  OK")

    # Write a fake batch file + rebuild
    batch_payload = {
        "batch_idx": 0,
        "resolved_assignments": [
            {
                "path_id": "p_batch_1",
                "harm_class_id": "HC001",
                "mechanism_class_id": "MC001",
                "axes": {},
                "confidence": 4,
                "source": "routing_batch_0000",
            },
        ],
    }
    M.atomic_write_json(R.BATCH_DIR / "batch_0000.json", batch_payload)
    n_rows = R._rebuild_routing_assignments_jsonl()
    assert n_rows == 11, f"expected 11 rows (10 pilot + 1 batch), got {n_rows}"
    print(f"    merged jsonl rebuilt: {n_rows} rows  OK")


def main():
    print("=" * 70)
    print("Opus routing pipeline smoke test (Class B — NO LLM tokens)")
    print("=" * 70)
    tmp = setup_tempdir()
    print(f"tempdir: {tmp}")
    try:
        test_init_catalog()
        test_pilot_seeding()
        test_prompt_makers()
        test_resolve_and_enforce()
        test_flags_and_jsonl_rebuild()
        print()
        print("=" * 70)
        print("ALL ROUTING SMOKE TESTS PASSED")
        print("=" * 70)
        return 0
    except AssertionError as e:
        print()
        print(f"!!! SMOKE TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1
    except Exception as e:
        print()
        print(f"!!! SMOKE TEST CRASHED: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return 2
    finally:
        try:
            shutil.rmtree(tmp)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
