"""unit_test_opus_review.py — Class B (no LLM tokens) smoke test for REVIEW_A/B/C
helpers in phase2_step4_phase2_doublet_llm_grouping.py.

Exercises:
  - prompt makers (make_review_a_prompt, make_review_b_prompt, make_review_c_prompt)
  - apply helpers (_apply_review_a_simple, _apply_review_b_decision, _apply_review_c_assignments)
  - remap persistence (_write_group_remap, _write_path_remap, _resolve_remap_transitive)
  - rebuild_merged_assignments_jsonl end-to-end with remaps + REVIEW_C output

Uses synthetic catalog + assignments in a tempdir; monkey-patches STEP1.
NO LLM CALLS. Should run in <5 seconds.
"""

from __future__ import annotations
import json
import sys
import tempfile
from pathlib import Path

import phase2_step4_phase2_doublet_llm_grouping as M


def _seed_catalog():
    return {
        "n_risk_groups": 3,
        "n_mechanism_groups": 3,
        "risk_groups": [
            {
                "group_id": "RG001",
                "group_name": "Misalignment",
                "group_description": "Goal-misalignment risk.",
            },
            {
                "group_id": "RG002",
                "group_name": "Reward hacking",
                "group_description": "Specification gaming.",
            },
            {
                "group_id": "RG003",
                "group_name": "Reward hack",
                "group_description": "Same as RG002 by mistake.",
            },
        ],
        "mechanism_groups": [
            {
                "group_id": "MG001",
                "group_name": "RLHF",
                "group_description": "Pref learning.",
            },
            {
                "group_id": "MG002",
                "group_name": "Constitutional AI",
                "group_description": "Self-critique training.",
            },
            {
                "group_id": "MG003",
                "group_name": "Debate",
                "group_description": "Adversarial dialog.",
            },
        ],
        "assignments": [
            {
                "path_id": f"path_{i:05d}",
                "risk_group_id": "RG001",
                "mechanism_group_id": "MG001",
            }
            for i in range(1, 5)
        ]
        + [
            {
                "path_id": f"path_{i:05d}",
                "risk_group_id": "RG002",
                "mechanism_group_id": "MG002",
            }
            for i in range(5, 9)
        ]
        + [
            {
                "path_id": "path_00009",
                "risk_group_id": "RG003",
                "mechanism_group_id": "MG003",
            },
        ],
    }


def _passb_batch():
    return {
        "batch_idx": 0,
        "model": "claude-sonnet-4-6",
        "n_input_paths": 5,
        "duration_sec": 60,
        "assignments": [],
        "resolved_assignments": [
            {
                "path_id": "path_00010",
                "risk_group_id": "RG001",
                "mechanism_group_id": "MG002",
            },
            {
                "path_id": "path_00011",
                "risk_group_id": "RG002",
                "mechanism_group_id": "MG001",
            },
            {
                "path_id": "path_00012",
                "risk_group_id": "RG003",
                "mechanism_group_id": "MG003",
            },
        ],
        "unassigned_rows": [
            {
                "path_id": "path_00013",
                "risk_group_id": None,
                "mechanism_group_id": "MG001",
                "risk_unassigned_reason": "Novel risk class about deceptive alignment",
                "mechanism_unassigned_reason": None,
                "batch_idx": 0,
            },
            {
                "path_id": "path_00014",
                "risk_group_id": "RG001",
                "mechanism_group_id": None,
                "risk_unassigned_reason": None,
                "mechanism_unassigned_reason": "Latent representation steering not in catalog",
                "batch_idx": 0,
            },
        ],
    }


def _fake_path_record(path_id):
    return {
        "path_id": path_id,
        "path": [101, 102, 103],
        "categories": ["risk", "design_rationale", "intervention"],
    }


def _fake_node_attrs():
    return {
        101: {
            "name": "Misaligned objective",
            "description": "Toy risk node.",
            "type": "concept",
        },
        102: {
            "name": "Constitutional self-critique",
            "description": "Toy body node.",
            "type": "concept",
            "concept_category": "design_rationale",
        },
        103: {
            "name": "RLHF preference trainer",
            "description": "Toy intervention node.",
            "type": "intervention",
        },
    }


def setup_tempdir():
    tmp = Path(tempfile.mkdtemp(prefix="opus_review_smoke_"))
    step1 = tmp / "phase2_results" / "step1_load_and_parse_umapwithoutlocalsatellites"
    step1.mkdir(parents=True, exist_ok=True)
    # Override STEP1 in the module
    M.STEP1 = step1
    # Drop in seed catalog
    seed_fp = step1 / "phase2_doublet_seed_catalog.json"
    seed_fp.write_text(json.dumps(_seed_catalog(), indent=2), encoding="utf-8")
    # Drop in one Sonnet Pass B batch
    passb_dir = step1 / "phase2_doublet_passb_batches"
    passb_dir.mkdir(parents=True, exist_ok=True)
    (passb_dir / "batch_0000.json").write_text(
        json.dumps(_passb_batch(), indent=2), encoding="utf-8"
    )
    # Unassigned log
    unassigned = _passb_batch()["unassigned_rows"]
    with open(
        step1 / "phase2_doublet_passb_unassigned.jsonl", "w", encoding="utf-8"
    ) as f:
        for r in unassigned:
            f.write(json.dumps(r) + "\n")
    return tmp


def test_prompt_makers():
    print("[1] prompt-maker tests")
    sentinel = "abc123def456"
    rg_list = _seed_catalog()["risk_groups"]
    mg_list = _seed_catalog()["mechanism_groups"]
    rg_counts = {"RG001": 4, "RG002": 4, "RG003": 1}
    mg_counts = {"MG001": 4, "MG002": 4, "MG003": 1}
    themes = ["(risk, seen 1x) Novel risk class about deceptive alignment"]
    p_a = M.make_review_a_prompt(
        rg_list, mg_list, rg_counts, mg_counts, themes, sentinel
    )
    assert f"END_SENTINEL_{sentinel}" in p_a, "REVIEW_A prompt missing sentinel"
    assert "RG001" in p_a and "MG001" in p_a, "REVIEW_A prompt missing group ids"
    assert "rg_decisions" in p_a and "mg_decisions" in p_a, "REVIEW_A schema missing"
    print(f"    REVIEW_A prompt OK ({len(p_a)} chars)")

    fake_member = {
        "path_id": "path_00001",
        "risk_group_id": "RG002",
        "mechanism_group_id": "MG002",
        "fmt_path_block": "[path_00001]\n  risk: x -- y\n  body[dr]: ...\n  interv: ...",
    }
    p_b = M.make_review_b_prompt(rg_list[1], "risk", [fake_member] * 3, sentinel)
    assert f"END_SENTINEL_{sentinel}" in p_b, "REVIEW_B prompt missing sentinel"
    assert "subgroups" in p_b and "decision" in p_b, "REVIEW_B schema missing"
    print(f"    REVIEW_B prompt OK ({len(p_b)} chars)")

    fake_unass = [
        {
            "path_id": "path_00013",
            "risk_group_id": None,
            "mechanism_group_id": "MG001",
            "risk_unassigned_reason": "Novel risk class",
            "mechanism_unassigned_reason": None,
            "fmt_path_block": "[path_00013]\n  risk: x -- y\n  interv: z -- w",
        }
    ]
    p_c = M.make_review_c_prompt(fake_unass, rg_list, mg_list, sentinel)
    assert f"END_SENTINEL_{sentinel}" in p_c, "REVIEW_C prompt missing sentinel"
    assert "assignments" in p_c and "path_00013" in p_c, "REVIEW_C content missing"
    print(f"    REVIEW_C prompt OK ({len(p_c)} chars)")


def test_apply_review_a_simple():
    print("[2] _apply_review_a_simple tests")
    rg_list = _seed_catalog()["risk_groups"]
    decisions = [
        {"group_id": "RG001", "decision": "keep"},
        {
            "group_id": "RG002",
            "decision": "rename",
            "new_name": "Specification gaming",
            "new_description": "Reward hacking / spec gaming.",
        },
        {"group_id": "RG003", "decision": "merge", "target_group_id": "RG002"},
    ]
    new_list, flagged, merge_remap = M._apply_review_a_simple(
        rg_list, decisions, "risk"
    )
    assert len(new_list) == 2, f"expected 2 RG after merge, got {len(new_list)}"
    assert merge_remap == {"RG003": "RG002"}, f"merge_remap wrong: {merge_remap}"
    assert len(flagged) == 0, f"expected 0 flagged, got {len(flagged)}"
    renamed = next(g for g in new_list if g["group_id"] == "RG002")
    assert renamed["group_name"] == "Specification gaming", "rename not applied"

    # invalid merge target → skipped
    bad_decisions = [
        {"group_id": "RG001", "decision": "merge", "target_group_id": "RG999"}
    ]
    new_list2, _, merge2 = M._apply_review_a_simple(rg_list, bad_decisions, "risk")
    assert merge2 == {}, "invalid merge target should be skipped"

    # deep_dive flag
    deep_decisions = [
        {
            "group_id": "RG002",
            "decision": "deep_dive",
            "deep_dive_reason": "Suspected fragmentation",
        }
    ]
    _, flagged2, _ = M._apply_review_a_simple(rg_list, deep_decisions, "risk")
    assert len(flagged2) == 1 and flagged2[0]["group_id"] == "RG002", (
        "deep_dive not flagged"
    )
    print("    OK")


def test_apply_review_b_decision():
    print("[3] _apply_review_b_decision tests")
    rg_list = _seed_catalog()["risk_groups"]
    group = rg_list[1]  # RG002
    # split into two subgroups
    decision = {
        "decision": "split",
        "rationale": "Two distinct mechanisms detected.",
        "subgroups": [
            {
                "new_name": "Reward-model exploitation",
                "new_description": "RM gaming.",
                "path_ids": ["path_00005", "path_00006", "path_00007"],
            },
            {
                "new_name": "Goal misgeneralization",
                "new_description": "OOD gen failures.",
                "path_ids": ["path_00008"],
            },  # < MIN_GROUP_SIZE; should fail validation
        ],
    }
    new_list, merge_add, path_overrides = M._apply_review_b_decision(
        group, "risk", decision, rg_list
    )
    # Only one valid subgroup → should be treated as keep
    assert len(new_list) == len(rg_list), (
        "split with <2 valid subgroups should be no-op"
    )
    assert merge_add == {}, "no merge expected"
    assert path_overrides == {}, "no path overrides expected"
    print("    split with <2 valid subgroups -> no-op: OK")

    # Now a valid split (both subgroups >= MIN_GROUP_SIZE)
    decision_v2 = {
        "decision": "split",
        "subgroups": [
            {
                "new_name": "Reward-model exploitation",
                "new_description": "RM gaming.",
                "path_ids": ["p1", "p2", "p3"],
            },
            {
                "new_name": "Goal misgeneralization",
                "new_description": "OOD gen failures.",
                "path_ids": ["p4", "p5", "p6"],
            },
        ],
    }
    new_list2, merge_add2, path_overrides2 = M._apply_review_b_decision(
        group, "risk", decision_v2, rg_list
    )
    assert any(g["group_name"] == "Reward-model exploitation" for g in new_list2), (
        "new split subgroup not appended"
    )
    assert "RG002" not in {g["group_id"] for g in new_list2}, (
        "original split group not removed"
    )
    assert merge_add2.get("RG002") == "SPLIT_REMOVED", "split-removal flag missing"
    assert "p1" in path_overrides2 and "risk_group_id" in path_overrides2["p1"], (
        "path override missing"
    )
    print("    valid split: new groups appended + path_overrides set: OK")

    # merge_with valid case
    decision_merge = {"decision": "merge_with", "target_group_id": "RG001"}
    new_list3, merge_add3, _ = M._apply_review_b_decision(
        group, "risk", decision_merge, rg_list
    )
    assert merge_add3.get("RG002") == "RG001", "merge_with not applied"
    assert "RG002" not in {g["group_id"] for g in new_list3}, "merged group not removed"
    print("    merge_with: OK")

    # rename
    decision_rename = {
        "decision": "rename",
        "new_name": "Reward gaming v2",
        "new_description": "Updated.",
    }
    rg_list_copy = json.loads(json.dumps(rg_list))  # deep copy
    grp = rg_list_copy[1]
    new_list4, _, _ = M._apply_review_b_decision(
        grp, "risk", decision_rename, rg_list_copy
    )
    assert (
        next(g for g in new_list4 if g["group_id"] == "RG002")["group_name"]
        == "Reward gaming v2"
    )
    print("    rename: OK")


def test_remap_persistence_and_rebuild(tmp_root):
    print("[4] remap persistence + rebuild_merged_assignments_jsonl tests")
    # Write a group_remap
    M._write_group_remap({"RG003": "RG002"}, {"MG003": "MG002"})
    rg_remap, mg_remap = M._load_group_remap()
    assert rg_remap == {"RG003": "RG002"}, f"rg_remap wrong: {rg_remap}"
    assert mg_remap == {"MG003": "MG002"}, f"mg_remap wrong: {mg_remap}"
    print("    write+read group_remap: OK")

    # Add a chained remap and check transitive resolution
    M._write_group_remap({"RG999": "RG003"}, {})  # RG999 -> RG003 -> RG002
    rg_remap2, _ = M._load_group_remap()
    assert rg_remap2.get("RG999") == "RG002", (
        f"transitive resolution failed: {rg_remap2}"
    )
    print("    transitive chain RG999 -> RG003 -> RG002 resolved to RG002: OK")

    # Path remap
    M._write_path_remap({"path_00005": {"risk_group_id": "RG_NEW1"}})
    pr = M._load_path_remap()
    assert pr.get("path_00005", {}).get("risk_group_id") == "RG_NEW1", (
        "path_remap not persisted"
    )
    print("    write+read path_remap: OK")

    # Rebuild merged jsonl and verify remaps applied
    out, n_rows = M.rebuild_merged_assignments_jsonl()
    rows = [
        json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_pid = {r["path_id"]: r for r in rows}
    assert by_pid["path_00005"]["risk_group_id"] == "RG_NEW1", (
        "path_remap override not applied"
    )
    assert by_pid["path_00009"]["risk_group_id"] == "RG002", (
        "group_remap (RG003->RG002) not applied"
    )
    assert by_pid["path_00009"]["mechanism_group_id"] == "MG002", (
        "MG003->MG002 not applied"
    )
    assert by_pid["path_00012"]["risk_group_id"] == "RG002", (
        "Pass B batch row not remapped"
    )
    print(
        f"    rebuild_merged_assignments_jsonl: {n_rows} rows, remaps applied correctly"
    )


def test_apply_review_c_assignments(tmp_root):
    print("[5] _apply_review_c_assignments tests")
    seed = _seed_catalog()
    rg_list = list(seed["risk_groups"])
    mg_list = list(seed["mechanism_groups"])
    parsed = {
        "assignments": [
            {
                "path_id": "path_00013",
                "risk_group": {
                    "new": {
                        "group_name": "Deceptive alignment",
                        "group_description": "Strategic deception risk.",
                    }
                },
                "mechanism_group": {"existing": "MG001"},
            },
            {
                "path_id": "path_00014",
                "risk_group": {"existing": "RG001"},
                "mechanism_group": {
                    "new": {
                        "group_name": "Latent steering",
                        "group_description": "Activation engineering.",
                    }
                },
            },
        ]
    }
    n_before_rg = len(rg_list)
    n_before_mg = len(mg_list)
    rg_list2, mg_list2, resolved, out_path = M._apply_review_c_assignments(
        parsed, rg_list, mg_list, 1
    )
    assert len(rg_list2) == n_before_rg + 1, (
        f"new RG not appended ({len(rg_list2)} vs {n_before_rg}+1)"
    )
    assert len(mg_list2) == n_before_mg + 1, (
        f"new MG not appended ({len(mg_list2)} vs {n_before_mg}+1)"
    )
    assert any(r["risk_group_id"] == "RG004" for r in resolved), "new RG id not used"
    assert any(r["mechanism_group_id"] == "MG004" for r in resolved), (
        "new MG id not used"
    )
    assert out_path.exists(), "review_c output file not written"
    print(
        f"    new groups appended (RG004, MG004); resolved={len(resolved)}; out={out_path.name}"
    )

    # Rebuild merged jsonl — REVIEW_C output should be included
    _, n_rows = M.rebuild_merged_assignments_jsonl()
    rows = [
        json.loads(line)
        for line in (M._passb_paths()["merged_assignments"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    by_pid = {r["path_id"]: r for r in rows}
    assert "path_00013" in by_pid, "review_c assignment not picked up by rebuild_merged"
    assert by_pid["path_00013"]["risk_group_id"] == "RG004", (
        "new RG not assigned correctly"
    )
    print(f"    rebuild_merged sees REVIEW_C output: {n_rows} rows total")


def test_unassigned_themes():
    print("[6] _extract_unassigned_themes tests")
    rows = [
        {
            "risk_unassigned_reason": "Novel risk class",
            "mechanism_unassigned_reason": None,
        },
        {
            "risk_unassigned_reason": "Novel risk class",
            "mechanism_unassigned_reason": "Latent steering",
        },
        {
            "risk_unassigned_reason": None,
            "mechanism_unassigned_reason": "Latent steering",
        },
    ]
    themes = M._extract_unassigned_themes(rows, k=5)
    assert any("Novel risk class" in t for t in themes), (
        "common risk reason not in themes"
    )
    assert any("Latent steering" in t for t in themes), (
        "common mech reason not in themes"
    )
    assert any("seen 2x" in t for t in themes), "frequency annotation missing"
    print("    OK")


def main():
    print("=" * 70)
    print("Opus REVIEW_A/B/C smoke test (Class B — NO LLM tokens)")
    print("=" * 70)
    tmp = setup_tempdir()
    print(f"tempdir: {tmp}")
    try:
        test_prompt_makers()
        test_apply_review_a_simple()
        test_apply_review_b_decision()
        test_remap_persistence_and_rebuild(tmp)
        test_apply_review_c_assignments(tmp)
        test_unassigned_themes()
        print()
        print("=" * 70)
        print("ALL SMOKE TESTS PASSED")
        print("=" * 70)
        return 0
    except AssertionError as e:
        print()
        print("!!! SMOKE TEST FAILED !!!")
        print(f"AssertionError: {e}")
        import traceback

        traceback.print_exc()
        return 1
    except Exception as e:
        print()
        print("!!! SMOKE TEST CRASHED !!!")
        print(f"{type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
