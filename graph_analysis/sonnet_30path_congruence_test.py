"""Sonnet 4.6 vs Opus 30-path congruence test.

Apples-to-apples comparison:
  - Catalog:  archived smoke catalog (20 RG + 27 MG that Opus created via combined
              seed in the 30-path smoke earlier today)
  - Paths:    the same 30 path_ids that Opus assigned in that smoke
  - Task:     assignments-only, read-only catalog (allow_new=False,
              allow_coherence=False) — Sonnet must pick from existing groups
  - Compare:  Sonnet's (RG, MG) per path vs Opus's (RG, MG) per path from the
              archived file
  - Report:   exact-match rate, RG-only-match, MG-only-match, disagreements
              per-pool agreement rate (RG agreement %, MG agreement %)

If Sonnet's exact-doublet match is high (>80% expected for routine routing),
the hybrid pipeline (Sonnet for bulk Pass B + periodic Opus review) is viable.

Cost (estimated, single Sonnet call):
  Input:  ~47k tokens (30 paths + 47 group entries + system)
  Output: ~900 tokens (30 assignments JSON)
  Shim:   ~30k preamble
  Total:  ~80k tokens. On Max plan Sonnet likely counts at ~1/3 Opus weight,
          so ~+4-12pp session depending on Max-plan policy.
"""

import json
import sys
import uuid
from pathlib import Path
from collections import Counter

# Load the main module (gives access to streaming_claude_call, fmt_path, etc.)
sys.path.insert(0, str(Path(__file__).parent))
import phase2_step4_phase2_doublet_llm_grouping as mod

# ============================================================
# Inputs
# ============================================================
SMOKE_CATALOG = (
    mod.STEP1 / "archive" / "phase2_doublet_seed_catalog_v2_smoke_30paths.json"
)
TEST_PARTIAL = Path("logfiles/sonnet_30path_test_partial.txt")
REPORT_PATH = Path("logfiles/sonnet_30path_congruence_report.json")

if not SMOKE_CATALOG.exists():
    print(f"ERROR: smoke catalog missing at {SMOKE_CATALOG}", flush=True)
    print(
        "  This is produced by `python phase2_step4_phase2_doublet_llm_grouping.py "
        "--mode seed --n-paths 30` (smoke run), then archived. This script "
        "does NOT regenerate or fall back to any other source.",
        flush=True,
    )
    sys.exit(1)

smoke = json.loads(SMOKE_CATALOG.read_text(encoding="utf-8"))
print(
    f"loaded smoke catalog: {smoke['n_risk_groups']} RG, "
    f"{smoke['n_mechanism_groups']} MG, "
    f"{len(smoke['input_path_ids'])} input paths, "
    f"{len(smoke['assignments'])} Opus assignments",
    flush=True,
)

opus_assignments_by_pid = {}
for a in smoke["assignments"]:
    pid = a["path_id"]

    # Schema in archived smoke: risk_group/mechanism_group could be {"existing": "RG017"}
    # or {"new": {...}} — but archived smoke was option-A so all are "existing" since
    # smoke generated catalog + assignments in one shot. Just extract group_id.
    def _extract_gid(field):
        if isinstance(field, dict):
            if "existing" in field:
                return field["existing"]
            if "new" in field:
                # New group was minted; the catalog has it. Cross-reference by name.
                return None  # placeholder — we'll resolve via catalog
        return field

    rg = _extract_gid(a.get("risk_group") or a.get("risk_group_id"))
    mg = _extract_gid(a.get("mechanism_group") or a.get("mechanism_group_id"))
    # The archived smoke has assignments already resolved to RG###/MG### IDs
    # in the catalog. Just use them directly if they are strings.
    if rg is None:
        rg = a.get("risk_group_id")
    if mg is None:
        mg = a.get("mechanism_group_id")
    opus_assignments_by_pid[pid] = {"rg": rg, "mg": mg}

# Sanity check
print(f"  parsed Opus assignments: {len(opus_assignments_by_pid)} paths", flush=True)
sample_pid = next(iter(opus_assignments_by_pid))
print(f"  example: {sample_pid} -> {opus_assignments_by_pid[sample_pid]}", flush=True)

# ============================================================
# Load paths + node_attrs, filter to the 30 smoke paths
# ============================================================
paths, node_attrs = mod.load_paths_and_attrs()
smoke_pids = set(smoke["input_path_ids"])
sample = [p for p in paths if p["path_id"] in smoke_pids]
print(f"loaded {len(sample)} paths matching smoke input_path_ids", flush=True)
if len(sample) != len(smoke_pids):
    print(f"  WARNING: expected {len(smoke_pids)} paths, got {len(sample)}", flush=True)

# ============================================================
# Build prompt (read-only catalog, no new groups, no coherence)
# ============================================================
sentinel = uuid.uuid4().hex[:12]
prompt = mod.make_assign_prompt(
    sample,
    node_attrs,
    smoke["risk_groups"],
    smoke["mechanism_groups"],
    sentinel,
    allow_new=False,
    allow_coherence=False,
)
print(f"prompt: {len(prompt)} chars (~{len(prompt) // 4} tokens)", flush=True)

system_prompt = (
    "You produce STRICT JSON output for an AI-safety doublet "
    "grouping pipeline. Never preamble, never use markdown fences, "
    "always emit valid JSON, always end your output with the "
    "requested sentinel."
)

# ============================================================
# Call Sonnet 4.6
# ============================================================
print(flush=True)
print("=" * 60, flush=True)
print("LAUNCHING SONNET 4.6 ASSIGNMENT CALL", flush=True)
print("=" * 60, flush=True)
text, dur, err = mod.streaming_claude_call(
    prompt, system_prompt, TEST_PARTIAL, model="claude-sonnet-4-6"
)
print(f"sonnet returned: {len(text)} chars in {dur:.1f}s, err={err}", flush=True)

if err:
    print(f"SONNET CALL FAILED: {err}", flush=True)
    print(f"  partial output preserved at: {TEST_PARTIAL}", flush=True)
    sys.exit(2)

# ============================================================
# Parse Sonnet's output
# ============================================================
end_marker = f"END_SENTINEL_{sentinel}"
trimmed = text.strip()
if not (trimmed.startswith("{") and trimmed.endswith(end_marker)):
    print(
        f"FAIL sentinel validation; first/last 100: {trimmed[:100]!r} ... {trimmed[-100:]!r}",
        flush=True,
    )
    sys.exit(3)
json_part = trimmed[: -len(end_marker)].rstrip()
try:
    parsed = json.loads(json_part)
except json.JSONDecodeError as e:
    print(f"FAIL JSON parse: {e}", flush=True)
    sys.exit(4)

sonnet_assignments_by_pid = {}
for a in parsed.get("assignments", []):
    pid = a["path_id"]
    rg_field = a.get("risk_group", {})
    mg_field = a.get("mechanism_group", {})
    rg = rg_field.get("existing") if isinstance(rg_field, dict) else rg_field
    mg = mg_field.get("existing") if isinstance(mg_field, dict) else mg_field
    sonnet_assignments_by_pid[pid] = {"rg": rg, "mg": mg}

print(f"parsed Sonnet assignments: {len(sonnet_assignments_by_pid)} paths", flush=True)

# ============================================================
# Congruence analysis
# ============================================================
common_pids = set(opus_assignments_by_pid) & set(sonnet_assignments_by_pid)
print(f"common path_ids for comparison: {len(common_pids)}", flush=True)

rg_match = 0
mg_match = 0
both_match = 0
neither_match = 0
disagreements = []

for pid in sorted(common_pids):
    opus = opus_assignments_by_pid[pid]
    sonnet = sonnet_assignments_by_pid[pid]
    rg_ok = opus["rg"] == sonnet["rg"]
    mg_ok = opus["mg"] == sonnet["mg"]
    if rg_ok and mg_ok:
        both_match += 1
        rg_match += 1
        mg_match += 1
    elif rg_ok:
        rg_match += 1
        disagreements.append(
            {"pid": pid, "type": "MG_only_diff", "opus": opus, "sonnet": sonnet}
        )
    elif mg_ok:
        mg_match += 1
        disagreements.append(
            {"pid": pid, "type": "RG_only_diff", "opus": opus, "sonnet": sonnet}
        )
    else:
        neither_match += 1
        disagreements.append(
            {"pid": pid, "type": "BOTH_diff", "opus": opus, "sonnet": sonnet}
        )

n = len(common_pids)
print(flush=True)
print("=" * 60, flush=True)
print("CONGRUENCE REPORT", flush=True)
print("=" * 60, flush=True)
print(f"common paths:              {n}", flush=True)
print(
    f"BOTH match  (exact):       {both_match}/{n} ({100 * both_match / n:.1f}%)",
    flush=True,
)
print(
    f"RG matches  (any):         {rg_match}/{n} ({100 * rg_match / n:.1f}%)", flush=True
)
print(
    f"MG matches  (any):         {mg_match}/{n} ({100 * mg_match / n:.1f}%)", flush=True
)
print(
    f"BOTH differ:               {neither_match}/{n} ({100 * neither_match / n:.1f}%)",
    flush=True,
)
print(flush=True)
if disagreements:
    print("Disagreements (first 10):", flush=True)
    for d in disagreements[:10]:
        print(
            f"  {d['pid']} [{d['type']}]: opus={d['opus']}, sonnet={d['sonnet']}",
            flush=True,
        )

# Group-coverage check — did Sonnet use a similar distribution of groups?
opus_rg_dist = Counter(a["rg"] for a in opus_assignments_by_pid.values())
sonnet_rg_dist = Counter(a["rg"] for a in sonnet_assignments_by_pid.values())
opus_mg_dist = Counter(a["mg"] for a in opus_assignments_by_pid.values())
sonnet_mg_dist = Counter(a["mg"] for a in sonnet_assignments_by_pid.values())

print(flush=True)
print("Top-5 RG distribution comparison:", flush=True)
print(f"  Opus:   {opus_rg_dist.most_common(5)}", flush=True)
print(f"  Sonnet: {sonnet_rg_dist.most_common(5)}", flush=True)
print("Top-5 MG distribution comparison:", flush=True)
print(f"  Opus:   {opus_mg_dist.most_common(5)}", flush=True)
print(f"  Sonnet: {sonnet_mg_dist.most_common(5)}", flush=True)

# Save report to disc
report = {
    "n_common_paths": n,
    "both_match": both_match,
    "rg_match": rg_match,
    "mg_match": mg_match,
    "neither_match": neither_match,
    "sonnet_duration_sec": dur,
    "sonnet_output_chars": len(text),
    "disagreements": disagreements,
    "opus_rg_distribution": dict(opus_rg_dist),
    "sonnet_rg_distribution": dict(sonnet_rg_dist),
    "opus_mg_distribution": dict(opus_mg_dist),
    "sonnet_mg_distribution": dict(sonnet_mg_dist),
}
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(
    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(flush=True)
print(f"wrote congruence report: {REPORT_PATH}", flush=True)
