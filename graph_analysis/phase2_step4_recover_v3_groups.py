"""
Recover the v3 NR seed taxonomy from a CLI-truncated raw response.

The Claude CLI subprocess captured only the back half of the JSON output for
the v3 NR seed call (opening `{"node_decisions":[...],"groups":[` + first ~10
group objects were dropped before stdout reached our subprocess). The captured
remainder still contains a clean run of `{"name":..., "description":...,
"representative_indices":[...]}` group objects + the closing `"notes": "..."`.

This script regex-extracts the recoverable groups and writes a "_recovered.json"
output with note explaining the truncation. The resulting group taxonomy is
the durable artifact — node_decisions for the 150 sample nodes are throwaway
(Pass-2 will re-derive against full 2,095 NR residuals anyway).
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

STEP1 = Path("phase2_results/step1_load_and_parse_umapwithoutlocalsatellites")
raw = (STEP1 / "phase2_seed_taxonomy_nr_v3_raw.txt").read_text(encoding="utf-8")

# Match {"name":"...","description":"...","representative_indices":[...]}
pattern = re.compile(
    r'\{"name":"((?:[^"\\]|\\.)*)","description":"((?:[^"\\]|\\.)*)","representative_indices":\[([0-9,\s]*)\]\}',
    re.DOTALL,
)
matches = pattern.findall(raw)
print(f"extracted {len(matches)} complete group objects from partial v3 response")

groups = []
for name, desc, idx_str in matches:
    idxs = [int(x.strip()) for x in idx_str.split(",") if x.strip()]
    groups.append({"name": name, "description": desc, "representative_indices": idxs})

size_hist = Counter(len(g["representative_indices"]) for g in groups)
print(f"member-count histogram: {dict(size_hist)}")
sizes = [len(g["representative_indices"]) for g in groups]
print(f"min={min(sizes)}, max={max(sizes)}, mean={sum(sizes) / len(sizes):.2f}")

notes_match = re.search(r'"notes":"((?:[^"\\]|\\.)*)"', raw)
notes = notes_match.group(1) if notes_match else ""

out = {
    "pool": "nr",
    "version": "v3_abstract_mechanism_class_recovered_from_partial_response",
    "note": (
        "CLI stdout truncation lost the opening half of the JSON for the v3 NR seed call. "
        "Recovered groups via regex extraction from the captured tail. "
        "node_decisions are NOT recovered — Pass-2 re-derives them against full 2,095 NR residuals. "
        "The N="
        + str(len(groups))
        + " group taxonomy below is the durable artifact for Pass-2."
    ),
    "n_residual": 2095,
    "n_sampled": 150,
    "n_groups_recovered": len(groups),
    "parsed": {
        "groups": groups,
        "node_decisions": [],
        "notes": notes,
    },
}
out_path = STEP1 / "phase2_seed_taxonomy_nr_v3_recovered.json"
out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"saved {out_path.name} with {len(groups)} groups")
print()
print("=== Notes ===")
print(notes[:500])
print()
print("=== ALL RECOVERED GROUPS ===")
for i, g in enumerate(groups):
    n = len(g["representative_indices"])
    print(f"  {i + 1:>2}. [{n}] {g['name']}")
