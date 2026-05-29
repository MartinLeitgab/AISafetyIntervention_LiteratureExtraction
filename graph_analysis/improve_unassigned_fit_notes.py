"""improve_unassigned_fit_notes.py — Class B (no LLM). Add topic-descriptor
tags to fit_notes of unassigned paths so future routing batches can detect
same-topic peers via grep on the attention queue.

The misfit_review confirmed these are honest non-AI-safety gaps. The "not AI
safety" framing is the NEGATIVE descriptor; this script adds the POSITIVE
topic descriptor (what the path IS) so peer detection works at scale.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import phase2_step5_opus_routing as R

# Topic descriptors from misfit_review_001 + path traversal inspection
TOPIC_TAGS = {
    "path_00385_dedup": "topic=math-pedagogy (abstract algebra reading curriculum)",
    "path_00576_dedup": "topic=process-mining (BPM SAT-precision verification, non-AI)",
    "path_00578_dedup": "topic=education-coordination (classroom team-formation tool, non-AI)",
    "path_01166_dedup": "topic=auction-theory (financial bidding agent, non-AI-safety)",
    "path_01879_dedup": "topic=biosecurity-engineering (Far-UVC public-health, non-AI)",
    "path_02038_dedup": "topic=marketing-causal-inference (ad holdout Goodhart metaphorical, non-AI)",
}


def main():
    if not R.ASSIGNMENTS_FP.exists():
        sys.exit(f"ERROR: {R.ASSIGNMENTS_FP} missing")
    rows = []
    for line in R.ASSIGNMENTS_FP.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))

    updated = 0
    for r in rows:
        pid = r["path_id"]
        if pid in TOPIC_TAGS:
            old = r.get("fit_note", "") or ""
            tag = TOPIC_TAGS[pid]
            if tag not in old:
                # Prepend topic tag, keep prior reason
                r["fit_note"] = f"{tag}; {old}" if old else tag
                r.setdefault("history", []).append(
                    {
                        "edit": f"fit_note topic tag added: {tag}",
                        "date": "2026-05-18",
                    }
                )
                updated += 1

    with open(R.ASSIGNMENTS_FP, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"updated {updated} fit_notes with topic descriptor tags", flush=True)
    print(
        "Now if future paths surface with topic=math-pedagogy or "
        "topic=auction-theory or topic=biosecurity-engineering or ..., "
        "the attention queue will surface them as same-topic peers.",
        flush=True,
    )


if __name__ == "__main__":
    main()
