#!/usr/bin/env python
"""Did every finding from the run list reach the manuscript?

The claim audit checks the forward direction: every number printed in the .tex matches a
receipt. This checks the reverse: every headline finding of every completed study appears
in the rendered text (not in a comment). Absence here is a lost result, not a wrong one.
"""

import re
from pathlib import Path

TEX = Path(
    "C:/Users/malei/0_project_work/eleutherAI_SOAR_step1knowledgegraphcreation/"
    "AISafetyIntervention_PaperA_shared/paperA_altstyle.tex"
)

# study -> [(what the finding is, a token that must appear in RENDERED text)]
FINDINGS = {
    "#152/#154 extraction cost": [
        ("total input tokens", "122.4M"),
        ("dollar band per 1,000 documents", "32 to 118"),
        ("cost moves by a factor of five across extractors", "factor of five"),
    ],
    "#153 stage separability": [
        ("probe accuracy", "98.8"),
        ("name-only TF-IDF ablation", "69.4"),
        ("centroid margins", "0.054"),
    ],
    "#156/#158 edge coverage": [
        ("edge-level omission", "18.1"),
        ("coverage-list rows", "777"),
        ("missing rows", "302"),
        ("no add_edges slot", "none for added edges"),
    ],
    "#157/#159 containment + release integrity": [
        ("no drop is a contiguous sub-path", "None of them is a contiguous sub-path"),
        ("chords-only share", "21.7"),
        ("distinct pairs lost", "18.0"),
        ("nodes lost", "6.1"),
        ("release ships no orphans", "no orphan"),
        ("exact-name residue", "1{,}140"),
    ],
    "#161/#162 stage agreement": [
        ("Cohen kappa", "0.84"),
        ("weakest stage", "0.756"),
        ("prediction half right", "half right"),
    ],
    "#163/#164 null-repair arm": [
        ("the stage is reported as confounded", "no null-repair arm"),
    ],
    "#165/#169 ablation": [
        ("schema-blind chain yield", "11 of the 28"),
        ("no ablated chain carries five stages", "no chain carries all five stages"),
        ("emergent labels", "144 distinct category labels"),
        ("labels mapping onto the five", "138"),
        ("shuffled-source yield", "30.0"),
        ("shuffled all-five share", "46.3"),
        ("reference-list arm", "2 of 25"),
        ("refusals", "returned no graph"),
        ("undirected traversal", "follows edges in both"),
        ("appendix protocol", "app:ablation"),
    ],
    "#166/#167 artifact comparison": [
        ("their input corpus", "7{,}011"),
        ("clustered papers matched", "534"),
        ("labels against pairs", "325 distinct"),
        ("co-membership converse", "2{,}986"),
        ("yield cross-check", "40.4"),
    ],
    "#168/#170 multi-model": [
        ("re-run node count", "21.1"),
        ("chain yield on re-run", "9 of the 18"),
        ("maturity split", "11 of them"),
        ("cosine agreement", "46.5"),
        ("lexical agreement", "19.0"),
        ("second model node count", "38.1"),
    ],
    "language + terminology (R4/R5)": [
        ("audit, not verification, in the abstract", "The audit ran on a"),
        ("audit stage in the Limitations heading", "The audit covers 100 documents"),
    ],
}


def main() -> None:
    raw = TEX.read_text(encoding="utf-8")
    rendered = "\n".join(
        line for line in raw.split("\n") if not line.lstrip().startswith("%")
    )
    # strip inline trailing comments too
    rendered = re.sub(r"(?<!\\)%.*", "", rendered)

    missing_total = 0
    for study, items in FINDINGS.items():
        misses = [(what, tok) for what, tok in items if tok not in rendered]
        status = "OK " if not misses else "MISS"
        print(f"[{status}] {study}: {len(items) - len(misses)}/{len(items)} present")
        for what, tok in misses:
            print(f"        NOT IN RENDERED TEXT: {what}  (looked for {tok!r})")
        missing_total += len(misses)
    print(f"\n{missing_total} finding(s) not found in rendered text")


if __name__ == "__main__":
    main()
