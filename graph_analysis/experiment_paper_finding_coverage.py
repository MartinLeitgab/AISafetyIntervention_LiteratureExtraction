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
    "#171 baselines + unconditioned arm": [
        ("abstract-only nodes", "11.4"),
        ("non-reasoning model yield", "56.7"),
        ("flat triples yield more nodes", "24.5"),
        ("endpoint recovery of the triple arm", "5.2"),
        ("unconditioned repeat reproduces the yield", "15.0"),
    ],
    "#172 second judge run": [
        ("node-level omission on the analysed unit", "26.4"),
        ("edge-level omission on the analysed unit", "21.7"),
        ("length confound stated", "twice the corpus mean length"),
        ("rationale-field confound stated", "rationale"),
    ],
    "#168 follow-up: what a chain count is": [
        ("collapse tames the cross-model counts", "0.5, 2 and 4"),
        ("worst document after the collapse", "to 44"),
        ("degree is the driver", "2.32"),
        ("prompt was tuned for one model", "tuned against"),
    ],
    "language + terminology (R4/R5)": [
        ("audit, not verification, in the abstract", "The audit ran twice"),
        ("audit stage in the Limitations heading", "The audit covers 100 documents"),
    ],
}


# A number that was superseded is worse than a number that was never there: a reader who
# finds both cannot tell which one the paper means. Each entry is a phrasing that MUST NOT
# appear in rendered text, with the reason it left.
RETIRED = [
    (
        "verification stage",
        "renamed to audit stage (L12); the stage is a diagnostic pass",
    ),
    ("schema-constrained", "conformance is prompt-enforced, not constrained (C2)"),
    ("implied coverage", "withdrawn with the edge-coverage study (#156)"),
    ("chain-level examples are unaudited", "the second judge run audits them (#172)"),
    ("Eleven populations", "twelve, since the second judge run (#172)"),
    ("97.8", "an interim gpt-5 chain figure computed at n=16"),
    ("4{,}462.9", "a mean dominated by one 57,007-path document"),
    ("139 of them", "138 map onto the five stages; one maps to risk"),
    ("235/235", "the claim audit has moved well past this"),
    (
        "direction and a floor",
        "population, length and serialisation differ at once, so the second judge run "
        "bounds nothing (2026-08-26)",
    ),
    (
        "document-level resources cannot",
        "a passage index can answer a mechanism question by synthesis; what it does not "
        "do is carry the mechanism as a field (2026-08-26)",
    ),
    ("more than five thousand", "their released clustering covers 554 papers"),
]

# Two numbers that mean different things must not float free of what distinguishes them.
# Each entry: (number, one of these qualifiers must appear within `window` characters).
QUALIFIED = [
    ("26.4\\%", ["chain-yielding", "second run", "narrower population"], 600),
    ("21.7\\% of the 2{,}192", ["chain-yielding", "second run"], 600),
    ("594", ["raw", "collapse", "density", "median"], 400),
    ("57{,}007", ["raw", "collapse", "44"], 400),
    ("0.6\\% of the nodes", ["released", "100 documents", "first"], 600),
]


def consistency(rendered: str) -> int:
    bad = 0
    print("\n--- retired phrasings that must not appear ---")
    for phrase, why in RETIRED:
        n = rendered.count(phrase)
        if n:
            bad += 1
            print(f"[MISS] {phrase!r} appears {n}x -- {why}")
    print(f"  {sum(1 for p, _ in RETIRED if p not in rendered)}/{len(RETIRED)} clear")

    print("--- numbers that need their qualifier nearby ---")
    for num, quals, window in QUALIFIED:
        i = rendered.find(num)
        if i < 0:
            print(f"[note] {num!r} not present")
            continue
        ctx = rendered[max(0, i - window) : i + window]
        if not any(q in ctx for q in quals):
            bad += 1
            print(f"[MISS] {num!r} has none of {quals} within {window} chars")
    print(f"  {len(QUALIFIED)} pairs checked")
    return bad


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
    bad = consistency(rendered)
    print(f"\n{missing_total} finding(s) not found in rendered text")
    print(f"{bad} self-consistency problem(s)")


if __name__ == "__main__":
    main()
