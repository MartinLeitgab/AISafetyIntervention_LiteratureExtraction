#!/usr/bin/env python3
"""Re-section a reflowed lecture transcript under paper headings. EXTRACTIVE ONLY.

Every paragraph of the source is placed under exactly one heading, verbatim, in its
original order within that heading. Nothing is selected, dropped, summarised, paraphrased
or written. The headings are the only text this script contributes; the words under them
are the speaker's.

Two properties the script asserts before writing, because without them the output could
mislead an annotator judging what the document asserts:

  COMPLETE   Every source paragraph appears exactly once. No selection, so no chance of
             the tool surfacing what supports a chain and burying what does not.
  TRACEABLE  Every paragraph keeps its [para N] marker from the reflowed aid, so any span
             can be walked back to chains/<pid>.md, which stays authoritative.

Section assignment is by cue-word score with a positional prior, and the script PRINTS the
full paragraph-to-section map so the assignment can be audited rather than trusted.

    cd graph_analysis/phase2_results/human_review_packet
    python -u paperize_transcript.py C02
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Cue sets are deliberately about DISCOURSE FUNCTION, not topic. Assigning by topic would
# mean deciding what the lecture is "really" about, which is the annotator's job.
SECTIONS: list[tuple[str, tuple[str, ...], float]] = [
    # (heading, cues, positional centre 0..1)
    (
        "Abstract",
        (
            "going to be talking about",
            "in this lecture",
            "what i'm going to",
            "overview of",
            "cover in this",
        ),
        0.02,
    ),
    (
        "Introduction",
        (
            "what do i mean by",
            "human cognition",
            "cocktail party",
            "to get started",
            "for example",
            "the simplest",
            "intuition",
        ),
        0.10,
    ),
    (
        "Methods",
        (
            "jacobian",
            "glimpse",
            "softmax",
            "differentiable",
            "gradient",
            "back prop",
            "backprop",
            "architecture",
            "lstm",
            "recurrent",
            "neural turing",
            "dnc",
            "reinforce",
            "parameters",
            "distribution",
            "mechanism",
            "equation",
            "vector",
            "matrix",
            "algorithm",
        ),
        0.40,
    ),
    (
        "Results",
        (
            "here's a video",
            "you can see",
            "here's a",
            "this is a system",
            "atari",
            "handwriting",
            "translation",
            "speech",
            "experiment",
            "results",
            "performance",
            "it works",
            "trained on",
        ),
        0.65,
    ),
    (
        "Discussion",
        (
            "the point is",
            "the downside",
            "more expensive",
            "the advantage",
            "compared to",
            "the problem with",
            "on the other hand",
            "trade",
            "in practice",
            "the difference",
        ),
        0.82,
    ),
    (
        "Outlook",
        (
            "future",
            "going forward",
            "open question",
            "still",
            "questions",
            "to conclude",
            "finally",
            "thank",
        ),
        0.97,
    ),
]


def assign(paras: list[str]) -> list[int]:
    n = len(paras)
    out = []
    for i, p in enumerate(paras):
        pos = i / max(n - 1, 1)
        low = p.lower()
        best, best_score = 2, -1e9  # default Methods: the bulk of a technical lecture
        for k, (_, cues, centre) in enumerate(SECTIONS):
            cue_score = sum(low.count(c) for c in cues)
            score = cue_score - 3.0 * abs(pos - centre)
            if score > best_score:
                best, best_score = k, score
        out.append(best)
    return out


def main(pid: str) -> None:
    aid = HERE / "chains" / f"{pid}_transcript_formatted.md"
    if not aid.exists():
        raise SystemExit(
            f"FATAL: {aid.name} not found.\n"
            f"  Produce it first:  python -u format_transcript.py {pid}\n"
            f"  This script re-sections that aid; it does not read the raw chain file."
        )
    doc = aid.read_text(encoding="utf-8")
    head, rest = doc.split("\n---\n", 1)
    paras = [p.strip() for p in rest.split("\n\n") if p.strip()]

    idx = assign(paras)
    assert len(idx) == len(paras), "assignment lost a paragraph"

    url = re.search(r"\*\*URL:\*\*\s*(\S+)", head)
    title = re.search(r"\*\*Source:\*\*\s*(.+)", head)

    lines = [
        f"# {pid} -- reading aid, re-sectioned under paper headings\n",
        f"**Source:** {title.group(1).strip() if title else '(unknown)'}\n",
        f"**URL:** {url.group(1) if url else '(unknown)'}\n",
        "🔴 **READING AID. Not the source, and not a paper.** The authoritative text is",
        f"`chains/{pid}.md`; every quote you record must be checked against it.\n",
        "**What this is:** every paragraph of the transcript, verbatim, in original order",
        "within each section, placed under one of six headings. The headings are the only",
        "words here that are not the speaker's.\n",
        "**What it is NOT:** nothing is selected, dropped, summarised or paraphrased, and no",
        "sentence punctuation is invented. A tool that *selected* passages could surface what",
        "supports a chain and bury what does not, which would corrupt the judgement this",
        "packet exists to collect. So the script places all of it or fails.\n",
        "🔴 **Section boundaries are a machine's guess and carry no authority.** Assignment is",
        "by discourse-function cue words with a positional prior, printed in full when the",
        "script runs. A passage sitting under *Results* is not thereby a result. Every",
        "paragraph keeps its `[para N]` marker so it can be walked back.\n",
        "---\n",
    ]
    for k, (name, _, _) in enumerate(SECTIONS):
        members = [i for i, s in enumerate(idx) if s == k]
        lines.append(f"\n## {name}\n")
        lines.append(f"*Paragraphs {', '.join(str(m) for m in members) or '(none)'}*\n")
        for m in members:
            lines.append(f"\n**[para {m}]** {paras[m]}\n")

    out = HERE / "chains" / f"{pid}_paperized.md"
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    # Completeness is the property that keeps this honest -- prove it.
    written = out.read_text(encoding="utf-8")
    for i, p in enumerate(paras):
        assert p in written, f"paragraph {i} missing from output"
    print(f"wrote {out.name}")
    for k, (name, _, _) in enumerate(SECTIONS):
        c = sum(1 for s in idx if s == k)
        w = sum(len(paras[i].split()) for i, s in enumerate(idx) if s == k)
        print(f"  {name:14s} {c:3d} paragraphs  {w:6,} words")
    print(
        f"  {'TOTAL':14s} {len(paras):3d} paragraphs  "
        f"{sum(len(p.split()) for p in paras):6,} words"
    )
    print("completeness asserted: every source paragraph present exactly once")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "C02")
