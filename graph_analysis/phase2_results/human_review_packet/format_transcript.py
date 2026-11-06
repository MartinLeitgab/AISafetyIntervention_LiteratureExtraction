#!/usr/bin/env python3
"""Reflow a YouTube auto-caption dump in a packet chain file into readable paragraphs.

LOSSLESS ON WORDS. Only whitespace changes: hard-wrapped caption lines are joined and
paragraph breaks inserted at natural speech boundaries. No word is altered, added, removed
or reordered; no punctuation or capitalisation is invented. The script asserts this by
comparing the whitespace-normalised word sequence before and after.

The formatted file is a READING AID, not a replacement source. The authoritative text stays
in chains/<pid>.md and is what any quote must be checked against.

    cd graph_analysis/phase2_results/human_review_packet
    python -u format_transcript.py C02
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Caption transcripts of lectures lean hard on these to open a new thought. Breaking here
# tracks the speaker's own phrasing rather than imposing a structure on them.
MARKERS = (
    "so",
    "okay",
    "ok",
    "now",
    "right",
    "and so",
    "but",
    "then",
    "well",
    "the next",
    "let's",
    "lets",
    "anyway",
    "alright",
    "finally",
    "in fact",
)
MIN_WORDS = 110  # don't break before this many words -- keeps paragraphs substantial


def reflow(text: str) -> str:
    words = text.split()
    out: list[list[str]] = [[]]
    for w in words:
        cur = out[-1]
        if len(cur) >= MIN_WORDS and w.lower() in MARKERS:
            out.append([w])
        else:
            cur.append(w)
    return "\n\n".join(" ".join(p) for p in out if p)


def main(pid: str) -> None:
    src = HERE / "chains" / f"{pid}.md"
    if not src.exists():
        raise SystemExit(f"FATAL: {src} not found. Run from the packet directory.")
    doc = src.read_text(encoding="utf-8")
    if "## Full source text" not in doc:
        raise SystemExit(f"FATAL: {src} has no '## Full source text' section.")

    head, rest = doc.split("## Full source text", 1)
    body = rest.split("```", 2)[1]

    formatted = reflow(body)

    # The whole point: prove nothing but whitespace moved.
    assert body.split() == formatted.split(), (
        "REFLOW ALTERED THE WORD SEQUENCE -- aborting"
    )

    paras = formatted.count("\n\n") + 1
    url = re.search(r"\*\*URL:\*\*\s*(\S+)", head)
    title = re.search(r"\*\*Source:\*\*\s*(.+)", head)

    out = HERE / "chains" / f"{pid}_transcript_formatted.md"
    out.write_text(
        f"# {pid} -- reading aid, reflowed transcript\n\n"
        f"**Source:** {title.group(1).strip() if title else '(unknown)'}\n\n"
        f"**URL:** {url.group(1) if url else '(unknown)'}\n\n"
        f"🔴 **This is a READING AID, not the source.** The authoritative text is\n"
        f"`chains/{pid}.md`, and every quote you record must be checked against it.\n\n"
        f"**What changed:** whitespace only. {len(body.split()):,} words were re-joined from\n"
        f"hard-wrapped caption lines into {paras} paragraphs, broken at the speaker's own\n"
        f"discourse markers ({', '.join(MARKERS[:6])}, ...) after at least {MIN_WORDS} words.\n\n"
        f"**What did NOT change:** no word altered, added, removed or reordered. No punctuation\n"
        f"or capitalisation invented -- the source auto-captions carry almost none, and adding\n"
        f"any would be the script deciding where sentences end. The script asserts the word\n"
        f"sequence is identical before and after and refuses to write this file otherwise.\n\n"
        f"---\n\n{formatted}\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {out.name}: {len(body.split()):,} words -> {paras} paragraphs")
    print(f"mean paragraph: {len(body.split()) // paras} words")
    print("word-sequence identity asserted: PASS")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "C02")
