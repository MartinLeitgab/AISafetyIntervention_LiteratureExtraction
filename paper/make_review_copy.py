#!/usr/bin/env python3
"""Build an anonymized review copy of the manuscript for external model reviewers.

The four rendered [GAP:] blocks and the author placeholder are team decisions still open.
A reviewer reads them as an unfinished draft, so the review copy replaces them with the
conventions of an anonymous submission -- and with nothing that asserts an unsettled fact:

    author placeholder      -> "Anonymous Authors"
    release URL gap         -> "an anonymized repository, linked after review"
    Acknowledgments section -> removed (anonymous submissions carry none)
    AI-drafting scope gap   -> removed; the disclosure sentence itself stays

Every replacement must match exactly once, or the script stops. The output goes to the
path given; the manuscript is never edited.

    python -u make_review_copy.py <paper_v4.tex> <out.tex>
"""

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    if not src.is_file():
        raise SystemExit(f"FATAL: manuscript not found: {src}")
    s = src.read_text(encoding="ascii")

    rules = [
        (r"\\author\{Author List Placeholder\}", r"\\author{Anonymous Authors}"),
        (
            r"\\OPEN\{\[GAP: release URL[^\]]*\]\}",
            r"an anonymized repository, linked after review",
        ),
        (
            r"%{20,}\n\\section\*\{Acknowledgments\}.*?(?=%{20,}\n\\section\*\{Impact Statement\})",
            "",
        ),
        (r"\\OPEN\{\[GAP: scope of the drafting claim[^\]]*\]\}\n", ""),
    ]
    for pat, rep in rules:
        new, n = re.subn(pat, rep, s, flags=re.S)
        if n != 1:
            raise SystemExit(
                f"FATAL: pattern matched {n} times, expected 1: {pat[:60]}"
            )
        s = new
        print(f"applied: {pat[:60]}")

    rendered = [
        line
        for line in s.split("\n")
        if not line.lstrip().startswith("%") and "[GAP:" in line
    ]
    if rendered or "Placeholder" in s.replace("% ", ""):
        raise SystemExit(f"FATAL: rendered gap left in review copy: {rendered[:3]}")
    out.write_text(s, encoding="ascii", newline="\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
