"""Force a .tex file to pure ASCII.

Why: Overleaf's editor renders a file as "binary" and refuses to edit it when it
contains characters outside the Basic Multilingual Plane (astral-plane, 4-byte
UTF-8). One LARGE RED CIRCLE (U+1F534) in a comment was enough. Non-ASCII BMP
characters compile fine under inputenc utf8 but are stripped here too, so the
question cannot come back.

Only comment/prose characters are rewritten; no LaTeX macro is touched.
"""

import sys
import unicodedata
from pathlib import Path

REPLACEMENTS = {
    "\U0001f534": "!!",  # LARGE RED CIRCLE -> attention marker in comments
    "\U0001f512": "[LOCKED]",  # LOCK
    "§": "Sec.",  # SECTION SIGN
    "—": "---",  # EM DASH
    "–": "--",  # EN DASH
    "→": "->",  # RIGHTWARDS ARROW
    "↔": "<->",
    "≥": ">=",
    "≤": "<=",
    "×": "x",  # MULTIPLICATION SIGN
    "≈": "~=",
    "…": "...",
    "±": "+/-",
    "−": "-",  # MINUS SIGN
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    " ": " ",  # NBSP
}

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
before = sum(1 for ch in text if ord(ch) > 127)

for src, dst in REPLACEMENTS.items():
    text = text.replace(src, dst)

leftover = sorted({ch for ch in text if ord(ch) > 127})
if leftover:
    print("  UNMAPPED characters remain -- add them to REPLACEMENTS:")
    for ch in leftover:
        print(f"    U+{ord(ch):04X} {unicodedata.name(ch, '?')}  x{text.count(ch)}")
    sys.exit(1)

path.write_text(text, encoding="utf-8", newline="\n")
after = sum(1 for ch in path.read_text(encoding="utf-8") if ord(ch) > 127)
print(f"  {path.name}: {before} non-ASCII -> {after}; LF line endings; pure ASCII")

# prove it: the file must round-trip through strict ASCII
raw = path.read_bytes()
raw.decode("ascii")
assert b"\x00" not in raw, "NUL byte present"
assert b"\r" not in raw, "CR present"
print(f"  verified: decodes as strict ASCII, no NUL, no CR, {len(raw)} bytes")
